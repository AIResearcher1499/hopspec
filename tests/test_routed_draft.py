"""Structure-routed drafting: FSM, scoped lookup, routers, and the replay
integration. CPU only, no network.

The two regressions worth naming here:

- the FSM must stop at a scaffold literal's boundary and never own the next
  CONTENT token's leading space (spec §4 — the bug that silently excluded the
  most important token in the project from every measurement);
- the lookup's scope must be enforced on the MATCH, not just on the copy:
  an identical n-gram in generated text is a different claim (P3) and must be
  ignored.
"""

import pytest
import torch

from hopspec.data.agent_pipeline import MockLLM, run_react_trajectory
from hopspec.data.collect import trajectory_to_record
from hopspec.data.schema import NO_PRIOR_HOP_DISTANCE, SegmentType
from hopspec.infer.chained_eval import (
    ChainedSpeculator,
    replay_record,
    summarize_rounds_by_source,
)
from hopspec.infer.routed_draft import (
    ACTION_PREFIX,
    DEFAULT_STEP_OPENING,
    EntropyRoutedProposer,
    RoutedProposer,
    ScaffoldFSM,
    ScopedLookup,
    fit_scaffold_fsm,
    last_token_entropy,
    scaffold_stats,
)
from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel

from conftest import HIDDEN_SIZE, ROUTED_VOCAB_SIZE, TinyTargetModel

GAMMA = 4
QUESTION = int(SegmentType.QUESTION)
PASSAGE = int(SegmentType.RETRIEVED_PASSAGE)
THOUGHT = int(SegmentType.THOUGHT)
TEMPLATE = int(SegmentType.TEMPLATE)


# ---- helpers ----

def routed_speculator(with_draft=True):
    tiny = TinyTargetModel(vocab_size=ROUTED_VOCAB_SIZE)
    draft = None
    if with_draft:
        config = HopSpecDraftConfig(
            target_hidden_size=HIDDEN_SIZE, vocab_size=ROUTED_VOCAB_SIZE,
            draft_hidden_size=32, num_heads=4,
        )
        draft = HopSpecDraftModel.from_target_embedding(
            config, tiny.get_input_embeddings()
        )
    return ChainedSpeculator(draft, tiny, device="cpu")


def region_starts(record):
    """Positions where a generated region begins (right after an inserted one)."""
    segments = record["segment_type_ids"]
    inserted = {QUESTION, PASSAGE}
    return [
        index for index in range(1, len(segments))
        if segments[index] not in inserted and segments[index - 1] in inserted
    ]


def first_position_where(fsm, record, literal):
    ids, segments = record["input_ids"], record["segment_type_ids"]
    for index in range(2, len(ids)):
        if fsm.next_literal(ids[:index], segments[:index]) == literal:
            return index
    raise AssertionError(f"no prefix of this record puts the FSM in state {literal!r}")


class FakeState:
    def __init__(self, ids, segments, last_logits=None):
        self.ids = list(ids)
        self.seg = list(segments)
        self.last_logits = last_logits


class FakeSource:
    """Stands in for scaffold/lookup with a fixed answer."""

    def __init__(self, span):
        self.span = list(span)
        self.calls = 0

    def next_span(self, ids, segments, max_tokens):
        self.calls += 1
        return self.span[:max_tokens]

    def propose(self, ids, segments, max_tokens):
        self.calls += 1
        return self.span[:max_tokens]


class FakeNeural:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.calls = []

    def propose(self, context_ids, num_tokens, forced_prefix=None):
        forced = list(forced_prefix or [])
        self.calls.append((len(context_ids), num_tokens, forced))
        return (forced + self.tokens)[:num_tokens]


# ---- ScaffoldFSM ----

def test_fsm_proposes_the_step_opening_after_a_passage(routed_records, routed_tokenizer):
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    starts = [p for p in region_starts(record) if segments[p - 1] == PASSAGE]
    assert starts, "the fixture must contain a passage followed by generation"
    for position in starts:
        span = fsm.next_span(ids[:position], segments[:position], GAMMA)
        assert span == routed_tokenizer.encode(DEFAULT_STEP_OPENING)
        assert routed_tokenizer.decode(span) == DEFAULT_STEP_OPENING
        # and it is what the recorded trajectory actually does next
        assert span == ids[position:position + len(span)]


def test_fsm_never_owns_the_next_content_tokens_leading_space(
    routed_records, routed_tokenizer
):
    """Spec §4: BPE attaches the leading space to the next word. A scaffold
    span that swallowed it would own a CONTENT token's first character."""
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = [p for p in region_starts(record) if segments[p - 1] == PASSAGE][0]
    span = fsm.next_span(ids[:position], segments[:position], GAMMA)
    following = ids[position + len(span)]
    # the regression is only exercised if the next real token owns a space
    assert routed_tokenizer.decode([following]).startswith(" ")
    assert following not in span
    assert not any(routed_tokenizer.decode([token]).startswith(" ") for token in span)


def test_fsm_stops_at_the_literal_boundary_everywhere(routed_records, routed_tokenizer):
    fsm = ScaffoldFSM(routed_tokenizer)
    for record in routed_records:
        ids, segments = record["input_ids"], record["segment_type_ids"]
        for index in range(2, len(ids)):
            literal = fsm.next_literal(ids[:index], segments[:index])
            if literal is None:
                continue
            span = fsm.next_span(ids[:index], segments[:index], 8)
            assert routed_tokenizer.decode(span) == literal


def test_fsm_proposes_the_action_prefix_after_the_thought_newline(
    routed_records, routed_tokenizer
):
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = first_position_where(fsm, record, ACTION_PREFIX)
    span = fsm.next_span(ids[:position], segments[:position], GAMMA)
    assert routed_tokenizer.decode(span) == ACTION_PREFIX
    assert span == ids[position:position + len(span)]


def test_fsm_proposes_a_newline_after_the_closing_bracket(
    routed_records, routed_tokenizer
):
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = first_position_where(fsm, record, "\n")
    assert routed_tokenizer.decode(ids[:position]).endswith("]")
    span = fsm.next_span(ids[:position], segments[:position], GAMMA)
    assert span == ids[position:position + 1]


def test_fsm_defers_inside_thought_content(routed_records, routed_tokenizer):
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = first_position_where(fsm, record, DEFAULT_STEP_OPENING)
    inside = position + 2  # "Thought:" + one content token
    assert routed_tokenizer.decode(ids[:inside]).endswith(" I")
    assert fsm.next_literal(ids[:inside], segments[:inside]) is None
    assert fsm.next_span(ids[:inside], segments[:inside], GAMMA) == []


def test_fsm_adds_the_newline_only_when_the_context_lacks_one(
    routed_records, routed_tokenizer
):
    """The newline may already live inside the last inserted token (Qwen
    merges "?\\n"); proposing another writes a line the target never writes."""
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    after_question = [p for p in region_starts(record) if segments[p - 1] == QUESTION][0]
    after_passage = [p for p in region_starts(record) if segments[p - 1] == PASSAGE][0]
    assert not routed_tokenizer.decode(ids[:after_question]).endswith("\n")
    assert routed_tokenizer.decode(ids[:after_passage]).endswith("\n")
    assert fsm.next_literal(ids[:after_question], segments[:after_question]) == (
        "\n" + DEFAULT_STEP_OPENING
    )
    assert fsm.next_literal(ids[:after_passage], segments[:after_passage]) == (
        DEFAULT_STEP_OPENING
    )


def test_fsm_truncates_to_the_budget_and_finishes_next_round(
    routed_records, routed_tokenizer
):
    record = routed_records[0]
    fsm = ScaffoldFSM(routed_tokenizer)
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = [p for p in region_starts(record) if segments[p - 1] == QUESTION][0]
    span = fsm.next_span(ids[:position], segments[:position], 1)
    assert span == routed_tokenizer.encode("\n")
    rest = fsm.next_span(ids[:position + 1], segments[:position + 1], GAMMA)
    assert routed_tokenizer.decode(rest) == DEFAULT_STEP_OPENING


def test_fsm_proposes_nothing_without_context(routed_tokenizer):
    fsm = ScaffoldFSM(routed_tokenizer)
    assert fsm.next_span([], [], GAMMA) == []
    assert fsm.next_span([1, 2], [TEMPLATE, TEMPLATE], 0) == []


def test_fsm_rejects_a_literal_ending_in_a_space(routed_tokenizer):
    with pytest.raises(ValueError, match="space"):
        ScaffoldFSM(routed_tokenizer, step_opening="Thought: ")


def test_fsm_rejects_a_literal_that_does_not_round_trip():
    class LossyTokenizer:
        def encode(self, text):
            return [1]

        def decode(self, ids):
            return "something else"

    with pytest.raises(ValueError, match="round-trip"):
        ScaffoldFSM(LossyTokenizer())


def test_fsm_action_verb_option_proposes_the_whole_action(
    routed_records, routed_tokenizer
):
    record = routed_records[0]
    plain = ScaffoldFSM(routed_tokenizer)
    betting = ScaffoldFSM(routed_tokenizer, action_verb="Search")
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = first_position_where(plain, record, ACTION_PREFIX)
    span = betting.next_span(ids[:position], segments[:position], 8)
    assert routed_tokenizer.decode(span) == "Action: Search["


# ---- fitting the grammar the agent actually writes ----

ACTION_ONLY_RESPONSES = [
    "Action: Search[book author]",
    "Action: Finish[Paris]",
]


def action_only_records(retriever, tokenizer, num_records=2):
    records = []
    for index in range(num_records):
        trajectory = run_react_trajectory(
            f"Question {index}: where was the author born?",
            MockLLM(list(ACTION_ONLY_RESPONSES)), retriever,
        )
        records.append(trajectory_to_record(trajectory, f"a{index}", tokenizer))
    return records


def test_scaffold_stats_counts_the_observed_openings(retriever, routed_tokenizer):
    records = action_only_records(retriever, routed_tokenizer)
    openings, verbs = scaffold_stats(records)
    assert openings[ACTION_PREFIX] == 4  # two steps per record
    assert openings[DEFAULT_STEP_OPENING] == 0
    assert verbs == {"Search": 2, "Finish": 2}


def test_fit_scaffold_fsm_follows_the_majority_opening(retriever, routed_tokenizer):
    """The 1.7B agent skips "Thought:" — assuming the spec grammar would score
    the scaffold arm at ~0 for a reason unrelated to the idea under test."""
    records = action_only_records(retriever, routed_tokenizer)
    fsm, stats = fit_scaffold_fsm(records, routed_tokenizer)
    assert stats["step_opening"] == ACTION_PREFIX
    assert stats["action_verb"] is None
    record = records[0]
    ids, segments = record["input_ids"], record["segment_type_ids"]
    position = [p for p in region_starts(record) if segments[p - 1] == PASSAGE][0]
    span = fsm.next_span(ids[:position], segments[:position], GAMMA)
    assert routed_tokenizer.decode(span) == ACTION_PREFIX
    assert span == ids[position:position + len(span)]


def test_fit_scaffold_fsm_can_bet_on_the_majority_verb(retriever, routed_tokenizer):
    records = action_only_records(retriever, routed_tokenizer)
    _fsm, stats = fit_scaffold_fsm(records, routed_tokenizer, use_verb=True)
    assert stats["action_verb"] in {"Search", "Finish"}


def test_fit_scaffold_fsm_falls_back_to_the_spec_grammar(routed_tokenizer):
    fsm, stats = fit_scaffold_fsm([], routed_tokenizer)
    assert stats["step_opening"] == DEFAULT_STEP_OPENING
    assert fsm.step_opening == DEFAULT_STEP_OPENING


# ---- ScopedLookup ----
#
# Positions 0-3 are an inserted passage, 4+ generated. The needle (7, 8)
# occurs both in scope and outside it.

SCOPED_IDS = [7, 8, 30, 31, 7, 8, 40, 7, 8]
SCOPED_SEGMENTS = [PASSAGE, PASSAGE, PASSAGE, PASSAGE, THOUGHT, THOUGHT, THOUGHT,
                   THOUGHT, THOUGHT]


def test_lookup_copies_the_continuation_of_an_in_scope_match():
    lookup = ScopedLookup()
    assert lookup.propose(SCOPED_IDS, SCOPED_SEGMENTS, 2) == [30, 31]


def test_lookup_ignores_an_identical_ngram_outside_the_scope():
    """Provenance is the claim (P3): the same tokens in generated text are a
    different source and must not be copied."""
    lookup = ScopedLookup()
    generated_only = [THOUGHT] * len(SCOPED_IDS)
    assert lookup.propose(SCOPED_IDS, generated_only, 2) == []


def test_lookup_stops_at_the_scope_boundary():
    ids = [7, 8, 30, 31, 7, 8]
    segments = [PASSAGE, PASSAGE, PASSAGE, THOUGHT, THOUGHT, THOUGHT]
    assert ScopedLookup().propose(ids, segments, 4) == [30]


def test_lookup_respects_the_budget():
    assert ScopedLookup().propose(SCOPED_IDS, SCOPED_SEGMENTS, 1) == [30]
    assert ScopedLookup().propose(SCOPED_IDS, SCOPED_SEGMENTS, 0) == []


def test_lookup_is_deterministic_and_takes_the_leftmost_match():
    """Two in-scope copies of the same 3-gram: the leftmost wins, always —
    the prompt-lookup reference implementation's rule, not a paper's."""
    ids = [1, 2, 3, 50, 9, 1, 2, 3, 60, 9, 1, 2, 3]
    segments = [PASSAGE] * 10 + [THOUGHT] * 3
    lookup = ScopedLookup()
    assert lookup.propose(ids, segments, 1) == [50] == lookup.propose(ids, segments, 1)


def test_lookup_backs_off_to_a_shorter_ngram():
    segments = [PASSAGE] * 5 + [THOUGHT, THOUGHT]
    # the 3-gram (0, 4, 6) never occurred and neither did the 2-gram (4, 6)
    assert ScopedLookup().propose([5, 6, 99, 0, 0, 4, 6], segments, 2) == []
    # the 2-gram (5, 6) did, in scope
    assert ScopedLookup().propose([5, 6, 99, 0, 0, 5, 6], segments, 2) == [99, 0]


def test_lookup_returns_nothing_without_a_match():
    assert ScopedLookup().propose([1, 2, 3], [PASSAGE] * 3, 2) == []


def test_lookup_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="equal lengths"):
        ScopedLookup().propose([1, 2, 3], [PASSAGE, PASSAGE], 2)


# ---- RoutedProposer ----

def test_router_prefers_the_scaffold():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    lookup = FakeSource([90, 91])
    neural = FakeNeural([70, 71])
    router = RoutedProposer(state, scaffold=FakeSource([50, 51]),
                            lookup=lookup, neural=neural)
    assert router.propose(state.ids, GAMMA) == [50, 51]
    assert router.last_source == "scaffold"
    assert router.last_token_sources == ["scaffold", "scaffold"]
    assert lookup.calls == 0 and neural.calls == []


def test_router_prefers_the_lookup_over_the_neural_draft():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    neural = FakeNeural([70, 71])
    router = RoutedProposer(state, scaffold=FakeSource([]),
                            lookup=FakeSource([90, 91]), neural=neural)
    assert router.propose(state.ids, GAMMA) == [90, 91]
    assert router.last_token_sources == ["lookup", "lookup"]
    assert neural.calls == []


def test_router_falls_through_to_the_neural_draft():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    router = RoutedProposer(state, scaffold=FakeSource([]), lookup=FakeSource([]),
                            neural=FakeNeural([70, 71, 72, 73, 74]))
    assert router.propose(state.ids, GAMMA) == [70, 71, 72, 73]
    assert router.last_source == "neural"
    assert router.last_token_sources == ["neural"] * GAMMA


def test_router_single_source_per_round_by_default():
    """A short scaffold span owns the whole round unless chaining is on —
    that difference is a measurement, not an implementation detail."""
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    neural = FakeNeural([70, 71, 72, 73])
    router = RoutedProposer(state, scaffold=FakeSource([50]),
                            lookup=FakeSource([90]), neural=neural)
    assert router.propose(state.ids, GAMMA) == [50]
    assert neural.calls == []


def test_router_chains_sources_up_to_gamma():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    neural = FakeNeural([70, 71, 72, 73])
    router = RoutedProposer(state, scaffold=FakeSource([50]),
                            lookup=FakeSource([90]), neural=neural, chain=True)
    assert router.propose(state.ids, GAMMA) == [50, 90, 70, 71]
    assert router.last_token_sources == ["scaffold", "lookup", "neural", "neural"]
    # the neural source is handed the earlier tokens so it chains on its own
    # predicted features over them
    assert neural.calls == [(3, GAMMA, [50, 90])]


def test_router_never_exceeds_the_budget():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    router = RoutedProposer(state, scaffold=FakeSource([1, 2, 3, 4, 5, 6]))
    proposed = router.propose(state.ids, 2)
    assert len(proposed) == 2 == len(router.last_token_sources)


def test_lookup_only_arm_proposes_nothing_on_a_miss():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, THOUGHT])
    router = RoutedProposer(state, lookup=ScopedLookup())
    assert router.propose(state.ids, GAMMA) == []
    assert router.last_source == "none"


def test_router_needs_at_least_one_source():
    with pytest.raises(ValueError, match="at least one source"):
        RoutedProposer(FakeState([], []))


def test_router_rejects_out_of_sync_labels():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE])
    router = RoutedProposer(state, lookup=ScopedLookup())
    with pytest.raises(ValueError, match="out of sync"):
        router.propose(state.ids, GAMMA)


# ---- EntropyRoutedProposer ----

def peaked_logits():
    logits = torch.full((ROUTED_VOCAB_SIZE,), -20.0)
    logits[3] = 20.0
    return logits


def flat_logits():
    return torch.zeros(ROUTED_VOCAB_SIZE)


def test_last_token_entropy_matches_the_distribution():
    peaked = last_token_entropy(FakeState([], [], peaked_logits()))
    assert peaked == pytest.approx(0.0, abs=1e-5)
    uniform = last_token_entropy(FakeState([], [], flat_logits()))
    assert uniform == pytest.approx(
        torch.tensor(float(ROUTED_VOCAB_SIZE)).log().item(), abs=1e-5
    )
    assert last_token_entropy(FakeState([], [], None)) is None


def test_entropy_router_takes_the_lookup_below_the_threshold():
    state = FakeState(SCOPED_IDS, SCOPED_SEGMENTS, peaked_logits())
    neural = FakeNeural([70, 71])
    router = EntropyRoutedProposer(state, lookup=ScopedLookup(), neural=neural,
                                   threshold=1.0)
    assert router.propose(state.ids, 2) == [30, 31]
    assert router.last_source == "lookup"
    assert neural.calls == []


def test_entropy_router_takes_the_neural_draft_above_the_threshold():
    state = FakeState(SCOPED_IDS, SCOPED_SEGMENTS, flat_logits())
    router = EntropyRoutedProposer(state, lookup=ScopedLookup(),
                                   neural=FakeNeural([70, 71]), threshold=1.0)
    assert router.propose(state.ids, 2) == [70, 71]
    assert router.last_token_sources == ["neural", "neural"]


def test_entropy_router_falls_back_when_the_lookup_misses():
    state = FakeState([1, 2, 3], [PASSAGE, PASSAGE, PASSAGE], peaked_logits())
    router = EntropyRoutedProposer(state, lookup=ScopedLookup(),
                                   neural=FakeNeural([70, 71]), threshold=1.0)
    assert router.propose(state.ids, 2) == [70, 71]
    assert router.last_source == "neural"


def test_entropy_router_uses_cached_logits_and_no_extra_target_forward(routed_records):
    """The entropy is read off the logits row the verifier already computed;
    a router that paid for its own target forward would not be free."""
    speculator = routed_speculator()
    record = routed_records[0]
    speculator.start()
    speculator.append_committed(
        record["input_ids"][:8], record["segment_type_ids"][:8],
        record["recency_bucket_ids"][:8],
    )
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)
    calls = []
    original = speculator.runner.extend

    def counting_extend(new_ids):
        calls.append(list(new_ids))
        return original(new_ids)

    speculator.runner.extend = counting_extend
    router = EntropyRoutedProposer(speculator, lookup=ScopedLookup(),
                                   threshold=float("inf"))
    router.propose(speculator.ids, GAMMA)
    assert calls == []
    assert router.last_entropy is not None


def test_entropy_router_needs_a_source():
    with pytest.raises(ValueError, match="needs a lookup or a neural source"):
        EntropyRoutedProposer(FakeState([], []))


# ---- integration: replay through the frozen evaluator ----

def build_proposer(name, speculator, tokenizer, records):
    lookup = ScopedLookup()
    fsm, _stats = fit_scaffold_fsm(records, tokenizer)
    if name == "scaffold":
        return RoutedProposer(speculator, scaffold=fsm)
    if name == "lookup":
        return RoutedProposer(speculator, lookup=lookup)
    if name == "routed":
        return RoutedProposer(speculator, scaffold=fsm, lookup=lookup,
                              neural=speculator)
    if name == "chained":
        return RoutedProposer(speculator, scaffold=fsm, lookup=lookup,
                              neural=speculator, chain=True)
    if name == "entropy":
        return EntropyRoutedProposer(speculator, lookup=lookup, neural=speculator,
                                     threshold=1.0)
    raise AssertionError(name)


@pytest.mark.parametrize(
    "source", ["scaffold", "lookup", "routed", "chained", "entropy"]
)
def test_replay_over_every_source_restores_the_rails(
    routed_records, routed_tokenizer, source
):
    speculator = routed_speculator()
    record = routed_records[0]
    proposer = build_proposer(source, speculator, routed_tokenizer, routed_records)
    rounds = replay_record(record, speculator, gamma=GAMMA, proposer=proposer)
    assert rounds
    assert speculator.ids == record["input_ids"]
    for row in rounds:
        assert 0 <= row["accepted"] <= GAMMA
        assert 1 <= row["emitted"] <= GAMMA + 1
        assert len(row["token_sources"]) <= GAMMA
        assert row["accepted"] <= len(row["token_sources"])
        assert row["source"] in {"scaffold", "lookup", "neural", "none"}


def replay_against_the_recorded_target(record, speculator, proposer, gamma=GAMMA):
    """Replay with a verifier that reproduces the record's own trajectory.

    The stand-in target is a random network, so realized acceptance against it
    is noise — that number is what the experiment measures, not something a
    unit test can assert. Substituting the recorded continuation makes the
    round's accepted count the proposer's true hit count against the
    trajectory the agent actually wrote, which IS testable on CPU. Acceptance
    stops at the first mismatch, so only the recorded prefix ever matters.
    """
    recorded = record["input_ids"]

    def recorded_next_tokens(context_ids, proposed):
        start = len(context_ids)
        return [
            recorded[start + offset] if start + offset < len(recorded)
            else recorded[-1]
            for offset in range(len(proposed) + 1)
        ]

    speculator.next_tokens = recorded_next_tokens
    return replay_record(record, speculator, gamma=gamma, proposer=proposer)


def test_replay_with_the_scaffold_accepts_template_spans(
    routed_records, routed_tokenizer
):
    """The FSM's whole claim: prompt-lookup cannot predict the scaffold
    because it follows novel passage text, but the grammar can."""
    speculator = routed_speculator()
    record = routed_records[0]
    proposer = build_proposer("scaffold", speculator, routed_tokenizer, routed_records)
    rounds = replay_against_the_recorded_target(record, speculator, proposer)
    by_source = summarize_rounds_by_source(rounds)
    assert set(by_source) <= {"scaffold", "none"}
    assert by_source["scaffold"]["proposed"] > 0
    assert by_source["scaffold"]["accepted"] > 0
    assert speculator.ids == record["input_ids"]


def test_scoped_lookup_cannot_cover_the_scaffold(routed_records, routed_tokenizer):
    """P5's surprise, as a test: the scaffold follows novel passage text, so
    no n-gram match for it exists. Whatever the lookup arm wins, it is not
    these tokens — which is why the FSM is load-bearing, not a nice-to-have."""
    speculator = routed_speculator()
    record = routed_records[0]
    scaffold_rounds = replay_against_the_recorded_target(
        record, speculator,
        build_proposer("scaffold", speculator, routed_tokenizer, routed_records),
    )
    speculator = routed_speculator()
    lookup_rounds = replay_against_the_recorded_target(
        record, speculator,
        build_proposer("lookup", speculator, routed_tokenizer, routed_records),
    )
    scaffold_accepted = summarize_rounds_by_source(scaffold_rounds)["scaffold"]
    lookup_summary = summarize_rounds_by_source(lookup_rounds)
    lookup_accepted = lookup_summary.get("lookup", {"accepted": 0})["accepted"]
    assert scaffold_accepted["accepted"] > lookup_accepted


def test_replay_source_labels_are_only_the_sources_given(
    routed_records, routed_tokenizer
):
    speculator = routed_speculator()
    proposer = build_proposer("lookup", speculator, routed_tokenizer, routed_records)
    rounds = replay_record(routed_records[0], speculator, gamma=GAMMA,
                           proposer=proposer)
    assert set(summarize_rounds_by_source(rounds)) <= {"lookup", "none"}


def test_replay_is_deterministic(routed_records, routed_tokenizer):
    record = routed_records[0]
    runs = []
    for _ in range(2):
        speculator = routed_speculator()
        proposer = build_proposer("routed", speculator, routed_tokenizer,
                                  routed_records)
        runs.append(replay_record(record, speculator, gamma=GAMMA,
                                  proposer=proposer))
    assert runs[0] == runs[1]


# ---- a speculator without a draft model ----

def test_model_free_speculator_verifies_but_refuses_to_propose(routed_records):
    speculator = routed_speculator(with_draft=False)
    record = routed_records[0]
    proposer = RoutedProposer(speculator, lookup=ScopedLookup())
    rounds = replay_record(record, speculator, gamma=GAMMA, proposer=proposer)
    assert rounds
    assert speculator.ids == record["input_ids"]
    with pytest.raises(RuntimeError, match="no draft model"):
        speculator.propose(speculator.ids, GAMMA)


def commit_prefix(speculator, record, length=8):
    speculator.start()
    speculator.append_committed(
        record["input_ids"][:length], record["segment_type_ids"][:length],
        record["recency_bucket_ids"][:length],
    )
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)


def test_forced_prefix_is_returned_verbatim(routed_records):
    speculator = routed_speculator()
    commit_prefix(speculator, routed_records[0])
    forced = [5, 6]
    proposed = speculator.propose(speculator.ids, GAMMA, forced_prefix=forced)
    assert proposed[:2] == forced
    assert len(proposed) == GAMMA


def test_forced_prefix_cannot_exceed_the_budget(routed_records):
    speculator = routed_speculator()
    commit_prefix(speculator, routed_records[0])
    with pytest.raises(ValueError, match="longer than the proposal budget"):
        speculator.propose(speculator.ids, 1, forced_prefix=[1, 2, 3])
