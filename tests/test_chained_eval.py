import pytest
import torch

from hopspec.data.schema import NO_PRIOR_HOP_DISTANCE, NUM_RECENCY_BUCKETS
from hopspec.infer.chained_eval import (
    ChainedSpeculator,
    chat_prompt_ids,
    chat_prompt_seam_ok,
    constant_gamma_policy,
    record_means,
    replay_record,
    summarize_rounds,
    summarize_rounds_by_source,
)
from hopspec.infer.routed_draft import RoutedProposer, ScopedLookup
from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel

from conftest import (
    HIDDEN_SIZE,
    ROUTED_VOCAB_SIZE,
    VOCAB_SIZE,
    FakeChatTokenizer,
    TinyTargetModel,
)

GAMMA = 3


def make_speculator(hop_signal_enabled=True):
    tiny = TinyTargetModel()
    config = HopSpecDraftConfig(
        target_hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
        draft_hidden_size=32, num_heads=4,
    )
    draft = HopSpecDraftModel.from_target_embedding(config, tiny.get_input_embeddings())
    return ChainedSpeculator(draft, tiny, device="cpu",
                             hop_signal_enabled=hop_signal_enabled)


def commit_prefix(speculator, record, length=8):
    speculator.start()
    speculator.append_committed(
        record["input_ids"][:length],
        record["segment_type_ids"][:length],
        record["recency_bucket_ids"][:length],
    )
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)


# ---- CachedTargetRunner: must equal an uncached full forward ----

def full_forward(model, ids):
    tensor = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    output_hidden_states=True)
    return out.hidden_states[-1][0].float(), out.logits[0]


def test_cache_matches_uncached_features_and_logits():
    from hopspec.infer.chained_eval import CachedTargetRunner

    tiny = TinyTargetModel()
    ids = [3, 9, 14, 2, 7, 11]
    runner = CachedTargetRunner(tiny, "cpu")
    hidden_a, logits_a = runner.extend(ids[:4])
    hidden_b, logits_b = runner.extend(ids[4:])
    hidden = torch.cat([hidden_a, hidden_b])
    expected_hidden, expected_logits = full_forward(tiny, ids)
    assert torch.allclose(hidden, expected_hidden, atol=1e-4)
    assert torch.allclose(
        torch.cat([logits_a, logits_b]), expected_logits, atol=1e-4
    )


def test_cache_rollback_restores_prefix_results():
    from hopspec.infer.chained_eval import CachedTargetRunner

    tiny = TinyTargetModel()
    runner = CachedTargetRunner(tiny, "cpu")
    runner.extend([3, 9, 14])
    runner.extend([5, 5, 5])            # speculated, to be discarded
    runner.rollback(3)
    hidden, _logits = runner.extend([2, 7])
    expected_hidden, _ = full_forward(tiny, [3, 9, 14, 2, 7])
    assert torch.allclose(hidden, expected_hidden[3:], atol=1e-4)
    assert runner.length == 5


def test_cache_rejects_forward_rollback():
    from hopspec.infer.chained_eval import CachedTargetRunner

    runner = CachedTargetRunner(TinyTargetModel(), "cpu")
    runner.extend([1, 2])
    with pytest.raises(ValueError):
        runner.rollback(5)


def test_speculated_tokens_do_not_persist_in_cache(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    length = speculator.runner.length
    speculator.next_tokens(speculator.ids, [4, 4, 4])
    assert speculator.runner.length == length


def test_committed_features_match_uncached(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0], length=12)
    expected, _ = full_forward(speculator.target, speculator.ids)
    assert torch.allclose(speculator.feats, expected, atol=1e-4)


# ---- constant_gamma_policy ----

def test_constant_policy_same_gamma_everywhere():
    policy = constant_gamma_policy(5)
    assert {policy.gamma_for(b) for b in range(NUM_RECENCY_BUCKETS)} == {5}


def test_constant_policy_allows_large_gamma():
    assert constant_gamma_policy(12).gamma_for(0) == 12


# ---- ChainedSpeculator primitives ----

def test_propose_returns_requested_count(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    proposed = speculator.propose(speculator.ids, GAMMA)
    assert len(proposed) == GAMMA
    assert all(0 <= token < VOCAB_SIZE for token in proposed)


def test_propose_rejects_out_of_sync_context(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    with pytest.raises(ValueError):
        speculator.propose(speculator.ids[:-1], GAMMA)


def test_propose_is_deterministic(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    first = speculator.propose(speculator.ids, GAMMA)
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)
    second = speculator.propose(speculator.ids, GAMMA)
    assert first == second


def test_verify_returns_len_plus_one(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    proposed = [1, 2, 3]
    continuation = speculator.next_tokens(speculator.ids, proposed)
    assert len(continuation) == len(proposed) + 1


def test_verify_matches_target_greedy(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    continuation = speculator.next_tokens(speculator.ids, [])
    tensor = torch.tensor([speculator.ids])
    logits = speculator.target(input_ids=tensor).logits
    assert continuation == [int(logits[0, -1].argmax())]


def test_commit_extends_state_and_features(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0])
    speculator.begin_round(0)
    length = len(speculator.ids)
    emitted = speculator.commit(speculator.ids + [7, 8])
    assert emitted == 2
    assert len(speculator.ids) == length + 2
    assert speculator.feats.shape[0] == length + 2
    assert len(speculator.seg) == len(speculator.buckets) == length + 2


def test_truncate_restores_length(records):
    speculator = make_speculator()
    commit_prefix(speculator, records[0], length=10)
    speculator.truncate_to(6)
    assert len(speculator.ids) == 6
    assert speculator.feats.shape[0] == 6


# ---- replay_record ----

@pytest.mark.parametrize("hop_signal_enabled", [True, False])
def test_replay_produces_rounds_and_restores_rails(records, hop_signal_enabled):
    speculator = make_speculator(hop_signal_enabled)
    rounds = replay_record(records[0], speculator, gamma=GAMMA)
    assert rounds
    # After replay the committed state is exactly the recorded trajectory.
    assert speculator.ids == records[0]["input_ids"]
    for row in rounds:
        assert 0 <= row["accepted"] <= GAMMA
        assert 1 <= row["emitted"] <= GAMMA + 1
        assert row["distance"] >= 0


def test_replay_logs_distance_zero_after_each_hop(records):
    speculator = make_speculator()
    rounds = replay_record(records[0], speculator, gamma=GAMMA)
    hops = len(records[0]["hop_boundary_positions"])
    zero_distance_hops = {
        row["hop_index"] for row in rounds if row["distance"] == 0
    }
    assert zero_distance_hops == set(range(hops))


def test_replay_pre_hop_rounds_are_no_prior(records):
    speculator = make_speculator()
    rounds = replay_record(records[0], speculator, gamma=GAMMA)
    pre_hop = [row for row in rounds if row["hop_index"] == -1]
    assert pre_hop
    assert all(row["distance"] == NO_PRIOR_HOP_DISTANCE for row in pre_hop)


def test_replay_distances_advance_within_region(records):
    speculator = make_speculator()
    rounds = replay_record(records[0], speculator, gamma=GAMMA)
    post_hop = [row for row in rounds if row["hop_index"] == 0]
    for previous, current in zip(post_hop, post_hop[1:]):
        if current["distance"] == 0:
            break  # next region
        assert current["distance"] == previous["distance"] + previous["emitted"]


def test_replay_respects_max_rounds_per_region(records):
    speculator = make_speculator()
    rounds = replay_record(records[0], speculator, gamma=1, max_rounds_per_region=1)
    from collections import Counter

    per_region = Counter(
        (row["hop_index"], row["region_tokens"]) for row in rounds
    )
    assert max(per_region.values()) <= 2  # <= 1 per region; key may repeat across regions


# ---- summarize_rounds ----

def test_summarize_rounds_means():
    rounds = [
        {"bucket": 0, "accepted": 1, "emitted": 2},
        {"bucket": 0, "accepted": 3, "emitted": 4},
        {"bucket": 5, "accepted": 0, "emitted": 1},
    ]
    summary = summarize_rounds(rounds)
    assert summary[0] == {"rounds": 2, "mean_accepted": 2.0, "mean_emitted": 3.0}
    assert summary[5]["rounds"] == 1


# ---- pluggable proposers: the verifier and the rails must not move ----

def make_proposer(name, speculator):
    if name == "neural":
        return None            # the speculator proposes for itself
    if name == "lookup":
        return RoutedProposer(speculator, lookup=ScopedLookup())
    raise AssertionError(name)


@pytest.mark.parametrize("source", ["neural", "lookup"])
def test_replay_restores_the_rails_for_every_draft_source(records, source):
    speculator = make_speculator()
    proposer = make_proposer(source, speculator)
    rounds = replay_record(records[0], speculator, gamma=GAMMA, proposer=proposer)
    assert rounds
    assert speculator.ids == records[0]["input_ids"]
    for row in rounds:
        assert 0 <= row["accepted"] <= GAMMA
        assert 1 <= row["emitted"] <= GAMMA + 1


def test_replay_defaults_to_the_neural_source(records):
    """The default path is byte-identical to what it was before proposers
    became pluggable — every §11.4 number came off it."""
    speculator = make_speculator()
    rounds = replay_record(records[0], speculator, gamma=GAMMA)
    assert {row["source"] for row in rounds} == {"neural"}
    for row in rounds:
        assert row["token_sources"] == ["neural"] * row["gamma"]


def test_replay_labels_a_model_free_source(records):
    speculator = make_speculator()
    proposer = RoutedProposer(speculator, lookup=ScopedLookup())
    rounds = replay_record(records[0], speculator, gamma=GAMMA, proposer=proposer)
    assert {row["source"] for row in rounds} <= {"lookup", "none"}
    for row in rounds:
        assert set(row["token_sources"]) <= {"lookup"}


# ---- per-source and per-record summaries ----

def test_summarize_by_source_counts_tokens_not_rounds():
    rounds = [
        {"source": "scaffold", "accepted": 2, "bucket": 0,
         "token_sources": ["scaffold", "scaffold", "neural"]},
        {"source": "lookup", "accepted": 0, "bucket": 0,
         "token_sources": ["lookup", "lookup"]},
    ]
    summary = summarize_rounds_by_source(rounds)
    assert summary["scaffold"] == {
        "rounds": 1, "proposed": 2, "accepted": 2, "acceptance": 1.0
    }
    assert summary["neural"] == {
        "rounds": 0, "proposed": 1, "accepted": 0, "acceptance": 0.0
    }
    assert summary["lookup"]["rounds"] == 1
    assert summary["lookup"]["acceptance"] == 0.0


def test_summarize_by_source_keeps_template_separate_from_content():
    """Scaffold tokens are TEMPLATE: excluded from decode-phase measurement,
    genuinely emitted in chained replay. Pooling the two is the §3 bug."""
    rounds = [{"source": "scaffold", "accepted": 3, "bucket": 0,
               "token_sources": ["scaffold", "scaffold", "neural"]}]
    summary = summarize_rounds_by_source(rounds)
    assert summary["scaffold"]["accepted"] == 2
    assert summary["neural"]["accepted"] == 1


def test_summarize_by_source_tolerates_rows_without_sources():
    assert summarize_rounds_by_source([{"accepted": 1, "bucket": 0}]) == {
        "neural": {"rounds": 1, "proposed": 0, "accepted": 0, "acceptance": 0.0}
    }


def test_record_means_average_per_record():
    rounds = [
        {"question_id": "q0", "accepted": 1},
        {"question_id": "q0", "accepted": 3},
        {"question_id": "q1", "accepted": 0},
    ]
    assert record_means(rounds) == {"q0": 2.0, "q1": 0.0}


def test_record_means_fall_back_to_the_record_index():
    assert record_means([{"record_index": 4, "accepted": 2}]) == {"4": 2.0}


# =====================================================================
# chat replay mode — prereg docs/prereg-chained-chat-replay-2026-08-28.md
#
# The raw mode feeds the target a document the agent never saw: at collection
# every step went through apply_chat_template(system + user(context so far)).
# That is the spec §4 bug family ("hidden states over a document that never
# existed") relocated into replay. `chat` re-renders the wrapper at each step
# boundary; `raw` stays bit for bit what it was, and the two modes' numbers
# are never comparable (spec §15).
# =====================================================================

PREFIX = [1, 2, 3]
SUFFIX = [4, 5]


def wrapped_speculator(prefix=PREFIX, suffix=SUFFIX, decode=None,
                       vocab_size=VOCAB_SIZE):
    tiny = TinyTargetModel(vocab_size=vocab_size)
    config = HopSpecDraftConfig(
        target_hidden_size=HIDDEN_SIZE, vocab_size=vocab_size,
        draft_hidden_size=32, num_heads=4,
    )
    draft = HopSpecDraftModel.from_target_embedding(config, tiny.get_input_embeddings())
    return ChainedSpeculator(draft, tiny, device="cpu", hop_signal_enabled=False,
                             prompt_prefix_ids=prefix, prompt_suffix_ids=suffix,
                             decode=decode)


# ---- chat_prompt_ids ----

def test_chat_prompt_ids_splits_the_rendered_wrapper(fake_chat_tokenizer):
    prefix, suffix = chat_prompt_ids(fake_chat_tokenizer, "SYS")
    assert fake_chat_tokenizer.decode(prefix) == "SYSTEM:\nSYS\nUSER:\n"
    assert fake_chat_tokenizer.decode(suffix) == "\nASSISTANT:\n"


def test_chat_prompt_ids_falls_back_when_enable_thinking_is_rejected():
    """HFTargetLLM has this branch; so must we, or the two prompts diverge."""
    tokenizer = FakeChatTokenizer(accepts_enable_thinking=False)
    prefix, suffix = chat_prompt_ids(tokenizer, "SYS")
    assert prefix and suffix
    assert tokenizer.decode(suffix) == "\nASSISTANT:\n"


def test_chat_prompt_ids_rejects_a_template_that_drops_the_user_content():
    tokenizer = FakeChatTokenizer(drop_user=True)
    with pytest.raises(ValueError, match="exactly once"):
        chat_prompt_ids(tokenizer, "SYS")


def test_chat_prompt_seam_holds_for_a_clean_context(fake_chat_tokenizer):
    prefix, suffix = chat_prompt_ids(fake_chat_tokenizer, "SYS")
    context = "Question: where was the author born?"
    assert chat_prompt_seam_ok(fake_chat_tokenizer, context, prefix, suffix, "SYS")


def test_chat_prompt_seam_check_catches_a_merging_tokenizer(fake_chat_tokenizer):
    """A context ending in whitespace merges with this stand-in's suffix. The
    guard has to notice — a silent merge shifts every label by a token."""
    prefix, suffix = chat_prompt_ids(fake_chat_tokenizer, "SYS")
    assert not chat_prompt_seam_ok(
        fake_chat_tokenizer, "Question: born?\n", prefix, suffix, "SYS"
    )


# ---- wrapper bookkeeping ----

def test_raw_mode_is_the_default_and_needs_no_wrapper():
    assert make_speculator().replay_mode == "raw"


def test_chat_mode_reports_its_mode():
    assert wrapped_speculator().replay_mode == "chat"


def test_half_a_wrapper_is_rejected():
    tiny = TinyTargetModel()
    with pytest.raises(ValueError, match="BOTH"):
        ChainedSpeculator(None, tiny, device="cpu", prompt_prefix_ids=[1, 2])


def test_start_prefills_the_wrapper_but_not_the_document(records):
    speculator = wrapped_speculator()
    assert speculator.runner.length == len(PREFIX)
    assert speculator.ids == [] and speculator.feats is None
    assert speculator.last_logits is not None


def test_open_turn_appends_the_suffix_and_close_turn_drops_it(records):
    speculator = wrapped_speculator()
    commit_prefix(speculator, records[0], length=6)
    assert speculator.runner.length == len(PREFIX) + 6
    speculator.open_turn()
    assert speculator.runner.length == len(PREFIX) + 6 + len(SUFFIX)
    speculator.commit(speculator.ids + [7, 8])
    assert speculator.runner.length == len(PREFIX) + 8 + len(SUFFIX)
    speculator.close_turn()
    assert speculator.runner.length == len(PREFIX) + 6


def test_open_turn_twice_is_an_error(records):
    speculator = wrapped_speculator()
    commit_prefix(speculator, records[0], length=6)
    speculator.open_turn()
    with pytest.raises(RuntimeError, match="already open"):
        speculator.open_turn()


def test_close_turn_without_a_turn_is_a_no_op(records):
    speculator = wrapped_speculator()
    commit_prefix(speculator, records[0], length=6)
    length = speculator.runner.length
    speculator.close_turn()
    assert speculator.runner.length == length


def test_raw_mode_turns_never_touch_the_cache(records):
    """The whole point of the mode switch: raw behaviour is unchanged."""
    speculator = make_speculator()
    commit_prefix(speculator, records[0], length=6)
    length, logits = speculator.runner.length, speculator.last_logits
    speculator.open_turn()
    assert speculator.runner.length == length
    assert speculator.last_logits is logits
    speculator.close_turn()
    assert speculator.runner.length == length
    assert speculator.last_logits is logits


# ---- the cache must equal an uncached forward of the WRAPPED sequence ----

def test_chat_features_match_an_uncached_wrapped_forward(records):
    speculator = wrapped_speculator()
    record = records[0]
    document = record["input_ids"][:10]
    speculator.append_committed(document, record["segment_type_ids"][:10],
                                record["recency_bucket_ids"][:10])
    speculator.open_turn()
    speculator.begin_round(NO_PRIOR_HOP_DISTANCE)
    generated = [7, 9]
    speculator.commit(list(speculator.ids) + generated)

    expected_hidden, expected_logits = full_forward(
        speculator.target, PREFIX + document + SUFFIX + generated
    )
    document_rows = expected_hidden[len(PREFIX):len(PREFIX) + len(document)]
    generated_rows = expected_hidden[len(PREFIX) + len(document) + len(SUFFIX):]
    assert torch.allclose(speculator.feats[:len(document)], document_rows, atol=1e-4)
    assert torch.allclose(speculator.feats[len(document):], generated_rows, atol=1e-4)
    assert torch.allclose(speculator.last_logits, expected_logits[-1], atol=1e-4)


def test_open_turn_logits_predict_the_first_step_token(records):
    """The suffix's last row is the prediction for the step's first token —
    the position raw mode could not verify at all."""
    speculator = wrapped_speculator()
    record = records[0]
    document = record["input_ids"][:10]
    speculator.append_committed(document, record["segment_type_ids"][:10],
                                record["recency_bucket_ids"][:10])
    speculator.open_turn()
    _hidden, expected_logits = full_forward(speculator.target, PREFIX + document + SUFFIX)
    assert torch.allclose(speculator.last_logits, expected_logits[-1], atol=1e-4)
    assert speculator.next_tokens(speculator.ids, []) == [
        int(expected_logits[-1].argmax())
    ]


def test_speculated_tokens_do_not_persist_in_cache_in_chat_mode(records):
    speculator = wrapped_speculator()
    commit_prefix(speculator, records[0])
    speculator.open_turn()
    length = speculator.runner.length
    speculator.next_tokens(speculator.ids, [4, 4, 4])
    assert speculator.runner.length == length


def test_truncate_maps_through_the_wrapper(records):
    speculator = wrapped_speculator()
    commit_prefix(speculator, records[0], length=10)
    speculator.truncate_to(6)
    assert len(speculator.ids) == 6
    assert speculator.runner.length == len(PREFIX) + 6


# ---- leading prompt tokens ----

def test_raw_mode_has_no_prompt_tokens(records):
    assert make_speculator().leading_prompt_tokens(records[0]["input_ids"]) == 0


def test_leading_prompt_tokens_counts_only_whitespace(routed_tokenizer):
    speculator = wrapped_speculator(decode=routed_tokenizer.decode,
                                    vocab_size=ROUTED_VOCAB_SIZE)
    ids = routed_tokenizer.encode("\nThought: I need")
    assert routed_tokenizer.decode(ids[:1]) == "\n"
    assert speculator.leading_prompt_tokens(ids) == 1
    assert speculator.leading_prompt_tokens(ids[1:]) == 0


def test_leading_prompt_tokens_never_consumes_a_whole_region(routed_tokenizer):
    speculator = wrapped_speculator(decode=routed_tokenizer.decode,
                                    vocab_size=ROUTED_VOCAB_SIZE)
    ids = routed_tokenizer.encode("\n")
    assert speculator.leading_prompt_tokens(ids) == 0


# ---- replay in chat mode ----

def test_chat_replay_restores_the_rails(routed_records, routed_tokenizer):
    speculator = wrapped_speculator(decode=routed_tokenizer.decode,
                                    vocab_size=ROUTED_VOCAB_SIZE)
    record = routed_records[0]
    rounds = replay_record(record, speculator, gamma=GAMMA)
    assert rounds
    assert speculator.ids == record["input_ids"]
    assert speculator.runner.length == len(PREFIX) + len(record["input_ids"])
    for row in rounds:
        assert 0 <= row["accepted"] <= GAMMA
        assert 1 <= row["emitted"] <= GAMMA + 1
        assert row["replay_mode"] == "chat"


def test_raw_replay_rows_are_labelled_raw(records):
    rounds = replay_record(records[0], make_speculator(), gamma=GAMMA)
    assert {row["replay_mode"] for row in rounds} == {"raw"}


def test_chat_replay_does_not_speculate_the_prompt_newline(
    routed_records, routed_tokenizer
):
    """The "\n" closing the question is PROMPT: the agent never generated it.
    Raw mode speculates it anyway (3 of 260 boundaries on the pilot shard)."""
    record = routed_records[0]
    chat = wrapped_speculator(decode=routed_tokenizer.decode,
                              vocab_size=ROUTED_VOCAB_SIZE)
    chat_rounds = replay_record(record, chat, gamma=GAMMA)
    raw = wrapped_speculator(prefix=[], suffix=[], vocab_size=ROUTED_VOCAB_SIZE)
    raw_rounds = replay_record(record, raw, gamma=GAMMA)
    # the first generated region really does start with a whitespace token
    segments = record["segment_type_ids"]
    start = next(i for i in range(1, len(segments))
                 if segments[i] not in (0, 3) and segments[i - 1] in (0, 3))
    assert routed_tokenizer.decode(record["input_ids"][start:start + 1]).strip() == ""
    first_region = [r for r in raw_rounds if r["hop_index"] == -1][0]["region_tokens"]
    chat_first = [r for r in chat_rounds if r["hop_index"] == -1][0]["region_tokens"]
    assert chat_first == first_region - 1
