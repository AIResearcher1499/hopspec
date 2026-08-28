"""Structure-routed drafting for agentic RAG.

Three draft sources, routed by structure the agent loop ALREADY knows —
segment type, hop recency, passage provenance — none of which costs a
forward pass:

- ``ScaffoldFSM``   deterministic proposer for ReAct scaffold spans. 35.4% of
  generated tokens are TEMPLATE and prompt-lookup cannot cover them (0.092
  acc on template positions) because the scaffold follows novel passage text:
  no n-gram match exists. Zero cost, and no published lookup method has it.
- ``ScopedLookup``  n-gram suffix match restricted to QUESTION and
  RETRIEVED_PASSAGE positions. 88% of correct lookup predictions copy from
  inserted content, so the scope costs ~nothing and shrinks the search space.
- the neural draft — ``ChainedSpeculator`` itself, passed in as ``neural``.
  Weak pooled, but the only source with signal right after a hop.

Everything here is a PROPOSER. Verification, the KV cache and the
rails-restoring replay live in ``chained_eval.py`` and are frozen: a routing
change must not be able to move the measurement.

Two things to keep straight when reading any number this produces:

- Scaffold tokens are TEMPLATE. ``decode_phase_mask`` excludes TEMPLATE from
  acceptance measurement, but chained replay genuinely proposes, accepts and
  emits them, so accepted-per-round counts them. That is correct for a
  systems claim and wrong for a content claim — which is why every round row
  carries per-token sources and ``summarize_rounds_by_source`` splits them.
  This is the spec §3 "why TEMPLATE exists" lesson in a new coat.
- The FSM's opening literal is FITTED from data, not assumed. The spec's
  ReAct grammar says every step opens with "Thought:"; measured on the 1.7B
  shard, 245 of 260 generated steps open with "Action:" instead. Fit on the
  TRAIN split only (``fit_scaffold_fsm``) and print what was fitted.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol, Sequence

import torch

from hopspec.infer.chained_eval import INSERTED_SEGMENTS, SPECULATED_SEGMENT

# Lookup scope: the segment types INSERTED into the context rather than
# generated. Of correct lookup predictions, 53.9% copy the question and 34.6%
# a retrieved passage; only 11.5% copy generated text.
SCOPE_SEGMENTS = INSERTED_SEGMENTS
_SCOPE = frozenset(int(segment) for segment in SCOPE_SEGMENTS)

DEFAULT_STEP_OPENING = "Thought:"
ACTION_PREFIX = "Action:"
_ACTION_TAIL_RE = re.compile(r"Action:\s*$", re.IGNORECASE)
_ACTION_VERB_RE = re.compile(r"Action:\s*(Search|Finish)\[", re.IGNORECASE)


class ScaffoldTokenizer(Protocol):
    """Encode/decode over the TARGET's vocabulary.

    The FSM works in token space (spec §4): scaffold literals are tokenized
    once with the target tokenizer, never assembled character by character.
    """

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class HFScaffoldTokenizer:
    """Adapter over a HuggingFace tokenizer.

    ``add_special_tokens=False`` and ``clean_up_tokenization_spaces=False``
    are both load-bearing: a literal that does not round-trip byte for byte
    would be rejected by ``ScaffoldFSM``, and silently mangled whitespace is
    exactly the failure mode §4 is about.
    """

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


def _last_inserted_position(segments: Sequence[int]) -> int:
    for position in range(len(segments) - 1, -1, -1):
        if int(segments[position]) in _SCOPE:
            return position
    return -1


def _is_open_thought(line: str) -> bool:
    lowered = line.lstrip().lower()
    return lowered.startswith("thought:") and "action:" not in lowered


class ScaffoldFSM:
    """Deterministic proposer for the ReAct scaffold.

    State is a pure function of the committed tail: the text generated since
    the last INSERTED token, which the replay loop labels for us. The states
    are the ones the pipeline itself enforces (one Thought + one Action per
    step, `agent_pipeline._split_generated_step`):

        fresh line, nothing open      -> the step opening literal
        fresh line after a Thought    -> "Action:"  (+ verb if configured)
        line ends at "Action:"        -> " Search[" / " Finish[" if configured
        inside THOUGHT / payload text -> nothing (defer)
        line ends at "]"              -> "\\n"

    Never proposes past a literal boundary, and never proposes a literal that
    ends with a space: BPE attaches a leading space to the following CONTENT
    token, so such a literal would own that token's first character — the
    §4 bug that once excluded the most important token in the project from
    every measurement.
    """

    def __init__(
        self,
        tokenizer: ScaffoldTokenizer,
        step_opening: str = DEFAULT_STEP_OPENING,
        action_verb: str | None = None,
    ):
        if not step_opening:
            raise ValueError("step_opening must be a non-empty literal")
        if action_verb is not None and not action_verb:
            raise ValueError("action_verb must be None or a non-empty verb")
        self.tokenizer = tokenizer
        self.step_opening = step_opening
        self.action_verb = action_verb
        self._encoded: dict[str, list[int]] = {}
        # Encode every literal the FSM can emit at construction time: a
        # tokenizer that mangles one should fail here, not mid-run.
        for literal in (step_opening, ACTION_PREFIX, "\n", self._action_literal(),
                        self._step_literal()):
            self._encode(literal)
        if action_verb is not None:
            self._encode(f" {action_verb}[")
            self._encode(f"{action_verb}[")

    # ---- literals ----

    def _action_literal(self) -> str:
        if self.action_verb is None:
            return ACTION_PREFIX
        return f"{ACTION_PREFIX} {self.action_verb}["

    def _step_literal(self) -> str:
        if self.step_opening.lower().startswith(ACTION_PREFIX.lower()):
            return self._action_literal()
        return self.step_opening

    def _encode(self, literal: str) -> list[int]:
        cached = self._encoded.get(literal)
        if cached is not None:
            return cached
        if literal.endswith(" "):
            raise ValueError(
                f"scaffold literal {literal!r} ends with a space: BPE attaches "
                "that space to the next CONTENT token, so proposing it would "
                "own that token's first character (spec §4)"
            )
        ids = list(self.tokenizer.encode(literal))
        if not ids:
            raise ValueError(f"scaffold literal {literal!r} encodes to nothing")
        round_trip = self.tokenizer.decode(ids)
        if round_trip != literal:
            raise ValueError(
                f"scaffold literal {literal!r} does not round-trip through the "
                f"tokenizer (got {round_trip!r}); the FSM would propose text "
                "the target never writes"
            )
        self._encoded[literal] = ids
        return ids

    # ---- state ----

    def next_literal(
        self, ids: Sequence[int], segments: Sequence[int]
    ) -> str | None:
        """The scaffold text due next, or None where the grammar defers."""
        if not ids:
            return None
        start = _last_inserted_position(segments) + 1
        region = self.tokenizer.decode(list(ids[start:])) if start < len(ids) else ""
        lines = region.split("\n")
        current = lines[-1]
        previous = lines[-2] if len(lines) >= 2 else None

        if current == "":
            if previous is not None and _is_open_thought(previous):
                literal = self._action_literal()
            else:
                literal = self._step_literal()
            # The newline may already be inside the last inserted token
            # (Qwen merges "?\n" and ".\n"), in which case proposing another
            # one writes a blank line the target never writes.
            if region == "" and not self.tokenizer.decode(list(ids[-1:])).endswith("\n"):
                literal = "\n" + literal
            return literal
        if self.action_verb is not None and _ACTION_TAIL_RE.search(current):
            verb = f"{self.action_verb}["
            return verb if current.endswith(" ") else f" {verb}"
        if current.endswith("]") and "[" in current and "action:" in current.lower():
            return "\n"
        return None

    def literal_token_ids(self) -> set[int]:
        """Every token id this FSM can emit, for offline template/content
        classification. Approximate by construction: a content token that
        shares an id with a literal's token counts as template, so any count
        built from this is an UPPER bound on the template share."""
        return {int(t) for ids in self._encoded.values() for t in ids}

    def next_span(
        self, ids: Sequence[int], segments: Sequence[int], max_tokens: int
    ) -> list[int]:
        """Token ids for the next scaffold span, truncated to the budget.

        A truncated literal is finished by the next round: the FSM re-derives
        its state from the committed tail, so a partial "Action:" leaves it in
        the verb state, not lost.
        """
        if max_tokens <= 0:
            return []
        literal = self.next_literal(ids, segments)
        if literal is None:
            return []
        return list(self._encode(literal))[:max_tokens]


class ScopedLookup:
    """Suffix n-gram lookup restricted to inserted content.

    Longest n first, and — following the prompt-lookup reference
    implementation rather than any paper's wording (spec §15) — the LEFTMOST
    match wins, so proposals are deterministic. Every position of the matched
    window must be in scope, and the continuation stops at the first
    out-of-scope position: a copy that runs off the end of a passage is not
    a copy of the passage.
    """

    def __init__(
        self,
        max_ngram: int = 3,
        min_ngram: int = 2,
        scope_segments: Sequence[int] = SCOPE_SEGMENTS,
    ):
        if min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.scope = frozenset(int(segment) for segment in scope_segments)

    def propose(
        self, ids: Sequence[int], segments: Sequence[int], max_tokens: int
    ) -> list[int]:
        if len(ids) != len(segments):
            raise ValueError("ids and segments must have equal lengths")
        if max_tokens <= 0:
            return []
        in_scope = [int(segment) in self.scope for segment in segments]
        for size in range(self.max_ngram, self.min_ngram - 1, -1):
            if len(ids) <= size:
                continue
            needle = list(ids[-size:])
            last = needle[-1]
            # Linear scan, cheapest test first: at these lengths an index or
            # a suffix automaton is an optimization, not a requirement.
            for start in range(len(ids) - size):
                if ids[start + size - 1] != last:
                    continue
                if list(ids[start:start + size]) != needle:
                    continue
                if not all(in_scope[start:start + size]):
                    continue
                continuation: list[int] = []
                cursor = start + size
                while (cursor < len(ids) and in_scope[cursor]
                       and len(continuation) < max_tokens):
                    continuation.append(int(ids[cursor]))
                    cursor += 1
                if continuation:
                    return continuation
        return []


class CommittedState(Protocol):
    """What a router needs from the speculator: the committed token ids and
    their segment labels (real for inserted spans, the documented placeholder
    for speculated ones)."""

    ids: list[int]
    seg: list[int]


class RoutedProposer:
    """Routes proposals between scaffold, scoped lookup and the neural draft.

    Implements the ``DraftProposer`` protocol, so ``run_speculative_round``
    and the replay loop need no changes. Precedence is structural: scaffold
    spans first (deterministic and free), then a scoped lookup hit, then the
    neural draft for whatever is left.

    ``chain=False`` (the default) is single-source-per-round: the first
    source that produces anything owns the round. ``chain=True`` stacks
    sources until gamma is filled — including into the neural draft, which
    chains on its own predicted features over the forced prefix, so the
    difference between the two is a measurement, not an implementation
    detail.

    Omit sources to get the single-source arms: ``RoutedProposer(state,
    lookup=...)`` is the lookup-only arm and proposes nothing when it misses
    (the round then emits the target's own token, exactly as an unassisted
    decode step would).
    """

    def __init__(
        self,
        state: CommittedState,
        scaffold: ScaffoldFSM | None = None,
        lookup: ScopedLookup | None = None,
        neural=None,
        chain: bool = False,
    ):
        if scaffold is None and lookup is None and neural is None:
            raise ValueError("a router needs at least one source")
        self.state = state
        self.scaffold = scaffold
        self.lookup = lookup
        self.neural = neural
        self.chain = chain
        self.last_source = "none"
        self.last_token_sources: list[str] = []

    def _segments(self, ids: list[int]) -> list[int]:
        segments = list(self.state.seg)
        if len(segments) != len(ids):
            raise ValueError(
                "committed segment labels are out of sync with the context "
                f"({len(segments)} labels for {len(ids)} tokens)"
            )
        return segments

    def propose(self, context_ids: list[int], num_tokens: int) -> list[int]:
        ids = list(context_ids)
        segments = self._segments(ids)
        proposed: list[int] = []
        sources: list[str] = []

        if self.scaffold is not None:
            span = list(self.scaffold.next_span(ids, segments, num_tokens))
            proposed += span
            sources += ["scaffold"] * len(span)

        if (self.lookup is not None and len(proposed) < num_tokens
                and (self.chain or not proposed)):
            span = list(self.lookup.propose(
                ids + proposed,
                segments + [SPECULATED_SEGMENT] * len(proposed),
                num_tokens - len(proposed),
            ))
            proposed += span
            sources += ["lookup"] * len(span)

        if (self.neural is not None and len(proposed) < num_tokens
                and (self.chain or not proposed)):
            if proposed:
                full = list(self.neural.propose(
                    ids, num_tokens, forced_prefix=proposed
                ))[:num_tokens]
            else:
                full = list(self.neural.propose(ids, num_tokens))[:num_tokens]
            sources += ["neural"] * max(0, len(full) - len(proposed))
            proposed = full

        proposed = proposed[:num_tokens]
        self.last_token_sources = sources[:len(proposed)]
        self.last_source = self.last_token_sources[0] if self.last_token_sources else "none"
        return proposed


def last_token_entropy(state: CommittedState) -> float | None:
    """Entropy (nats) of the target's distribution for the next position.

    Read off ``state.last_logits``, the row the verifier already computed for
    the last committed token — no extra target forward. Returns None when no
    committed logits are available (right after a truncate).
    """
    logits = getattr(state, "last_logits", None)
    if logits is None:
        return None
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(-(log_probs.exp() * log_probs).sum())


class EntropyRoutedProposer:
    """ReSpec-style comparator: route by the TARGET's last-token entropy.

    Low entropy means the continuation is predictable, which is where the
    retrieval draft wins; high entropy routes to the model draft. That
    direction is the measured one — within every entropy tercile, lookup
    acceptance is highest at low entropy — not a guess.

    This is the baseline structure routing has to beat. Both signals are free
    in this replay, but they are not the same kind of signal: entropy needs
    the target's distribution over the current prefix, structure is known
    from the agent loop's own state. The threshold is a fitted parameter and
    must be tuned on the TRAIN split (``tune_entropy_threshold``); structure
    routing has none.

    An optional ``scaffold`` makes the "structure + entropy" combination the
    plan asks for before concluding anything from a loss to this baseline.
    """

    def __init__(
        self,
        state: CommittedState,
        lookup: ScopedLookup | None = None,
        neural=None,
        threshold: float = 1.0,
        scaffold: ScaffoldFSM | None = None,
    ):
        if lookup is None and neural is None:
            raise ValueError("an entropy router needs a lookup or a neural source")
        self.state = state
        self.lookup = lookup
        self.neural = neural
        self.threshold = float(threshold)
        self.scaffold = scaffold
        self.last_source = "none"
        self.last_token_sources: list[str] = []
        self.last_entropy: float | None = None

    def propose(self, context_ids: list[int], num_tokens: int) -> list[int]:
        ids = list(context_ids)
        segments = list(self.state.seg)
        if len(segments) != len(ids):
            raise ValueError("committed segment labels are out of sync with the context")
        proposed: list[int] = []
        sources: list[str] = []

        if self.scaffold is not None:
            span = list(self.scaffold.next_span(ids, segments, num_tokens))
            proposed += span
            sources += ["scaffold"] * len(span)

        self.last_entropy = last_token_entropy(self.state)
        if not proposed:
            if (self.lookup is not None and self.last_entropy is not None
                    and self.last_entropy <= self.threshold):
                span = list(self.lookup.propose(ids, segments, num_tokens))
                proposed += span
                sources += ["lookup"] * len(span)
            if not proposed and self.neural is not None:
                span = list(self.neural.propose(ids, num_tokens))[:num_tokens]
                proposed += span
                sources += ["neural"] * len(span)

        proposed = proposed[:num_tokens]
        self.last_token_sources = sources[:len(proposed)]
        self.last_source = self.last_token_sources[0] if self.last_token_sources else "none"
        return proposed


# ---- fitting the scaffold to the grammar the agent actually writes ----

def generated_regions(record: dict) -> list[str]:
    """Text of each maximal run of non-inserted steps, in order."""
    regions: list[str] = []
    current: list[str] = []
    for step in record["steps"]:
        if int(step["segment_type"]) in _SCOPE:
            if current:
                regions.append("".join(current))
                current = []
        else:
            current.append(step["text"])
    if current:
        regions.append("".join(current))
    return regions


def scaffold_stats(records: list[dict]) -> tuple[Counter, Counter]:
    """How generated steps actually open, and which action verb they use.

    The spec's ReAct grammar says every step opens with "Thought:". Measured
    on the 1.7B shard, 245 of 260 steps open with "Action:" — the agent skips
    the thought. Assuming the grammar would have made the scaffold arm score
    ~0 for a reason that has nothing to do with the idea under test, so the
    literal is fitted. TRAIN RECORDS ONLY.
    """
    openings: Counter = Counter()
    verbs: Counter = Counter()
    for record in records:
        for index, region in enumerate(generated_regions(record)):
            if index == 0 and region.lstrip().lower().startswith("question:"):
                continue  # the "Question: " prefix, not a generated step
            line = region.lstrip("\n").split("\n", 1)[0].lstrip().lower()
            if line.startswith("thought:"):
                openings[DEFAULT_STEP_OPENING] += 1
            elif line.startswith(ACTION_PREFIX.lower()):
                openings[ACTION_PREFIX] += 1
            else:
                openings["other"] += 1
            for match in _ACTION_VERB_RE.finditer(region):
                verbs[match.group(1).capitalize()] += 1
    return openings, verbs


def fit_scaffold_fsm(
    records: list[dict], tokenizer: ScaffoldTokenizer, use_verb: bool = False
) -> tuple[ScaffoldFSM, dict]:
    """A ScaffoldFSM configured from the observed grammar of `records`.

    TRAIN RECORDS ONLY — fitting the proposer on held-out trajectories would
    make the held-out acceptance number meaningless. Returns the FSM and the
    stats behind the choice so the caller can print them; an opening that is
    neither literal falls back to the spec grammar.
    """
    openings, verbs = scaffold_stats(records)
    step_opening = DEFAULT_STEP_OPENING
    if openings:
        best, _count = openings.most_common(1)[0]
        if best in (DEFAULT_STEP_OPENING, ACTION_PREFIX):
            step_opening = best
    action_verb = verbs.most_common(1)[0][0] if (use_verb and verbs) else None
    fsm = ScaffoldFSM(tokenizer, step_opening=step_opening, action_verb=action_verb)
    stats = {
        "openings": dict(openings),
        "verbs": dict(verbs),
        "step_opening": step_opening,
        "action_verb": action_verb,
    }
    return fsm, stats


def tune_entropy_threshold(
    records: list[dict],
    speculator,
    lookup: ScopedLookup,
    thresholds: Sequence[float],
    gamma: int = 4,
    max_rounds_per_region: int | None = None,
    scaffold: ScaffoldFSM | None = None,
) -> tuple[float, list[dict]]:
    """Pick the entropy threshold with the best mean accepted/round.

    TRAIN RECORDS ONLY. Tuning the comparator on held-out data is the classic
    way to manufacture a baseline that loses; the caller passes the train
    split and prints the grid that was searched. Ties go to the smaller
    threshold (less routing to lookup, the more conservative choice).
    """
    from hopspec.infer.chained_eval import replay_record

    table: list[dict] = []
    for threshold in thresholds:
        proposer = EntropyRoutedProposer(
            speculator, lookup=lookup, neural=speculator,
            threshold=threshold, scaffold=scaffold,
        )
        rounds: list[dict] = []
        for record in records:
            rounds += replay_record(
                record, speculator, gamma=gamma, proposer=proposer,
                max_rounds_per_region=max_rounds_per_region,
            )
        mean = sum(row["accepted"] for row in rounds) / len(rounds) if rounds else 0.0
        table.append({
            "threshold": float(threshold),
            "rounds": len(rounds),
            "mean_accepted": mean,
        })
    best = max(table, key=lambda row: (row["mean_accepted"], -row["threshold"]))
    return best["threshold"], table
