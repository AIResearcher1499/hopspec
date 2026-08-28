"""Chained speculative evaluation against real models — the decisive
experiment (spec §11.4).

Every §11 measurement is single-step and teacher-forced: the draft is handed
the target's own f_{q-2}, which already encodes the retrieved passage. Here
the draft chains several tokens per round off its OWN predicted features
(EAGLE-style), and the target verifies. This is the regime where a context
shift should bite if it bites anywhere.

Measurement design:

- Collected trajectories are replayed token-region by token-region. Inserted
  regions (QUESTION, RETRIEVED_PASSAGE) are committed verbatim. For each
  generated region, speculative rounds run from the TRUE recorded prefix
  until the region's recorded length is reached; then the speculated tokens
  are discarded and the recorded tokens restored, so every region starts on
  the recorded rails.
- Per round we log the tracker distance at round start, gamma, and the
  accepted count. Accepted-per-round vs distance is the systems quantity of
  interest; note that a round right after a hop includes easy TEMPLATE tokens
  ("Thought:"), which genuinely speeds decoding but is NOT evidence about
  content tokens — read near-hop rounds with that in mind.

Two replay modes, selected by whether the speculator is given a chat wrapper
(`--replay-mode`, prereg `docs/prereg-chained-chat-replay-2026-08-28.md`):

- `raw` — the target sees the recorded document verbatim. This is the original
  behaviour and is kept bit for bit so old artifacts stay reproducible. It has
  a known validity defect: the agent never saw that document, so at each step
  boundary the target has no reason to open a ReAct step, and it reproduces
  the recorded step-opening token only ~5% of the time versus ~70% elsewhere.
- `chat` — the wrapper is re-rendered at every step boundary, exactly as
  `HFTargetLLM.generate` did at collection: a fixed prefix
  (`<|im_start|>system … <|im_start|>user\n`), the context so far, a fixed
  suffix (`<|im_end|>\n<|im_start|>assistant\n…`), then the step. This is the
  deployed loop.

NUMBERS FROM THE TWO MODES ARE NOT COMPARABLE — the measurement definition
differs, and §15 forbids it.

The target runs with a KV cache that is cropped on rollback, so verify and
commit forwards cost O(new tokens) rather than O(prefix). Without it a round
costs two full-prefix forwards of the target and a 12-record replay does not
finish in a sitting. `CachedTargetRunner` is checked against uncached
forwards in the tests — a silently wrong cache would corrupt every feature.
In chat mode the cached common prefix is `wrapper prefix + document[:turn
start]`, which only grows; only the suffix and the current step are recomputed
per step.
"""

from __future__ import annotations

import torch

from hopspec.data.agent_pipeline import SYSTEM_PROMPT
from hopspec.data.schema import (
    NO_PRIOR_HOP_DISTANCE,
    NUM_RECENCY_BUCKETS,
    SegmentType,
    recency_bucket_id,
)
from hopspec.eval.diagnostic import resolve_recency_buckets_for_model
from hopspec.infer.adaptive_length import HopAwareLengthPolicy
from hopspec.infer.speculative_generate import RecencyStateTracker, run_speculative_round

INSERTED_SEGMENTS = (SegmentType.QUESTION, SegmentType.RETRIEVED_PASSAGE)
# Speculated tokens have no recorded label; the draft input segment for them
# is a documented placeholder (content-span). Baseline arms never see it.
SPECULATED_SEGMENT = int(SegmentType.THOUGHT)


# A string the chat template will pass through untouched, used to split the
# rendered wrapper around the user content.
CHAT_CONTEXT_SENTINEL = "\x00hopspec-context\x00"


def chat_prompt_ids(
    hf_tokenizer, system_prompt: str = SYSTEM_PROMPT
) -> tuple[list[int], list[int]]:
    """Token ids of the chat wrapper, split around the user content.

    Mirrors `HFTargetLLM.generate` exactly — same messages, same
    `add_generation_prompt`, same `enable_thinking=False` with the `TypeError`
    fallback for templates that reject the kwarg — so the replay target is
    prompted the way the agent actually was.

    The split is done on the RENDERED STRING around a sentinel, not by
    guessing where the template's halves are, because the document's own
    tokenization must survive: verified over the shard that
    `encode(prefix + context + suffix) == prefix_ids + encode(context) +
    suffix_ids` at every boundary tried. A tokenizer where that fails would
    silently shift every label.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": CHAT_CONTEXT_SENTINEL},
    ]
    try:
        rendered = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:  # templates that reject the enable_thinking kwarg
        rendered = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if rendered.count(CHAT_CONTEXT_SENTINEL) != 1:
        raise ValueError(
            "chat template did not place the user content exactly once; "
            "cannot split it into a fixed prefix and suffix"
        )
    before, after = rendered.split(CHAT_CONTEXT_SENTINEL)
    prefix = list(hf_tokenizer.encode(before, add_special_tokens=False))
    suffix = list(hf_tokenizer.encode(after, add_special_tokens=False))
    if not prefix or not suffix:
        raise ValueError("the chat wrapper must have a non-empty prefix and suffix")
    return prefix, suffix


def chat_prompt_seam_ok(
    hf_tokenizer,
    context: str,
    prefix_ids: list[int],
    suffix_ids: list[int],
    system_prompt: str = SYSTEM_PROMPT,
) -> bool:
    """Does wrapping `context` leave ITS OWN tokenization untouched?

    Splitting the wrapper into fixed prefix/suffix ids is only sound if the
    seams do not merge with the document (BPE happily merges `"...2017"` with
    whatever follows). Qwen's wrapper ends in special tokens, so it does not —
    verified over the pilot shard — but another tokenizer may differ, and a
    silent merge would shift every label by a token. Call this before spending
    GPU time (spec §15: validate before you pay).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    try:
        rendered = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    joint = list(hf_tokenizer.encode(rendered, add_special_tokens=False))
    split = (list(prefix_ids)
             + list(hf_tokenizer.encode(context, add_special_tokens=False))
             + list(suffix_ids))
    return joint == split


def constant_gamma_policy(gamma: int) -> HopAwareLengthPolicy:
    """Fixed proposal length for every bucket — the adaptive policy would
    confound the acceptance-vs-distance measurement."""
    return HopAwareLengthPolicy(
        {bucket: gamma for bucket in range(NUM_RECENCY_BUCKETS)},
        gamma_min=1, gamma_max=max(8, gamma),
    )


class CachedTargetRunner:
    """Target forwards over a growing prefix, backed by a croppable KV cache.

    `extend(new_ids)` returns (hidden_states, logits) for the NEW positions
    only; `rollback(length)` crops the cache back to a prefix length. Results
    are identical to a full forward — the tests assert that.
    """

    def __init__(self, target_model, device: str, feature_layer: int = -1):
        self.model = target_model
        self.device = device
        self.feature_layer = feature_layer
        self.reset()

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.length = 0

    def rollback(self, length: int) -> None:
        if length > self.length:
            raise ValueError("cannot roll a cache forward")
        if length < self.length:
            # Negative form: remove this many tokens (positive absolute
            # lengths are deprecated in transformers >= 5.16).
            self.cache.crop(-(self.length - length))
            self.length = length

    def extend(self, new_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not new_ids:
            raise ValueError("nothing to extend with")
        tokens = torch.tensor([new_ids], dtype=torch.long, device=self.device)
        total = self.length + len(new_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=torch.ones((1, total), dtype=torch.long,
                                          device=self.device),
                past_key_values=self.cache,
                use_cache=True,
                output_hidden_states=True,
            )
        self.cache = outputs.past_key_values
        self.length = total
        return (
            outputs.hidden_states[self.feature_layer][0].float(),
            outputs.logits[0],
        )


def token_regions(segment_ids) -> list[tuple[bool, int, int]]:
    """Maximal runs of positions as (is_inserted, start, end).

    Inserted regions (QUESTION, RETRIEVED_PASSAGE) are put into the context by
    the loop; generated regions are what the agent itself wrote.
    """
    inserted_types = {int(segment) for segment in INSERTED_SEGMENTS}
    regions: list[tuple[bool, int, int]] = []
    for index, segment in enumerate(segment_ids):
        flag = int(segment) in inserted_types
        if regions and regions[-1][0] == flag:
            regions[-1] = (flag, regions[-1][1], index + 1)
        else:
            regions.append((flag, index, index + 1))
    return regions


def leading_prompt_token_count(ids, start: int, end: int, decode) -> int:
    """Leading tokens of a generated region that were PROMPT, not generation.

    The newline closing the question does not always merge into the question's
    last token, so it can head the generated region although the agent never
    generated it.
    """
    if decode is None:
        return 0
    count = 0
    for token in ids[start:end - 1]:
        text = decode([int(token)])
        if not text or text.strip():
            break
        count += 1
    return count


def assistant_turns(ids, segment_ids, decode=None) -> list[tuple[int, int]]:
    """(turn_start, region_end) for every assistant turn chat replay opens.

    Mirrors `replay_record` exactly, and is the single definition of the chat
    feature layout shared by replay and by training: a region with fewer than
    two document tokens before it is committed verbatim (the "Question: "
    prefix), leading whitespace is prompt, and an empty remainder opens no
    turn. Training features built from this cannot drift from serving
    features built from the same function.
    """
    turns: list[tuple[int, int]] = []
    for flag, start, end in token_regions(segment_ids):
        if flag or start < 2:
            continue
        turn = start + leading_prompt_token_count(ids, start, end, decode)
        if turn < end:
            turns.append((turn, end))
    return turns


def _tracker_with_distance(distance: int) -> RecencyStateTracker:
    tracker = RecencyStateTracker()
    if distance != NO_PRIOR_HOP_DISTANCE:
        tracker.on_hop_boundary()
        tracker.on_tokens_appended(distance)
    return tracker


class ChainedSpeculator:
    """Stateful DraftProposer + TargetVerifier over one replayed trajectory.

    Committed state: token ids, segment ids, bucket ids, and target features
    for every committed position. During a round, features for speculated
    positions are the draft's own predictions — that is the point.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        device: str = "cpu",
        feature_layer: int = -1,
        hop_signal_enabled: bool = True,
        prompt_prefix_ids: list[int] | tuple[int, ...] = (),
        prompt_suffix_ids: list[int] | tuple[int, ...] = (),
        decode=None,
    ):
        # None is allowed: the model-free arms (scaffold, lookup) need the
        # verifier and the committed state but never a draft checkpoint.
        self.draft = None
        if draft_model is not None:
            self.draft = draft_model.to(device)
            self.draft.eval()
        self.target = target_model.to(device)
        self.target.eval()
        self.device = device
        self.feature_layer = feature_layer
        self.hop_signal_enabled = hop_signal_enabled
        self.lm_head_weight = target_model.get_output_embeddings().weight
        # Empty wrapper == raw mode == the original behaviour, bit for bit.
        self._prefix_ids = list(prompt_prefix_ids)
        self._suffix_ids = list(prompt_suffix_ids)
        if bool(self._prefix_ids) != bool(self._suffix_ids):
            raise ValueError(
                "chat replay needs BOTH a wrapper prefix and a wrapper suffix"
            )
        self._decode = decode
        self.runner = CachedTargetRunner(target_model, device, feature_layer)
        self.start()

    @property
    def replay_mode(self) -> str:
        return "chat" if self._suffix_ids else "raw"

    def start(self) -> None:
        self.ids: list[int] = []
        self.seg: list[int] = []
        self.buckets: list[int] = []
        self.feats: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None
        self._round_distance: int | None = None
        self._round_start: int | None = None
        self._spec_feats: list[torch.Tensor] = []
        # Round rows carry the proposing source; the speculator IS the neural
        # source, so it labels its own proposals like any other proposer.
        self.last_source = "neural"
        self.last_token_sources: list[str] = []
        self._turn_open = False
        self._turn_start = 0
        self.runner.reset()
        if self._prefix_ids:
            # The wrapper is NOT part of the document: it gets no segment
            # label, no bucket and no entry in `feats`. Only the cache knows
            # about it.
            _hidden, logits = self.runner.extend(list(self._prefix_ids))
            self.last_logits = logits[-1]

    # ---- document position <-> cache position ----

    def _runner_length(self, doc_length: int) -> int:
        """Cache length holding `doc_length` document tokens.

        Raw mode is the identity. In chat mode the wrapper prefix sits before
        every document token, and while an assistant turn is open the wrapper
        suffix sits between the turn's start and the tokens generated after it.
        """
        extra = (
            len(self._suffix_ids)
            if self._turn_open and doc_length >= self._turn_start
            else 0
        )
        return len(self._prefix_ids) + doc_length + extra

    def open_turn(self) -> None:
        """Begin an assistant turn at the current document position.

        In chat mode this is what makes the target expect a new ReAct step:
        the suffix carries `<|im_end|>` and the assistant header. No-op in raw
        mode, which is exactly why raw numbers are unchanged.
        """
        if self._turn_open:
            raise RuntimeError("an assistant turn is already open")
        self._turn_start = len(self.ids)
        self._turn_open = True
        if self._suffix_ids:
            _hidden, logits = self.runner.extend(list(self._suffix_ids))
            # The suffix's last row predicts the FIRST token of the step —
            # which is precisely the position raw mode could not verify.
            self.last_logits = logits[-1]

    def close_turn(self) -> None:
        """End the turn, dropping the suffix and every speculated token from
        the cache. Idempotent; pure bookkeeping in raw mode."""
        if not self._turn_open:
            return
        self._turn_open = False
        if self._suffix_ids:
            self.runner.rollback(len(self._prefix_ids) + self._turn_start)
            self.last_logits = None

    def leading_prompt_tokens(self, region_ids: list[int]) -> int:
        """How many leading tokens of a generated region were PROMPT.

        The newline closing the question does not always merge into the
        question's last token (3 of 260 boundaries on the pilot shard), so it
        can land at the head of the generated region although the agent never
        generated it. Committing those tokens as prompt makes every boundary
        agree with the context the agent was actually given. Raw mode has no
        decoder and returns 0, leaving its behaviour untouched.
        """
        if self._decode is None:
            return 0
        count = 0
        for token in region_ids[:-1]:   # never consume the whole region
            text = self._decode([int(token)])
            if not text or text.strip():
                break
            count += 1
        return count

    # ---- committed-state management ----

    def append_committed(
        self, ids: list[int], segment_ids: list[int], bucket_ids: list[int]
    ) -> None:
        if not (len(ids) == len(segment_ids) == len(bucket_ids)):
            raise ValueError("committed arrays must have equal lengths")
        hidden, logits = self.runner.extend(list(ids))
        self.ids += list(ids)
        self.seg += list(segment_ids)
        self.buckets += list(bucket_ids)
        self.feats = hidden if self.feats is None else torch.cat([self.feats, hidden])
        self.last_logits = logits[-1]

    def truncate_to(self, length: int) -> None:
        self.ids = self.ids[:length]
        self.seg = self.seg[:length]
        self.buckets = self.buckets[:length]
        self.feats = self.feats[:length]
        self.runner.rollback(self._runner_length(length))
        # The logits row for the new last position is not cached; the next
        # append_committed refreshes it, and replay always appends after a
        # truncate. Guard against use in between.
        self.last_logits = None

    def begin_round(self, distance: int) -> None:
        self._round_distance = distance
        self._round_start = len(self.ids)
        self._spec_feats = []

    def _bucket_at(self, position: int) -> int:
        if position < self._round_start:
            return self.buckets[position]
        if self._round_distance == NO_PRIOR_HOP_DISTANCE:
            return recency_bucket_id(NO_PRIOR_HOP_DISTANCE)
        return recency_bucket_id(
            self._round_distance + (position - self._round_start)
        )

    # ---- DraftProposer protocol ----

    def propose(
        self,
        context_ids: list[int],
        num_tokens: int,
        forced_prefix: list[int] | None = None,
    ) -> list[int]:
        """Chain `num_tokens` draft tokens off the committed state.

        `forced_prefix` lets a router hand the draft tokens another source
        already proposed. Those positions still run a draft forward — that is
        how the draft gets its OWN predicted feature for each of them, which
        is the whole point of chained speculation — but the proposed token is
        the forced one, not the draft's argmax.
        """
        if self.draft is None:
            raise RuntimeError(
                "this speculator has no draft model; use a model-free proposer"
            )
        if list(context_ids) != self.ids:
            raise ValueError("context out of sync with committed state")
        if len(self.ids) < 2:
            raise ValueError("chained proposal needs >= 2 committed tokens")
        forced = list(forced_prefix or [])
        if len(forced) > num_tokens:
            raise ValueError("forced prefix is longer than the proposal budget")
        committed_len = len(self.ids)
        proposed: list[int] = []
        for step in range(num_tokens):
            all_ids = self.ids + proposed
            total = len(all_ids)
            # Training layout, slot i (i = 0..total-2):
            #   token e_{i+1}, feature f_i, segment at i+1, bucket at q = i+2;
            #   output ~ f_{i+1}, whose logits give token q = i+2.
            feature_rows = [
                self.feats[pos] if pos < committed_len
                else self._spec_feats[pos - committed_len]
                for pos in range(total - 1)
            ]
            features = torch.stack(feature_rows).unsqueeze(0)
            tokens = torch.tensor([all_ids[1:]], dtype=torch.long, device=self.device)
            segments = torch.tensor(
                [[self.seg[p] if p < committed_len else SPECULATED_SEGMENT
                  for p in range(1, total)]],
                dtype=torch.long, device=self.device,
            )
            bucket_inputs = torch.tensor(
                [[self._bucket_at(q) for q in range(2, total + 1)]],
                dtype=torch.long, device=self.device,
            )
            segments, bucket_inputs = resolve_recency_buckets_for_model(
                segments, bucket_inputs, self.hop_signal_enabled
            )
            with torch.no_grad():
                predicted = self.draft(tokens, features, segments, bucket_inputs)
                logits = self.draft.predict_logits(
                    predicted[:, -1:], self.lm_head_weight
                )
            next_token = forced[step] if step < len(forced) else int(logits[0, -1].argmax())
            # The last slot's output is the predicted feature of position
            # total-1. Keep it ONLY for speculated positions — committed ones
            # keep their real target features.
            if total - 1 >= committed_len:
                index = total - 1 - committed_len
                assert index == len(self._spec_feats)
                self._spec_feats.append(predicted[0, -1].float())
            proposed.append(next_token)
        self.last_token_sources = ["neural"] * len(proposed)
        self.last_source = "neural" if proposed else "none"
        return proposed

    # ---- TargetVerifier protocol ----

    def next_tokens(self, context_ids: list[int], proposed: list[int]) -> list[int]:
        if list(context_ids) != self.ids:
            raise ValueError("context out of sync with committed state")
        if self.last_logits is None:
            raise RuntimeError("no committed logits; append_committed first")
        # The prediction for the first speculated position is already cached.
        continuations = [int(self.last_logits.argmax())]
        if proposed:
            committed_length = self._runner_length(len(self.ids))
            _hidden, logits = self.runner.extend(list(proposed))
            continuations += [int(row.argmax()) for row in logits]
            # Speculated tokens must not stay in the cache: only what the
            # round actually commits may persist.
            self.runner.rollback(committed_length)
        return continuations

    # ---- after run_speculative_round ----

    def commit(self, new_ids: list[int]) -> int:
        """Absorb the round's emitted tokens; returns how many were emitted."""
        emitted = len(new_ids) - len(self.ids)
        for offset in range(emitted):
            self.seg.append(SPECULATED_SEGMENT)
            self.buckets.append(self._bucket_at(self._round_start + offset))
        appended = list(new_ids[len(self.ids):])
        hidden, logits = self.runner.extend(appended)
        self.ids = list(new_ids)
        self.feats = torch.cat([self.feats, hidden])
        self.last_logits = logits[-1]
        self._spec_feats = []
        return emitted


def replay_record(
    record: dict,
    speculator: ChainedSpeculator,
    gamma: int = 4,
    max_rounds_per_region: int | None = None,
    proposer=None,
) -> list[dict]:
    """Replay one collected record, speculating over its generated regions.

    Returns one row per speculative round:
    {distance, bucket, gamma, accepted, emitted, hop_index, region_tokens,
    source, token_sources}. After each generated region the recorded tokens
    are restored, so every region — and every later hop boundary — sits on
    the recorded rails.

    `proposer` is any DraftProposer; it defaults to the speculator itself, so
    the neural arm and every test written against it are unchanged. The
    speculator stays the verifier and the state holder in every arm — routing
    must not be able to move the measurement. A proposer that also sets
    `last_source` / `last_token_sources` gets its per-token sources logged,
    which is what separates TEMPLATE (scaffold) acceptance from content
    acceptance downstream.
    """
    policy = constant_gamma_policy(gamma)
    draft = speculator if proposer is None else proposer
    ids = record["input_ids"]
    segments = record["segment_type_ids"]
    buckets = record["recency_bucket_ids"]
    distances = record["recency_distances"]

    regions = token_regions(segments)

    speculator.start()
    tracker = RecencyStateTracker()
    rounds: list[dict] = []
    hop_count = 0
    passage_type = int(SegmentType.RETRIEVED_PASSAGE)

    for flag, start, end in regions:
        if flag or len(speculator.ids) < 2:
            speculator.append_committed(ids[start:end], segments[start:end],
                                        buckets[start:end])
            if any(segments[i] == passage_type for i in range(start, end)):
                hop_count += 1
        else:
            # Any leading whitespace-only tokens of this region were PROMPT,
            # not generation (chat mode only; raw mode returns 0 here).
            lead = speculator.leading_prompt_tokens(ids[start:end])
            if lead:
                speculator.append_committed(
                    ids[start:start + lead], segments[start:start + lead],
                    buckets[start:start + lead],
                )
                tracker = _tracker_with_distance(distances[start + lead])
            base_length = len(speculator.ids)
            region_tokens = end - start - lead
            produced = 0
            round_count = 0
            if region_tokens > 0:
                speculator.open_turn()
            while produced < region_tokens and (
                max_rounds_per_region is None or round_count < max_rounds_per_region
            ):
                distance = tracker.distance
                round_gamma = policy.gamma_for(recency_bucket_id(distance))
                new_ids, accepted = run_speculative_round(
                    _begin(speculator, distance), draft, speculator,
                    tracker, policy,
                )
                emitted = speculator.commit(new_ids)
                token_sources = list(
                    getattr(draft, "last_token_sources", None) or []
                )[:round_gamma]
                rounds.append({
                    "distance": distance,
                    "bucket": recency_bucket_id(distance),
                    "gamma": round_gamma,
                    "accepted": accepted,
                    "emitted": emitted,
                    "hop_index": hop_count - 1,
                    "region_tokens": region_tokens,
                    "source": getattr(draft, "last_source", "neural"),
                    "token_sources": token_sources,
                    "replay_mode": speculator.replay_mode,
                })
                produced += emitted
                round_count += 1
            if region_tokens > 0:
                speculator.close_turn()
                speculator.truncate_to(base_length)
                speculator.append_committed(ids[base_length:end],
                                            segments[base_length:end],
                                            buckets[base_length:end])
        # Resync the tracker to the recorded labels for the next position.
        next_distance = distances[end] if end < len(distances) else NO_PRIOR_HOP_DISTANCE
        tracker = _tracker_with_distance(next_distance)

    return rounds


def _begin(speculator: ChainedSpeculator, distance: int) -> list[int]:
    speculator.begin_round(distance)
    return speculator.ids


def summarize_rounds(rounds: list[dict]) -> dict[int, dict]:
    """bucket -> {rounds, mean_accepted, mean_emitted}.

    The per-source breakdown is `summarize_rounds_by_source`, deliberately a
    separate table: pooling scaffold (TEMPLATE) acceptance into a bucket mean
    is how a systems number turns into a content claim by accident.
    """
    by_bucket: dict[int, list[dict]] = {}
    for row in rounds:
        by_bucket.setdefault(row["bucket"], []).append(row)
    return {
        bucket: {
            "rounds": len(rows),
            "mean_accepted": sum(r["accepted"] for r in rows) / len(rows),
            "mean_emitted": sum(r["emitted"] for r in rows) / len(rows),
        }
        for bucket, rows in sorted(by_bucket.items())
    }


def summarize_rounds_by_source(rounds: list[dict]) -> dict[str, dict]:
    """source -> {rounds, proposed, accepted, acceptance}.

    `rounds` counts the rounds a source OPENED; `proposed` and `accepted` are
    per TOKEN, read off `token_sources` (a round accepts a prefix, so the
    first `accepted` entries are the accepted ones).

    The scaffold source proposes only TEMPLATE literals, so its `accepted` IS
    the template share of the win. Report it beside the content sources and
    never let the two blur: TEMPLATE is excluded from `decode_phase_mask` for
    exactly this reason (spec §3, §10), while chained replay really does emit
    those tokens.
    """
    from collections import Counter

    opened: Counter = Counter()
    proposed: Counter = Counter()
    accepted: Counter = Counter()
    for row in rounds:
        opened[row.get("source", "neural")] += 1
        for index, source in enumerate(row.get("token_sources") or []):
            proposed[source] += 1
            if index < row["accepted"]:
                accepted[source] += 1
    summary = {}
    for source in sorted(set(opened) | set(proposed)):
        count = proposed[source]
        summary[source] = {
            "rounds": opened[source],
            "proposed": count,
            "accepted": accepted[source],
            "acceptance": accepted[source] / count if count else 0.0,
        }
    return summary


def record_means(rounds: list[dict]) -> dict[str, float]:
    """record key -> mean accepted per round.

    Rounds are NOT byte-identical across draft sources — the sequences
    diverge as soon as two sources propose differently — so paired McNemar
    over positions (spec §10) does not apply between arms here. Compare these
    per-record means with a paired test across records instead, and say so.
    """
    by_record: dict[str, list[int]] = {}
    for row in rounds:
        key = str(row.get("question_id", row.get("record_index")))
        by_record.setdefault(key, []).append(row["accepted"])
    return {
        key: sum(values) / len(values)
        for key, values in sorted(by_record.items())
    }
