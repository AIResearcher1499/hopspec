"""Online recency tracking and one speculative round.

Wiring `run_speculative_round` to REAL draft/target models is still unbuilt,
and it is the decisive experiment (§11.4 of the spec): every measurement so
far is single-step and teacher-forced — the draft is handed the target's own
f_{q-2}, which already encodes the retrieved passage. Real speculative
decoding chains several tokens per round off the draft's OWN predicted
features, and that is where a context shift should bite. Until this runs,
"no effect" means "no effect in the easiest regime".
"""

from __future__ import annotations

from typing import Protocol

from hopspec.data.schema import NO_PRIOR_HOP_DISTANCE, recency_bucket_id
from hopspec.infer.adaptive_length import HopAwareLengthPolicy


class RecencyStateTracker:
    """Tracks distance since the last completed hop during generation.

    Read-then-advance: the token AT the anchor has distance 0, so read
    `bucket_id` before advancing with `on_tokens_appended`.
    """

    def __init__(self) -> None:
        self._distance = NO_PRIOR_HOP_DISTANCE

    def on_hop_boundary(self) -> None:
        """Call once the retrieved passage has fully landed in the context,
        before generation resumes."""
        self._distance = 0

    def on_tokens_appended(self, num_tokens: int) -> None:
        """Advance by GENERATED tokens only (never passage tokens)."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self._distance != NO_PRIOR_HOP_DISTANCE:
            self._distance += num_tokens

    @property
    def distance(self) -> int:
        return self._distance

    @property
    def bucket_id(self) -> int:
        return recency_bucket_id(self._distance)


class DraftProposer(Protocol):
    def propose(self, context_ids: list[int], num_tokens: int) -> list[int]: ...


class TargetVerifier(Protocol):
    def next_tokens(self, context_ids: list[int], proposed: list[int]) -> list[int]:
        """Greedy target continuations: element i is the target's next token
        given context + proposed[:i]. Must return len(proposed) + 1 tokens
        (the last one is the bonus token when everything is accepted)."""
        ...


def run_speculative_round(
    context_ids: list[int],
    draft: DraftProposer,
    target: TargetVerifier,
    tracker: RecencyStateTracker,
    policy: HopAwareLengthPolicy,
) -> tuple[list[int], int]:
    """One greedy speculative round. Returns (new_context_ids, num_accepted).

    Reads the tracker BEFORE proposing (the proposal length is decided by the
    state at the position being speculated), accepts the proposed prefix up to
    the first rejection, appends the target's correction/bonus token, and
    advances the tracker by every token actually appended.
    """
    gamma = policy.gamma_for(tracker.bucket_id)
    proposed = list(draft.propose(list(context_ids), gamma))[:gamma]
    target_next = list(target.next_tokens(list(context_ids), proposed))
    if len(target_next) < len(proposed) + 1:
        raise ValueError(
            "target verifier must return len(proposed) + 1 tokens "
            f"(got {len(target_next)} for {len(proposed)} proposed)"
        )
    num_accepted = 0
    for draft_token, target_token in zip(proposed, target_next):
        if draft_token != target_token:
            break
        num_accepted += 1
    emitted = proposed[:num_accepted] + [target_next[num_accepted]]
    tracker.on_tokens_appended(len(emitted))
    return list(context_ids) + emitted, num_accepted
