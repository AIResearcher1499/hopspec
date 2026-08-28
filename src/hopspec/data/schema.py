"""Core label schema: segment types, recency buckets, trajectory containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SegmentType(IntEnum):
    QUESTION = 0
    THOUGHT = 1
    TOOL_CALL = 2
    RETRIEVED_PASSAGE = 3
    ANSWER = 4
    OTHER = 5      # padding and malformed steps
    TEMPLATE = 6   # fixed ReAct scaffolding ("Thought:", "Action: Search[", ...)


NUM_SEGMENT_TYPES = len(SegmentType)  # 7

# (lo, hi) inclusive ranges over token distance since the last closed passage.
# Fitted to the measured post-hop span distribution (p50=27, p95=74, max=152),
# not chosen a priori.
RECENCY_BUCKETS = ((0, 0), (1, 2), (3, 8), (9, 24), (25, 48), (49, None))

# Tokens before ANY retrieval get their own bucket. Pinning them to the
# farthest distance bucket conflates "settled after a hop" with "before any
# retrieval ever" — on real data that made the far bucket 99.6% pre-retrieval.
NO_PRIOR_HOP_BUCKET_ID = len(RECENCY_BUCKETS)      # 6
NUM_RECENCY_BUCKETS = len(RECENCY_BUCKETS) + 1     # 7
NO_PRIOR_HOP_DISTANCE = 10**9


def recency_bucket_id(distance: int) -> int:
    """Map a token distance to its recency bucket id.

    Only the exact NO_PRIOR_HOP_DISTANCE sentinel maps to
    NO_PRIOR_HOP_BUCKET_ID; every other non-negative distance falls in a
    distance bucket (the last one is open-ended).
    """
    if distance < 0:
        raise ValueError(f"recency distance must be non-negative, got {distance}")
    if distance == NO_PRIOR_HOP_DISTANCE:
        return NO_PRIOR_HOP_BUCKET_ID
    for bucket_id, (lo, hi) in enumerate(RECENCY_BUCKETS):
        if distance >= lo and (hi is None or distance <= hi):
            return bucket_id
    raise AssertionError("RECENCY_BUCKETS must end with an open-ended bucket")


@dataclass
class TrajectoryStep:
    segment_type: SegmentType
    text: str
    hop_index: int | None = None


@dataclass
class Trajectory:
    question: str
    steps: list[TrajectoryStep]
    final_answer: str | None
    is_complete: bool
    context: str


@dataclass
class LabeledSequence:
    input_ids: list[int]
    segment_type_ids: list[int]
    recency_bucket_ids: list[int]
    recency_distances: list[int]
    hop_boundary_positions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        lengths = {
            "input_ids": len(self.input_ids),
            "segment_type_ids": len(self.segment_type_ids),
            "recency_bucket_ids": len(self.recency_bucket_ids),
            "recency_distances": len(self.recency_distances),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"per-token arrays must have equal lengths, got {lengths}")

    def __len__(self) -> int:
        return len(self.input_ids)
