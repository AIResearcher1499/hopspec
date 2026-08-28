"""Adaptive speculation length: recency bucket -> proposal length gamma.

Composable with EAGLE-2's confidence-based dynamic tree (it sets the tree's
target length), not a replacement for it.
"""

from __future__ import annotations

from hopspec.data.schema import NO_PRIOR_HOP_BUCKET_ID, NUM_RECENCY_BUCKETS, RECENCY_BUCKETS

# Hand-set placeholders — tune before trusting.
DEFAULT_GAMMA_BY_BUCKET = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 8, NO_PRIOR_HOP_BUCKET_ID: 8}


class HopAwareLengthPolicy:
    def __init__(
        self,
        gamma_by_bucket: dict[int, int] | None = None,
        gamma_min: int = 1,
        gamma_max: int = 8,
    ):
        if gamma_min < 1 or gamma_max < gamma_min:
            raise ValueError("need 1 <= gamma_min <= gamma_max")
        table = dict(DEFAULT_GAMMA_BY_BUCKET if gamma_by_bucket is None else gamma_by_bucket)
        missing = set(range(NUM_RECENCY_BUCKETS)) - set(table)
        if missing:
            raise ValueError(f"gamma_by_bucket missing bucket entries: {sorted(missing)}")
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_by_bucket = {
            bucket: min(max(int(gamma), gamma_min), gamma_max)
            for bucket, gamma in table.items()
        }

    def gamma_for(self, bucket_id: int) -> int:
        if bucket_id not in self.gamma_by_bucket:
            raise ValueError(f"unknown recency bucket {bucket_id}")
        return self.gamma_by_bucket[bucket_id]

    def is_monotonic(self) -> bool:
        """Sanity check: gamma non-decreasing over the distance buckets
        (the no-prior bucket is not part of the distance ordering)."""
        gammas = [self.gamma_by_bucket[b] for b in range(len(RECENCY_BUCKETS))]
        return all(a <= b for a, b in zip(gammas, gammas[1:]))
