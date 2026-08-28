"""Basic acceptance metrics over raw per-position columns."""

from __future__ import annotations


def acceptance_rate_by_bucket(
    recency_buckets: list[int], correct: list[int]
) -> dict[int, tuple[int, float]]:
    """bucket -> (n, acceptance rate)."""
    if len(recency_buckets) != len(correct):
        raise ValueError("column lengths differ")
    totals: dict[int, int] = {}
    hits: dict[int, int] = {}
    for bucket, is_correct in zip(recency_buckets, correct):
        totals[bucket] = totals.get(bucket, 0) + 1
        hits[bucket] = hits.get(bucket, 0) + int(bool(is_correct))
    return {
        bucket: (totals[bucket], hits[bucket] / totals[bucket])
        for bucket in sorted(totals)
    }
