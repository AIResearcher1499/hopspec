"""Measurement masks and degeneracy diagnostics."""

from __future__ import annotations

from collections import Counter

import torch

from hopspec.data.schema import SegmentType

# QUESTION and RETRIEVED_PASSAGE are prefill-only — inserted, never generated,
# never a speculation target. TEMPLATE is generated but excluded as
# template-determined: the first version counted every position, and because
# "Observation: " is a mechanically inserted literal it reported ~100%
# acceptance at buckets 0/1 and looked like a finding.
DECODE_SEGMENT_TYPES = (SegmentType.THOUGHT, SegmentType.TOOL_CALL, SegmentType.ANSWER)


def decode_phase_mask(
    segment_type_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    keep = torch.zeros_like(segment_type_ids, dtype=torch.bool)
    for segment in DECODE_SEGMENT_TYPES:
        keep |= segment_type_ids == int(segment)
    return keep & attention_mask.bool()


DISABLED_RECENCY_BUCKET = 0


def resolve_recency_buckets_for_model(
    segment_type_ids: torch.Tensor,
    recency_bucket_ids: torch.Tensor,
    hop_signal_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """What the MODEL sees. Disabled arm gets constant rows for both aux
    tables (identical architecture and parameter count, zero signal). Results
    are always grouped by the REAL bucket regardless."""
    if hop_signal_enabled:
        return segment_type_ids, recency_bucket_ids
    return (
        torch.zeros_like(segment_type_ids),
        torch.full_like(recency_bucket_ids, DISABLED_RECENCY_BUCKET),
    )


DEGENERATE_DISTINCT_THRESHOLD = 10
MAJORITY_DEGENERACY_THRESHOLD = 0.5


def bucket_token_stats(
    recency_buckets: list[int], target_tokens: list[int]
) -> dict[int, dict]:
    """Per-bucket distinct-token count and majority-class rate. Report these
    next to every acceptance number, always.

    Majority rate is the load-bearing signal: a count-only threshold is a
    leak on its own — a bucket once had 18 distinct tokens (passing a naive
    count check) while one token was still 64% of answers, and a constant
    predictor beat the "improved" model there.
    """
    tokens_by_bucket: dict[int, Counter] = {}
    for bucket, token in zip(recency_buckets, target_tokens):
        tokens_by_bucket.setdefault(bucket, Counter())[token] += 1
    stats = {}
    for bucket, counts in sorted(tokens_by_bucket.items()):
        n = sum(counts.values())
        distinct = len(counts)
        majority_rate = counts.most_common(1)[0][1] / n
        stats[bucket] = {
            "n": n,
            "distinct": distinct,
            "majority_rate": majority_rate,
            "degenerate": (
                distinct <= DEGENERATE_DISTINCT_THRESHOLD
                or majority_rate >= MAJORITY_DEGENERACY_THRESHOLD
            ),
        }
    return stats
