"""Offline summarizers over the raw per-position columns.

Everything here is a pure re-slicing of columns already saved by
`evaluate_draft_model` (--raw-out). The measurement definition changed four
times during development and each change previously forced a fresh GPU run;
with raw columns saved, re-bucketing, cohort filters and novelty filters are
offline operations. The expensive pass must never have to be repeated for an
analysis change.
"""

from __future__ import annotations

from dataclasses import dataclass

from hopspec.data.schema import (
    NO_PRIOR_HOP_BUCKET_ID,
    NO_PRIOR_HOP_DISTANCE,
    RECENCY_BUCKETS,
    SegmentType,
)
from hopspec.eval.diagnostic import bucket_token_stats

RAW_COLUMNS = (
    "recency_distance",
    "recency_bucket",
    "hop_index",
    "target_token",
    "correct",
    "record_index",
)


@dataclass
class BucketSummary:
    bucket: int
    n: int
    acceptance: float
    distinct_tokens: int
    majority_rate: float
    degenerate: bool


def _check_columns(columns: dict[str, list[int]]) -> None:
    missing = [key for key in RAW_COLUMNS if key not in columns]
    if missing:
        raise ValueError(f"missing raw columns: {missing}")
    lengths = {key: len(columns[key]) for key in RAW_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"raw columns must have equal lengths, got {lengths}")


def novel_hop_flags(records: list[dict]) -> dict[tuple[int, int], bool]:
    """(record_index, hop_index) -> did the retrieval bring text not already
    in context? 17-18% of hops re-retrieve a passage already present (a stuck
    agent repeating its query); those boundaries shift nothing and cannot
    test the hypothesis."""
    flags: dict[tuple[int, int], bool] = {}
    for record_index, record in enumerate(records):
        prefix = ""
        for step in record["steps"]:
            if step["segment_type"] == int(SegmentType.RETRIEVED_PASSAGE):
                core = step["text"]
                if core.startswith("Observation: "):
                    core = core[len("Observation: "):]
                core = core.strip()
                flags[(record_index, step["hop_index"])] = bool(core) and core not in prefix
            prefix += step["text"]
    return flags


def hop_span_lengths(columns: dict[str, list[int]]) -> dict[tuple[int, int], int]:
    """(record_index, hop_index) -> generated-span length reached by that hop,
    measured as max finite distance among measured positions + 1."""
    _check_columns(columns)
    spans: dict[tuple[int, int], int] = {}
    for record_index, hop_index, distance in zip(
        columns["record_index"], columns["hop_index"], columns["recency_distance"]
    ):
        if hop_index < 0 or distance == NO_PRIOR_HOP_DISTANCE:
            continue
        key = (record_index, hop_index)
        spans[key] = max(spans.get(key, 0), distance + 1)
    return spans


def select(
    columns: dict[str, list[int]],
    *,
    exclude_no_prior: bool = False,
    min_hop_span: int | None = None,
    novel_flags: dict[tuple[int, int], bool] | None = None,
) -> dict[str, list[int]]:
    """Filter raw columns.

    min_hop_span builds the matched cohort: far distance buckets are only
    reachable by hops with long generated spans, so comparing near-hop vs
    far-hop across all hops compares different populations. Restricting to
    hops that reach the far bucket makes every bucket come from the same hops.
    """
    _check_columns(columns)
    spans = hop_span_lengths(columns) if min_hop_span is not None else None
    keep_indices = []
    for i in range(len(columns["record_index"])):
        distance = columns["recency_distance"][i]
        hop_key = (columns["record_index"][i], columns["hop_index"][i])
        if exclude_no_prior and distance == NO_PRIOR_HOP_DISTANCE:
            continue
        if spans is not None:
            if columns["hop_index"][i] < 0 or spans.get(hop_key, 0) < min_hop_span:
                continue
        if novel_flags is not None:
            if columns["hop_index"][i] < 0 or not novel_flags.get(hop_key, False):
                continue
        keep_indices.append(i)
    return {key: [columns[key][i] for i in keep_indices] for key in RAW_COLUMNS}


def rebucket(
    columns: dict[str, list[int]],
    boundaries: tuple[tuple[int, int | None], ...] = RECENCY_BUCKETS,
) -> dict[str, list[int]]:
    """Recompute recency_bucket from stored raw distances — no GPU pass."""
    _check_columns(columns)

    def bucket_of(distance: int) -> int:
        if distance == NO_PRIOR_HOP_DISTANCE:
            return len(boundaries)
        for bucket_id, (lo, hi) in enumerate(boundaries):
            if distance >= lo and (hi is None or distance <= hi):
                return bucket_id
        raise ValueError(f"distance {distance} fits no bucket in {boundaries}")

    out = {key: list(columns[key]) for key in RAW_COLUMNS}
    out["recency_bucket"] = [bucket_of(d) for d in columns["recency_distance"]]
    return out


def summarize(columns: dict[str, list[int]]) -> list[BucketSummary]:
    _check_columns(columns)
    stats = bucket_token_stats(columns["recency_bucket"], columns["target_token"])
    hits: dict[int, int] = {}
    for bucket, is_correct in zip(columns["recency_bucket"], columns["correct"]):
        hits[bucket] = hits.get(bucket, 0) + int(bool(is_correct))
    return [
        BucketSummary(
            bucket=bucket,
            n=info["n"],
            acceptance=hits.get(bucket, 0) / info["n"],
            distinct_tokens=info["distinct"],
            majority_rate=info["majority_rate"],
            degenerate=info["degenerate"],
        )
        for bucket, info in stats.items()
    ]


def format_table(summaries: list[BucketSummary], title: str | None = None) -> str:
    lines = []
    if title:
        lines.append(f"== {title} ==")
    header = f"{'bucket':>6} {'n':>8} {'accept':>8} {'distinct':>8} {'majority':>8}  verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for summary in summaries:
        bucket_label = (
            "no-hop" if summary.bucket == NO_PRIOR_HOP_BUCKET_ID else str(summary.bucket)
        )
        verdict = "DEGENERATE" if summary.degenerate else "ok"
        lines.append(
            f"{bucket_label:>6} {summary.n:>8} {summary.acceptance:>8.4f} "
            f"{summary.distinct_tokens:>8} {summary.majority_rate:>8.4f}  {verdict}"
        )
    if not summaries:
        lines.append("(no measured positions)")
    return "\n".join(lines)
