import torch

from hopspec.data.schema import SegmentType
from hopspec.eval.diagnostic import (
    DECODE_SEGMENT_TYPES,
    DISABLED_RECENCY_BUCKET,
    bucket_token_stats,
    decode_phase_mask,
    resolve_recency_buckets_for_model,
)


def test_decode_segment_types():
    assert set(DECODE_SEGMENT_TYPES) == {
        SegmentType.THOUGHT, SegmentType.TOOL_CALL, SegmentType.ANSWER,
    }


def test_mask_drops_template_keeps_thought():
    segments = torch.tensor([[int(SegmentType.TEMPLATE), int(SegmentType.THOUGHT)]])
    mask = decode_phase_mask(segments, torch.ones_like(segments))
    assert mask.tolist() == [[False, True]]


def test_mask_drops_prefill_segments():
    segments = torch.tensor([[
        int(SegmentType.QUESTION),
        int(SegmentType.RETRIEVED_PASSAGE),
        int(SegmentType.TOOL_CALL),
        int(SegmentType.ANSWER),
        int(SegmentType.OTHER),
    ]])
    mask = decode_phase_mask(segments, torch.ones_like(segments))
    assert mask.tolist() == [[False, False, True, True, False]]


def test_mask_drops_padding():
    segments = torch.tensor([[int(SegmentType.THOUGHT), int(SegmentType.THOUGHT)]])
    attention = torch.tensor([[1, 0]])
    assert decode_phase_mask(segments, attention).tolist() == [[True, False]]


def test_resolve_enabled_passthrough():
    segments = torch.tensor([[1, 2]])
    buckets = torch.tensor([[0, 3]])
    out_segments, out_buckets = resolve_recency_buckets_for_model(segments, buckets, True)
    assert torch.equal(out_segments, segments)
    assert torch.equal(out_buckets, buckets)


def test_resolve_disabled_gives_constants():
    segments = torch.tensor([[1, 2, 6]])
    buckets = torch.tensor([[0, 3, 5]])
    out_segments, out_buckets = resolve_recency_buckets_for_model(segments, buckets, False)
    assert out_segments.unique().tolist() == [0]
    assert out_buckets.unique().tolist() == [DISABLED_RECENCY_BUCKET]


def test_resolve_disabled_does_not_mutate():
    segments = torch.tensor([[1, 2]])
    buckets = torch.tensor([[0, 3]])
    resolve_recency_buckets_for_model(segments, buckets, False)
    assert buckets.tolist() == [[0, 3]]


def test_stats_flags_low_distinct_as_degenerate():
    buckets = [0] * 20
    tokens = list(range(5)) * 4  # 5 distinct <= 10
    stats = bucket_token_stats(buckets, tokens)
    assert stats[0]["degenerate"]


def test_stats_flags_high_majority_even_with_many_distinct():
    """Mandatory test 8: majority rate is the load-bearing signal. 18 distinct
    tokens passes a naive count check while one token is still 64%."""
    tokens = [7] * 64 + list(range(100, 117)) * 2  # 18 distinct, majority 64%
    buckets = [0] * len(tokens)
    stats = bucket_token_stats(buckets, tokens)
    assert stats[0]["distinct"] == 18
    assert stats[0]["majority_rate"] > 0.5
    assert stats[0]["degenerate"]


def test_stats_healthy_bucket_not_degenerate():
    tokens = list(range(30)) * 2  # 30 distinct, majority ~3%
    buckets = [1] * len(tokens)
    stats = bucket_token_stats(buckets, tokens)
    assert not stats[1]["degenerate"]


def test_stats_reports_n_and_majority():
    stats = bucket_token_stats([0, 0, 0, 1], [5, 5, 6, 9])
    assert stats[0]["n"] == 3
    assert stats[0]["majority_rate"] == 2 / 3
    assert stats[1]["n"] == 1
