import pytest

from hopspec.data.schema import (
    NO_PRIOR_HOP_BUCKET_ID,
    NO_PRIOR_HOP_DISTANCE,
    NUM_RECENCY_BUCKETS,
    NUM_SEGMENT_TYPES,
    RECENCY_BUCKETS,
    LabeledSequence,
    SegmentType,
    recency_bucket_id,
)


def test_segment_type_values():
    assert SegmentType.QUESTION == 0
    assert SegmentType.THOUGHT == 1
    assert SegmentType.TOOL_CALL == 2
    assert SegmentType.RETRIEVED_PASSAGE == 3
    assert SegmentType.ANSWER == 4
    assert SegmentType.OTHER == 5
    assert SegmentType.TEMPLATE == 6


def test_num_segment_types():
    assert NUM_SEGMENT_TYPES == 7


def test_bucket_counts():
    assert len(RECENCY_BUCKETS) == 6
    assert NO_PRIOR_HOP_BUCKET_ID == 6
    assert NUM_RECENCY_BUCKETS == 7


@pytest.mark.parametrize(
    "distance,expected",
    [(0, 0), (1, 1), (2, 1), (3, 2), (8, 2), (9, 3), (24, 3), (25, 4), (48, 4),
     (49, 5), (1000, 5)],
)
def test_recency_bucket_id_boundaries(distance, expected):
    assert recency_bucket_id(distance) == expected


def test_no_prior_hop_maps_to_own_bucket_never_farthest():
    assert recency_bucket_id(NO_PRIOR_HOP_DISTANCE) == NO_PRIOR_HOP_BUCKET_ID
    assert recency_bucket_id(NO_PRIOR_HOP_DISTANCE) != len(RECENCY_BUCKETS) - 1


def test_large_finite_distance_is_far_bucket_not_no_prior():
    assert recency_bucket_id(NO_PRIOR_HOP_DISTANCE - 1) == len(RECENCY_BUCKETS) - 1


def test_negative_distance_raises():
    with pytest.raises(ValueError):
        recency_bucket_id(-1)


def test_labeled_sequence_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        LabeledSequence(
            input_ids=[1, 2, 3],
            segment_type_ids=[0, 0],
            recency_bucket_ids=[6, 6, 6],
            recency_distances=[NO_PRIOR_HOP_DISTANCE] * 3,
        )


def test_labeled_sequence_accepts_equal_lengths():
    seq = LabeledSequence(
        input_ids=[1, 2],
        segment_type_ids=[0, 0],
        recency_bucket_ids=[6, 6],
        recency_distances=[NO_PRIOR_HOP_DISTANCE] * 2,
        hop_boundary_positions=[],
    )
    assert len(seq) == 2
