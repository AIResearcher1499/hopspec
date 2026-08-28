import unicodedata

from hopspec.data.schema import (
    NO_PRIOR_HOP_DISTANCE,
    SegmentType,
    Trajectory,
    TrajectoryStep,
    recency_bucket_id,
)
from hopspec.data.segment_labeling import (
    assign_tokens_to_steps,
    label_trajectory,
    trajectory_text,
)

from conftest import simple_offset_tokenizer


def make_trajectory(steps):
    return Trajectory(
        question="q", steps=steps, final_answer=None, is_complete=True,
        context="".join(s.text for s in steps),
    )


def test_trajectory_text_joins_steps(two_hop_trajectory):
    assert trajectory_text(two_hop_trajectory) == two_hop_trajectory.context


def test_straddling_token_attributed_to_first_char_step():
    steps = [
        TrajectoryStep(SegmentType.THOUGHT, "Hel"),
        TrajectoryStep(SegmentType.OTHER, "lo world"),
    ]
    trajectory = make_trajectory(steps)
    # One token "Hello" spans both steps -> attributed to step 0.
    offsets = [(0, 5), (5, 11)]
    assert assign_tokens_to_steps(trajectory, offsets) == [0, 1]


def test_joint_tokenization_single_call(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    ids, offsets = simple_offset_tokenizer(trajectory_text(two_hop_trajectory))
    assert labeled.input_ids == ids
    assert len(labeled.input_ids) == len(offsets)


def test_labels_have_equal_lengths(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    n = len(labeled.input_ids)
    assert len(labeled.segment_type_ids) == n
    assert len(labeled.recency_bucket_ids) == n
    assert len(labeled.recency_distances) == n


def test_pre_retrieval_tokens_are_no_prior(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    first_boundary = labeled.hop_boundary_positions[0]
    # Everything before the first passage has never seen a hop close.
    passage_end = first_boundary
    for i in range(passage_end):
        assert labeled.recency_distances[i] == NO_PRIOR_HOP_DISTANCE


def test_distance_anchored_at_passage_close_not_start(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    passage_id = int(SegmentType.RETRIEVED_PASSAGE)
    # Find the end of the first passage.
    start = labeled.hop_boundary_positions[0]
    end = start
    while end < len(labeled.input_ids) and labeled.segment_type_ids[end] == passage_id:
        end += 1
    # First token AFTER the closed passage has distance 0, the next 1, ...
    assert labeled.recency_distances[end] == 0
    assert labeled.recency_distances[end + 1] == 1
    # Passage-internal tokens are NOT distance 0 from their own passage.
    assert labeled.recency_distances[start] == NO_PRIOR_HOP_DISTANCE


def test_first_post_hop_token_gets_bucket_zero(two_hop_trajectory):
    """Mandatory test 7: the first post-hop token is what gets labeled bucket 0."""
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    passage_id = int(SegmentType.RETRIEVED_PASSAGE)
    for boundary in labeled.hop_boundary_positions:
        end = boundary
        while end < len(labeled.input_ids) and labeled.segment_type_ids[end] == passage_id:
            end += 1
        if end < len(labeled.input_ids):
            assert labeled.recency_bucket_ids[end] == 0


def test_passage_internal_tokens_count_from_previous_hop(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    second_boundary = labeled.hop_boundary_positions[1]
    # Second passage's internal tokens keep counting from the FIRST hop's close.
    distance = labeled.recency_distances[second_boundary]
    assert distance != NO_PRIOR_HOP_DISTANCE
    assert distance > 0


def test_hop_boundary_positions_are_passage_starts(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    assert len(labeled.hop_boundary_positions) == 2
    passage_id = int(SegmentType.RETRIEVED_PASSAGE)
    for boundary in labeled.hop_boundary_positions:
        assert labeled.segment_type_ids[boundary] == passage_id
        assert boundary == 0 or labeled.segment_type_ids[boundary - 1] != passage_id


def test_buckets_are_function_of_distances(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    for bucket, distance in zip(labeled.recency_bucket_ids, labeled.recency_distances):
        assert bucket == recency_bucket_id(distance)


def test_segment_types_match_owning_steps(two_hop_trajectory):
    labeled = label_trajectory(two_hop_trajectory, simple_offset_tokenizer)
    text = trajectory_text(two_hop_trajectory)
    _ids, offsets = simple_offset_tokenizer(text)
    assignments = assign_tokens_to_steps(two_hop_trajectory, offsets)
    for token_index, step_index in enumerate(assignments):
        expected = int(two_hop_trajectory.steps[step_index].segment_type)
        assert labeled.segment_type_ids[token_index] == expected


def test_nfd_text_offsets_stay_monotonic():
    # NFD "é" = "e" + combining acute. Offsets index the ORIGINAL string and
    # must stay monotonic; labels recompute identically.
    nfd = unicodedata.normalize("NFD", "café time")
    steps = [
        TrajectoryStep(SegmentType.THOUGHT, nfd[:5]),
        TrajectoryStep(SegmentType.OTHER, nfd[5:]),
    ]
    trajectory = make_trajectory(steps)
    labeled = label_trajectory(trajectory, simple_offset_tokenizer)
    _ids, offsets = simple_offset_tokenizer(nfd)
    assert all(a[1] <= b[0] or a[0] < b[0] for a, b in zip(offsets, offsets[1:]))
    relabeled = label_trajectory(trajectory, simple_offset_tokenizer)
    assert relabeled.segment_type_ids == labeled.segment_type_ids
