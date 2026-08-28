"""Token-level labeling by character offset over ONE joint tokenization.

Per-step tokenization then concatenating ids cannot reproduce the tokenization
of the joined string, because subword merges cross step boundaries — and the
discrepancies land exactly at step boundaries, which is where the measurement
looks. So: tokenize the whole trajectory in one call and map tokens back to
steps by the character offset of their FIRST character.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Callable

from hopspec.data.schema import (
    NO_PRIOR_HOP_DISTANCE,
    LabeledSequence,
    SegmentType,
    Trajectory,
    recency_bucket_id,
)

# text -> (input_ids, [(char_start, char_end), ...])
OffsetTokenizer = Callable[[str], tuple[list[int], list[tuple[int, int]]]]


def hf_offset_tokenizer(hf_tokenizer) -> OffsetTokenizer:
    def tokenize(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    return tokenize


def trajectory_text(trajectory: Trajectory) -> str:
    return "".join(step.text for step in trajectory.steps)


def assign_tokens_to_steps(
    trajectory: Trajectory, offsets: list[tuple[int, int]]
) -> list[int]:
    """Step index per token. A token straddling a step boundary is attributed
    to the step its FIRST character falls in."""
    cumulative_ends: list[int] = []
    total = 0
    for step in trajectory.steps:
        total += len(step.text)
        cumulative_ends.append(total)
    assignments = []
    for start, _end in offsets:
        if start >= total:
            raise ValueError(
                f"token offset {start} beyond trajectory text length {total}"
            )
        assignments.append(bisect_right(cumulative_ends, start))
    return assignments


def label_trajectory(trajectory: Trajectory, tokenize: OffsetTokenizer) -> LabeledSequence:
    text = trajectory_text(trajectory)
    input_ids, offsets = tokenize(text)
    step_of_token = assign_tokens_to_steps(trajectory, offsets)

    segment_type_ids = [int(trajectory.steps[s].segment_type) for s in step_of_token]

    # Distance counts tokens since the END of the most recently CLOSED
    # RETRIEVED_PASSAGE step — since the model resumed generating — not since
    # its start. Anchoring at the start puts the fine-grained near-zero
    # buckets inside mechanically inserted passage content that is never a
    # speculation target. Passage-internal tokens keep counting from the
    # previous hop's close; the online tracker never sees them, and the two
    # agree at every position that is actually speculated on.
    recency_distances: list[int] = []
    hop_boundary_positions: list[int] = []
    last_anchor: int | None = None
    num_tokens = len(input_ids)
    for i in range(num_tokens):
        step_index = step_of_token[i]
        step = trajectory.steps[step_index]
        is_passage = step.segment_type is SegmentType.RETRIEVED_PASSAGE
        if is_passage and (i == 0 or step_of_token[i - 1] != step_index):
            hop_boundary_positions.append(i)
        recency_distances.append(
            NO_PRIOR_HOP_DISTANCE if last_anchor is None else i - last_anchor
        )
        passage_closes_here = is_passage and (
            i == num_tokens - 1 or step_of_token[i + 1] != step_index
        )
        if passage_closes_here:
            last_anchor = i + 1

    recency_bucket_ids = [recency_bucket_id(d) for d in recency_distances]
    return LabeledSequence(
        input_ids=input_ids,
        segment_type_ids=segment_type_ids,
        recency_bucket_ids=recency_bucket_ids,
        recency_distances=recency_distances,
        hop_boundary_positions=hop_boundary_positions,
    )
