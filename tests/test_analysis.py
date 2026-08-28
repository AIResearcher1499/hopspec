import pytest

from hopspec.data.schema import (
    NO_PRIOR_HOP_BUCKET_ID,
    NO_PRIOR_HOP_DISTANCE,
    SegmentType,
    recency_bucket_id,
)
from hopspec.eval.analysis import (
    format_table,
    hop_span_lengths,
    novel_hop_flags,
    rebucket,
    select,
    summarize,
)


def make_columns():
    distances = [NO_PRIOR_HOP_DISTANCE, 0, 1, 5, 30, 0, 2]
    return {
        "recency_distance": distances,
        "recency_bucket": [recency_bucket_id(d) for d in distances],
        "hop_index": [-1, 0, 0, 0, 0, 1, 1],
        "target_token": [10, 11, 12, 13, 14, 15, 16],
        "correct": [1, 0, 1, 1, 0, 1, 1],
        "record_index": [0, 0, 0, 0, 0, 0, 0],
    }


# ---- novel_hop_flags ----

def passage(text, hop_index):
    return {
        "segment_type": int(SegmentType.RETRIEVED_PASSAGE),
        "text": f"Observation: - {text}\n",
        "hop_index": hop_index,
    }


def thought(text):
    return {"segment_type": int(SegmentType.THOUGHT), "text": text, "hop_index": None}


def test_novel_hop_flags_detects_repeat_retrieval():
    records = [{"steps": [
        thought("Question: q\n"),
        passage("the sky is blue", 0),
        thought("Thought: hmm\n"),
        passage("the sky is blue", 1),  # stuck agent, same passage again
        passage("grass is green", 2),
    ]}]
    flags = novel_hop_flags(records)
    assert flags[(0, 0)] is True
    assert flags[(0, 1)] is False
    assert flags[(0, 2)] is True


def test_novel_hop_flags_real_records(records):
    flags = novel_hop_flags(records)
    # Two distinct retrievals per record in the fixture -> all novel.
    assert all(flags.values())
    assert len(flags) == 2 * len(records)


# ---- span lengths and select ----

def test_hop_span_lengths():
    spans = hop_span_lengths(make_columns())
    assert spans[(0, 0)] == 31  # max distance 30 -> span 31
    assert spans[(0, 1)] == 3


def test_select_exclude_no_prior():
    out = select(make_columns(), exclude_no_prior=True)
    assert NO_PRIOR_HOP_DISTANCE not in out["recency_distance"]
    assert len(out["correct"]) == 6


def test_select_matched_cohort():
    out = select(make_columns(), min_hop_span=10)
    # Only hop 0 reaches span >= 10; hop 1 and the pre-hop position drop out.
    assert set(out["hop_index"]) == {0}
    assert len(out["correct"]) == 4


def test_select_novel_flags():
    flags = {(0, 0): True, (0, 1): False}
    out = select(make_columns(), novel_flags=flags)
    assert set(out["hop_index"]) == {0}


def test_select_preserves_column_alignment():
    out = select(make_columns(), exclude_no_prior=True)
    lengths = {len(v) for v in out.values()}
    assert len(lengths) == 1


def test_select_missing_column_rejected():
    with pytest.raises(ValueError):
        select({"recency_distance": [1]})


# ---- rebucket ----

def test_rebucket_recomputes_from_distances():
    out = rebucket(make_columns(), boundaries=((0, 4), (5, None)))
    assert out["recency_bucket"] == [2, 0, 0, 1, 1, 0, 0]


def test_rebucket_default_matches_schema():
    columns = make_columns()
    assert rebucket(columns)["recency_bucket"] == columns["recency_bucket"]


def test_rebucket_no_prior_gets_extra_bucket():
    out = rebucket(make_columns(), boundaries=((0, None),))
    assert out["recency_bucket"][0] == 1  # len(boundaries)


# ---- summarize / format_table ----

def test_summarize_acceptance():
    summaries = {s.bucket: s for s in summarize(make_columns())}
    assert summaries[0].n == 2  # distances 0, 0
    assert summaries[0].acceptance == 0.5
    assert summaries[NO_PRIOR_HOP_BUCKET_ID].n == 1


def test_summarize_marks_small_buckets_degenerate():
    for summary in summarize(make_columns()):
        assert summary.degenerate  # every bucket here is tiny


def test_format_table_includes_majority_column():
    table = format_table(summarize(make_columns()), title="test")
    assert "majority" in table
    assert "DEGENERATE" in table
    assert "test" in table


def test_format_table_empty():
    assert "no measured positions" in format_table([])
