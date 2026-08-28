import json

from hopspec.data.agent_pipeline import MockLLM
from hopspec.data.collect import collect_shard, trajectory_to_record
from hopspec.data.schema import SegmentType, recency_bucket_id

from conftest import TWO_HOP_RESPONSES, simple_offset_tokenizer


def test_record_has_all_fields(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    for key in ("question_id", "question", "final_answer", "is_complete", "context",
                "steps", "input_ids", "segment_type_ids", "recency_bucket_ids",
                "recency_distances", "hop_boundary_positions"):
        assert key in record
    assert record["question_id"] == "qid-1"
    assert record["is_complete"] is True


def test_record_steps_reproduce_context(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    assert "".join(step["text"] for step in record["steps"]) == record["context"]


def test_record_arrays_have_equal_lengths(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    n = len(record["input_ids"])
    assert len(record["segment_type_ids"]) == n
    assert len(record["recency_bucket_ids"]) == n
    assert len(record["recency_distances"]) == n


def test_record_buckets_consistent_with_distances(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    for bucket, distance in zip(record["recency_bucket_ids"], record["recency_distances"]):
        assert bucket == recency_bucket_id(distance)


def test_record_is_json_serializable(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    assert json.loads(json.dumps(record)) == record


def test_record_step_types_are_ints(two_hop_trajectory):
    record = trajectory_to_record(two_hop_trajectory, "qid-1", simple_offset_tokenizer)
    passage = int(SegmentType.RETRIEVED_PASSAGE)
    assert any(step["segment_type"] == passage for step in record["steps"])
    assert all(isinstance(step["segment_type"], int) for step in record["steps"])


def test_collect_shard_writes_jsonl(tmp_path, retriever):
    out = tmp_path / "shard.jsonl"
    questions = [("q1", "Where was the author born?")]
    llm = MockLLM(TWO_HOP_RESPONSES)
    count = collect_shard(questions, llm, retriever, simple_offset_tokenizer, str(out))
    assert count == 1
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question_id"] == "q1"


def test_collect_shard_appends(tmp_path, retriever):
    out = tmp_path / "shard.jsonl"
    for qid in ("q1", "q2"):
        collect_shard(
            [(qid, "Where was the author born?")], MockLLM(TWO_HOP_RESPONSES),
            retriever, simple_offset_tokenizer, str(out),
        )
    lines = out.read_text().strip().splitlines()
    assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2"]


def test_collect_shard_on_record_callback(tmp_path, retriever):
    seen = []
    collect_shard(
        [("q1", "Where was the author born?")], MockLLM(TWO_HOP_RESPONSES),
        retriever, simple_offset_tokenizer, str(tmp_path / "s.jsonl"),
        on_record=seen.append,
    )
    assert len(seen) == 1
