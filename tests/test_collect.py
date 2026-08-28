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


# ---- --resume (migration plan §4) ----
#
# collect_shard opens the output for APPEND, so a killed run leaves valid
# partial output but a naive restart re-collects everything and duplicates
# every id. On a rented box a killed run is normal, not exceptional.

from hopspec.data.collect import collected_question_ids


def test_collected_question_ids_reads_a_shard(retriever, tmp_path):
    out = tmp_path / "shard.jsonl"
    questions = [(f"q{i}", f"Question {i}?") for i in range(3)]
    collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 3), retriever,
                  simple_offset_tokenizer, str(out))
    assert collected_question_ids(str(out)) == {"q0", "q1", "q2"}


def test_collected_question_ids_on_a_missing_file():
    assert collected_question_ids("/nonexistent/shard.jsonl") == set()


def test_collected_question_ids_tolerates_a_truncated_tail(retriever, tmp_path):
    """A killed collect can leave a half-written final line. Surviving that is
    the entire point of resuming."""
    out = tmp_path / "shard.jsonl"
    collect_shard([("q0", "Question 0?")], MockLLM(list(TWO_HOP_RESPONSES)),
                  retriever, simple_offset_tokenizer, str(out))
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"question_id": "q1", "context": "half a rec')
    assert collected_question_ids(str(out)) == {"q0"}


def test_resume_appends_only_the_missing_records(retriever, tmp_path):
    out = tmp_path / "shard.jsonl"
    questions = [(f"q{i}", f"Question {i}?") for i in range(5)]

    first = collect_shard(questions[:3], MockLLM(TWO_HOP_RESPONSES * 3), retriever,
                          simple_offset_tokenizer, str(out))
    assert first == 3

    # the same full question list, resumed: only the two new ones are collected
    second = collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 5), retriever,
                           simple_offset_tokenizer, str(out), resume=True)
    assert second == 2

    ids = [json.loads(line)["question_id"] for line in open(out, encoding="utf-8")]
    assert ids == ["q0", "q1", "q2", "q3", "q4"]
    # the duplicate-id check in 06_validate_shard.py must stay clean
    assert len(ids) == len(set(ids))


def test_resume_is_a_no_op_when_everything_is_collected(retriever, tmp_path):
    out = tmp_path / "shard.jsonl"
    questions = [(f"q{i}", f"Question {i}?") for i in range(3)]
    collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 3), retriever,
                  simple_offset_tokenizer, str(out))
    again = collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 3), retriever,
                          simple_offset_tokenizer, str(out), resume=True)
    assert again == 0
    assert sum(1 for _ in open(out, encoding="utf-8")) == 3


def test_without_resume_a_restart_duplicates_every_id(retriever, tmp_path):
    """Pins the failure mode --resume exists to prevent, so nobody 'simplifies'
    the flag away."""
    out = tmp_path / "shard.jsonl"
    questions = [(f"q{i}", f"Question {i}?") for i in range(3)]
    for _ in range(2):
        collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 3), retriever,
                      simple_offset_tokenizer, str(out))
    ids = [json.loads(line)["question_id"] for line in open(out, encoding="utf-8")]
    assert len(ids) == 6 and len(set(ids)) == 3
