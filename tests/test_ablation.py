import copy
import inspect
import json

import pytest
import torch

from hopspec.data.schema import NUM_RECENCY_BUCKETS, NUM_SEGMENT_TYPES
from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel
from hopspec.train.ablation import (
    batches,
    evaluate_draft_model,
    load_records,
    split_records,
    train_draft_model,
)
from hopspec.train.target_policy import GeneratedTokensOnlyPolicy

from conftest import HIDDEN_SIZE, VOCAB_SIZE, TinyTargetModel

PAD_ID = 0


# ---- record IO ----

def test_load_records_rejects_missing_question_id(tmp_path):
    path = tmp_path / "shard.jsonl"
    path.write_text(json.dumps({"input_ids": [1, 2, 3]}) + "\n")
    with pytest.raises(ValueError):
        load_records(str(path))


def test_load_records_roundtrip(tmp_path, records):
    path = tmp_path / "shard.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    loaded = load_records(str(path))
    assert len(loaded) == len(records)
    assert loaded[0]["question_id"] == records[0]["question_id"]


def test_split_records_deterministic_and_disjoint(records):
    train1, heldout1 = split_records(records, eval_fraction=0.25, seed=3)
    train2, heldout2 = split_records(records, eval_fraction=0.25, seed=3)
    assert [r["question_id"] for r in train1] == [r["question_id"] for r in train2]
    train_ids = {r["question_id"] for r in train1}
    heldout_ids = {r["question_id"] for r in heldout1}
    assert not train_ids & heldout_ids
    assert train_ids | heldout_ids == {r["question_id"] for r in records}
    assert heldout1 and train1


def test_batches_chunking():
    items = list(range(5))
    chunks = list(batches(items, 2))
    assert chunks == [[0, 1], [2, 3], [4]]


def test_batches_rejects_zero():
    with pytest.raises(ValueError):
        list(batches([1], 0))


# ---- evaluation contract ----

def test_evaluate_signature_has_no_loss_target_policy():
    # Evaluation must be identical across all arms.
    params = inspect.signature(evaluate_draft_model).parameters
    assert "loss_target_policy" not in params
    assert "policy" not in params


# ---- mandatory end-to-end test (spec §14.12) ----

def make_draft(tiny):
    config = HopSpecDraftConfig(
        target_hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
        draft_hidden_size=32, num_heads=4,
    )
    return HopSpecDraftModel.from_target_embedding(config, tiny.get_input_embeddings())


@pytest.mark.parametrize("hop_signal_enabled", [True, False])
def test_end_to_end_train_and_evaluate(records, hop_signal_enabled):
    tiny = TinyTargetModel()
    draft = make_draft(tiny)
    before = copy.deepcopy(
        {k: v for k, v in draft.state_dict().items() if "token_embedding" not in k}
    )

    train_draft_model(
        draft, tiny, records, PAD_ID, "cpu",
        epochs=1, batch_size=2, lr=1e-3, log_every=0,
        hop_signal_enabled=hop_signal_enabled,
    )
    after = {k: v for k, v in draft.state_dict().items() if "token_embedding" not in k}
    assert any(not torch.equal(before[k], after[k]) for k in before), "model did not update"

    columns = evaluate_draft_model(
        draft, tiny, records, PAD_ID, "cpu",
        batch_size=2, hop_signal_enabled=hop_signal_enabled,
    )
    lengths = {key: len(values) for key, values in columns.items()}
    assert len(set(lengths.values())) == 1
    assert lengths["correct"] > 0

    # Every label the data can produce fits the aux embedding tables.
    for record in records:
        assert max(record["segment_type_ids"]) < NUM_SEGMENT_TYPES
        assert max(record["recency_bucket_ids"]) < NUM_RECENCY_BUCKETS


def test_frozen_embedding_not_updated_by_training(records):
    tiny = TinyTargetModel()
    draft = make_draft(tiny)
    embedding_before = draft.token_embedding.weight.clone()
    train_draft_model(
        draft, tiny, records, PAD_ID, "cpu", epochs=1, batch_size=2, lr=1e-2,
        log_every=0,
    )
    assert torch.equal(draft.token_embedding.weight, embedding_before)


def test_training_accepts_generated_policy(records):
    tiny = TinyTargetModel()
    draft = make_draft(tiny)
    train_draft_model(
        draft, tiny, records, PAD_ID, "cpu", epochs=1, batch_size=2, log_every=0,
        loss_target_policy=GeneratedTokensOnlyPolicy(),
    )


def test_evaluation_positions_identical_across_arms(records):
    """The paired-comparison precondition: all arms are evaluated on
    byte-identical positions — only `correct` may differ."""
    tiny = TinyTargetModel()
    columns_on = evaluate_draft_model(
        make_draft(tiny), tiny, records, PAD_ID, "cpu", hop_signal_enabled=True
    )
    columns_off = evaluate_draft_model(
        make_draft(tiny), tiny, records, PAD_ID, "cpu", hop_signal_enabled=False
    )
    for key in ("recency_distance", "recency_bucket", "hop_index", "target_token",
                "record_index"):
        assert columns_on[key] == columns_off[key], key


def test_evaluate_groups_by_real_bucket_when_disabled(records):
    tiny = TinyTargetModel()
    columns = evaluate_draft_model(
        make_draft(tiny), tiny, records, PAD_ID, "cpu", hop_signal_enabled=False
    )
    # The disabled arm feeds constant buckets to the model, but results are
    # grouped by the REAL bucket — so more than one bucket must appear.
    assert len(set(columns["recency_bucket"])) > 1


def test_evaluate_record_index_tracks_records(records):
    tiny = TinyTargetModel()
    columns = evaluate_draft_model(
        make_draft(tiny), tiny, records, PAD_ID, "cpu", batch_size=2
    )
    assert set(columns["record_index"]) == set(range(len(records)))
