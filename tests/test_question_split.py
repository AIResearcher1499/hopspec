import json

import pytest

from hopspec.data.question_split import get_or_create_split, split_question_ids

IDS = [f"q{i}" for i in range(50)]


def test_split_disjoint_and_exhaustive():
    collect_ids, eval_ids = split_question_ids(IDS, eval_fraction=0.2, seed=0)
    assert not set(collect_ids) & set(eval_ids)
    assert set(collect_ids) | set(eval_ids) == set(IDS)


def test_split_fraction():
    _collect, eval_ids = split_question_ids(IDS, eval_fraction=0.2, seed=0)
    assert len(eval_ids) == 10


def test_split_order_independent():
    forward = split_question_ids(IDS, seed=1)
    backward = split_question_ids(list(reversed(IDS)), seed=1)
    assert forward == backward


def test_split_seed_determinism():
    assert split_question_ids(IDS, seed=2) == split_question_ids(IDS, seed=2)
    assert split_question_ids(IDS, seed=2) != split_question_ids(IDS, seed=3)


def test_split_deduplicates():
    collect_ids, eval_ids = split_question_ids(IDS + IDS, eval_fraction=0.2)
    assert len(collect_ids) + len(eval_ids) == len(IDS)


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_question_ids(IDS, eval_fraction=1.5)


def test_get_or_create_persists(tmp_path):
    path = tmp_path / "split.json"
    first = get_or_create_split(IDS, str(path))
    assert path.exists()
    second = get_or_create_split(IDS, str(path))
    assert first == second


def test_persisted_split_immutable_when_ids_shift(tmp_path):
    """A split must never silently move a question between pools after
    collection has started."""
    path = tmp_path / "split.json"
    first = get_or_create_split(IDS, str(path))
    shifted = get_or_create_split(IDS + ["brand-new-id"], str(path))
    assert shifted == first
    assert "brand-new-id" not in shifted[0] + shifted[1]


def test_persisted_file_records_parameters(tmp_path):
    path = tmp_path / "split.json"
    get_or_create_split(IDS, str(path), eval_fraction=0.3, seed=7)
    stored = json.loads(path.read_text())
    assert stored["eval_fraction"] == 0.3
    assert stored["seed"] == 7
