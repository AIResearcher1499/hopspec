import pytest

from hopspec.eval.metrics import acceptance_rate_by_bucket


def test_rates_by_bucket():
    buckets = [0, 0, 1, 1, 1]
    correct = [1, 0, 1, 1, 0]
    out = acceptance_rate_by_bucket(buckets, correct)
    assert out[0] == (2, 0.5)
    assert out[1] == (3, pytest.approx(2 / 3))


def test_sorted_bucket_keys():
    out = acceptance_rate_by_bucket([5, 0, 3], [1, 1, 1])
    assert list(out) == [0, 3, 5]


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        acceptance_rate_by_bucket([0, 1], [1])


def test_empty_input():
    assert acceptance_rate_by_bucket([], []) == {}
