import pytest

from hopspec.data.schema import NO_PRIOR_HOP_BUCKET_ID, NUM_RECENCY_BUCKETS
from hopspec.infer.adaptive_length import DEFAULT_GAMMA_BY_BUCKET, HopAwareLengthPolicy


def test_defaults_cover_every_bucket():
    assert set(DEFAULT_GAMMA_BY_BUCKET) == set(range(NUM_RECENCY_BUCKETS))


def test_default_policy_values():
    policy = HopAwareLengthPolicy()
    assert policy.gamma_for(0) == 1
    assert policy.gamma_for(5) == 8
    assert policy.gamma_for(NO_PRIOR_HOP_BUCKET_ID) == 8


def test_missing_bucket_entry_rejected():
    table = dict(DEFAULT_GAMMA_BY_BUCKET)
    del table[3]
    with pytest.raises(ValueError):
        HopAwareLengthPolicy(table)


def test_values_clipped_to_range():
    table = {b: 100 for b in range(NUM_RECENCY_BUCKETS)}
    table[0] = 0
    policy = HopAwareLengthPolicy(table, gamma_min=1, gamma_max=8)
    assert policy.gamma_for(0) == 1
    assert policy.gamma_for(1) == 8


def test_is_monotonic_true_for_defaults():
    assert HopAwareLengthPolicy().is_monotonic()


def test_is_monotonic_false_when_decreasing():
    table = dict(DEFAULT_GAMMA_BY_BUCKET)
    table[4] = 1
    assert not HopAwareLengthPolicy(table).is_monotonic()


def test_unknown_bucket_rejected():
    with pytest.raises(ValueError):
        HopAwareLengthPolicy().gamma_for(99)


def test_invalid_range_rejected():
    with pytest.raises(ValueError):
        HopAwareLengthPolicy(gamma_min=0)
    with pytest.raises(ValueError):
        HopAwareLengthPolicy(gamma_min=5, gamma_max=2)
