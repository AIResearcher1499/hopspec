"""Algorithm R reservoir sampling over a single-pass iterable."""

from __future__ import annotations

import random
from typing import Iterable, TypeVar

T = TypeVar("T")


def reservoir_sample(iterable: Iterable[T], k: int, seed: int = 0) -> list[T]:
    if k < 0:
        raise ValueError("sample size must be non-negative")
    rng = random.Random(seed)
    reservoir: list[T] = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
