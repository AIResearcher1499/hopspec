"""Deterministic collect/eval question-id split — the leakage guard.

The pilot corpus guarantees gold-passage coverage only for the benchmarks'
VALIDATION splits, so trajectory collection draws from validation too — the
same pool the eval plan names. Without this guard, training and eval
questions come from the same set.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable


def split_question_ids(
    ids: Iterable[str], eval_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Disjoint, exhaustive, order-independent split.

    Ids are sorted before shuffling so the split does not depend on dataset
    iteration order (which varies across `datasets` versions and mirrors).
    """
    if not 0.0 <= eval_fraction <= 1.0:
        raise ValueError("eval_fraction must be in [0, 1]")
    unique_ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    num_eval = int(round(len(unique_ids) * eval_fraction))
    eval_ids = sorted(unique_ids[:num_eval])
    collect_ids = sorted(unique_ids[num_eval:])
    return collect_ids, eval_ids


def get_or_create_split(
    ids: Iterable[str],
    split_path: str,
    eval_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Compute once, persist, then load UNCHANGED.

    A persisted split is returned as-is even if the input id set later
    shifts — a split must never silently move a question between pools after
    collection has started.
    """
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as f:
            stored = json.load(f)
        return list(stored["collect_ids"]), list(stored["eval_ids"])
    collect_ids, eval_ids = split_question_ids(ids, eval_fraction=eval_fraction, seed=seed)
    payload = {
        "collect_ids": collect_ids,
        "eval_ids": eval_ids,
        "eval_fraction": eval_fraction,
        "seed": seed,
    }
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return collect_ids, eval_ids
