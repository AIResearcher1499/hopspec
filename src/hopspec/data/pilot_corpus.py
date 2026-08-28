"""Pilot corpus: gold-title passages ∪ reservoir-sampled random background."""

from __future__ import annotations

import random
from typing import Iterable

from hopspec.data.retriever import Document


def build_pilot_corpus(
    documents: Iterable[Document],
    gold_titles: set[str],
    num_background: int,
    seed: int = 0,
) -> list[Document]:
    """Single pass: keep every passage whose title is gold; reservoir-sample
    ``num_background`` passages from the rest (Algorithm R inline so the
    stream is consumed once)."""
    rng = random.Random(seed)
    gold_docs: list[Document] = []
    reservoir: list[Document] = []
    seen_background = 0
    for doc in documents:
        if doc.title in gold_titles:
            gold_docs.append(doc)
            continue
        if seen_background < num_background:
            reservoir.append(doc)
        else:
            j = rng.randint(0, seen_background)
            if j < num_background:
                reservoir[j] = doc
        seen_background += 1
    return gold_docs + reservoir
