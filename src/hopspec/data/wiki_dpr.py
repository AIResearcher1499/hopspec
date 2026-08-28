"""Streaming reader for the DPR psgs_w100 Wikipedia TSV dump.

The file is standard QUOTE_MINIMAL TSV (header: id, text, title). Do NOT force
csv.QUOTE_NONE: that left literal quote characters attached to any title that
needed quoting and broke gold-title matching by ~10x (HotpotQA match rate
7.9% -> 84.5% after the fix).
"""

from __future__ import annotations

import csv
import sys
from typing import Iterator

from hopspec.data.retriever import Document

_HEADER = ["id", "text", "title"]


def stream_psgs_w100(path: str) -> Iterator[Document]:
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")  # default QUOTE_MINIMAL
        for row in reader:
            if not row:
                continue
            if row == _HEADER:
                continue
            if len(row) != 3:
                raise ValueError(f"malformed psgs_w100 row with {len(row)} fields: {row[:1]}")
            doc_id, text, title = row
            yield Document(doc_id=doc_id, title=title, text=text)
