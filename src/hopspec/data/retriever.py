"""Retrieval: Document, BM25 over an in-memory corpus, dense FAISS retriever."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


class BaseRetriever:
    def search(self, query: str, k: int = 3) -> list[Document]:
        raise NotImplementedError


_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class InMemoryBM25Retriever(BaseRetriever):
    """Okapi BM25 over title + text. Dependency-free; used for tests, smoke
    runs, and any corpus small enough to hold in memory."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._doc_terms = [Counter(_terms(d.title + " " + d.text)) for d in self.documents]
        self._doc_lens = [sum(c.values()) for c in self._doc_terms]
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        df: Counter = Counter()
        for counts in self._doc_terms:
            df.update(counts.keys())
        n = len(self.documents)
        self._idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(self, query: str, k: int = 3) -> list[Document]:
        query_terms = _terms(query)
        scored = []
        for i, doc in enumerate(self.documents):
            counts = self._doc_terms[i]
            dl = self._doc_lens[i]
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_len, 1e-9))
                score += self._idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, doc.doc_id, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _score, _doc_id, doc in scored[:k]]


class DenseRetriever(BaseRetriever):
    """bge-base + FAISS IndexFlatIP. Load on CPU so GPUs stay free for LLMs."""

    INDEX_FILE = "index.faiss"
    DOCS_FILE = "documents.jsonl"
    META_FILE = "meta.json"

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.documents: list[Document] = []
        self._index = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    @staticmethod
    def _faiss():
        import faiss

        return faiss

    def _encode(self, texts: list[str], batch_size: int = 64):
        return self._get_encoder().encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def build(self, documents: list[Document], batch_size: int = 64) -> None:
        faiss = self._faiss()
        self.documents = list(documents)
        embeddings = self._encode(
            [f"{d.title}\n{d.text}" for d in self.documents], batch_size=batch_size
        )
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, query: str, k: int = 3) -> list[Document]:
        if self._index is None:
            raise RuntimeError("index not built or loaded")
        query_emb = self._encode([query])
        _scores, indices = self._index.search(query_emb, k)
        return [self.documents[i] for i in indices[0] if i >= 0]

    def save(self, dir_path: str) -> None:
        faiss = self._faiss()
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(dir_path, self.INDEX_FILE))
        with open(os.path.join(dir_path, self.DOCS_FILE), "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
        with open(os.path.join(dir_path, self.META_FILE), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)

    @classmethod
    def load(cls, dir_path: str, device: str = "cpu") -> "DenseRetriever":
        with open(os.path.join(dir_path, cls.META_FILE), encoding="utf-8") as f:
            meta = json.load(f)
        retriever = cls(model_name=meta["model_name"], device=device)
        retriever._index = retriever._faiss().read_index(
            os.path.join(dir_path, cls.INDEX_FILE)
        )
        with open(os.path.join(dir_path, cls.DOCS_FILE), encoding="utf-8") as f:
            retriever.documents = [Document(**json.loads(line)) for line in f if line.strip()]
        return retriever

    @classmethod
    def merge(cls, shard_dirs: list[str], out_dir: str, device: str = "cpu") -> "DenseRetriever":
        if not shard_dirs:
            raise ValueError("no shard directories to merge")
        merged = cls.load(shard_dirs[0], device=device)
        faiss = merged._faiss()
        for shard_dir in shard_dirs[1:]:
            shard = cls.load(shard_dir, device=device)
            if shard.model_name != merged.model_name:
                raise ValueError("cannot merge shards built with different encoders")
            vectors = shard._index.reconstruct_n(0, shard._index.ntotal)
            merged._index.add(vectors)
            merged.documents.extend(shard.documents)
        del faiss
        merged.save(out_dir)
        return merged
