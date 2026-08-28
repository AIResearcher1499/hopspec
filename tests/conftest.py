"""Shared fixtures: an offset tokenizer, a tiny stand-in target model, and a
record factory that goes through the REAL collection path (MockLLM ->
run_react_trajectory -> trajectory_to_record). All CPU, no network."""

from __future__ import annotations

import re
import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hopspec.data.agent_pipeline import MockLLM, run_react_trajectory
from hopspec.data.collect import trajectory_to_record
from hopspec.data.retriever import Document, InMemoryBM25Retriever

VOCAB_SIZE = 64
HIDDEN_SIZE = 16

_PIECE_RE = re.compile(r" ?\S+|\s+")


def simple_offset_tokenizer(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Deterministic word-level tokenizer with BPE-like leading-space
    attachment (a token is an optional leading space + the following word)."""
    ids, offsets = [], []
    for match in _PIECE_RE.finditer(text):
        ids.append(zlib.crc32(match.group().encode("utf-8")) % VOCAB_SIZE)
        offsets.append((match.start(), match.end()))
    return ids, offsets


class TinyTargetModel(nn.Module):
    """HF-shaped causal LM stand-in: returns .hidden_states (embeddings +
    one tensor per layer, final one post-'norm') and exposes embedding/head."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = 2):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                past_key_values=None, use_cache=False):
        x = self.embedding(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            hidden_states.append(x)
        hidden_states[-1] = self.norm(hidden_states[-1])
        if past_key_values is not None:
            # Position-wise by construction, so cached and uncached outputs
            # agree; the dummy entries exist so cache length and crop() behave
            # exactly as they do for a real model.
            batch, seq_len, hidden = x.shape
            dummy = x.new_zeros(batch, 1, seq_len, hidden)
            for layer_idx in range(len(self.layers)):
                past_key_values.update(dummy, dummy, layer_idx)
        return SimpleNamespace(
            hidden_states=tuple(hidden_states),
            logits=self.lm_head(hidden_states[-1]),
            past_key_values=past_key_values,
        )

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


TWO_HOP_RESPONSES = [
    "Thought: I need to find who wrote the book.\nAction: Search[book author]",
    "Thought: Now I need the author's birthplace.\nAction: Search[author birthplace]",
    "Thought: I can answer now.\nAction: Finish[Paris]",
]

CORPUS = [
    Document("d1", "Book", "The book's author is Alice Smith."),
    Document("d2", "Alice Smith", "The birthplace of Alice Smith is Paris."),
    Document("d3", "Unrelated", "Something about trains and stations."),
]


@pytest.fixture
def retriever() -> InMemoryBM25Retriever:
    return InMemoryBM25Retriever(CORPUS)


def make_two_hop_trajectory(retriever, question="Where was the book's author born?"):
    return run_react_trajectory(question, MockLLM(TWO_HOP_RESPONSES), retriever)


@pytest.fixture
def two_hop_trajectory(retriever):
    return make_two_hop_trajectory(retriever)


def make_records(retriever, num_records: int = 4, tokenize=None) -> list[dict]:
    """Records from the real collection path, labeled with the test tokenizer."""
    tokenize = simple_offset_tokenizer if tokenize is None else tokenize
    records = []
    for i in range(num_records):
        trajectory = make_two_hop_trajectory(
            retriever, question=f"Question number {i}: where was the author born?"
        )
        records.append(trajectory_to_record(trajectory, f"q{i}", tokenize))
    return records


@pytest.fixture
def records(retriever) -> list[dict]:
    return make_records(retriever)


@pytest.fixture
def tiny_target() -> TinyTargetModel:
    return TinyTargetModel()


# ---- reversible tokenizer, for anything that must DECODE ids ----
#
# `simple_offset_tokenizer` hashes pieces into 64 slots, so it cannot decode
# and it collides. The scaffold FSM works in token space but has to read the
# committed tail back as text, so it needs a real inverse.

ROUTED_VOCAB_SIZE = 512


class ReversibleWordTokenizer:
    """Deterministic, invertible offset tokenizer with BPE-like spacing.

    A piece is `optional leading space + word`, or a whitespace run, so a
    content token OWNS its leading space exactly as BPE does — that is what
    makes the spec §4 scaffold-boundary regression testable at all. Ids are
    assigned in first-seen order, so build every record of one test from one
    instance.
    """

    def __init__(self, vocab_size: int = ROUTED_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._ids: dict[str, int] = {}
        self._pieces: list[str] = []

    def _piece_id(self, piece: str) -> int:
        if piece not in self._ids:
            if len(self._pieces) >= self.vocab_size:
                raise ValueError("test tokenizer vocabulary overflowed")
            self._ids[piece] = len(self._pieces)
            self._pieces.append(piece)
        return self._ids[piece]

    def __call__(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets = [], []
        for match in _PIECE_RE.finditer(text):
            ids.append(self._piece_id(match.group()))
            offsets.append((match.start(), match.end()))
        return ids, offsets

    def encode(self, text: str) -> list[int]:
        return self(text)[0]

    def decode(self, ids) -> str:
        # A real tokenizer can decode any id in its vocabulary; unassigned
        # slots stand in for pieces this fixture never saw (speculated tokens
        # are the target's argmax over the whole vocabulary). The placeholder
        # is deliberately free of scaffold characters and whitespace.
        return "".join(
            self._pieces[int(i)] if int(i) < len(self._pieces) else f"<{int(i)}>"
            for i in ids
        )


@pytest.fixture
def routed_tokenizer() -> ReversibleWordTokenizer:
    return ReversibleWordTokenizer()


@pytest.fixture
def routed_records(retriever, routed_tokenizer) -> list[dict]:
    """Records tokenized so they can be decoded back to text."""
    return make_records(retriever, num_records=2, tokenize=routed_tokenizer)


class FakeChatTokenizer:
    """Minimal HF-tokenizer stand-in that renders a chat template.

    `accepts_enable_thinking=False` raises TypeError on that kwarg, which is
    the branch `HFTargetLLM.generate` and `chat_prompt_ids` both carry for
    templates that reject it — untested code is how that branch would rot.
    """

    def __init__(self, word_tokenizer=None, accepts_enable_thinking=True,
                 drop_user=False):
        self._tokenizer = word_tokenizer or ReversibleWordTokenizer()
        self._accepts_enable_thinking = accepts_enable_thinking
        self._drop_user = drop_user

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        if "enable_thinking" in kwargs and not self._accepts_enable_thinking:
            raise TypeError(
                "apply_chat_template() got an unexpected keyword argument "
                "'enable_thinking'"
            )
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        if self._drop_user:
            user = ""
        rendered = f"SYSTEM:\n{system}\nUSER:\n{user}"
        if add_generation_prompt:
            rendered += "\nASSISTANT:\n"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return self._tokenizer.encode(text)

    def decode(self, ids, **kwargs):
        return self._tokenizer.decode(ids)


@pytest.fixture
def fake_chat_tokenizer() -> FakeChatTokenizer:
    return FakeChatTokenizer()
