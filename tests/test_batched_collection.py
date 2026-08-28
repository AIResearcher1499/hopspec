"""Batched trajectory collection.

The whole design exists to keep the blast radius inside the generate call.
Everything that has ever produced a bug — the step/context invariant, the span
slicing, the truncation, the labelling — stays on `_absorb_step`, shared with
the sequential path. These four tests are what make "shared, not
reimplemented" a fact rather than a claim.
"""

import re

import pytest
import torch

from hopspec.data.agent_pipeline import (
    HFTargetLLM,
    MockLLM,
    run_react_trajectories_batched,
    run_react_trajectory,
)
from hopspec.data.collect import collect_shard, trajectory_to_record
from hopspec.data.retriever import Document, InMemoryBM25Retriever
from hopspec.data.segment_labeling import label_trajectory

from conftest import CORPUS, TWO_HOP_RESPONSES, simple_offset_tokenizer

# Three questions with deliberately different shapes: one finishes on the
# first turn, one takes two hops, one emits a malformed step with no action.
SCRIPTS = {
    "one-hop": ["Thought: I know this.\nAction: Finish[Paris]"],
    "two-hop": [
        "Thought: I need the author.\nAction: Search[book author]",
        "Thought: Now the birthplace.\nAction: Finish[Paris]",
    ],
    "malformed": [
        "Thought: I am thinking out loud with no action at all.",
        "Thought: Now I will act.\nAction: Finish[Rome]",
    ],
}


class ScriptedBatchLLM:
    """MockLLM's batch sibling: per-question scripts, served in lockstep.

    Deliberately serves the SAME strings a sequential MockLLM would, so any
    difference between the two paths is the driver's fault, not the model's.
    """

    def __init__(self, scripts: dict[str, list[str]], order: list[str]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.order = order
        self.cursor = {k: 0 for k in scripts}
        self.batches: list[list[str]] = []

    def _key_for(self, context: str) -> str:
        for key in self.order:
            if context.startswith(f"Question: {key}"):
                return key
        raise AssertionError(f"no script for context {context[:40]!r}")

    def generate_batch(self, contexts: list[str]) -> list[str]:
        self.batches.append(list(contexts))
        out = []
        for context in contexts:
            key = self._key_for(context)
            script = self.scripts[key]
            index = self.cursor[key]
            out.append(script[index] if index < len(script) else script[-1])
            self.cursor[key] += 1
        return out


def sequential(question: str, retriever, max_hops=4):
    return run_react_trajectory(question, MockLLM(list(SCRIPTS[question])),
                                retriever, max_hops=max_hops)


@pytest.fixture
def bm25():
    return InMemoryBM25Retriever(CORPUS)


# ---- 1. batched == sequential, in steps, context and labels ----

def test_batched_matches_sequential_exactly(bm25):
    order = list(SCRIPTS)
    batched = run_react_trajectories_batched(
        order, ScriptedBatchLLM(SCRIPTS, order), bm25
    )
    for question, got in zip(order, batched):
        want = sequential(question, InMemoryBM25Retriever(CORPUS))
        assert got.context == want.context
        assert got.final_answer == want.final_answer
        assert got.is_complete == want.is_complete
        assert [(int(s.segment_type), s.text, s.hop_index) for s in got.steps] == \
               [(int(s.segment_type), s.text, s.hop_index) for s in want.steps]
        # and the labels the trainer would see
        a = label_trajectory(got, simple_offset_tokenizer)
        b = label_trajectory(want, simple_offset_tokenizer)
        assert a.input_ids == b.input_ids
        assert a.segment_type_ids == b.segment_type_ids
        assert a.recency_bucket_ids == b.recency_bucket_ids
        assert a.recency_distances == b.recency_distances
        assert a.hop_boundary_positions == b.hop_boundary_positions


# ---- 2. trajectories retire independently ----

def test_the_active_set_shrinks_as_trajectories_finish(bm25):
    order = list(SCRIPTS)
    llm = ScriptedBatchLLM(SCRIPTS, order)
    trajectories = run_react_trajectories_batched(order, llm, bm25)

    # round 1 has all three; "one-hop" finishes and must not appear again
    assert len(llm.batches[0]) == 3
    assert len(llm.batches) >= 2
    assert all(not c.startswith("Question: one-hop") for c in llm.batches[1]), \
        "a finished trajectory was re-sent to the model"
    assert len(llm.batches[1]) == 2

    shapes = {q: t for q, t in zip(order, trajectories)}
    assert shapes["one-hop"].is_complete and shapes["one-hop"].final_answer == "Paris"
    assert shapes["two-hop"].is_complete and shapes["two-hop"].final_answer == "Paris"
    assert shapes["malformed"].is_complete and shapes["malformed"].final_answer == "Rome"
    # the malformed step took an extra turn and produced no hop
    assert len(shapes["malformed"].steps) > 3
    for trajectory in trajectories:
        assert "".join(s.text for s in trajectory.steps) == trajectory.context


def test_max_hops_retires_a_trajectory_that_never_finishes(bm25):
    scripts = {"looping": ["Thought: still going.\nAction: Search[book author]"]}
    llm = ScriptedBatchLLM(scripts, ["looping"])
    trajectories = run_react_trajectories_batched(["looping"], llm, bm25, max_hops=2)
    assert len(llm.batches) == 2                      # not more
    assert not trajectories[0].is_complete
    assert "".join(s.text for s in trajectories[0].steps) == trajectories[0].context


def test_driver_rejects_a_mismatched_batch(bm25):
    class Broken:
        def generate_batch(self, contexts):
            return contexts[:-1]

    with pytest.raises(ValueError, match="returned 1 outputs for 2"):
        run_react_trajectories_batched(["one-hop", "two-hop"], Broken(), bm25)


# ---- 3. left-padding equivalence, at the tensor level ----

_PIECE = re.compile(r" ?\S+|\s+")


class PaddingTokenizer:
    """Stand-in with the padding behaviour that matters. pad_token_id=0."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self._pieces = ["<pad>"]
        self._ids = {"<pad>": 0}

    def _id(self, piece):
        if piece not in self._ids:
            self._ids[piece] = len(self._pieces)
            self._pieces.append(piece)
        return self._ids[piece]

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        user = next(m["content"] for m in messages if m["role"] == "user")
        return f"U:{user}A:"

    def _encode(self, text):
        return [self._id(m.group()) for m in _PIECE.finditer(text)]

    def __call__(self, texts, return_tensors=None, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._encode(t) for t in texts]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for row in rows:
            pad = [0] * (width - len(row))
            if self.padding_side == "left":
                ids.append(pad + row)
                mask.append([0] * len(pad) + [1] * len(row))
            else:
                ids.append(row + pad)
                mask.append([1] * len(row) + [0] * len(pad))
        return _Batch({"input_ids": torch.tensor(ids),
                       "attention_mask": torch.tensor(mask)})

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._pieces[int(i)] for i in ids if int(i) != 0)


class _Batch(dict):
    def to(self, device):
        return self


class RecordingModel:
    """Records what it was handed, then appends a fixed generated tail."""

    def __init__(self, tokenizer, reply="Thought: ok\nAction: Finish[x]"):
        self.tokenizer = tokenizer
        self.reply = reply
        self.seen = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.seen.append({"input_ids": input_ids.clone(),
                          "attention_mask": attention_mask.clone()})
        tail = torch.tensor([self.tokenizer._encode(self.reply)] * input_ids.shape[0])
        return torch.cat([input_ids, tail], dim=1)


def make_llm(tokenizer, model):
    llm = HFTargetLLM.__new__(HFTargetLLM)      # no download, no network
    llm.tokenizer, llm.model = tokenizer, model
    llm.device, llm.max_new_tokens, llm._torch = "cpu", 16, torch
    return llm


def test_generate_batch_left_pads_and_preserves_each_prompt():
    """Right padding would continue generation from pad tokens. Assert the
    padding is on the LEFT and that each row's real content and mask match the
    single-sequence encoding."""
    tokenizer = PaddingTokenizer()
    model = RecordingModel(tokenizer)
    llm = make_llm(tokenizer, model)
    contexts = ["Question: short one\n", "Question: a considerably longer context here\n"]

    llm.generate_batch(contexts)

    seen = model.seen[0]
    ids, mask = seen["input_ids"], seen["attention_mask"]
    assert ids.shape[0] == 2
    for row, context in enumerate(contexts):
        expected = tokenizer._encode(tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": context}],
            add_generation_prompt=True))
        real = ids[row][mask[row] == 1].tolist()
        assert real == expected, "row content differs from the single-sequence encoding"
        pads = (mask[row] == 0).nonzero().flatten().tolist()
        assert pads == list(range(len(pads))), "padding is not on the LEFT"
        assert ids[row][-1] != tokenizer.pad_token_id, "row ends on a pad token"


def test_generate_batch_restores_the_tokenizer_padding_side():
    tokenizer = PaddingTokenizer()
    llm = make_llm(tokenizer, RecordingModel(tokenizer))
    llm.generate_batch(["Question: a\n", "Question: bb\n"])
    assert tokenizer.padding_side == "right"


def test_generate_batch_slices_the_generated_part_only():
    tokenizer = PaddingTokenizer()
    llm = make_llm(tokenizer, RecordingModel(tokenizer))
    out = llm.generate_batch(["Question: a\n", "Question: much longer question\n"])
    assert out == ["Thought: ok\nAction: Finish[x]"] * 2, \
        "the prompt leaked into the output, or the slice offset is wrong"


def test_generate_batch_on_no_contexts():
    tokenizer = PaddingTokenizer()
    model = RecordingModel(tokenizer)
    assert make_llm(tokenizer, model).generate_batch([]) == []
    assert model.seen == []


# ---- 4. --batch-size 1 is the pre-change path ----

def test_batch_size_one_matches_the_sequential_shard(bm25, tmp_path):
    questions = [(f"q{i}", f"Question number {i}: where was the author born?")
                 for i in range(3)]

    def shard(path, **kwargs):
        collect_shard(questions, MockLLM(TWO_HOP_RESPONSES * 3), bm25,
                      simple_offset_tokenizer, str(path), **kwargs)
        return path.read_text(encoding="utf-8")

    assert shard(tmp_path / "default.jsonl") == \
           shard(tmp_path / "explicit.jsonl", batch_size=1)


def test_collect_shard_rejects_a_zero_batch(bm25, tmp_path):
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        collect_shard([("q0", "Q?")], MockLLM(list(TWO_HOP_RESPONSES)), bm25,
                      simple_offset_tokenizer, str(tmp_path / "s.jsonl"),
                      batch_size=0)


def test_collect_shard_batched_writes_the_same_records(bm25, tmp_path):
    """The shard is what the trainer eats: batching must not change a record."""
    order = list(SCRIPTS)
    questions = [(q, q) for q in order]

    collect_shard(questions, ScriptedBatchLLM(SCRIPTS, order), bm25,
                  simple_offset_tokenizer, str(tmp_path / "batched.jsonl"),
                  batch_size=3)
    batched = (tmp_path / "batched.jsonl").read_text(encoding="utf-8")

    expected = "".join(
        __import__("json").dumps(
            trajectory_to_record(sequential(q, InMemoryBM25Retriever(CORPUS)),
                                 q, simple_offset_tokenizer),
            ensure_ascii=False) + "\n"
        for q in order
    )
    assert batched == expected
