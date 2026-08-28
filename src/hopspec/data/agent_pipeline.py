"""ReAct agent loop and the span-slicing step splitter.

THE CENTRAL INVARIANT of this module (asserted throughout the tests):

    "".join(step.text for step in trajectory.steps) == trajectory.context

Every step's ``text`` is the verbatim slice of the agent's context it
contributed — template prefixes and trailing newlines included. Steps are
produced by slicing the rendered text by character span, never by
regex-extract-and-reformat. Reformatting once dropped the "Thought: " prefix
and every newline, so the target model's hidden states were computed over a
document that never existed and the resulting 4.9% post-hop acceptance was
reported as a confirming result.
"""

from __future__ import annotations

import re
from typing import Protocol

from hopspec.data.retriever import BaseRetriever, Document
from hopspec.data.schema import SegmentType, Trajectory, TrajectoryStep

SYSTEM_PROMPT = (
    "You are answering a multi-hop question by searching a document collection.\n"
    "At each turn write exactly one 'Thought:' line followed by exactly one action:\n"
    "either 'Action: Search[query]' to retrieve evidence, or 'Action: Finish[answer]'\n"
    "once you can answer. Never write an 'Observation:' line yourself; observations\n"
    "are provided to you after each search."
)

_SEARCH_RE = re.compile(r"Action:\s*Search\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_FINISH_RE = re.compile(r"Action:\s*Finish\[(.*?)\]", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=Action:|$)", re.IGNORECASE | re.DOTALL)
# NO trailing \s*: BPE attaches the leading space to the next word (" The" is
# one token). A TEMPLATE span that swallowed the space would own that token's
# first character and offset attribution would classify the first CONTENT
# token as scaffolding — silently excluding the most important token in the
# project from every measurement.
_THOUGHT_LABEL_RE = re.compile(r"\s*Thought:", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLM(Protocol):
    def generate(self, context: str) -> str: ...


class MockLLM:
    """Returns scripted, already-clean steps in order. No truncation needed."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._cursor = 0

    def generate(self, context: str) -> str:
        if self._cursor >= len(self._responses):
            raise RuntimeError("MockLLM ran out of scripted responses")
        out = self._responses[self._cursor]
        self._cursor += 1
        return out


class HFTargetLLM:
    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # sdpa: eager attention materializes a full [T,T] matrix per head per
        # layer and OOM'd a 49 GB A6000 on ~2,600-token trajectories.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", attn_implementation="sdpa"
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def _prompt(self, context: str) -> str:
        """The ONE chat-template path. `generate` and `generate_batch` both go
        through it, so a batched shard cannot be prompted differently from a
        sequential one."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:  # templates that reject the enable_thinking kwarg
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _clean(self, text: str) -> str:
        return _truncate_to_first_action(_THINK_BLOCK_RE.sub("", text))

    def generate_batch(self, contexts: list[str]) -> list[str]:
        """One generate call for many live trajectories.

        **Left padding is required, not a preference.** `generate` continues
        from the right edge of the tensor; with right padding the shorter
        prompts would be continued from their pad tokens and produce garbage.
        Left padding also makes every row share one prompt length, which is
        what lets the generated part be sliced at a single offset.

        Batched matmuls are not bit-identical to single-sequence ones, so a
        greedy token can flip and a batched shard will differ slightly from a
        sequential one. That is fine — a shard is data and every downstream
        arm uses one shard — but it belongs in the shard's provenance and must
        never be used to explain away a result.
        """
        if not contexts:
            return []
        prompts = [self._prompt(context) for context in contexts]
        tokenizer = self.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = previous_side
        inputs = inputs.to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        prompt_length = inputs["input_ids"].shape[1]
        return [
            self._clean(self.tokenizer.decode(row[prompt_length:],
                                              skip_special_tokens=True))
            for row in out
        ]

    def generate(self, context: str) -> str:
        prompt = self._prompt(context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Real models ramble past their action and hallucinate their own
        # "Observation:"; truncating stops that entering the context.
        return self._clean(text)


def _first_action_match(text: str):
    """Earliest Search/Finish match and the SegmentType of its payload."""
    search = _SEARCH_RE.search(text)
    finish = _FINISH_RE.search(text)
    if search and finish:
        return (search, SegmentType.TOOL_CALL) if search.start() < finish.start() else (
            finish, SegmentType.ANSWER)
    if search:
        return search, SegmentType.TOOL_CALL
    if finish:
        return finish, SegmentType.ANSWER
    return None, None


def _truncate_to_first_action(text: str) -> str:
    match, _ = _first_action_match(text)
    if match is None:
        return text
    return text[: match.end()]


def _format_passage(docs: list[Document]) -> str:
    return "Observation: " + "\n".join("- " + d.text for d in docs)


def _split_generated_step(rendered: str) -> list[TrajectoryStep]:
    """Split one rendered agent step into verbatim, contiguous spans.

    Guarantees "".join(s.text for s in result) == rendered. The Finish answer
    is a sub-span of the action payload, not a separate appended step — that
    is where it actually occurs in the context.
    """
    steps: list[TrajectoryStep] = []

    def emit(segment_type: SegmentType, start: int, end: int) -> None:
        if end > start:
            steps.append(TrajectoryStep(segment_type, rendered[start:end]))

    match, payload_type = _first_action_match(rendered)
    if match is None:
        label = _THOUGHT_LABEL_RE.match(rendered)
        if label:
            emit(SegmentType.TEMPLATE, 0, label.end())
            emit(SegmentType.THOUGHT, label.end(), len(rendered))
        else:
            emit(SegmentType.OTHER, 0, len(rendered))
        return steps

    # Region before the action: template label + thought content + trailing
    # whitespace (the whitespace is template — it is inter-step scaffolding).
    label = _THOUGHT_LABEL_RE.match(rendered)
    content_start = label.end() if label and label.end() <= match.start() else 0
    if content_start:
        emit(SegmentType.TEMPLATE, 0, content_start)
    content = rendered[content_start: match.start()]
    stripped_len = len(content.rstrip())
    emit(SegmentType.THOUGHT, content_start, content_start + stripped_len)
    emit(SegmentType.TEMPLATE, content_start + stripped_len, match.start())

    # "Action: Search[" / "Action: Finish[" scaffolding, payload, "]" + tail.
    emit(SegmentType.TEMPLATE, match.start(), match.start(1))
    emit(payload_type, match.start(1), match.end(1))
    emit(SegmentType.TEMPLATE, match.end(1), len(rendered))
    return steps


class _TrajectoryState:
    """One in-flight trajectory. Exists so the batched driver can hold many at
    once; the per-step work itself is `_absorb_step`, shared with the
    sequential path so the two cannot drift apart."""

    __slots__ = ("question", "steps", "context", "final_answer", "is_complete",
                 "hop", "turns")

    def __init__(self, question: str):
        self.question = question
        self.steps = [
            TrajectoryStep(SegmentType.TEMPLATE, "Question: "),
            TrajectoryStep(SegmentType.QUESTION, question),
            TrajectoryStep(SegmentType.TEMPLATE, "\n"),
        ]
        self.context = f"Question: {question}\n"
        self.final_answer: str | None = None
        self.is_complete = False
        self.hop = 0
        self.turns = 0

    def done(self, max_hops: int) -> bool:
        return self.is_complete or self.turns >= max_hops

    def trajectory(self) -> Trajectory:
        return Trajectory(
            question=self.question, steps=self.steps,
            final_answer=self.final_answer, is_complete=self.is_complete,
            context=self.context,
        )


def _absorb_step(state: _TrajectoryState, raw: str, retriever: BaseRetriever,
                 num_docs: int) -> None:
    """Absorb one generated step into a trajectory.

    THE one place that turns model output into steps and context. Everything
    that has ever produced a bug lives here — the step/context invariant, the
    span slicing, the truncation, the retrieval append — so the batched driver
    reuses it verbatim rather than reimplementing it.
    """
    raw = _truncate_to_first_action(raw)  # idempotent; HFTargetLLM already applies
    rendered = raw.rstrip("\n") + "\n"
    state.steps.extend(_split_generated_step(rendered))
    state.context += rendered
    state.turns += 1

    match, payload_type = _first_action_match(rendered)
    if match is not None and payload_type is SegmentType.ANSWER:
        state.final_answer = match.group(1).strip()
        state.is_complete = True
        return
    if match is not None and payload_type is SegmentType.TOOL_CALL:
        docs = retriever.search(match.group(1).strip(), num_docs)
        passage = _format_passage(docs) + "\n"
        state.steps.append(
            TrajectoryStep(SegmentType.RETRIEVED_PASSAGE, passage, hop_index=state.hop)
        )
        state.context += passage
        state.hop += 1
    # Malformed step (no action): keep it in context and give the model
    # another turn; the step was emitted as THOUGHT/OTHER above.


def run_react_trajectory(
    question: str,
    llm: LLM,
    retriever: BaseRetriever,
    max_hops: int = 4,
    num_docs: int = 3,
) -> Trajectory:
    """Run the ReAct loop, building context and verbatim-slice steps in lockstep."""
    state = _TrajectoryState(question)
    for _ in range(max_hops):
        _absorb_step(state, llm.generate(state.context), retriever, num_docs)
        if state.is_complete:
            break
    return state.trajectory()


def run_react_trajectories_batched(
    questions: list[str],
    llm,
    retriever: BaseRetriever,
    max_hops: int = 4,
    num_docs: int = 3,
) -> list[Trajectory]:
    """Run many ReAct loops in lockstep, one `generate_batch` per round.

    An active set, not a fixed grid: trajectories retire independently the
    moment they emit a Finish or exhaust `max_hops`, and the batch shrinks.
    A 1-hop question does not wait for a 4-hop one.

    Per-trajectory work is `_absorb_step`, unchanged and shared with
    `run_react_trajectory`, so for a given sequence of model outputs the two
    produce identical trajectories — steps, context and labels. Only the
    generate call differs.
    """
    states = [_TrajectoryState(question) for question in questions]
    for _ in range(max_hops):
        active = [state for state in states if not state.done(max_hops)]
        if not active:
            break
        outputs = llm.generate_batch([state.context for state in active])
        if len(outputs) != len(active):
            raise ValueError(
                f"generate_batch returned {len(outputs)} outputs for "
                f"{len(active)} contexts"
            )
        for state, raw in zip(active, outputs):
            _absorb_step(state, raw, retriever, num_docs)
    return [state.trajectory() for state in states]
