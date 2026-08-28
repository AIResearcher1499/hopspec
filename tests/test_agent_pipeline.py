import pytest

from hopspec.data.agent_pipeline import (
    MockLLM,
    _format_passage,
    _split_generated_step,
    _THOUGHT_LABEL_RE,
    _truncate_to_first_action,
    run_react_trajectory,
)
from hopspec.data.retriever import Document
from hopspec.data.schema import SegmentType

from conftest import CORPUS, TWO_HOP_RESPONSES


def joined(steps):
    return "".join(step.text for step in steps)


# ---- truncation ----

def test_truncate_keeps_through_first_action():
    text = "Thought: X.\nAction: Search[q]\nObservation: I made this up"
    assert _truncate_to_first_action(text) == "Thought: X.\nAction: Search[q]"


def test_truncate_finish():
    text = "Thought: done.\nAction: Finish[42]\nmore rambling"
    assert _truncate_to_first_action(text) == "Thought: done.\nAction: Finish[42]"


def test_truncate_no_action_returns_unchanged():
    assert _truncate_to_first_action("just rambling") == "just rambling"


def test_truncate_picks_earliest_action():
    text = "Action: Search[a]\nAction: Finish[b]"
    assert _truncate_to_first_action(text) == "Action: Search[a]"


# ---- the span table from the spec ----

def test_split_search_step_span_table():
    rendered = "Thought: I need X.\nAction: Search[q]\n"
    steps = _split_generated_step(rendered)
    assert [(s.segment_type, s.text) for s in steps] == [
        (SegmentType.TEMPLATE, "Thought:"),
        (SegmentType.THOUGHT, " I need X."),
        (SegmentType.TEMPLATE, "\n"),
        (SegmentType.TEMPLATE, "Action: Search["),
        (SegmentType.TOOL_CALL, "q"),
        (SegmentType.TEMPLATE, "]\n"),
    ]


def test_split_reconstructs_exactly():
    rendered = "Thought: I need X.\nAction: Search[q]\n"
    assert joined(_split_generated_step(rendered)) == rendered


def test_split_finish_payload_is_answer_subspan():
    rendered = "Thought: done.\nAction: Finish[Paris]\n"
    steps = _split_generated_step(rendered)
    answers = [s for s in steps if s.segment_type is SegmentType.ANSWER]
    assert len(answers) == 1
    assert answers[0].text == "Paris"
    # The answer is a sub-span of the action, never a separate appended copy.
    assert joined(steps) == rendered


def test_split_no_action_with_thought_label():
    rendered = "Thought: I wonder.\n"
    steps = _split_generated_step(rendered)
    assert steps[0].segment_type is SegmentType.TEMPLATE
    assert steps[0].text == "Thought:"
    assert steps[1].segment_type is SegmentType.THOUGHT
    assert joined(steps) == rendered


def test_split_no_action_no_label_is_other():
    rendered = "complete nonsense\n"
    steps = _split_generated_step(rendered)
    assert [s.segment_type for s in steps] == [SegmentType.OTHER]
    assert joined(steps) == rendered


def test_thought_label_regex_does_not_consume_trailing_space():
    # BPE attaches the leading space to the next word (" The" is one token);
    # a TEMPLATE span swallowing it would misattribute the first CONTENT token.
    match = _THOUGHT_LABEL_RE.match("Thought: The answer")
    assert match.group() == "Thought:"


def test_split_multiline_thought():
    rendered = "Thought: line one\nline two.\nAction: Search[q]\n"
    steps = _split_generated_step(rendered)
    thought = [s for s in steps if s.segment_type is SegmentType.THOUGHT]
    assert thought[0].text == " line one\nline two."
    assert joined(steps) == rendered


# ---- passage formatting ----

def test_format_passage():
    docs = [Document("1", "A", "first"), Document("2", "B", "second")]
    assert _format_passage(docs) == "Observation: - first\n- second"


# ---- MockLLM ----

def test_mock_llm_returns_in_order_and_exhausts():
    llm = MockLLM(["a", "b"])
    assert llm.generate("ctx") == "a"
    assert llm.generate("ctx") == "b"
    with pytest.raises(RuntimeError):
        llm.generate("ctx")


# ---- THE CENTRAL INVARIANT, on every trajectory shape ----

def _bm25():
    from hopspec.data.retriever import InMemoryBM25Retriever

    return InMemoryBM25Retriever(CORPUS)


def test_invariant_two_hop(two_hop_trajectory):
    assert joined(two_hop_trajectory.steps) == two_hop_trajectory.context
    assert two_hop_trajectory.is_complete
    assert two_hop_trajectory.final_answer == "Paris"


def test_invariant_malformed_step():
    llm = MockLLM([
        "I forgot the format entirely",
        "Thought: recovering.\nAction: Finish[ok]",
    ])
    trajectory = run_react_trajectory("q?", llm, _bm25())
    assert joined(trajectory.steps) == trajectory.context
    assert trajectory.is_complete


def test_invariant_incomplete():
    llm = MockLLM(["Thought: hmm.\nAction: Search[trains]"] * 4)
    trajectory = run_react_trajectory("q?", llm, _bm25(), max_hops=4)
    assert joined(trajectory.steps) == trajectory.context
    assert not trajectory.is_complete
    assert trajectory.final_answer is None


def test_invariant_immediate_finish():
    llm = MockLLM(["Thought: trivial.\nAction: Finish[yes]"])
    trajectory = run_react_trajectory("q?", llm, _bm25())
    assert joined(trajectory.steps) == trajectory.context
    assert trajectory.final_answer == "yes"


# ---- trajectory structure ----

def test_context_starts_with_question_line(two_hop_trajectory):
    assert two_hop_trajectory.context.startswith(
        "Question: Where was the book's author born?\n"
    )


def test_passage_steps_have_hop_indices(two_hop_trajectory):
    passages = [
        s for s in two_hop_trajectory.steps
        if s.segment_type is SegmentType.RETRIEVED_PASSAGE
    ]
    assert [p.hop_index for p in passages] == [0, 1]


def test_passage_steps_end_with_newline(two_hop_trajectory):
    for step in two_hop_trajectory.steps:
        if step.segment_type is SegmentType.RETRIEVED_PASSAGE:
            assert step.text.startswith("Observation: ")
            assert step.text.endswith("\n")


def test_generated_steps_have_no_hop_index(two_hop_trajectory):
    for step in two_hop_trajectory.steps:
        if step.segment_type is not SegmentType.RETRIEVED_PASSAGE:
            assert step.hop_index is None


def test_no_empty_steps(two_hop_trajectory):
    assert all(step.text for step in two_hop_trajectory.steps)


def test_search_truncates_hallucinated_observation():
    llm = MockLLM([
        "Thought: X.\nAction: Search[book author]\nObservation: fabricated",
        "Thought: done.\nAction: Finish[ok]",
    ])
    trajectory = run_react_trajectory("q?", llm, _bm25())
    assert "fabricated" not in trajectory.context
    assert joined(trajectory.steps) == trajectory.context


def test_two_hop_uses_scripted_responses(retriever):
    llm = MockLLM(TWO_HOP_RESPONSES)
    trajectory = run_react_trajectory("q?", llm, retriever)
    assert trajectory.context.count("Observation:") == 2
