"""Grammar labelers for the entropy-generality probe.

The invariant that matters is the §4 one: a labeler SLICES, it never
reformats. If `"".join(span.text) != text` the labeler is not shipped
(prereg §8).
"""

import pytest

from hopspec.probe.agent_grammars import (
    BOUNDARY,
    CONTENT,
    GRAMMARS,
    code_agent_spans,
    function_call_spans,
    label_positions,
    label_tokens,
)

FUNCTION_CALL = (
    'Sure.\n<functioncall> {"name": "get_weather", "arguments": '
    '{"city": "Hanoi", "unit": "celsius"}}\n'
)
CODE_AGENT = (
    "DISCUSSION\nI need to find where the bug lives before editing.\n"
    "```\nfind_file \"config.py\" src\n```\n"
)


@pytest.mark.parametrize("name,text", [
    ("function_call", FUNCTION_CALL),
    ("code_agent", CODE_AGENT),
    ("function_call", ""),
    ("code_agent", "plain prose with no structure at all"),
    ("function_call", '{"a": 1}'),
    ("code_agent", "```\nls\n```"),
])
def test_labelers_reconstruct_their_input_verbatim(name, text):
    spans = GRAMMARS[name](text)
    assert "".join(s.text for s in spans) == text


def test_function_call_marks_the_tag_and_keys_as_boundary():
    spans = function_call_spans(FUNCTION_CALL)
    boundary = "".join(s.text for s in spans if s.kind == BOUNDARY)
    assert "<functioncall>" in boundary
    assert '"name":' in boundary.replace(" ", "")
    assert '"arguments":' in boundary.replace(" ", "")
    content = "".join(s.text for s in spans if s.kind == CONTENT)
    assert "get_weather" in content and "Hanoi" in content
    assert "<functioncall>" not in content


def test_code_agent_marks_fences_and_the_command_verb():
    spans = code_agent_spans(CODE_AGENT)
    boundary = "".join(s.text for s in spans if s.kind == BOUNDARY)
    content = "".join(s.text for s in spans if s.kind == CONTENT)
    assert "```" in boundary
    assert "DISCUSSION" in boundary
    assert "find_file" in boundary, "the command verb is scaffolding"
    assert "config.py" in content, "the argument is payload"
    assert "I need to find where the bug lives" in content


def test_a_verb_outside_a_fence_is_not_a_boundary():
    """Prose that happens to start with a lowercase word must not be labelled
    as a command."""
    text = "find the file yourself\n"
    spans = code_agent_spans(text)
    assert all(s.kind == CONTENT for s in spans)


def test_label_positions_is_one_kind_per_character():
    kinds = label_positions(FUNCTION_CALL, function_call_spans)
    assert len(kinds) == len(FUNCTION_CALL)
    assert kinds[FUNCTION_CALL.index("<functioncall>")] == BOUNDARY
    assert kinds[FUNCTION_CALL.index("Hanoi")] == CONTENT


def test_label_positions_rejects_a_reformatting_grammar():
    def broken(text):
        from hopspec.probe.agent_grammars import Span
        return [Span(BOUNDARY, text.strip())]      # drops whitespace

    with pytest.raises(ValueError, match="verbatim"):
        label_positions("  padded  ", broken)


def test_tokens_take_the_kind_of_their_first_character():
    """Spec §5: a token straddling a boundary belongs to the span its FIRST
    character falls in."""
    kinds = [BOUNDARY] * 3 + [CONTENT] * 5
    offsets = [(0, 2), (2, 5), (5, 8)]
    assert label_tokens(offsets, kinds) == [BOUNDARY, BOUNDARY, CONTENT]


QWEN_TOOL_CALL = (
    '<tool_call>\n{"name": "get_news_headlines", "arguments": {"country": "United States"}}\n'
    "</tool_call>"
)


def test_function_call_grammar_also_handles_the_qwen_wrapper():
    """Self-generated mode uses Qwen's native format, teacher-forced uses
    glaive's. One labeler must serve both or the two modes are not comparable."""
    spans = function_call_spans(QWEN_TOOL_CALL)
    assert "".join(s.text for s in spans) == QWEN_TOOL_CALL
    boundary = "".join(s.text for s in spans if s.kind == BOUNDARY)
    content = "".join(s.text for s in spans if s.kind == CONTENT)
    assert "<tool_call>" in boundary and "</tool_call>" in boundary
    assert "United States" in content
    assert "<tool_call>" not in content
