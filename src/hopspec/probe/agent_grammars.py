"""Boundary/content grammars for agent output formats.

Same discipline as `data/agent_pipeline._split_generated_step` (spec §4):
**span slicing over the verbatim generated string, never
regex-extract-and-reformat.** Every labeler guarantees

    "".join(span.text for span in labeler(text)) == text

which the tests assert, because the one bug that cost this project most was a
labeler that reformatted instead of slicing.

BOUNDARY = the format's own scaffolding, which the model emits because the
grammar requires it. CONTENT = the payload a user would call the answer.
The probe asks whether the target is near-deterministic at BOUNDARY spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BOUNDARY = "boundary"
CONTENT = "content"


@dataclass(frozen=True)
class Span:
    kind: str
    text: str


def _slice(text: str, cuts: list[tuple[int, int, str]]) -> list[Span]:
    """Turn (start, end, kind) marks into a gapless span list.

    Anything not claimed by a mark is CONTENT. Marks must be sorted and
    non-overlapping; the result always reconstructs `text` exactly.
    """
    spans: list[Span] = []
    cursor = 0
    for start, end, kind in cuts:
        if start < cursor or end > len(text) or end < start:
            raise ValueError(f"bad mark ({start},{end}) at cursor {cursor}")
        if start > cursor:
            spans.append(Span(CONTENT, text[cursor:start]))
        if end > start:
            spans.append(Span(kind, text[start:end]))
        cursor = end
    if cursor < len(text):
        spans.append(Span(CONTENT, text[cursor:]))
    return spans


# ---- workload 1: JSON / function calling (glaive format) ----
#
# Assistant turns look like:
#   <functioncall> {"name": "get_weather", "arguments": '{"city": "Hanoi"}'}
# The scaffolding is the tag, the keys, and the JSON punctuation. The payload
# is the value text.

_FC_MARKS = [
    # Qwen's native tool-calling wrapper and glaive's, so the same labeler
    # serves self-generated (Qwen format) and teacher-forced (glaive format).
    re.compile(r"</?tool_call>"),
    re.compile(r"<functioncall>"),
    re.compile(r'"name"\s*:\s*'),
    re.compile(r'"arguments"\s*:\s*'),
    re.compile(r'FUNCTION RESPONSE'),
    re.compile(r'[{}\[\]]'),
]


def function_call_spans(text: str) -> list[Span]:
    """Boundaries: the call tag, the reserved keys, and JSON structural
    punctuation. Content: everything else, i.e. the values."""
    marks: list[tuple[int, int, str]] = []
    for pattern in _FC_MARKS:
        for m in pattern.finditer(text):
            marks.append((m.start(), m.end(), BOUNDARY))
    marks.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in marks:
        if merged and start < merged[-1][1]:
            continue                      # overlapping mark, keep the first
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end, kind)
        else:
            merged.append((start, end, kind))
    return _slice(text, merged)


# ---- workload 2: code agent (SWE-agent step format) ----
#
# A step is a thought followed by a fenced command:
#   <reasoning text>
#   ```
#   find_file "foo.py"
#   ```
# Boundaries: the fence lines and the command's leading verb. Content: the
# reasoning prose and the command's arguments.

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\n?", re.MULTILINE)
_CMD_VERB_RE = re.compile(r"(?m)^(?P<verb>[a-z_]+[a-z0-9_]*)(?=[ \n])")
_DISCUSSION_RE = re.compile(r"(?m)^(DISCUSSION|THOUGHT|Thought:|Action:)")


def code_agent_spans(text: str) -> list[Span]:
    """Boundaries: fences, section labels, and the leading verb of a command
    line inside a fence. Content: reasoning prose and command arguments."""
    marks: list[tuple[int, int, str]] = []
    fences = [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]
    for start, end in fences:
        marks.append((start, end, BOUNDARY))
    for m in _DISCUSSION_RE.finditer(text):
        marks.append((m.start(), m.end(), BOUNDARY))
    # the command verb: only on lines inside a fence
    inside: list[tuple[int, int]] = []
    for i in range(0, len(fences) - 1, 2):
        inside.append((fences[i][1], fences[i + 1][0]))
    for lo, hi in inside:
        for m in _CMD_VERB_RE.finditer(text, lo, hi):
            marks.append((m.start("verb"), m.end("verb"), BOUNDARY))
    marks.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in marks:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, kind))
    return _slice(text, merged)


GRAMMARS = {
    "function_call": function_call_spans,
    "code_agent": code_agent_spans,
}


def label_positions(text: str, grammar) -> list[str]:
    """Per-CHARACTER kind, for attributing tokens by first character — the
    same rule `assign_tokens_to_steps` uses."""
    spans = grammar(text)
    if "".join(s.text for s in spans) != text:
        raise ValueError("grammar did not reconstruct its input verbatim")
    out: list[str] = []
    for span in spans:
        out.extend([span.kind] * len(span.text))
    return out


def label_tokens(offsets, kinds: list[str]) -> list[str]:
    """Token kind = the kind of its FIRST character (spec §5)."""
    return [kinds[start] if start < len(kinds) else CONTENT for start, _e in offsets]
