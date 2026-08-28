# Prereg: is entropy degeneracy general to agentic loops? — 2026-08-29

**FROZEN once any entropy measurement file exists.** Amendments appended with
a date; no threshold below may be edited after a number is seen
(`hopspc.md` §15). Written before any data was downloaded or any model run.

## 1. Why this decides the venue, not just a claim

Twice measured in our ReAct-RAG loop: the target's next-token entropy at the
positions where a router must decide is essentially zero — **median 0.000
nats, 86.7% of routing decisions ≤ 0.25 nats**, tuning tables flat across an
order of magnitude of thresholds. A ReSpec-style entropy signal has nothing to
route on there.

That is currently a **workload-specific** observation about one ReAct-RAG
harness. `docs/novelty-verdict-2026-08-28.md` frames the contribution as a
measurement study, and the venue question follows directly:

- If the degeneracy is **general to agentic loops** — formats with a rigid
  scaffold the model emits itself — then "confidence signals are uninformative
  at the structural boundaries of agent generation, and structure is free
  there" is a generalisable thesis about agent serving, and a top-tier
  measurement-study target is defensible.
- If it is **specific to ReAct-RAG**, the honest frame is a workload-specific
  system and ARR/Findings is the right home.

**Either outcome is a result.** The failure mode this prereg guards against is
running the probe, seeing a mixed picture, and choosing the framing afterwards.

## 2. The quantity

Target-model **next-token entropy in nats** at **structural boundary**
positions versus **content** positions, within the same trajectory, from the
same model, under the same prompt.

    H(p) = -sum_v p_v log p_v      over the full vocabulary, natural log

A position is scored by the distribution predicting the token **at** that
position (i.e. the logits row at position-1), which is the quantity a router
would see before proposing.

Boundary and content are assigned by a **grammar labeler per workload**,
built with the same discipline as `_split_generated_step` (spec §4): span
slicing over the verbatim generated string, never regex-extract-and-reformat,
with `"".join(span.text for span in spans) == generated` asserted in tests.
Positions are attributed to a span by their **first character**, exactly as
`assign_tokens_to_steps` does.

## 3. Workloads

At least two beyond RAG. Chosen for being public and ungated (checked
2026-08-29):

| # | workload | data | boundaries | content |
|---|---|---|---|---|
| 1 | JSON / function calling | `glaiveai/glaive-function-calling-v2` | the tool-call block opening, the `"name"` / `"arguments"` keys, structural braces and quotes | argument values, natural-language turns |
| 2 | Code agent | `nebius/SWE-agent-trajectories` (traces) + `princeton-nlp/SWE-bench_Lite` (task instances) | the step's action/command prefix, the thought→action transition, the command fence | command body, reasoning text |
| 3 | optional | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | a second function-calling format from a different framework | as above |

RAG is the incumbent and is **not** re-measured here; nothing in this probe
may be compared with any table in the scale-up run (different models,
prompts, formats and definitions).

## 4. Two measurement modes, both required

**Self-generated (primary; the gate is read off this).** Run
`Qwen/Qwen3-1.7B` — and 4B if it is cheap on this hardware — on **100–200
prompts per workload**, using each framework's own system prompt and format,
log per-token entropy over the model's own generation, and label boundary vs
content with that workload's grammar. This is the exact quantity: the
degeneracy claim is about a model's own generation, so only self-generation
measures it.

**Teacher-forced on public traces (secondary).** Forward the same models over
**100–200 public trajectories** and take the entropy at boundary vs content
positions of text the model did not write. The absolute level is confounded —
another model wrote the trace — so only the **ratio boundary/content** and the
boundary median are reported, and neither may be used to decide the gate.

## 5. The gate, fixed before running

**The generality thesis is SUPPORTED if, for ≥ 2 non-RAG workloads in
self-generated mode, all three hold:**

1. **median boundary entropy ≤ 0.25 nats**, and
2. **≥ 80% of boundary positions ≤ 0.25 nats**, and
3. **median(boundary) / median(content) < 0.2**.

If the gate fails on either workload, **say so plainly: that is the ARR
outcome and it is fine.** No post-hoc threshold, no "close enough", no
dropping a workload that came out inconvenient. A workload may only be
dropped for a reason recorded *before* its numbers were seen — a grammar
labeler that fails its invariant test, or data that turns out unusable.

Ratio 3 uses medians because the content distribution is heavy-tailed; if
median(content) is 0 the ratio is undefined and is reported as such, which
would itself mean content is degenerate too and the contrast is empty.

## 6. Reporting requirements

Per workload, both modes:

- n positions, split boundary/content, and the fitted grammar in full.
- Entropy median, p75, p95, max, and fraction ≤ 0.25, for each class.
- **Per spec §10, at boundary positions: the distinct-token count and the
  majority-class rate — and what a constant predictor scores there.** This is
  exactly where a constant predictor does well, and a low boundary entropy
  that merely restates "the majority token is 90% of boundaries" is not an
  interesting finding. Say which it is.
- The invariant assertion result for each grammar labeler.

## 7. Constraints

- **No GPU rental.** Mac/MPS or CPU is enough for 1.7B at these sizes; a pod
  may be reused only if one is already up for another reason.
- Grammar labelers get CPU-only, no-network tests, like every other labeler in
  this repo.
- Nothing here may be compared with any scale-up table.

## 8. Stopping rules

- A grammar labeler that cannot satisfy `"".join(spans) == generated` is not
  shipped and its workload is reported as unmeasurable, not approximated.
- If the model's self-generated output does not follow the framework's format
  often enough to label (< 50% of generations parse), the workload is reported
  as **inapplicable to self-generated mode at this model size** — that is a
  finding about 1.7B, not about the thesis, and the gate is read on the
  remaining workloads.
- If a confound outside §4 appears, stop and report before building around it.

---

## Amendment 2026-08-29 — the prompt format for workload 1, and a loader bug

Appended before any entropy number existed: every run so far produced
`n = 0` boundary positions, so nothing below was chosen with a result in view.

**Workload 1 (function calling) could not be measured as specified.** §4 says
to use "each framework's own system prompt and format". The glaive system
prompt *describes* the available functions but never states the output syntax —
`<functioncall> {...}` was taught to glaive's models by fine-tuning, not by the
prompt. Qwen3-1.7B therefore answers in prose:

    system: "...you have access to the following functions... {get_news_headlines...}"
    user:   "Can you tell me the latest news headlines for the United States?"
    glaive: <functioncall> {"name": "get_news_headlines", "arguments": ...}
    Qwen:   "Sure! I can help you with that. Please wait a moment while I fetch..."

Parse rate 0/5. Under §8 that would make the workload "inapplicable to
self-generated mode", but that verdict would be wrong: it measures a missing
instruction, not the model's behaviour under a real function-calling
framework.

**Change:** workload 1 keeps glaive's real tool *schemas* as data, but the
prompt is built with **Qwen's own native tool-calling chat template**
(`apply_chat_template(..., tools=[...])`), which is the framework format a
deployed Qwen agent actually uses. Boundaries become that format's
scaffolding — the `<tool_call>` wrapper, the `"name"` / `"arguments"` keys and
the JSON punctuation — which is the same *kind* of boundary §3 specified, in
the syntax the model is actually asked to produce.

Teacher-forced mode for workload 1 is unaffected and keeps glaive's own
`<functioncall>` format, since there the trace is glaive's.

**Loader bug, workload 2, no measurement consequence.** In
`nebius/SWE-agent-trajectories` the assistant role is `ai` (not `assistant`)
and message text is under `text` (not `content`); the system prompt is on
`system_prompt`. The loader read the wrong keys and returned zero examples.
Fixed.

Nothing else changes: the gate in §5, the thresholds and the reporting
requirements in §6 stand exactly as written.
