# Result: is entropy degeneracy general to agentic loops? — 2026-08-29

Executes `docs/prereg-entropy-generality-probe-2026-08-29.md`. The gate, the
thresholds and the reporting requirements were fixed before any data was
downloaded. Nothing here may be compared with any scale-up table (different
models, prompts, formats and definitions).

Ran on the Mac (MPS), `Qwen/Qwen3-1.7B`. No GPU was rented.

## The gate: FAILS as pre-registered

The thesis is supported only if **≥ 2 non-RAG workloads** pass all three
criteria in self-generated mode. One does.

| workload (self-generated) | 1) median ≤ 0.25 | 2) ≥ 80% ≤ 0.25 | 3) ratio < 0.2 | verdict |
|---|---|---|---|---|
| function calling | 0.0000 **PASS** | 0.999 **PASS** | 150.89 **FAIL** | **FAIL** |
| code agent | 0.0008 **PASS** | 0.836 **PASS** | 0.17 **PASS** | **PASS** |

**This is the ARR outcome, and per §5 it is stated plainly rather than
reframed.** The generality thesis is not supported at the strength the prereg
demanded, so the defensible frame is a workload-specific system, not a
general claim about agent serving.

## But read *why* it fails — the two criteria come apart

Criteria 1 and 2 say *boundaries are near-deterministic*. Criterion 3 says
*boundaries are distinctively more deterministic than content*. They are
different claims and this probe separates them:

- **Boundary degeneracy IS general.** Both non-RAG workloads have a boundary
  median of essentially zero and ≥ 83% of boundary positions ≤ 0.25 nats.
- **The boundary/content *contrast* is not.** In function calling the content
  is equally deterministic — median 0.00000, 99.3% ≤ 0.25 — so the ratio
  explodes to 150.89. Nothing is uncertain anywhere.

The prereg anticipated this exact shape: *"if median(content) is 0 … that
would itself mean content is degenerate too and the contrast is empty."* It
is empty. The reason is mechanical: a tool call's argument values are copied
from the user's request, so they are as predictable as the braces around them.

So the honest pair of sentences is:

1. **A target-entropy signal has nothing to route on in any of the three
   workloads measured.** That claim got *stronger*, not weaker — in function
   calling it is uninformative everywhere, not merely at boundaries.
2. **"Structure marks where the model is certain" does not generalise.** It
   holds for the code agent (ratio 0.17) and for our RAG loop; in function
   calling structure marks nothing, because everything is certain.

Claim 1 is about entropy's *usefulness as a router signal*. Claim 2 is about
*structure's usefulness as a proxy*. The paper has been leaning on both; only
the first survives beyond ReAct-RAG.

## Full tables

Self-generated (primary; the gate is read from these):

| workload | class | n | median | p75 | p95 | max | ≤0.25 |
|---|---|---|---|---|---|---|---|
| function calling | boundary | 1090 | 0.00000 | 0.0000 | 0.0077 | 0.256 | 0.999 |
| function calling | content | 2496 | 0.00000 | 0.0000 | 0.0000 | 1.573 | 0.993 |
| code agent | boundary | 1962 | 0.00079 | 0.0901 | 0.8334 | 1.253 | 0.836 |
| code agent | content | 11079 | 0.00454 | 0.2720 | 0.8886 | 1.485 | 0.744 |

Teacher-forced (secondary — another model wrote the trace, so only the ratio
and the boundary median are reported, and neither decides the gate):

| workload | boundary median | ≤0.25 | content median | ratio |
|---|---|---|---|---|
| function calling | 0.00013 | 0.771 | 0.00001 | 10.15 |
| code agent | 0.04018 | 0.731 | 0.05320 | 0.76 |

The teacher-forced ratios sit much closer to 1 than the self-generated ones,
which is what the confound predicts: forced over another model's text, our
model is uncertain about content it would not have written, and the contrast
washes out.

## Spec §10: what a constant predictor scores at boundaries

Required because a near-zero boundary entropy that merely restates "one token
is most boundaries" would be uninteresting.

| workload (self) | distinct tokens | majority rate | constant predictor |
|---|---|---|---|
| function calling | 12 | 0.212 | **21.2%** |
| code agent | 19 | 0.187 | **18.7%** |

**Neither is degenerate by the project's own test** (`distinct ≤ 10 or
majority ≥ 0.5`). So the near-zero boundary entropy is *not* an artifact of
one token dominating: the model is confidently predicting **different** tokens
at different boundaries — 12–19 of them, none above 21%. A constant predictor
scores ~20% where the model is ~100% certain. That is the finding, and it is
the strongest single number in this probe.

Top boundary tokens — function calling: `":` (231), `name` (116), ` "` (116),
`<tool_call>` (115), `{"` (115). Code agent: ` `` ` (367), `` `\n `` (255),
`DIS`/`CU`/`SSION` (232 each — the `DISCUSSION` label, tokenised in three
pieces).

## Method notes and what could be wrong

- **Grammars slice, never reformat.** Both labelers assert
  `"".join(span.text) == text` and are covered by 13 CPU-only tests. Tokens
  take the kind of their first character (spec §5).
- **Parse rates**: code agent 150/150; function calling 115/150 self and
  114/150 forced — above the §8 floor of 50%, so the workload is measured
  rather than declared inapplicable.
- **The function-calling prompt had to be amended** before any number existed
  (amendment 2026-08-29): glaive's stored system prompt never states its
  output syntax, so Qwen answered in prose at a 0/5 parse rate. Workload 1
  keeps glaive's real tool schemas but is prompted with Qwen's own native
  tool-calling template. The change is recorded in the prereg, not here.
- **One model, one size.** Everything is Qwen3-1.7B. A larger model could be
  less certain at content positions, which would *widen* the contrast and
  could flip criterion 3 for function calling. The cheapest next step is the
  same probe at 4B/8B; until then the negative is specific to this size.
- **Two workloads is the pre-registered minimum.** BFCL, listed as optional,
  was not run.

## Verdict for the venue question

**ARR / Findings, measurement-study frame** — the outcome the prereg named as
acceptable. The generalisable half is narrower than hoped but real: *entropy
is uninformative as a routing signal across three agent formats, while the
structural cue that replaces it is only sometimes discriminative.* Written
that way, the RAG loop remains the contribution's home and the probe becomes
a scope statement rather than a failed bid.

Do not write "structure beats entropy in agentic decoding". Two independent
results now contradict it: the like-for-like signal test inside the RAG loop
(a tie, twice) and criterion 3 here.
