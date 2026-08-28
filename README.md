# HopSpec

Hop-aware speculative decoding for agentic multi-hop RAG — rebuilt from
`hopspc.md`, which is the authoritative specification. Read it before touching
anything: roughly half of it is a record of bugs that produced confident,
plausible, completely wrong numbers, and the invariants here exist to keep
them from coming back.

**Status: the central hypothesis is not supported** (spec §11). There is no
post-hop acceptance dip; the apparent distance effect is a within-sentence
confound; the hop signal moves nothing. The codebase's value is the
measurement machinery that makes that negative trustworthy.

Chained (non-teacher-forced) speculation is now built and measured
(`infer/chained_eval.py`, `scripts/07_chained_eval.py`). It has two replay
modes and they are not comparable: `raw` feeds the recorded document, `chat`
re-renders the collection-time chat wrapper at every step boundary and is the
deployed loop. See `data/README-chained-rounds.md` before quoting any chained
number.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dense-retrieval]" --python .venv/bin/python
```

Do not use `--system-site-packages` (real ABI breakage; see spec §1).

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

All tests run on CPU with no model download and no network. Many pin specific
shipped bugs (spec §14); do not delete a failing one without reading the spec
section it cites.

## Layout

- `src/hopspec/data/` — ReAct agent loop, verbatim-span step slicing,
  offset-based token labeling, benchmark collection, retrieval, leakage split.
- `src/hopspec/model/` — EAGLE-1-style draft model + hop-aware embeddings.
- `src/hopspec/train/` — EAGLE-aligned batches, recency-weighted losses,
  shared ablation runner (arms differ ONLY in the switch under test).
- `src/hopspec/infer/` — recency tracker, adaptive speculation length, one
  speculative round (the unbuilt-with-real-models decisive experiment).
- `src/hopspec/eval/` — decode-phase mask, degeneracy diagnostics, offline
  re-slicing of raw per-position columns.
- `scripts/00…06` — smoke test → index build → collection → training arms →
  relabel/validate. Always run `06_validate_shard.py` before spending GPU
  time on a shard.

## Working rules (short form; full list in spec §15)

- Never quote an acceptance number without its bucket's majority-class rate.
- Never compare against a number from an earlier run.
- Ask what a constant predictor scores before believing any large effect.
- Decode the actual token sequence and look at it.
- Quarantine invalidated artifacts into `data/_invalidated/` with a README.
