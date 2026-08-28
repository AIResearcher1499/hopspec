# Plan: move the running to an A6000, keep the dev loop on the laptop

Why: the Mac is a working machine. An 8–12 hour collect is not free there,
and the project needs at least three more collections (scale-up, a larger
target model, possibly a second benchmark). Sequential collection on the
A6000 is ~3–4h per shard; batched it is well under an hour. That is what
makes the migration worth its code risk — not raw device speed.

Written for a fresh agent. Read `hopspc.md` §1, §9, §14–15 first, plus
`docs/chat-trained-draft-result-2026-08-29.md` for the current state.

## 0. What is NOT a problem

A device audit found **no hardcoded device anywhere**: every entry point
takes `--device`, already defaulting to `cuda`
(`collect.py`, `train_draft.py`, `_experiment_common.py`,
`07_chained_eval.py`, `00_smoke_test.py`, `01_build_dense_index.py`,
`HFTargetLLM.__init__`). `CachedTargetRunner` and `chat_layout_features`
take the device as a string. `torch_dtype="auto"` gives bf16 on both MPS
and Ampere. `attn_implementation="sdpa"` is required on the A6000 (spec §9:
eager materialises `[T,T]` per head per layer and OOM'd a 49 GB A6000 at
batch 4 on ~2,600-token trajectories) and is already the default at every
load site.

So porting is not a code task. Items 1–4 below are.

## 1. Environment + smoke (do first, ~1h)

- `uv venv .venv && uv pip install -e ".[dense-retrieval]"` on the box.
  Never `--system-site-packages` (spec §1).
- Run the full test suite on the box: **328 tests must pass**. They are
  CPU-only, so a failure means an environment problem, not a device one.
- `00_smoke_test.py --device cuda --num-questions 5`.
- Shared-box hygiene (CLAUDE.md): before reading or writing any data file,
  check `pwd` and `git remote -v`. Long jobs: `nohup … > log 2>&1 </dev/null
  & disown`, confirm `PPID=1`, report the PID.

## 2. Bridge measurement (discipline, ~1h, do before any new science)

Device numerics differ (kernels, bf16 reductions); greedy argmax can flip
near ties. Before trusting any CUDA number against the MPS history:

- Copy `data/shard_1p7b.jsonl` and `data/ckpt_*_1p7b.pt` to the box.
- Re-run **two arms** (neural, routed-verb) in chat mode on CUDA over the
  same shard, same split, same checkpoints.
- Report per-arm pooled accepted/round and the paired per-record delta vs
  the MPS artifacts, in a short `docs/device-bridge-<date>.md`.

Interpretation, fixed in advance: a small delta (within the per-record
bootstrap interval) licenses treating the MPS pilot as directionally
comparable and saying so in the paper. A large delta means the MPS numbers
are pilot-only and every table must be regenerated on CUDA. Either outcome
is fine; not knowing which is not.

## 3. Batched collection (the actual speed win, the actual risk)

Current shape: `run_react_trajectory` loops one question at a time and
calls `llm.generate(context)` once per hop — the only generate call site
(`agent_pipeline.py:191`). GPU utilisation is near zero between steps.

Design, chosen to keep the blast radius inside one class:

- Add `HFTargetLLM.generate_batch(contexts: list[str]) -> list[str]`:
  build each prompt with the SAME `apply_chat_template` path (including
  the `enable_thinking` TypeError fallback), **left-pad**, generate with
  `do_sample=False`, slice each output by its own prompt length, strip
  `<think>`, apply `_truncate_to_first_action`.
- Add `run_react_trajectories_batched(questions, llm, retriever, …)`: an
  active-set lockstep driver. Each round, collect the contexts of all live
  trajectories, one `generate_batch`, then per trajectory do the existing
  per-step work — `_split_generated_step`, `_format_passage`, retrieval,
  context append — **unchanged**. Trajectories retire on Finish or
  `max_hops`; the active set shrinks.
- `collect.py` gains `--batch-size` (default 1 → the existing sequential
  path, untouched).

Everything that has ever produced a bug — the step/context invariant, the
span slicing, the truncation, the labelling — stays on the unchanged code
path. Only the generate call is new.

Tests (CPU, no network, mandatory):

1. With a `MockLLM`-style batch stub, batched trajectories are **identical**
   to sequential ones for the same questions — steps, context, labels.
2. The driver retires trajectories independently: a batch mixing a
   1-hop Finish, a 2-hop, and a malformed step yields exactly the
   sequential results, and the invariant `"".join(step.text) == context`
   holds for each.
3. Left-padding equivalence at the tensor level: a stub model records the
   prompts it received; assert each sequence's non-pad content and its
   attention mask match the single-sequence case.
4. `--batch-size 1` is bit-identical to the pre-change path.

Known and to be documented, not fixed: with a real model, batched matmuls
are not bit-identical to single-sequence ones, so a greedy token can flip
and a batched shard will differ slightly from a sequential one. That is
fine — a shard is data, and every downstream arm uses one shard — but it
must be stated in the shard's provenance and never used to explain away a
result. Validate once on the box: 5 questions batched vs sequential,
report how many trajectories match exactly.

## 4. Resume for long jobs (cheap, high value on a shared box)

`collect_shard` already appends and flushes per record, so a killed job
leaves valid partial output — but restarting re-runs everything. Add
`--resume`: read the existing output file, collect its `question_id`s,
skip them. Test: writing 3 records then resuming over the same question
list appends only the missing ones and never duplicates an id
(`06_validate_shard.py`'s duplicate-id check must stay clean).

## 5. Division of labour

| Machine | Work |
|---|---|
| Mac | dev loop, the 328 tests, offline analysis over raw columns (`eval/analysis.py`), prereg and result docs, everything in `docs/` |
| A6000 | collection, training (both feature modes), chained evaluation, anything loading a target model |

The offline-analysis design (raw per-position columns, `rebucket`,
`select`) exists precisely so re-slicing never needs the GPU. Keep it that
way: no analysis-only step should require the box.

## 6. Order of work

1. §1 environment + 328 tests green on the box.
2. §2 bridge measurement, written up.
3. §4 resume (30 min, do it before the first long job, not after losing one).
4. §3 batched collection + its four tests; validate 5 questions batched vs
   sequential on the box.
5. Only then the scale-up collection (500–1000), which by now runs in
   under an hour and can be repeated cheaply for the larger target model.

Note on ordering: §2 uses the existing sequential path, so it does not
depend on §3. Do it first — it is the measurement that tells you what the
whole MPS history is worth.
