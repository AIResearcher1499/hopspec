# Plan: rented-GPU execution — 2026-08-29

The single plan for what still has to be built and where each step runs.
Supersedes `plan-gpu-migration-2026-08-28.md`, which was written for a
specific A6000 box that is no longer being used; that document is deleted
rather than kept, because a plan whose premise has changed reads as current
and misleads. Its still-live technical content is carried over below in full.
Git history has the original.

Two machines only:

- **Mac** — the development machine (`mac-mini.local`, MPS). Free.
- **Pod** — a rented RunPod instance. Billed by the hour.

Read `hopspc.md` §1 and §14–15 first, plus
`docs/chat-trained-draft-result-2026-08-29.md` for the current state.

## 0. Porting is not a code task

A device audit found **no hardcoded device anywhere**: every entry point takes
`--device`, already defaulting to `cuda` (`collect.py`, `train_draft.py`,
`_experiment_common.py`, `07_chained_eval.py`, `00_smoke_test.py`,
`01_build_dense_index.py`, `HFTargetLLM.__init__`). `CachedTargetRunner` and
`chat_layout_features` take the device as a string. `torch_dtype="auto"` gives
bf16 on both MPS and CUDA. `attn_implementation="sdpa"` is required on CUDA
(spec §9: eager materialises `[T,T]` per head per layer and once OOM'd a 49 GB
card at batch 4 on ~2,600-token trajectories) and is already the default at
every load site.

So nothing needs porting. What follows is the work that remains, and the rule
that keeps it cheap.

## 1. The one rule that controls the bill

**The pod is rented for GPU work only.** Anything that runs on a CPU runs on
the Mac, for free, before or after the pod exists: the whole test suite, shard
validation, every analysis, every prereg and every result document. The
offline-analysis design — raw per-position columns, `rebucket`, `select` —
exists precisely so re-slicing never needs a GPU. Keep it that way.

Four consequences, each of which costs real money if ignored:

1. **Keep a persistent volume** for the HuggingFace cache and the draft
   checkpoints. Re-downloading `Qwen/Qwen3-1.7B` (~3.4 GB) every session is
   billed; re-training a checkpoint you already paid for is worse.
2. **Launch long jobs detached** — `nohup … > log 2>&1 < /dev/null & disown`,
   confirm `PPID=1` (spec §1). A dropped connection must not kill a job you
   are paying for.
3. **`--resume` before the first long collect** (done — see §3).
4. **Bring back only small artifacts.** Round files, raw per-position columns
   and summaries are KB–MB. Checkpoints are 1.2 GB and stay on the volume.

## 2. Cost model, and how much of it is measured

Measured on the Mac (MPS, `Qwen/Qwen3-1.7B`, shard mean 524 tokens/record):

| Work | Measured (MPS) |
|---|---|
| Collect | 1.4 s/record → 1000 records ≈ 23 min |
| Train one arm, 102 records × 3 epochs, chat features | 78 min |
| Nine chained arms, 18 held-out records | ~70 min |

Where the compute goes, for 850 train records × 3 epochs:

| Component | TFLOP |
|---|---|
| Target forwards building features (chat mode, ~2 passes) | ~9,500 |
| `predict_logits`: `[T,2048] @ [2048,151936]` over the full sequence | ~2,500 |

**Collection is not the expensive part at this scale.** The superseded plan
assumed an 8–12 hour collect; measured, it is minutes. At 1.7B the agent emits
~6 tokens per step. The cost is training, dominated by the target's feature
forwards.

> **The GPU figures below are extrapolations, not measurements** — MPS timings
> plus FLOP counting. Session 1 produces the first real CUDA datapoint;
> recalibrate before budgeting session 2.

## 3. Status of the remaining work

| Item | Where | Status |
|---|---|---|
| §4 resume | Mac | **DONE** — `--resume` on `collect.py`, 6 tests, 334 total green |
| Device bridge | Pod, then Mac | ready: `scripts/08_device_bridge.sh`, `scripts/09_device_bridge_compare.py` |
| Batched collection | Mac (build+test), Pod (validate) | **NOT STARTED** — §4 below |
| Scale-up prereg | Mac | **NOT STARTED** |

## 4. Batched collection — the remaining build, and it is free

Current shape: `run_react_trajectory` loops one question at a time and calls
`llm.generate(context)` once per hop — the only generate call site
(`agent_pipeline.py`). GPU utilisation is near zero between steps.

Design, chosen to keep the blast radius inside one class:

- Add `HFTargetLLM.generate_batch(contexts: list[str]) -> list[str]`: build
  each prompt with the SAME `apply_chat_template` path (including the
  `enable_thinking` TypeError fallback), **left-pad**, generate with
  `do_sample=False`, slice each output by its own prompt length, strip
  `<think>`, apply `_truncate_to_first_action`.
- Add `run_react_trajectories_batched(questions, llm, retriever, …)`: an
  active-set lockstep driver. Each round, collect the contexts of all live
  trajectories, one `generate_batch`, then per trajectory do the existing
  per-step work — `_split_generated_step`, `_format_passage`, retrieval,
  context append — **unchanged**. Trajectories retire on Finish or
  `max_hops`; the active set shrinks.
- `collect.py` gains `--batch-size` (default 1 → the existing sequential path,
  untouched).

Everything that has ever produced a bug — the step/context invariant, the span
slicing, the truncation, the labelling — stays on the unchanged code path.
Only the generate call is new.

Tests (CPU, no network, mandatory, all on the Mac):

1. With a `MockLLM`-style batch stub, batched trajectories are **identical**
   to sequential ones for the same questions — steps, context, labels.
2. The driver retires trajectories independently: a batch mixing a 1-hop
   Finish, a 2-hop, and a malformed step yields exactly the sequential
   results, and `"".join(step.text) == context` holds for each.
3. Left-padding equivalence at the tensor level: a stub model records the
   prompts it received; assert each sequence's non-pad content and its
   attention mask match the single-sequence case.
4. `--batch-size 1` is bit-identical to the pre-change path.

Known and to be documented, not fixed: with a real model, batched matmuls are
not bit-identical to single-sequence ones, so a greedy token can flip and a
batched shard will differ slightly from a sequential one. That is fine — a
shard is data, and every downstream arm uses one shard — but it must be stated
in the shard's provenance and never used to explain away a result. Validate
once on the pod: 5 questions batched vs sequential, report how many
trajectories match exactly.

## 5. Session 1 — bridge (~30 min billed)

```bash
git clone https://github.com/AIResearcher1499/hopspec.git && cd hopspec
uv venv .venv && uv pip install -e ".[dense-retrieval]" --python .venv/bin/python
.venv/bin/python -m pytest tests -q                     # 334 passed, else env problem
.venv/bin/python scripts/00_smoke_test.py --device cuda --num-questions 5
time bash scripts/08_device_bridge.sh cuda              # keep the timing
```

Bring back (≈280 KB): `data/rounds_cuda_{lookup,scaffold}_1p7b.jsonl` and the
wall-clock. Then **destroy the pod.** The comparison runs on the Mac:

```bash
.venv/bin/python scripts/09_device_bridge_compare.py --device cuda
```

Both bridged arms are model-free and deterministic — no draft network, no
sampling — so the only thing that can differ across devices is the target's
greedy argmax. The verdict is exact: the round files match, or a countable
number of rounds diverged. Nothing further is scheduled until it is written
up.

What the bridge does NOT cover: the draft network's own forward is not
exercised by these two arms. That is acceptable only because the draft is
retrained on CUDA anyway, after which no number depends on an MPS-trained
draft. Until then the neural-arm results in
`chat-trained-draft-result-2026-08-29.md` are **pilot-only** — which is the
fallback the original bridge design pre-registered for a large delta. Say it
that way in the paper; do not quote an MPS neural number as a CUDA one.

## 6. Mac between sessions — free

Build and test §4's batched collection. Freeze the scale-up prereg. This is
the largest remaining block of work and none of it is billed.

## 7. Session 2 — collect, train, evaluate (~3–5 h billed)

Only after session 1 is written up and the batching is tested.

```bash
# validate batching against the sequential path on the real model
… --batch-size 1 --max-questions 5 --out /tmp/seq.jsonl
… --batch-size 8 --max-questions 5 --out /tmp/bat.jsonl   # report exact matches

# the collect — detached, resumable, PPID=1
nohup .venv/bin/python -u scripts/01_generate_trajectories.py \
  --benchmark hotpotqa --split validation --split-file data/hotpot_split.json \
  --retriever bm25-distractor --target-model-name Qwen/Qwen3-1.7B --device cuda \
  --max-questions 1000 --batch-size 8 --resume \
  --out data/shard_1p7b_scale.jsonl > collect.log 2>&1 < /dev/null & disown
```

**Bring the shard back (~16 MB) and validate it on the Mac** —
`06_validate_shard.py` takes no `--device` and loads no model, so paying for
it is waste. Only once it passes do you spend GPU time training on it, then
run the arm sweep per the scale-up prereg.

Budget: collect 5–15 min, training 0.5–1.5 h for both arms, arm sweep 1–2 h on
~150 held-out records.

## 8. Later — the larger target model

This is where the money actually is. Going from 1.7B to 8B multiplies the
target forwards by ~5, and a better-behaved agent writes full `Thought:` lines
instead of six tokens, multiplying generated length by up to ~8. Collection
stops being free (2–3 h) and a full cycle lands at 4–8 h. Two or three cycles:
**~15–30 GPU-hours for everything foreseeable.** Rent in bounded sessions;
there is no case for a long-lived pod.

## 9. Cost lever, noted but not recommended

`predict_logits` runs over the **whole sequence** against a 151,936-row head —
about 2,500 TFLOP of the training budget. Computing the token loss on a subset
of positions would cut it substantially. It also **changes the loss
definition**, so it needs its own prereg and no number produced under it may
be compared with any number above (spec §15). Recorded so the lever is known
to exist, not proposed.
