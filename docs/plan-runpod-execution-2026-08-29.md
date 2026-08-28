# Plan: what runs local, what runs on a rented GPU — 2026-08-29

Companion to `plan-gpu-migration-2026-08-28.md` (which says *how* to move) and
its §2' amendment (which says what the bridge measurement became once the
weights could not be copied). This one says *where each step runs and what it
costs*, because the box is now billed by the hour.

Read `hopspc.md` §1 and §15 first. The §5 division of labour there already
says analysis never needs the GPU — this plan is that principle applied to a
meter that is running.

## 0. The one rule that controls the bill

**The pod is rented for GPU work only.** Anything that runs on a CPU runs on
the laptop, for free, before or after the pod exists. That includes the whole
test suite, shard validation, every analysis, every prereg and every result
document.

Four consequences, all of which cost real money if ignored:

1. **Keep a persistent volume** for the HuggingFace cache and the draft
   checkpoints. Re-downloading `Qwen/Qwen3-1.7B` (~3.4 GB) at the start of
   every session is billed, and re-training a checkpoint you already paid for
   is worse.
2. **Launch long jobs detached** — `nohup … > log 2>&1 < /dev/null & disown`,
   confirm `PPID=1` (spec §1). A dropped SSH must not kill a job you are
   paying for.
3. **Add `--resume` (migration plan §4) before the first long collect**, not
   after losing one. A killed collect otherwise restarts from zero.
4. **Bring back only small artifacts.** Round files, raw per-position columns
   and summaries are KB–MB. Checkpoints are 1.2 GB and stay on the volume.

## 1. Cost model, and how much of it is measured

Measured on the laptop (MPS, `Qwen/Qwen3-1.7B`, shard mean 524 tokens/record):

| Work | Measured (MPS) |
|---|---|
| Collect | 1.4 s/record → 1000 records ≈ 23 min |
| Train one arm, 102 records × 3 epochs, chat features | 78 min |
| Nine chained arms, 18 held-out records | ~70 min |

Where the compute actually goes, for 850 train records × 3 epochs:

| Component | TFLOP |
|---|---|
| Target forwards building features (chat mode, ~2 passes) | ~9,500 |
| `predict_logits`: `[T,2048] @ [2048,151936]` over the full sequence | ~2,500 |

**Collection is not the expensive part at this scale** — a point worth making
because the migration plan assumed it was ("an 8–12 hour collect"). At 1.7B
with an agent that emits ~6 tokens per step, it is minutes. The cost is
training, and it is dominated by the target's feature forwards.

> **These GPU figures are extrapolations, not measurements.** They come from
> MPS timings plus FLOP counting. Session 1 produces the first real CUDA
> datapoint; recalibrate this table from it before budgeting session 3.

## 2. Local, free — do all of this before renting anything

| Step | Command | Notes |
|---|---|---|
| Test suite | `.venv/bin/python -m pytest tests -q` | 328 must pass; CPU-only |
| Batched collection (§3) + its 4 tests | — | The tests use stubs; write and test it **entirely locally**. Only the 5-question real-model validation needs the GPU |
| Resume flag (§4) + its test | — | Same: pure CPU |
| Shard validation | `scripts/06_validate_shard.py --shard … --split-file …` | No `--device`, no model load — **runs local** |
| Every prereg and result doc | — | Freeze preregs before the pod is even booted |
| Every analysis | `scripts/09_device_bridge_compare.py`, `eval/analysis.py` | Raw per-position columns exist so re-slicing never needs a GPU |

## 3. Session 1 — bridge (~30 min billed)

Purpose: the device bridge (§2'), plus the first honest CUDA timing.

```bash
git clone https://github.com/AIResearcher1499/hopspec.git && cd hopspec
uv venv .venv && uv pip install -e ".[dense-retrieval]" --python .venv/bin/python
.venv/bin/python -m pytest tests -q                     # 328 passed, else env problem
.venv/bin/python scripts/00_smoke_test.py --device cuda --num-questions 5
time bash scripts/08_device_bridge.sh cuda              # keep the timing
```

Bring back (≈280 KB): `data/rounds_cuda_{lookup,scaffold}_1p7b.jsonl`, plus
the wall-clock of the bridge run.

Then **stop and destroy the pod.** The comparison and the write-up happen on
the laptop:

```bash
.venv/bin/python scripts/09_device_bridge_compare.py --device cuda
```

Two arms are model-free and deterministic, so the verdict is exact: the round
files match or a countable number of rounds diverged. Nothing further is
scheduled until that is written up.

## 4. Local between sessions — free

Write §3 batched collection and §4 resume with their tests, entirely on CPU.
Freeze the scale-up prereg. This is the largest block of work in the whole
project and none of it is billed.

## 5. Session 2 — collect, train, evaluate (~3–5 h billed)

Only after session 1's bridge is written up and the batching is tested.

```bash
# validate batching against the sequential path on the real model
.venv/bin/python scripts/01_generate_trajectories.py … --batch-size 1  --max-questions 5 --out /tmp/seq.jsonl
.venv/bin/python scripts/01_generate_trajectories.py … --batch-size 8  --max-questions 5 --out /tmp/bat.jsonl
# report how many of the 5 trajectories match exactly (migration plan §3)

# the collect — detached, resumable, PPID=1
nohup .venv/bin/python -u scripts/01_generate_trajectories.py \
  --benchmark hotpotqa --split validation --split-file data/hotpot_split.json \
  --retriever bm25-distractor --target-model-name Qwen/Qwen3-1.7B --device cuda \
  --max-questions 1000 --batch-size 8 --resume \
  --out data/shard_1p7b_scale.jsonl > collect.log 2>&1 < /dev/null & disown
```

**Bring the shard back (~16 MB) and validate it on the laptop** —
`06_validate_shard.py` needs no GPU, so paying for it is waste. Only once it
passes do you spend GPU time training on it:

```bash
.venv/bin/python scripts/03_run_diagnostic.py       … --feature-mode chat --checkpoint-out data/ckpt_base_chat_scale.pt
.venv/bin/python scripts/04_train_hop_signal_model.py … --feature-mode chat --checkpoint-out data/ckpt_hop_chat_scale.pt
bash scripts/…                                       # the arm sweep, per the scale-up prereg
```

Checkpoints stay on the volume. Bring back the round files, `raw_*.json` and
`summary_*.json` — MB at most.

Budget: collect 5–15 min, training 0.5–1.5 h for both arms, the arm sweep
1–2 h on ~150 held-out records. **~3–5 h.**

## 6. Later — the larger target model

This is where the money actually is, and the only part that justifies the
migration on its own. Going from 1.7B to 8B multiplies the target forwards by
~5, and a better-behaved agent writes full `Thought:` lines instead of six
tokens, multiplying generated length by up to ~8. Collection stops being free
(2–3 h) and a full cycle lands at 4–8 h.

Two or three such cycles: **~15–30 GPU-hours for everything foreseeable.**
Rent by the hour in bounded sessions; there is no case for a long-lived pod.

## 7. Cost lever, noted but not recommended

`predict_logits` runs over the **whole sequence** against a 151,936-row head —
about 2,500 TFLOP of the training budget. Computing the token loss on a subset
of positions would cut it substantially.

It also **changes the loss definition**, so it is not a free optimisation: it
needs its own prereg, and no number produced under it may be compared with any
number above (spec §15). Recorded so the lever is known to exist, not proposed.
