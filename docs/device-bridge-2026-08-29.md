# Device bridge: CUDA vs MPS — 2026-08-29

Executes the bridge measurement of `plan-runpod-execution-2026-08-29.md` §5.
It answers one question, fixed in advance: **is the MPS pilot comparable to
what a rented CUDA box will produce, or is it pilot-only?**

## Setup

RTX PRO 4500 Blackwell (32 GB, driver 580.159.04), torch 2.8.0.dev+cu128,
`Qwen/Qwen3-1.7B`, `data/shard_1p7b.jsonl`, the existing 15%/seed-0 split, 18
held-out records, gamma=4, `--replay-mode chat`. 334 tests green on the pod
before anything else ran. Bridge wall-clock: **1 min 24 s**.

The two bridged arms use **no draft checkpoint** — `lookup` and `scaffold`
propose by deterministic rules over the committed token ids. No draft network,
no sampling. So the only thing that can differ between devices is the
**target's greedy argmax**, which is exactly the risk the bridge was designed
around. This is also why no 1.2 GB weight transfer was needed.

## Result

| arm | rounds MPS/CUDA | records identical | pooled MPS | pooled CUDA | per-record delta [95% boot] | sign p |
|---|---|---|---|---|---|---|
| lookup | 449 / 448 | **16/18** | 0.321 | 0.324 | −0.0058 [−0.0562, +0.0389] | 1.000 |
| scaffold | 478 / 482 | **13/18** | 0.238 | 0.234 | −0.0110 [−0.0458, +0.0167] | 1.000 |

Not byte-identical: 7 of 36 record-arms diverge. But the divergence is rare,
unsigned, and does not move the measurement — both bootstrap intervals
straddle zero, both sign tests are p=1.000, and the two arms move in
*opposite* directions (+0.003 and −0.004 pooled), which is what noise looks
like and not what a systematic device effect looks like.

## What divergence actually looks like

The mechanism is exactly the predicted one, and worth recording because it
explains why round *counts* differ at all:

- **lookup**, record `5a713f39…`, first divergence at round #7:
  `accepted: 4 → 2`, `emitted: 5 → 3`. A greedy argmax flipped mid-span, so
  CUDA accepted two fewer tokens of the same lookup proposal. From there the
  two replays are at different positions and every later round is a different
  round: 21 rounds on MPS, 26 on CUDA for that record.
- **scaffold**, record `5a710bb1…`, first divergence at round #17:
  `source: none → scaffold`. Nothing about the FSM changed; the committed text
  had already diverged, so the FSM found itself in a state where it fires.

So one flip cascades into a structurally different replay. That is why "how
many rounds differ" is the wrong question and "how many records diverge at
all, and does the aggregate move" is the right one.

## Verdict, against the criterion fixed in advance

The pre-registered rule was: identical → device-stable, report as a result;
a countable number of diverging rounds → report the count, and *"if divergence
is not rare, every table must be regenerated on CUDA."*

**Divergence is rare and its aggregate effect is null.** The MPS pilot is
**directionally comparable** to CUDA and may be described that way in the
paper, with this measurement cited. Tables do not need regenerating on that
account.

Two limits, both to be stated wherever this is used:

1. **Not byte-identical, so do not merge.** A CUDA run and an MPS run of the
   same arm are not interchangeable at the round level and must never be
   pooled into one table. Compare arms *within* a device.
2. **This says nothing about the draft network.** Neither bridged arm runs it.
   The neural-arm numbers in `chat-trained-draft-result-2026-08-29.md` stay
   **pilot-only** until the draft is retrained on CUDA — which is scheduled
   anyway, after which nothing depends on an MPS-trained draft.

## Cost, and two mistakes worth recording

The session cost roughly **$0.20**. Two things went wrong and both are now
fixed in code rather than in memory:

1. **A "probe" mutation deployed a real pod.** `podFindAndDeployOnDemand` is
   not a dry run. It was terminated within a minute, but the lesson is that
   RunPod's GraphQL has no dry-run mode and every deploy mutation spends money.
2. **The bridge script defaulted to `.venv/bin/python`**, which does not exist
   on a pod that installs into the image's own python. The first billed run
   exited in 0.004 s. `08_device_bridge.sh` now falls back to `python` and
   verifies `hopspec` is importable before doing anything.

A third item is a pricing trap for the next session: RunPod reports a
`communityPrice` for cards that are **not on community cloud at all**
(`communityCloud: false`). RTX PRO 4500 advertises $0.34 and actually bills
$0.72 on secure cloud. Cards genuinely available on community: A5000 $0.16,
RTX 3090 $0.22, **A6000 $0.33**. Check `communityCloud` before quoting a
price, and deploy with `cloudType: COMMUNITY`.
