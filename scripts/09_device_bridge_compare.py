"""Device bridge comparison: CUDA rounds vs the MPS baselines.

See docs/plan-runpod-execution-2026-08-29.md.

The two bridged arms are model-free — `lookup` and `scaffold` propose by
deterministic rules over the committed token ids, with no draft network and no
sampling. So the ONLY thing that can differ between devices is the target's
greedy argmax, which is exactly the risk the plan named. That makes the test
exact rather than statistical: either the round files match, or a countable
number of rounds diverged.

The paired per-record statistics are reported only when something actually
diverged — there is nothing to be statistical about otherwise.

Run on the Mac, not the rented pod — analysis never needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import os

# Fields that ARE the measurement. record_index/question_id identify the row;
# draft_source and replay_mode are provenance and must match by construction.
MEASURED = ("distance", "bucket", "gamma", "accepted", "emitted", "hop_index",
            "region_tokens", "source", "token_sources")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_record_means(rounds: list[dict]) -> dict[str, float]:
    by_record: dict[str, list[int]] = {}
    for row in rounds:
        by_record.setdefault(str(row.get("question_id", row.get("record_index"))),
                             []).append(row["accepted"])
    return {k: sum(v) / len(v) for k, v in by_record.items()}


def compare(arm: str, baseline_path: str, device_path: str) -> dict:
    baseline, device = load(baseline_path), load(device_path)
    result = {"arm": arm, "baseline_rounds": len(baseline),
              "device_rounds": len(device)}

    if len(baseline) != len(device):
        # A different number of rounds means the two replays took different
        # paths; there is no row-to-row correspondence to report.
        result["identical"] = False
        result["diverging_rounds"] = None
        result["note"] = "round counts differ — the replays diverged structurally"
    else:
        diverging = [
            index for index, (a, b) in enumerate(zip(baseline, device))
            if any(a.get(k) != b.get(k) for k in MEASURED)
        ]
        result["identical"] = not diverging
        result["diverging_rounds"] = len(diverging)
        result["first_divergence"] = (
            None if not diverging else {
                "round_index": diverging[0],
                "baseline": {k: baseline[diverging[0]].get(k) for k in MEASURED},
                "device": {k: device[diverging[0]].get(k) for k in MEASURED},
            }
        )

    result["baseline_pooled"] = sum(r["accepted"] for r in baseline) / len(baseline)
    result["device_pooled"] = sum(r["accepted"] for r in device) / len(device)
    base_means, dev_means = per_record_means(baseline), per_record_means(device)
    shared = sorted(set(base_means) & set(dev_means))
    result["records"] = len(shared)
    result["per_record_delta"] = (
        sum(dev_means[k] - base_means[k] for k in shared) / len(shared)
        if shared else 0.0
    )
    result["max_abs_record_delta"] = (
        max(abs(dev_means[k] - base_means[k]) for k in shared) if shared else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda",
                        help="device tag used in data/rounds_<device>_<arm>_1p7b.jsonl")
    parser.add_argument("--arms", default="lookup,scaffold")
    parser.add_argument("--baseline-prefix", default="data/rounds_ct_")
    args = parser.parse_args()

    results = []
    for arm in args.arms.split(","):
        baseline = f"{args.baseline_prefix}{arm}_1p7b.jsonl"
        device = f"data/rounds_{args.device}_{arm}_1p7b.jsonl"
        for path in (baseline, device):
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}")
        results.append(compare(arm, baseline, device))

    print(f"device bridge: {args.device} vs the MPS baselines "
          f"(model-free arms, deterministic proposers)\n")
    print(f"{'arm':>10} {'rounds':>14} {'identical':>10} {'diverging':>10} "
          f"{'pooled MPS':>11} {'pooled dev':>11} {'per-rec delta':>14}")
    for r in results:
        rounds = f"{r['baseline_rounds']}/{r['device_rounds']}"
        div = "-" if r["diverging_rounds"] is None else str(r["diverging_rounds"])
        print(f"{r['arm']:>10} {rounds:>14} {str(r['identical']):>10} {div:>10} "
              f"{r['baseline_pooled']:>11.3f} {r['device_pooled']:>11.3f} "
              f"{r['per_record_delta']:>+14.3f}")

    all_identical = all(r["identical"] for r in results)
    print()
    if all_identical:
        print("VERDICT: byte-identical measurements on both devices.")
        print("The verification path is device-stable on this workload. Report")
        print("that as a result. It does NOT extend to the draft network, which")
        print("is not exercised by these arms. See the plan for why that is")
        print("acceptable: the draft is retrained on CUDA anyway.")
    else:
        print("VERDICT: the devices disagree. Report the diverging-round counts")
        print("above, not a hand-wave. Greedy argmax flipping near ties is the")
        print("expected cause; if divergence is not rare, every table must be")
        print("regenerated on CUDA.")
        for r in results:
            if r.get("first_divergence"):
                print(f"\nfirst divergence in {r['arm']}:")
                print(json.dumps(r["first_divergence"], indent=2)[:800])
    print("\nNOTE: these arms use no draft checkpoint, so this says nothing about")
    print("MPS-trained draft weights. Those numbers stay pilot-only until the")
    print("draft is retrained on CUDA.")
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
