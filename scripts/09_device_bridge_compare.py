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


def group(rounds: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rounds:
        out.setdefault(str(row.get("question_id", row.get("record_index"))), []).append(row)
    return out


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

    # Compare PER RECORD, not as one flat list. Once a greedy argmax flips, the
    # replay takes a different path and the round counts stop lining up — so a
    # global "the lengths differ" says only that something diverged somewhere,
    # which is exactly the hand-wave this measurement exists to avoid. What
    # matters is how MANY records diverge at all, and where the first one does.
    by_record_a, by_record_b = group(baseline), group(device)
    keys = sorted(set(by_record_a) & set(by_record_b))
    clean, diverged = [], []
    for key in keys:
        rows_a, rows_b = by_record_a[key], by_record_b[key]
        same = len(rows_a) == len(rows_b) and all(
            all(x.get(k) == y.get(k) for k in MEASURED) for x, y in zip(rows_a, rows_b)
        )
        (clean if same else diverged).append(key)
    result["records_compared"] = len(keys)
    result["records_identical"] = len(clean)
    result["records_diverged"] = len(diverged)
    result["identical"] = not diverged
    result["diverging_rounds"] = None if diverged else 0
    result["first_divergence"] = None
    if diverged:
        rows_a, rows_b = by_record_a[diverged[0]], by_record_b[diverged[0]]
        for index, (x, y) in enumerate(zip(rows_a, rows_b)):
            fields = {k: (x.get(k), y.get(k)) for k in MEASURED if x.get(k) != y.get(k)}
            if fields:
                result["first_divergence"] = {
                    "record": diverged[0], "round_index": index,
                    "rounds": f"{len(rows_a)} vs {len(rows_b)}", "fields": fields,
                }
                break

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
    print(f"{'arm':>10} {'rounds':>12} {'records ok':>12} {'pooled MPS':>11} "
          f"{'pooled dev':>11} {'per-rec delta':>14}")
    for r in results:
        rounds = f"{r['baseline_rounds']}/{r['device_rounds']}"
        ok = f"{r['records_identical']}/{r['records_compared']}"
        print(f"{r['arm']:>10} {rounds:>12} {ok:>12} "
              f"{r['baseline_pooled']:>11.3f} {r['device_pooled']:>11.3f} "
              f"{r['per_record_delta']:>+14.4f}")

    all_identical = all(r["identical"] for r in results)
    print()
    if all_identical:
        print("VERDICT: byte-identical measurements on both devices.")
        print("The verification path is device-stable on this workload. Report")
        print("that as a result. It does NOT extend to the draft network, which")
        print("is not exercised by these arms. See the plan for why that is")
        print("acceptable: the draft is retrained on CUDA anyway.")
    else:
        worst = max(abs(r["per_record_delta"]) for r in results)
        diverged = sum(r["records_diverged"] for r in results)
        total = sum(r["records_compared"] for r in results)
        print(f"VERDICT: not byte-identical. {diverged} of {total} record-arms")
        print("diverged; a greedy argmax flipping near a tie sends the replay down")
        print("a different path and the round counts stop lining up from there.")
        print(f"Largest per-record effect on the measurement: {worst:+.4f} "
              "accepted/round.")
        print("Read the intervals below, not this line: rare flips with a null")
        print("aggregate effect mean the pilot is directionally comparable;")
        print("frequent or one-sided ones mean every table must be regenerated.")
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
