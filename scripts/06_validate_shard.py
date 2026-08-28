"""Audit a shard before spending GPU time on it. Exits non-zero on any hard
check failure. Every check corresponds to a defect that actually shipped."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter

from hopspec.data.schema import SegmentType, recency_bucket_id
from hopspec.eval.diagnostic import DECODE_SEGMENT_TYPES, bucket_token_stats

_ARRAY_KEYS = (
    "input_ids", "segment_type_ids", "recency_bucket_ids", "recency_distances",
)


def validate_record(record: dict, line_number: int, errors: list[str]) -> None:
    def fail(message: str) -> None:
        errors.append(f"record {line_number} ({record.get('question_id')}): {message}")

    # Step/context invariant.
    joined = "".join(step["text"] for step in record["steps"])
    if joined != record["context"]:
        fail("step texts do not concatenate to context")

    # Array lengths.
    lengths = {key: len(record[key]) for key in _ARRAY_KEYS}
    if len(set(lengths.values())) != 1:
        fail(f"per-token array length mismatch: {lengths}")
        return

    # bucket == f(distance).
    for bucket, distance in zip(record["recency_bucket_ids"], record["recency_distances"]):
        if bucket != recency_bucket_id(distance):
            fail(f"bucket {bucket} != recency_bucket_id({distance})")
            break

    # Empty steps.
    if any(step["text"] == "" for step in record["steps"]):
        fail("empty step text")

    passage_type = int(SegmentType.RETRIEVED_PASSAGE)
    for step in record["steps"]:
        if step["segment_type"] != passage_type:
            # "Observation:" leaking into thoughts.
            if step["segment_type"] == int(SegmentType.THOUGHT) and \
                    "Observation:" in step["text"]:
                fail("'Observation:' leaked into a THOUGHT step")
            continue
        if not step["text"].endswith("\n"):
            fail("retrieved passage does not end in newline")
        core = step["text"].removeprefix("Observation: ").strip()
        if not core:
            fail("empty retrieved passage")

    # The token right after each closed passage must be TEMPLATE-labeled
    # (in ReAct it is literally 'Thought' — if it is not excluded as template,
    # post-hop acceptance measures template predictability).
    seg_ids = record["segment_type_ids"]
    num_tokens = len(seg_ids)
    i = 0
    while i < num_tokens:
        if seg_ids[i] == passage_type:
            while i < num_tokens and seg_ids[i] == passage_type:
                i += 1
            if i < num_tokens and seg_ids[i] != int(SegmentType.TEMPLATE):
                fail(
                    f"token {i} after a closed passage is "
                    f"{SegmentType(seg_ids[i]).name}, expected TEMPLATE"
                )
        else:
            i += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--split-file", default=None,
                        help="leakage check: no record may come from the eval pool")
    parser.add_argument("--target-model-name", default=None,
                        help="enables the NFC decode round-trip check")
    args = parser.parse_args()

    errors: list[str] = []
    records: list[dict] = []
    with open(args.shard, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            validate_record(record, line_number, errors)

    # Duplicate ids.
    id_counts = Counter(record["question_id"] for record in records)
    for question_id, count in id_counts.items():
        if count > 1:
            errors.append(f"duplicate question_id {question_id} ({count} records)")

    # Eval-pool leakage.
    if args.split_file:
        with open(args.split_file, encoding="utf-8") as f:
            eval_ids = set(json.load(f)["eval_ids"])
        leaked = sorted(id_counts.keys() & eval_ids)
        if leaked:
            errors.append(f"{len(leaked)} records leak from the eval pool: {leaked[:5]}")

    # NFC round-trip (tokenizer normalization is expected; labels unaffected).
    if args.target_model_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
        for line_number, record in enumerate(records, start=1):
            decoded = tokenizer.decode(record["input_ids"])
            if unicodedata.normalize("NFC", decoded) != unicodedata.normalize(
                "NFC", record["context"]
            ):
                errors.append(
                    f"record {line_number}: NFC(decode(input_ids)) != NFC(context)"
                )

    # ---- soft stats ----
    passage_type = int(SegmentType.RETRIEVED_PASSAGE)
    total_hops = 0
    repeat_hops = 0
    for record in records:
        prefix = ""
        for step in record["steps"]:
            if step["segment_type"] == passage_type:
                core = step["text"].removeprefix("Observation: ").strip()
                total_hops += 1
                if core and core in prefix:
                    repeat_hops += 1
            prefix += step["text"]
    if total_hops:
        print(f"repeat-hop rate: {repeat_hops}/{total_hops} "
              f"({repeat_hops / total_hops:.1%})")

    # Per-bucket measurement-viability table over decode-phase positions.
    decode_types = {int(t) for t in DECODE_SEGMENT_TYPES}
    buckets, tokens = [], []
    for record in records:
        for token, segment, bucket in zip(
            record["input_ids"], record["segment_type_ids"], record["recency_bucket_ids"]
        ):
            if segment in decode_types:
                buckets.append(bucket)
                tokens.append(token)
    print("\nper-bucket measurement viability (decode-phase positions):")
    for bucket, stats in bucket_token_stats(buckets, tokens).items():
        verdict = "DEGENERATE" if stats["degenerate"] else "ok"
        print(f"  bucket {bucket}: n={stats['n']} distinct={stats['distinct']} "
              f"majority={stats['majority_rate']:.3f}  {verdict}")

    incomplete = sum(1 for record in records if not record.get("is_complete"))
    print(f"\n{len(records)} records, {incomplete} incomplete")

    if errors:
        print(f"\n{len(errors)} HARD CHECK FAILURES:", file=sys.stderr)
        for error in errors[:50]:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("shard passed all hard checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
