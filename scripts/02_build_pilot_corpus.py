"""Build the pilot corpus: gold-title passages ∪ random background.

Gold titles come from the VALIDATION splits of HotpotQA / 2WikiMultihopQA /
MuSiQue; background is reservoir-sampled from the Dec-2018 Wikipedia DPR dump
(psgs_w100 TSV).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psgs-w100", required=True, help="path to psgs_w100.tsv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-background", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["hotpotqa", "2wikimultihopqa", "musique"],
    )
    args = parser.parse_args()

    from hopspec.data.collect import load_benchmark
    from hopspec.data.gold_titles import GOLD_TITLE_EXTRACTORS
    from hopspec.data.pilot_corpus import build_pilot_corpus
    from hopspec.data.wiki_dpr import stream_psgs_w100

    gold_titles: set[str] = set()
    for name in args.benchmarks:
        extractor = GOLD_TITLE_EXTRACTORS[name]
        dataset = load_benchmark(name, split="validation")
        for example in dataset:
            gold_titles |= extractor(example)
    print(f"{len(gold_titles)} gold titles across {args.benchmarks}")

    corpus = build_pilot_corpus(
        stream_psgs_w100(args.psgs_w100), gold_titles,
        num_background=args.num_background, seed=args.seed,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    print(f"wrote {len(corpus)} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
