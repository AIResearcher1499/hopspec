"""Chained speculative evaluation — the decisive experiment (spec §11.4),
now with structure-routed draft sources.

Replays held-out trajectories with a chosen draft source, verified by the
real target, and reports accepted tokens per round by recency bucket and by
source. The verifier, the KV cache and the rails-restoring replay are the
same machinery for every source: only the proposer changes.

    --draft-source neural     the trained draft checkpoint (the §11.4 arm)
                   lookup     n-gram copy scoped to question + passages
                   scaffold   deterministic ReAct scaffold FSM
                   routed     scaffold -> lookup -> neural
                   entropy    ReSpec-style: lookup vs neural by target entropy

Two things are FITTED, and both are fitted on the TRAIN split only, then
printed:

- the scaffold's opening literal. The spec's ReAct grammar says every step
  opens with "Thought:"; the 1.7B agent opens 245 of 260 steps with "Action:"
  instead. Hardcoding the grammar would score the scaffold arm at ~0 for a
  reason that has nothing to do with the idea under test.
- the entropy threshold of the comparator arm (--tune-entropy). Tuning a
  baseline on held-out data is how you manufacture a baseline that loses.

Caveats printed with the results, both mandatory reading before quoting a
number:

- a round right after a hop contains easy TEMPLATE tokens, which genuinely
  speed decoding but say nothing about content tokens. The per-source table
  splits them out: everything the scaffold accepts is TEMPLATE.
- rounds are NOT byte-identical across sources (the sequences diverge as
  soon as two sources propose differently), so paired McNemar over positions
  does not apply between arms. Compare the per-record means instead.
"""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-name", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="draft checkpoint from 03/04 --checkpoint-out; "
                             "required for every source that uses the neural draft")
    parser.add_argument("--hop-signal", action="store_true",
                        help="MUST match how the checkpoint was trained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-rounds-per-region", type=int, default=None)
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--rounds-out", default=None, help="raw per-round jsonl")
    parser.add_argument("--replay-mode", default="raw", choices=["raw", "chat"],
                        help="'chat' re-renders the collection-time wrapper at "
                             "every step boundary (the deployed loop); 'raw' "
                             "feeds the recorded document, the original "
                             "behaviour. Numbers from the two modes are NEVER "
                             "comparable — see spec 15.")

    parser.add_argument("--draft-source", default="neural",
                        choices=["neural", "lookup", "scaffold", "routed", "entropy"])
    parser.add_argument("--chain", action="store_true",
                        help="routed: stack sources within one round instead of "
                             "one source per round")
    parser.add_argument("--scaffold-opening", default="fit",
                        choices=["fit", "thought", "action"],
                        help="'fit' takes the majority opening on the TRAIN split")
    parser.add_argument("--scaffold-verb", default="none",
                        choices=["none", "fit", "Search", "Finish"],
                        help="bet on the action verb instead of stopping at the "
                             "shared 'Action:' prefix")
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-ngram", type=int, default=2)
    parser.add_argument("--entropy-threshold", type=float, default=1.0,
                        help="nats; below it the entropy arm takes the lookup")
    parser.add_argument("--tune-entropy", default=None,
                        help="comma-separated grid to tune on the TRAIN split, "
                             "e.g. '0.25,0.5,1.0,2.0'")
    parser.add_argument("--tune-max-records", type=int, default=6)
    parser.add_argument("--entropy-scaffold", action="store_true",
                        help="structure + entropy: give the entropy arm the "
                             "scaffold FSM too (measure the combination before "
                             "concluding anything from a loss to it)")
    return parser


NEEDS_CHECKPOINT = {"neural", "routed", "entropy"}
NEEDS_TOKENIZER = {"scaffold", "routed"}


def main() -> int:
    args = build_parser().parse_args()
    if args.draft_source in NEEDS_CHECKPOINT and not args.checkpoint:
        raise SystemExit(f"--draft-source {args.draft_source} needs --checkpoint")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hopspec.infer.chained_eval import (
        ChainedSpeculator,
        chat_prompt_ids,
        chat_prompt_seam_ok,
        record_means,
        replay_record,
        summarize_rounds,
        summarize_rounds_by_source,
    )
    from hopspec.infer.chained_eval import INSERTED_SEGMENTS
    from hopspec.infer.routed_draft import (
        ACTION_PREFIX,
        DEFAULT_STEP_OPENING,
        EntropyRoutedProposer,
        HFScaffoldTokenizer,
        RoutedProposer,
        ScaffoldFSM,
        ScopedLookup,
        fit_scaffold_fsm,
        tune_entropy_threshold,
    )
    from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel
    from hopspec.train.ablation import load_records, split_records

    records = load_records(args.trajectory_file)
    # Same in-shard split as 03/04: measure on the SAME held-out pool, and
    # fit anything that needs fitting on the train side of it.
    train, heldout = split_records(
        records, eval_fraction=args.eval_fraction, seed=args.split_seed
    )
    if args.max_records is not None:
        heldout = heldout[: args.max_records]
    print(f"replaying {len(heldout)} held-out records "
          f"(mode={args.replay_mode}, source={args.draft_source}, "
          f"gamma={args.gamma}, hop_signal={args.hop_signal}); "
          f"{len(train)} train records for fitting")

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, torch_dtype="auto", attn_implementation="sdpa"
    ).to(args.device)
    target_model.eval()

    draft_model = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = HopSpecDraftConfig(**checkpoint["config"])
        draft_model = HopSpecDraftModel(
            config, torch.nn.Embedding(config.vocab_size, config.target_hidden_size)
        )
        draft_model.load_state_dict(checkpoint["state_dict"])
        draft_model.token_embedding.weight.requires_grad_(False)
        # A draft trained on one feature layout and served the other is a
        # train/serve mismatch that nothing downstream would report.
        trained_mode = checkpoint.get("feature_mode")
        if trained_mode is None:
            print("WARNING: checkpoint records no feature_mode (trained before "
                  "the flag existed); assuming 'raw'.")
            trained_mode = "raw"
        if trained_mode != args.replay_mode:
            print(f"WARNING: checkpoint was trained with --feature-mode "
                  f"{trained_mode} but this run replays in {args.replay_mode} "
                  "mode. The draft is being served a feature layout it was not "
                  "trained on; say so wherever this run is quoted.")
        else:
            print(f"checkpoint feature_mode={trained_mode} matches the replay mode")

    tokenizer = None
    if args.replay_mode == "chat" or args.draft_source in NEEDS_TOKENIZER \
            or args.entropy_scaffold:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)

    prefix_ids: list[int] = []
    suffix_ids: list[int] = []
    decode = None
    if args.replay_mode == "chat":
        prefix_ids, suffix_ids = chat_prompt_ids(tokenizer)

        def decode(ids):
            return tokenizer.decode(list(ids), skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)

        # Validate before spending GPU time (spec 15): if the wrapper's seams
        # merge with the document, every label shifts by a token and nothing
        # downstream would say so.
        probe = heldout[0]
        segments = probe["segment_type_ids"]
        inserted = {int(segment) for segment in INSERTED_SEGMENTS}
        cuts = [len(probe["input_ids"])] + [
            index for index in range(1, len(segments))
            if segments[index] not in inserted and segments[index - 1] in inserted
        ]
        for cut in cuts:
            context = decode(probe["input_ids"][:cut])
            if not chat_prompt_seam_ok(tokenizer, context, prefix_ids, suffix_ids):
                raise SystemExit(
                    f"chat wrapper changes the document's tokenization at cut "
                    f"{cut}; the fixed prefix/suffix split is unsound for this "
                    "tokenizer and every label would shift"
                )
        print(f"chat replay: {len(prefix_ids)}-token prefix, "
              f"{len(suffix_ids)}-token suffix, seam verified at {len(cuts)} cuts")

    speculator = ChainedSpeculator(
        draft_model, target_model, device=args.device,
        feature_layer=args.feature_layer, hop_signal_enabled=args.hop_signal,
        prompt_prefix_ids=prefix_ids, prompt_suffix_ids=suffix_ids, decode=decode,
    )

    scaffold = None
    if args.draft_source in NEEDS_TOKENIZER or args.entropy_scaffold:
        scaffold_tokenizer = HFScaffoldTokenizer(tokenizer)
        fitted = {"step_opening": DEFAULT_STEP_OPENING, "action_verb": None}
        if args.scaffold_opening == "fit" or args.scaffold_verb == "fit":
            _fitted_fsm, fitted = fit_scaffold_fsm(
                train, scaffold_tokenizer, use_verb=args.scaffold_verb == "fit"
            )
            print(f"scaffold fitted on the TRAIN split: {fitted}")
        openings = {"fit": fitted["step_opening"],
                    "thought": DEFAULT_STEP_OPENING, "action": ACTION_PREFIX}
        verbs = {"none": None, "fit": fitted["action_verb"],
                 "Search": "Search", "Finish": "Finish"}
        scaffold = ScaffoldFSM(
            scaffold_tokenizer,
            step_opening=openings[args.scaffold_opening],
            action_verb=verbs[args.scaffold_verb],
        )
        print(f"scaffold: step_opening={scaffold.step_opening!r} "
              f"action_verb={scaffold.action_verb!r}")

    lookup = ScopedLookup(max_ngram=args.max_ngram, min_ngram=args.min_ngram)

    threshold = args.entropy_threshold
    if args.draft_source == "entropy" and args.tune_entropy:
        grid = [float(value) for value in args.tune_entropy.split(",")]
        tune_records = train[: args.tune_max_records]
        print(f"tuning the entropy threshold on {len(tune_records)} TRAIN records "
              f"over {grid}")
        threshold, table = tune_entropy_threshold(
            tune_records, speculator, lookup, grid, gamma=args.gamma,
            max_rounds_per_region=args.max_rounds_per_region,
            scaffold=scaffold if args.entropy_scaffold else None,
        )
        for row in table:
            print(f"  threshold {row['threshold']:>6.3f}: "
                  f"{row['mean_accepted']:.3f} accepted/round "
                  f"over {row['rounds']} rounds")
        print(f"chosen threshold: {threshold}")

    if args.draft_source == "neural":
        proposer = None                       # the speculator proposes for itself
    elif args.draft_source == "lookup":
        proposer = RoutedProposer(speculator, lookup=lookup)
    elif args.draft_source == "scaffold":
        proposer = RoutedProposer(speculator, scaffold=scaffold)
    elif args.draft_source == "routed":
        proposer = RoutedProposer(speculator, scaffold=scaffold, lookup=lookup,
                                  neural=speculator, chain=args.chain)
    else:
        proposer = EntropyRoutedProposer(
            speculator, lookup=lookup, neural=speculator, threshold=threshold,
            scaffold=scaffold if args.entropy_scaffold else None,
        )

    all_rounds = []
    for index, record in enumerate(heldout):
        rounds = replay_record(
            record, speculator, gamma=args.gamma, proposer=proposer,
            max_rounds_per_region=args.max_rounds_per_region,
        )
        for row in rounds:
            row["record_index"] = index
            row["question_id"] = record["question_id"]
            row["draft_source"] = args.draft_source
        all_rounds.extend(rounds)
        print(f"  [{index + 1}/{len(heldout)}] {record['question_id']}: "
              f"{len(rounds)} rounds")

    if args.rounds_out:
        with open(args.rounds_out, "w", encoding="utf-8") as f:
            for row in all_rounds:
                f.write(json.dumps(row) + "\n")
        print(f"raw rounds -> {args.rounds_out}")

    if not all_rounds:
        print("no rounds — nothing to summarize")
        return 1

    print(f"\naccepted tokens per round by recency bucket "
          f"(gamma={args.gamma}, {len(all_rounds)} rounds):")
    print(f"{'bucket':>6} {'rounds':>7} {'accepted':>9} {'emitted':>8}")
    for bucket, stats in summarize_rounds(all_rounds).items():
        print(f"{bucket:>6} {stats['rounds']:>7} {stats['mean_accepted']:>9.3f} "
              f"{stats['mean_emitted']:>8.3f}")

    by_source = summarize_rounds_by_source(all_rounds)
    print("\nper draft source (proposed/accepted are TOKENS, not rounds):")
    print(f"{'source':>9} {'rounds':>7} {'proposed':>9} {'accepted':>9} "
          f"{'acceptance':>11}  kind")
    for source, stats in by_source.items():
        kind = "TEMPLATE" if source == "scaffold" else (
            "content" if source in ("lookup", "neural") else "-")
        print(f"{source:>9} {stats['rounds']:>7} {stats['proposed']:>9} "
              f"{stats['accepted']:>9} {stats['acceptance']:>11.3f}  {kind}")

    total_accepted = sum(row["accepted"] for row in all_rounds)
    means = record_means(all_rounds)
    template_accepted = by_source.get("scaffold", {}).get("accepted", 0)
    print(f"\npooled: {total_accepted / len(all_rounds):.3f} accepted tokens/round "
          f"over {len(all_rounds)} rounds")
    print(f"mean of per-record means: "
          f"{sum(means.values()) / len(means):.3f} over {len(means)} records")
    print(f"template (scaffold) share of accepted tokens: "
          f"{template_accepted}/{total_accepted}")

    print("\nNOTE: near-hop rounds include easy TEMPLATE tokens ('Thought:',"
          "\n'Action:'); the per-source table above is the split — read it"
          "\nbefore quoting the pooled number as a content result."
          "\nNOTE: rounds are not aligned across draft sources, so compare arms"
          "\nwith a paired test over the per-record means, not McNemar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
