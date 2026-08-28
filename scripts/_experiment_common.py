"""Shared runner for 03_run_diagnostic and 04_train_hop_signal_model.

Both scripts route through hopspec.train.ablation so the ONLY difference
between arms is the switch under test. This module holds the shared CLI and
the three-views reporting (all positions / novel hops only / matched cohort),
each with n, acceptance, distinct-token count, majority-class rate and a
verdict column.
"""

from __future__ import annotations

import argparse
import json


def build_shared_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target-model-name", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", default=None, help="summary json")
    parser.add_argument("--raw-out", default=None,
                        help="raw per-position columns json (never lose these)")
    parser.add_argument("--loss-targets", choices=["all", "generated"], default="all")
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--v-w", type=float, default=1.0)
    parser.add_argument("--p-w", type=float, default=0.1)
    parser.add_argument("--min-hop-span", type=int, default=25,
                        help="matched-cohort threshold for the third view")
    parser.add_argument("--checkpoint-out", default=None,
                        help="save the trained draft model (needed for 07)")
    parser.add_argument("--feature-mode", default="raw", choices=["raw", "chat"],
                        help="'chat' computes target features in the layout the "
                             "DEPLOYED loop produces (07 --replay-mode chat); "
                             "'raw' forwards the bare document, the original "
                             "behaviour. Encode the mode in --checkpoint-out: a "
                             "draft trained on one layout and served the other "
                             "is a train/serve mismatch nothing downstream "
                             "would report.")
    return parser


def run_experiment(args, *, hop_signal_enabled: bool, alpha: float, tau: float,
                   checkpoint_out: str | None = None) -> dict:
    import functools

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hopspec.eval.analysis import format_table, novel_hop_flags, select, summarize
    from hopspec.infer.chained_eval import chat_prompt_ids, chat_prompt_seam_ok
    from hopspec.train.train_draft import chat_layout_features
    from hopspec.model.draft_model import HopSpecDraftConfig, HopSpecDraftModel
    from hopspec.train.ablation import (
        evaluate_draft_model,
        load_records,
        split_records,
        train_draft_model,
    )
    from hopspec.train.target_policy import GeneratedTokensOnlyPolicy, LossTargetPolicy

    records = load_records(args.trajectory_file)
    train_records, heldout_records = split_records(
        records, eval_fraction=args.eval_fraction, seed=args.split_seed
    )
    print(f"{len(train_records)} train / {len(heldout_records)} held-out records")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, torch_dtype="auto", attn_implementation="sdpa"
    ).to(args.device)
    target_model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    config = HopSpecDraftConfig(
        target_hidden_size=target_model.config.hidden_size,
        vocab_size=target_model.config.vocab_size,
    )
    draft_model = HopSpecDraftModel.from_target_embedding(
        config, target_model.get_input_embeddings()
    )

    feature_mode = getattr(args, "feature_mode", "raw")
    feature_builder = None
    if feature_mode == "chat":
        prefix_ids, suffix_ids = chat_prompt_ids(tokenizer)

        def decode(ids):
            return tokenizer.decode(list(ids), skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)

        # Validate before spending GPU time (spec 15). If a wrapper seam merges
        # with the document, every label shifts by a token and no downstream
        # check would say so. Probe the boundaries training will actually use.
        from hopspec.infer.chained_eval import token_regions

        checked = 0
        for record in (train_records + heldout_records)[:5]:
            cuts = [len(record["input_ids"])] + [
                start for flag, start, _end in token_regions(record["segment_type_ids"])
                if not flag and start >= 2
            ]
            for cut in cuts:
                if not chat_prompt_seam_ok(
                    tokenizer, decode(record["input_ids"][:cut]), prefix_ids, suffix_ids
                ):
                    raise SystemExit(
                        f"chat wrapper changes the document's tokenization at cut "
                        f"{cut} of {record['question_id']}; the fixed prefix/suffix "
                        "split is unsound for this tokenizer"
                    )
                checked += 1
        print(f"chat feature layout: {len(prefix_ids)}-token prefix, "
              f"{len(suffix_ids)}-token suffix, seam verified at {checked} cuts")
        feature_builder = functools.partial(
            chat_layout_features, prefix_ids=prefix_ids, suffix_ids=suffix_ids,
            decode=decode, feature_layer=args.feature_layer,
        )

    policy = (
        GeneratedTokensOnlyPolicy() if args.loss_targets == "generated"
        else LossTargetPolicy()
    )
    train_draft_model(
        draft_model, target_model, train_records, pad_id, args.device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        log_every=args.log_every, hop_signal_enabled=hop_signal_enabled,
        alpha=alpha, tau=tau, loss_target_policy=policy,
        feature_layer=args.feature_layer, v_w=args.v_w, p_w=args.p_w,
        feature_builder=feature_builder,
    )
    # Evaluation takes no loss-target policy — it must be identical across arms.
    columns = evaluate_draft_model(
        draft_model, target_model, heldout_records, pad_id, args.device,
        batch_size=args.batch_size, hop_signal_enabled=hop_signal_enabled,
        feature_layer=args.feature_layer, feature_builder=feature_builder,
    )

    if args.raw_out:
        with open(args.raw_out, "w", encoding="utf-8") as f:
            json.dump(columns, f)
        print(f"raw per-position columns -> {args.raw_out}")

    flags = novel_hop_flags(heldout_records)
    views = {
        "all_positions": columns,
        "novel_hops_only": select(columns, exclude_no_prior=True, novel_flags=flags),
        "matched_cohort": select(
            columns, exclude_no_prior=True, min_hop_span=args.min_hop_span
        ),
    }
    summary = {"hop_signal_enabled": hop_signal_enabled, "alpha": alpha, "tau": tau,
               "feature_mode": feature_mode, "views": {}}
    for view_name, view_columns in views.items():
        table = summarize(view_columns)
        print()
        print(format_table(table, title=view_name))
        summary["views"][view_name] = [vars(row) for row in table]

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"summary -> {args.out}")
    if checkpoint_out:
        torch.save(
            # feature_mode travels WITH the weights: a checkpoint trained on
            # one layout and served the other is a silent train/serve mismatch.
            {"state_dict": draft_model.state_dict(), "config": vars(config),
             "feature_mode": feature_mode},
            checkpoint_out,
        )
        print(f"checkpoint -> {checkpoint_out}")
    return summary
