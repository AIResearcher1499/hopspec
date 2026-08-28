"""Train the HOP-SIGNAL arm (signal on, alpha=2.0), measure, save a checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment_common import build_shared_parser, run_experiment  # noqa: E402


def main() -> int:
    parser = build_shared_parser(__doc__)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=16.0)
    args = parser.parse_args()
    run_experiment(
        args, hop_signal_enabled=True, alpha=args.alpha, tau=args.tau,
        checkpoint_out=args.checkpoint_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
