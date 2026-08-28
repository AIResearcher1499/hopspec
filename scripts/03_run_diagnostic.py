"""Train the BASELINE arm (hop signal off, alpha=0) and measure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment_common import build_shared_parser, run_experiment  # noqa: E402


def main() -> int:
    parser = build_shared_parser(__doc__)
    parser.add_argument("--tau", type=float, default=16.0)
    args = parser.parse_args()
    run_experiment(args, hop_signal_enabled=False, alpha=0.0, tau=args.tau,
                   checkpoint_out=args.checkpoint_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
