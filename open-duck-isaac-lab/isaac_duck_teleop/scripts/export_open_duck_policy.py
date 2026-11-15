from __future__ import annotations

import argparse
from pathlib import Path

from duck_policy_runtime import export_torchscript


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Open Duck RSL-RL actor to TorchScript."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--obs-dim", type=int, default=101)
    parser.add_argument("--action-dim", type=int, default=14)
    args = parser.parse_args()

    export_torchscript(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
    )
    print(f"exported TorchScript policy to {args.output}")


if __name__ == "__main__":
    main()
