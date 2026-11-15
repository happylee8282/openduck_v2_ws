#!/usr/bin/env python3
"""Import the external Gym tasks, then run the selected Isaac Lab launcher.

Run through isaaclab.sh -p. No mutation of the Isaac Lab checkout is needed.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=["train", "play", "list", "random"])
    parser.add_argument(
        "--isaaclab-root", type=Path, default=os.environ.get("ISAACLAB_PATH")
    )
    parser.add_argument("--fixed-noise-std", type=float)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    args, rest = parser.parse_known_args()
    if args.isaaclab_root is None:
        import importlib.util

        spec = importlib.util.find_spec("isaaclab")
        if spec is None:
            raise RuntimeError("Use isaaclab.sh -p or set --isaaclab-root")
        args.isaaclab_root = Path(spec.origin).resolve().parents[3]
    paths = {
        "train": "reinforcement_learning/rsl_rl/train.py",
        "play": "reinforcement_learning/rsl_rl/play.py",
        "list": "environments/list_envs.py",
        "random": "environments/random_agent.py",
    }
    script = args.isaaclab_root / "scripts" / paths[args.mode]
    if not script.exists():
        raise FileNotFoundError(script)
    sys.path.insert(0, str(ROOT))
    # Registration only imports gymnasium; simulator classes load after AppLauncher.
    import open_duck_complex_terrain  # noqa: F401
    import open_duck_imitation  # noqa: F401

    if args.mode == "train":
        from open_duck_v2.runtime import install_checkpoint_contract_hook

        install_checkpoint_contract_hook()
        if args.fixed_noise_std is not None:
            from open_duck_v2.training_policy import install_straight_policy_hook

            install_straight_policy_hook(
                args.fixed_noise_std, args.warm_start_checkpoint
            )
    sys.path.insert(0, str(script.parent))
    if args.mode in ("train", "play", "random") and "--task" not in rest:
        rest = ["--task", "Isaac-Open-Duck-Direct-v0", *rest]
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
