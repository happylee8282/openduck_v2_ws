#!/usr/bin/env python3
"""Fine-tune the validated full-mass v2 actor with reproducible straight PPO defaults."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Actor initialization; fresh critic/optimizer, 40 critic warm-up updates",
    )
    p.add_argument(
        "--randomized",
        action="store_true",
        help="Enable configured mass/friction/torque/sensor/delay randomization",
    )
    args, rest = p.parse_known_args()
    if args.checkpoint is None:
        manifest = ROOT / "config/validated_flat_policy.json"
        if manifest.exists():
            args.checkpoint = ROOT / json.loads(manifest.read_text())["checkpoint"]
        else:
            args.checkpoint = (
                ROOT / "validation/straight_walk_2026-11-10/official_teacher.pt"
            )
    if not args.checkpoint.is_file():
        p.error(
            "Initialization missing. Pass --checkpoint /path/model.pt or import the official ONNX with tools/import_official_policy.py."
        )
    from open_duck_v2.runtime import load_rsl_rl_actor
    from torch import nn

    actor = load_rsl_rl_actor(args.checkpoint)
    hidden = [
        layer.out_features for layer in actor.net if isinstance(layer, nn.Linear)
    ][:-1]
    defaults = [
        "train",
        "--task",
        "Isaac-Open-Duck-StraightV6-Direct-v0",
        "--num_envs",
        "1024",
        "--max_iterations",
        "600",
        "--seed",
        "42",
        "--fixed-noise-std",
        "0.05",
        "--warm-start-checkpoint",
        str(args.checkpoint.resolve()),
        "agent.experiment_name=open_duck_v2_straight",
        "agent.num_steps_per_env=32",
        "agent.save_interval=100",
        "agent.empirical_normalization=True",
        "agent.policy.actor_hidden_dims=[" + ",".join(map(str, hidden)) + "]",
        "agent.algorithm.entropy_coef=0.0",
        "agent.algorithm.learning_rate=0.0001",
        "agent.algorithm.schedule=fixed",
        "agent.algorithm.gamma=0.99",
        "env.heading_hold_kp=" + str(actor.control_profile.get("heading_hold_kp", 2.0)),
        "env.heading_hold_max_rate="
        + str(actor.control_profile.get("heading_hold_max_rate", 0.35)),
        "env.heading_hold_filter_tau="
        + str(actor.control_profile.get("heading_hold_filter_tau", 0.25)),
        "env.forward_command_scale="
        + str(actor.control_profile.get("forward_command_scale", 1.0)),
        "env.lateral_command_trim="
        + str(actor.control_profile.get("lateral_command_trim", 0.0)),
    ]
    if not args.randomized:
        defaults += [
            "env.randomize_domain=False",
            "env.observation_noise=False",
            "env.action_delay_min=0",
            "env.action_delay_max=0",
            "env.imu_delay_min=0",
            "env.imu_delay_max=0",
        ]
    sys.argv = [str(ROOT / "tools/launch.py"), *defaults, *rest]
    from launch import main as launch

    launch()


if __name__ == "__main__":
    main()
