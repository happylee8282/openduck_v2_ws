#!/usr/bin/env python3
"""Train an isolated reverse or posture skill without changing the forward model."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("skill", choices=["backward", "posture"])
    p.add_argument("--checkpoint", type=Path)
    args, rest = p.parse_known_args()
    reverse = args.skill == "backward"
    checkpoint = args.checkpoint or (
        ROOT / "policies/flat_straight_v2.pt" if reverse else None
    )
    options = [
        "train",
        "--task",
        "Isaac-Open-Duck-Backward-Direct-v0"
        if reverse
        else "Isaac-Open-Duck-Posture-Direct-v0",
        "--num_envs",
        "512" if reverse else "256",
        "--max_iterations",
        "600" if reverse else "300",
        "--seed",
        "60" if reverse else "70",
        "--fixed-noise-std",
        "0.12" if reverse else "0.04",
        "agent.experiment_name=open_duck_v2_" + args.skill,
        "agent.num_steps_per_env=32",
        "agent.save_interval=" + ("100" if reverse else "50"),
        "agent.empirical_normalization=True",
        "agent.policy.init_noise_std=" + ("0.12" if reverse else "0.04"),
        "agent.algorithm.entropy_coef=0.0",
        "agent.algorithm.learning_rate=" + ("0.0001" if reverse else "0.00005"),
        "agent.algorithm.schedule=fixed",
        "env.randomize_domain=False",
        "env.observation_noise=False",
        "env.action_delay_min=0",
        "env.action_delay_max=0",
        "env.imu_delay_min=0",
        "env.imu_delay_max=0",
    ]
    if checkpoint:
        if not checkpoint.is_file():
            p.error("Initialization checkpoint missing")
        from torch import nn

        if reverse:
            from open_duck_v2.runtime import load_rsl_rl_actor

            actor = load_rsl_rl_actor(checkpoint)
        else:
            from open_duck_v2.posture_runtime import load_posture_actor

            actor = load_posture_actor(
                checkpoint, ROOT / "config/posture_reference_v1.json"
            )
        dims = [
            layer.out_features for layer in actor.net if isinstance(layer, nn.Linear)
        ][:-1]
        options += [
            "--warm-start-checkpoint",
            str(checkpoint.resolve()),
            "agent.policy.actor_hidden_dims=[" + ",".join(map(str, dims)) + "]",
        ]
    sys.argv = [str(ROOT / "tools/launch.py"), *options, *rest]
    from launch import main as launch

    launch()


if __name__ == "__main__":
    main()
