#!/usr/bin/env python3
"""Export real TensorBoard scalars and a compact training figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--straight", action="store_true")
    p.add_argument(
        "--max-iteration",
        type=int,
        help="Only include data available at the selected checkpoint",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    acc = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0}).Reload()
    tags = acc.Tags()["scalars"]
    values = {}
    with (args.output / "training_scalars.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "iteration", "wall_time", "value"])
        for tag in tags:
            rows = [
                r
                for r in acc.Scalars(tag)
                if args.max_iteration is None or r.step <= args.max_iteration
            ]
            for row in rows:
                writer.writerow([tag, row.step, row.wall_time, row.value])
            values[tag] = rows
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    configs = [
        ([("Train/mean_reward", "reward", 1)], "Episode return", "reward"),
        (
            [("Train/mean_episode_length", "duration", 0.02)],
            "Mean episode duration",
            "seconds",
        ),
        (
            [
                ("Stats/commanded_speed", "command", 1),
                ("Stats/velocity_along_command", "velocity along command", 1),
            ],
            "Translation tracking",
            "m/s",
        ),
        (
            [("Reward/lin_vel_track", "tracking reward", 1)],
            "Velocity tracking reward",
            "weighted reward / step",
        ),
        (
            [("Penalty/no_motion", "no-motion penalty", 1)],
            "No-motion penalty",
            "penalty / step",
        ),
        (
            [("Policy/mean_noise_std", "action noise", 1)],
            "Exploration",
            "standard deviation",
        ),
    ]
    if args.straight:
        configs[3:] = [
            (
                [("Stats/heading_error_abs_rad", "mean abs heading", 180 / np.pi)],
                "Heading error during training",
                "degrees",
            ),
            (
                [("Stats/lateral_path_error_abs_m", "mean abs path error", 1)],
                "Lateral path error during training",
                "m",
            ),
            (
                [("Stats/head_velocity_rms", "head joint RMS", 1)],
                "Head motion (full mass)",
                "rad/s",
            ),
        ]
    for ax, (lines, title, unit) in zip(axs.flat, configs):
        for tag, label, scale in lines:
            rows = values.get(tag, [])
            if rows:
                x = np.array([r.step for r in rows])
                y = np.array([r.value * scale for r in rows])
                ax.plot(x, y, alpha=0.25)
                n = min(10, len(y))
                ax.plot(
                    x[n - 1 :],
                    np.convolve(y, np.ones(n) / n, mode="valid"),
                    label=label,
                )
        ax.set(title=title, xlabel="PPO iteration", ylabel=unit)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Open Duck Mini v2 - flat-ground training (actual TensorBoard data)",
        fontsize=15,
    )
    fig.savefig(args.output / "learning_curves.png", dpi=160)
    fig.savefig(args.output / "learning_curves.pdf")
    plt.close(fig)
    summary = {}
    for tag in [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Stats/commanded_speed",
        "Stats/velocity_along_command",
        "Reward/lin_vel_track",
        "Penalty/no_motion",
        "Stats/heading_error_abs_rad",
        "Stats/lateral_path_error_abs_m",
        "Stats/head_velocity_rms",
    ]:
        rows = values.get(tag, [])
        if rows:
            n = min(10, len(rows))
            summary[tag] = dict(
                first_window_mean=float(np.mean([r.value for r in rows[:n]])),
                last_window_mean=float(np.mean([r.value for r in rows[-n:]])),
                last_iteration=rows[-1].step,
                window=n,
            )
    (args.output / "training_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
