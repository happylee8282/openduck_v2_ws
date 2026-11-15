#!/usr/bin/env python3
"""Plot measured multiskill validation and selected-checkpoint TensorBoard data."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation", type=Path, default=ROOT / "validation/multiskill_2026-11-10"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    args.output.mkdir(parents=True, exist_ok=True)
    assessments = [
        json.loads((args.validation / f"smooth_seed{s}/assessment.json").read_text())
        for s in (62, 63)
    ]
    trials = [r for a in assessments for r in a["trials"]]
    summary = {}
    for case in ("stand", "forward", "backward", "posture", "demo"):
        group = [r for r in trials if r["command"] == case]
        summary[case] = dict(
            trials=len(group),
            accepted=sum(r["accepted"] for r in group),
            survived=sum(r["survived"] for r in group),
            duration_s=group[0]["duration_s"],
        )
        for key in (
            "steady_mean_velocity",
            "world_displacement_xy_m",
            "max_abs_heading_rad",
            "height_rmse_m",
            "max_foot_drift_m",
            "final_planar_speed_m_s",
            "max_trunk_tilt_rad",
            "mean_torque_saturation",
            "standing_return_error_m",
        ):
            if key in group[0]["metrics"]:
                x = np.array([r["metrics"][key] for r in group])
                summary[case][key] = dict(
                    mean=np.mean(x, axis=0).tolist(),
                    minimum=np.min(x, axis=0).tolist(),
                    maximum=np.max(x, axis=0).tolist(),
                )
        if case == "posture":
            cycles = [c for r in group for c in r["metrics"]["cycles"]]
            summary[case]["cycles"] = len(cycles)
            for key in (
                "low_height_m",
                "standing_height_m",
                "returned_height_m",
                "height_reduction_m",
                "return_height_error_m",
                "both_contact_fraction",
            ):
                x = np.array([c[key] for c in cycles])
                summary[case][key] = dict(
                    mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max())
                )
    (args.output / "metrics_summary.json").write_text(
        json.dumps(
            dict(
                accepted=all(r["accepted"] for r in trials),
                seeds=[62, 63],
                summary=summary,
            ),
            indent=2,
        )
        + "\n"
    )
    d = np.load(args.validation / "smooth_seed63/rollout.npz")
    v = d["values"]
    labels = d["labels"]
    dt = float(d["dt"])
    t = (np.arange(len(v)) + 1) * dt
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    def meanband(ax, x, y, color, label):
        ax.plot(x, y.mean(1), color=color, label=label)
        ax.fill_between(x, y.min(1), y.max(1), alpha=0.16, color=color)

    n = 27
    for case, color, sign in [("forward", "#278156", 1), ("backward", "#3158ac", -1)]:
        ids = labels == case
        a = v[:, ids]
        mask = t <= 20 + 1e-6
        speed = np.stack(
            [
                np.convolve(a[mask, i, 0], np.ones(n) / n, mode="valid")
                for i in range(a.shape[1])
            ],
            axis=1,
        )
        meanband(axs[0, 0], t[mask][n - 1 :], speed, color, case)
        axs[0, 0].axhline(0.1 * sign, color=color, ls="--", alpha=0.5)
        delta = a[mask, :, 3:5] - a[0, :, 3:5]
        for j in range(a.shape[1]):
            axs[0, 1].plot(
                delta[:, j, 0],
                delta[:, j, 1],
                color=color,
                alpha=0.4,
                label=case if j == 0 else None,
            )
        q = a[mask, :, 31:35]
        w, x, y, z = np.moveaxis(q, -1, 0)
        yaw = np.unwrap(
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), axis=0
        )
        yaw -= yaw[0]
        meanband(axs[0, 2], t[mask], yaw * 180 / np.pi, color, case)
    axs[0, 0].set(
        title="Walking speed: 0.54s cycle average",
        ylabel="body vx (m/s)",
        xlabel="seconds",
    )
    axs[0, 1].set(
        title="Actual ground displacement", xlabel="world X (m)", ylabel="world Y (m)"
    )
    axs[0, 2].set(title="Walking heading deviation", ylabel="degrees", xlabel="seconds")
    axs[0, 2].axhline(0.15 * 180 / np.pi, color="red", ls=":", label="acceptance limit")
    axs[0, 2].axhline(-0.15 * 180 / np.pi, color="red", ls=":")
    ids = labels == "posture"
    a = v[:, ids]
    mask = t <= 48 + 1e-6
    meanband(axs[1, 0], t[mask], a[mask, :, 6] * 1000, "#8b4ca4", "measured height")
    axs[1, 0].plot(
        t[mask], d["height_target"][mask][:, ids][:, 0] * 1000, "k--", label="target"
    )
    axs[1, 0].set(
        title="Both-feet squat and stand: three cycles",
        ylabel="base height (mm)",
        xlabel="seconds",
    )
    feet = a[:, :, 42:48].reshape(len(a), -1, 2, 3)
    base = feet[np.flatnonzero(t >= 2)[0], :, :, :2]
    drift = np.linalg.norm(feet[:, :, :, :2] - base[None], axis=-1).max(-1) * 1000
    meanband(axs[1, 1], t[mask], drift[mask], "#b77b22", "max foot drift")
    axs[1, 1].axvspan(0, 2, color="grey", alpha=0.15)
    axs[1, 1].axhline(25, color="red", ls=":", label="25mm criterion")
    axs[1, 1].set(
        title="Foot drift (first 2s excluded from criterion)",
        ylabel="horizontal drift (mm)",
        xlabel="seconds",
    )
    ids = labels == "demo"
    a = v[:, ids]
    speed = np.stack(
        [
            np.convolve(a[:, i, 0], np.ones(n) / n, mode="valid")
            for i in range(a.shape[1])
        ],
        axis=1,
    )
    meanband(axs[1, 2], t[n - 1 :], speed, "#3158ac", "vx")
    axs[1, 2].plot(
        t, d["command_trace"][:, ids, 0].mean(1), "k--", label="requested vx"
    )
    twin = axs[1, 2].twinx()
    twin.plot(t, a[:, :, 6].mean(1) * 1000, color="#8b4ca4", alpha=0.75)
    twin.set_ylabel("base height (mm)", color="#8b4ca4")
    axs[1, 2].set(
        title="52s sequence: F / stop / B / low / stand / F",
        ylabel="body vx (m/s)",
        xlabel="seconds",
    )
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Open Duck Mini v2: actual Isaac Lab data (seed 63, eight robots per case)",
        fontsize=15,
    )
    fig.savefig(args.output / "validation_metrics.png", dpi=160)
    fig.savefig(args.output / "validation_metrics.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    training = {}
    with (args.output / "selected_training_scalars.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["skill", "tag", "iteration", "wall_time", "value"])
        for row, (skill, run, selected, extra) in enumerate(
            [
                (
                    "backward",
                    "2026-11-10_04-25-43_backward_v1",
                    100,
                    "Stats/velocity_along_command",
                ),
                (
                    "posture",
                    "2026-11-10_04-33-25_posture_residual_v1",
                    50,
                    "Stats/height_error_abs_m",
                ),
            ]
        ):
            acc = EventAccumulator(
                str(ROOT / "logs/rsl_rl" / ("open_duck_v2_" + skill) / run),
                size_guidance={"scalars": 0},
            ).Reload()
            metrics = {}
            for tag in acc.Tags()["scalars"]:
                values = [r for r in acc.Scalars(tag) if r.step <= selected]
                for value in values:
                    writer.writerow(
                        [skill, tag, value.step, value.wall_time, value.value]
                    )
                if values:
                    metrics[tag] = dict(
                        first_10_mean=float(np.mean([r.value for r in values[:10]])),
                        last_10_mean=float(np.mean([r.value for r in values[-10:]])),
                        last_step=values[-1].step,
                    )
            training[skill] = dict(selected_iteration=selected, metrics=metrics)
            for col, tag in enumerate(
                ["Train/mean_reward", "Train/mean_episode_length", extra]
            ):
                vals = [r for r in acc.Scalars(tag) if r.step <= selected]
                x = np.array([r.step for r in vals])
                y = np.array([r.value for r in vals])
                y = y * 0.02 if col == 1 else y
                axs[row, col].plot(x, y, alpha=0.35)
                width = min(10, len(y))
                axs[row, col].plot(
                    x[width - 1 :], np.convolve(y, np.ones(width) / width, mode="valid")
                )
                axs[row, col].set(
                    title=skill + ": " + tag,
                    xlabel="PPO iteration",
                    ylabel="seconds"
                    if col == 1
                    else "m/s"
                    if tag.endswith("command")
                    else "m"
                    if tag.endswith("_m")
                    else "return",
                )
                axs[row, col].grid(alpha=0.2)
    fig.suptitle(
        "Actual TensorBoard learning curves, truncated at the selected checkpoints"
    )
    fig.savefig(args.output / "learning_curves.png", dpi=160)
    fig.savefig(args.output / "learning_curves.pdf")
    plt.close(fig)
    (args.output / "selected_training_metrics.json").write_text(
        json.dumps(training, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
