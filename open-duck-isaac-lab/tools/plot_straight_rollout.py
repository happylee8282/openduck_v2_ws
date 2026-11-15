#!/usr/bin/env python3
"""Plot raw and one-cycle motion, accumulated drift, and head motion."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--cycle-period", type=float, default=0.54)
    args = p.parse_args()
    a = np.load(args.rollout)
    v = a["values"][:, args.trial]
    fields = a["columns"].tolist()
    t = (np.arange(len(v)) + 1) * float(a["dt"])
    target = (
        a["command_trace"][:, args.trial]
        if "command_trace" in a
        else np.tile(a["commands"][args.trial], (len(v), 1))
    )
    n = round(args.cycle_period / float(a["dt"]))

    def col(name):
        return v[:, fields.index(name)]

    fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for i, name in enumerate(("vx", "vy", "yaw_rate")):
        ax = axs[0, i]
        ax.plot(t, col(name), alpha=0.22, label="raw physics")
        if len(v) >= n:
            ax.plot(
                t[n - 1 :],
                np.convolve(col(name), np.ones(n) / n, mode="valid"),
                label="0.54s cycle mean",
            )
        ax.plot(t, target[:, i], "--", label="command")
        ax.set(title=name, xlabel="time / s", ylabel="rad/s" if i == 2 else "m/s")
    qw, qx, qy, qz = [col(k) for k in ("qw", "qx", "qy", "qz")]
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    yaw = np.rad2deg(yaw - yaw[0])
    axs[0, 3].plot(t, yaw, label="heading change")
    axs[0, 3].axhline(0, color="k", linestyle="--")
    axs[0, 3].set(title="Accumulated heading", xlabel="time / s", ylabel="degrees")
    x, y = col("x") - col("x")[0], col("y") - col("y")[0]
    ax = axs[1, 0]
    ax.plot(x, y, label="world trajectory")
    ax.plot(
        [0, (target[:, 0] * float(a["dt"])).sum()],
        [0, 0],
        "--",
        label="commanded straight path",
    )
    ax.plot(x[0], y[0], "go", label="start")
    ax.plot(x[-1], y[-1], "ro", label="end")
    ax.set(title="World displacement", xlabel="x / m", ylabel="y / m")
    ax.axis("equal")
    ax = axs[1, 1]
    for j, side in enumerate(("left", "right")):
        ax.step(t, col("stable_contact_" + side) + 1.4 * j, label=side, where="post")
    ax.set(title="Foot contact (60ms debounce)", xlabel="time / s", ylabel="contact")
    ax = axs[1, 2]
    for side in ("left", "right"):
        ax.plot(t, col("foot_clearance_" + side), label=side)
    ax.set(title="Foot link origin height", xlabel="time / s", ylabel="m")
    ax = axs[1, 3]
    for j, name in enumerate(("neck pitch", "head pitch", "head yaw", "head roll")):
        key = "head_velocity_" + str(j)
        if key in fields:
            ax.plot(t, col(key), label=name, alpha=0.7)
    ax.set(title="Full-mass head joint motion", xlabel="time / s", ylabel="rad/s")
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Actual Isaac Sim playback: " + str(a["labels"][args.trial]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
