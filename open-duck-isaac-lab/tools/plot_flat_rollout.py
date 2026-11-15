#!/usr/bin/env python3
"""Plot actual fixed-command telemetry, including terminal frames."""

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
    args = p.parse_args()
    a = np.load(args.rollout)
    values = a["values"][:, args.trial]
    fields = a["columns"].tolist()
    target = a["commands"][args.trial]
    command_trace = (
        a["command_trace"][:, args.trial]
        if "command_trace" in a
        else np.tile(target, (len(values), 1))
    )
    t = (np.arange(len(values)) + 1) * float(a["dt"])

    def col(name):
        return values[:, fields.index(name)]

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axs[0, 0]
    for i, name in enumerate(["vx", "vy"]):
        ax.plot(t, col(name), label=name, alpha=0.75)
        ax.plot(t, command_trace[:, i], linestyle="--", label="target " + name)
    ax.set(title="Body-frame translation velocity", xlabel="seconds", ylabel="m/s")
    ax = axs[0, 1]
    ax.plot(t, col("yaw_rate"), label="measured")
    ax.plot(t, command_trace[:, 2], linestyle="--", label="target")
    ax.set(title="Yaw tracking", xlabel="seconds", ylabel="rad/s")
    ax = axs[0, 2]
    ax.plot(t, np.rad2deg(col("tilt_rad")), label="trunk tilt")
    ax.axhline(np.rad2deg(0.75), linestyle="--", color="red", label="fall threshold")
    ax.set(title="Balance", xlabel="seconds", ylabel="degrees")
    ax = axs[1, 0]
    for name in ["contact_left", "contact_right"]:
        key = "stable_" + name if "stable_" + name in fields else name
        ax.step(t, col(key) + (1.4 if "left" in name else 0), where="post", label=key)
    ax.set(title="Foot contact (60ms debounce)", xlabel="seconds", ylabel="contact")
    ax = axs[1, 1]
    for name in ["foot_clearance_left", "foot_clearance_right"]:
        if name in fields:
            ax.plot(t, col(name), label=name)
    ax.set(title="Foot link origin height above ground", xlabel="seconds", ylabel="m")
    ax = axs[1, 2]
    x, y = col("x") - col("x")[0], col("y") - col("y")[0]
    ax.plot(x, y, label="root trajectory")
    ax.plot(x[0], y[0], "go", label="start")
    ax.plot(x[-1], y[-1], "ro", label="end")
    ax.axis("equal")
    ax.set(title="World-frame displacement", xlabel="x / m", ylabel="y / m")
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(f"Actual Isaac Sim policy playback - {a['labels'][args.trial]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
