#!/usr/bin/env python3
"""Assess straight walking, standing and stop/go transitions from recorded physics.

A 0.54 s boxcar measures net command tracking across natural biped sway.
Raw velocities, world displacement and accumulated heading remain in the report.
Targets are declared engineering acceptance criteria, not hardware certification.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def cycle_average(values, window):
    if len(values) < window:
        return np.empty((0,) + values.shape[1:])
    total = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)])
    return (total[window:] - total[:-window]) / window


def assess(evaluation_path, period=0.54):
    evaluation_path = Path(evaluation_path)
    report = json.loads(evaluation_path.read_text())
    with np.load(evaluation_path.parent / "rollout.npz") as data:
        values, fields = data["values"], data["columns"].tolist()
        dt = float(data["dt"])
        trace = (
            data["command_trace"]
            if "command_trace" in data
            else np.broadcast_to(
                data["commands"][None], (len(values),) + data["commands"].shape
            )
        )
        records = []
        for i, original in enumerate(report["trials"]):
            row = dict(original)
            a = values[:, i]
            live = a[:, fields.index("alive")] > 0.5
            t = (np.arange(len(a)) + 1) * dt
            pos = a[:, [fields.index("x"), fields.index("y")]]
            q = a[:, [fields.index(k) for k in ("qw", "qx", "qy", "qz")]]
            qw, qx, qy, qz = q.T
            heading = np.unwrap(
                np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            )
            heading -= heading[0]
            motion = a[:, [fields.index(k) for k in ("vx", "vy", "yaw_rate")]]
            target = trace[:, i]
            n = max(1, round(period / dt))
            valid_end = np.flatnonzero(live)
            if not len(valid_end):
                row.update(checks={"survived_full_trial": False}, accepted=False)
                records.append(row)
                continue
            end = valid_end[-1] + 1
            smooth = cycle_average(motion[:end], n)
            smooth_target = cycle_average(target[:end], n)
            smooth_t = t[n - 1 : end]
            steady = smooth_t >= 2.0
            error = smooth[steady] - smooth_target[steady]
            d = pos[end - 1] - pos[0]
            lift = []
            for side in ("left", "right"):
                h = a[:, fields.index("foot_clearance_" + side)]
                contact = a[:, fields.index("contact_" + side)] > 0.5
                stable = a[:, fields.index("stable_contact_" + side)] > 0.5
                stance, swing = h[live & contact], h[live & ~stable]
                lift.append(
                    float(max(0, swing.max() - np.median(stance)))
                    if len(stance) and len(swing)
                    else 0.0
                )
            metrics = dict(
                cycle_period_s=period,
                cycle_xy_rmse_m_s=float(
                    np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))
                )
                if len(error)
                else None,
                cycle_yaw_mae_rad_s=float(np.mean(np.abs(error[:, 2])))
                if len(error)
                else None,
                final_heading_rad=float(heading[end - 1]),
                max_abs_heading_rad=float(np.abs(heading[:end]).max()),
                world_displacement_xy_m=d.tolist(),
                peak_swing_lift_above_stance_m=lift,
            )
            checks = {"survived_full_trial": original["survived"]}
            command = original["command"]
            if command in ("forward", "backward"):
                direction = 1 if command == "forward" else -1
                mean = motion[live & (t >= 2), :].mean(0)
                metrics["steady_mean_velocity"] = mean.tolist()
                checks.update(
                    speed_along_command_0_08_to_0_12_m_s=bool(
                        0.08 <= direction * mean[0] <= 0.12
                    ),
                    cycle_xy_rmse_at_most_0_04_m_s=bool(
                        len(error) and metrics["cycle_xy_rmse_m_s"] <= 0.04
                    ),
                    cycle_yaw_mae_at_most_0_06_rad_s=bool(
                        len(error) and metrics["cycle_yaw_mae_rad_s"] <= 0.06
                    ),
                    distance_80_to_120_percent_of_target=bool(
                        0.08 * report["duration_s"]
                        <= direction * d[0]
                        <= 0.12 * report["duration_s"]
                    ),
                    net_lateral_at_most_0_15_m=bool(abs(d[1]) <= 0.15),
                    final_heading_at_most_0_12_rad=bool(abs(heading[end - 1]) <= 0.12),
                    max_heading_at_most_0_15_rad=bool(
                        np.abs(heading[:end]).max() <= 0.15
                    ),
                    both_feet_at_least_3_stable_landings=bool(
                        min(original["stable_foot_landings"]) >= 3
                    ),
                    both_feet_lift_at_least_10_mm=bool(min(lift) >= 0.01),
                )
            elif command == "stand":
                steady_mask = live & (t >= 2)
                speed = np.linalg.norm(motion[steady_mask, :2], axis=1)
                metrics["steady_planar_speed_mean_m_s"] = (
                    float(speed.mean()) if len(speed) else None
                )
                checks.update(
                    steady_planar_speed_at_most_0_015_m_s=bool(
                        len(speed) and speed.mean() <= 0.015
                    ),
                    net_drift_at_most_0_05_m=bool(np.linalg.norm(d) <= 0.05),
                    max_heading_at_most_0_1_rad=bool(
                        np.abs(heading[:end]).max() <= 0.1
                    ),
                )
            elif command == "stop_go_stop":
                phases = {}
                for name, lo, hi in (
                    ("initial_stop", 1, 3),
                    ("forward", 5, 15),
                    ("final_stop", 17, report["duration_s"] + dt / 2),
                ):
                    mask = live & (t >= lo) & (t < hi)
                    idx = np.flatnonzero(mask)
                    smooth_mask = (smooth_t >= lo + period) & (smooth_t < hi)
                    e = smooth[smooth_mask] - smooth_target[smooth_mask]
                    phases[name] = dict(
                        samples=len(idx),
                        mean_velocity=motion[mask].mean(0).tolist()
                        if len(idx)
                        else None,
                        planar_speed_mean_m_s=float(
                            np.linalg.norm(motion[mask, :2], axis=1).mean()
                        )
                        if len(idx)
                        else None,
                        displacement_xy_m=(pos[idx[-1]] - pos[idx[0]]).tolist()
                        if len(idx)
                        else None,
                        cycle_xy_rmse_m_s=float(
                            np.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1)))
                        )
                        if len(e)
                        else None,
                        cycle_yaw_mae_rad_s=float(np.abs(e[:, 2]).mean())
                        if len(e)
                        else None,
                    )
                metrics["phases"] = phases
                for name in ("initial_stop", "final_stop"):
                    phase = phases[name]
                    checks[name + "_planar_speed_at_most_0_015_m_s"] = bool(
                        phase["samples"] and phase["planar_speed_mean_m_s"] <= 0.015
                    )
                    checks[name + "_net_drift_at_most_0_03_m"] = bool(
                        phase["samples"]
                        and np.linalg.norm(phase["displacement_xy_m"]) <= 0.03
                    )
                forward = phases["forward"]
                checks["forward_mean_0_08_to_0_12_m_s"] = bool(
                    forward["samples"] and 0.08 <= forward["mean_velocity"][0] <= 0.12
                )
                checks["forward_cycle_xy_rmse_at_most_0_04_m_s"] = bool(
                    forward["cycle_xy_rmse_m_s"] is not None
                    and forward["cycle_xy_rmse_m_s"] <= 0.04
                )
                checks["max_heading_at_most_0_15_rad"] = bool(
                    np.abs(heading[:end]).max() <= 0.15
                )
                checks["net_lateral_at_most_0_15_m"] = bool(abs(d[1]) <= 0.15)
            else:
                checks["supported_command"] = False
            row.update(metrics=metrics, checks=checks, accepted=all(checks.values()))
            records.append(row)
    summary = {}
    for command in dict.fromkeys(row["command"] for row in records):
        group = [row for row in records if row["command"] == command]
        summary[command] = dict(
            trials=len(group),
            survived=sum(row["survived"] for row in group),
            accepted=sum(row["accepted"] for row in group),
            acceptance_rate=sum(row["accepted"] for row in group) / len(group),
            mean_world_displacement_xy_m=np.mean(
                [
                    row["metrics"]["world_displacement_xy_m"]
                    for row in group
                    if "metrics" in row
                ],
                axis=0,
            ).tolist(),
            mean_final_heading_rad=float(
                np.mean(
                    [
                        row["metrics"]["final_heading_rad"]
                        for row in group
                        if "metrics" in row
                    ]
                )
            ),
            mean_cycle_xy_rmse_m_s=float(
                np.mean(
                    [
                        row["metrics"]["cycle_xy_rmse_m_s"]
                        for row in group
                        if row.get("metrics", {}).get("cycle_xy_rmse_m_s") is not None
                    ]
                )
            ),
            mean_cycle_yaw_mae_rad_s=float(
                np.mean(
                    [
                        row["metrics"]["cycle_yaw_mae_rad_s"]
                        for row in group
                        if row.get("metrics", {}).get("cycle_yaw_mae_rad_s") is not None
                    ]
                )
            ),
        )
    return dict(
        checkpoint=report["checkpoint"],
        source_evaluation=str(evaluation_path.resolve()),
        criteria_note="Declared engineering targets for this experiment. Cycle-averaged errors exclude the first 2 s; raw errors remain in each trial. World displacement and heading independently catch slow drift. Each trial must meet every listed check; each command needs >=90% accepted trials. No post-fall samples or respawns count.",
        head_control_mode=report.get("head_control_mode"),
        controller={
            key: report.get(key)
            for key in (
                "heading_hold_kp",
                "heading_hold_max_rate",
                "heading_hold_filter_tau",
                "forward_command_scale",
                "lateral_command_trim",
                "heading_sensor",
            )
        },
        randomized=report["randomized"],
        seed=report["seed"],
        accepted=bool(
            summary and all(row["acceptance_rate"] >= 0.9 for row in summary.values())
        ),
        summary=summary,
        trials=records,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cycle-period", type=float, default=0.54)
    args = p.parse_args()
    result = assess(args.evaluation, args.cycle_period)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
