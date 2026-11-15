#!/usr/bin/env python3
"""Declared engineering acceptance for preserved walking, reverse and squat/stand."""

import argparse
import json
from pathlib import Path
import numpy as np
from assess_straight_walk import cycle_average


def assess(path):
    path = Path(path)
    evaluation = json.loads(path.read_text())
    a = np.load(path.parent / "rollout.npz")
    values = a["values"]
    fields = a["columns"].tolist()
    trace = a["command_trace"]
    height = a["height_target"]
    dt = float(a["dt"])
    times = (np.arange(len(values)) + 1) * dt
    records = []
    for i, original in enumerate(evaluation["trials"]):
        case = original["command"]
        duration = original["duration_s"]
        valid = (values[:, i, fields.index("alive")] > 0.5) & (times <= duration + 1e-5)
        data = values[valid, i]
        t = times[valid]
        target = trace[valid, i]
        goal = height[valid, i]
        checks = {"survived_full_trial": original["survived"]}
        m = {}
        if len(data) < 3:
            checks["enough_samples"] = False
            records.append(dict(original, checks=checks, metrics=m, accepted=False))
            continue

        def col(name):
            return data[:, fields.index(name)]

        q = data[:, [fields.index(k) for k in ("qw", "qx", "qy", "qz")]]
        w, x, y, z = q.T
        heading = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        heading -= heading[0]
        velocity = data[:, :3]
        position = data[:, [fields.index("x"), fields.index("y")]]
        delta = position[-1] - position[0]
        n = round(0.54 / dt)
        smooth = cycle_average(velocity, n)
        cmd = cycle_average(target, n)
        smooth_t = t[n - 1 :]
        steady = smooth_t >= 2
        error = smooth[steady] - cmd[steady]
        m.update(
            world_displacement_xy_m=delta.tolist(),
            final_heading_rad=float(heading[-1]),
            max_abs_heading_rad=float(abs(heading).max()),
            cycle_xy_rmse_m_s=float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1))))
            if len(error)
            else None,
            cycle_yaw_mae_rad_s=float(abs(error[:, 2]).mean()) if len(error) else None,
            mean_velocity=velocity.mean(0).tolist(),
            mean_torque_saturation=float(col("torque_saturation").mean()),
        )
        if case in ("forward", "backward"):
            sign = 1 if case == "forward" else -1
            mean = velocity[t >= 2].mean(0)
            lift = []
            for side in ("left", "right"):
                h = col("foot_clearance_" + side)
                contact = col("contact_" + side) > 0.5
                stable = col("stable_contact_" + side) > 0.5
                lift.append(
                    float(max(0, h[~stable].max() - np.median(h[contact])))
                    if (~stable).any() and contact.any()
                    else 0.0
                )
            m.update(
                steady_mean_velocity=mean.tolist(), peak_swing_lift_above_stance_m=lift
            )
            checks.update(
                speed_along_command_0_08_to_0_12_m_s=bool(
                    0.08 <= sign * mean[0] <= 0.12
                ),
                cycle_xy_rmse_at_most_0_04_m_s=bool(
                    m["cycle_xy_rmse_m_s"] is not None
                    and m["cycle_xy_rmse_m_s"] <= 0.04
                ),
                cycle_yaw_mae_at_most_0_06_rad_s=bool(
                    m["cycle_yaw_mae_rad_s"] is not None
                    and m["cycle_yaw_mae_rad_s"] <= 0.06
                ),
                distance_80_to_120_percent_of_target=bool(
                    0.08 * duration <= sign * delta[0] <= 0.12 * duration
                ),
                net_lateral_at_most_0_15_m=bool(abs(delta[1]) <= 0.15),
                final_heading_at_most_0_12_rad=bool(abs(heading[-1]) <= 0.12),
                max_heading_at_most_0_15_rad=bool(abs(heading).max() <= 0.15),
                both_feet_at_least_3_stable_landings=bool(
                    min(original["stable_foot_landings"]) >= 3
                ),
                both_feet_lift_at_least_10_mm=bool(min(lift) >= 0.01),
            )
        elif case == "stand":
            speed = np.linalg.norm(velocity[t >= 2, :2], axis=1).mean()
            m["steady_planar_speed_m_s"] = float(speed)
            checks.update(
                steady_planar_speed_at_most_0_015_m_s=bool(speed <= 0.015),
                net_drift_at_most_0_05_m=bool(np.linalg.norm(delta) <= 0.05),
                max_heading_at_most_0_1_rad=bool(abs(heading).max() <= 0.1),
            )
        elif case == "posture":
            feet = data[
                :,
                [
                    fields.index("foot_" + s + "_" + axis)
                    for s in ("left", "right")
                    for axis in ("x", "y", "z")
                ],
            ].reshape(-1, 2, 3)
            initial_feet = feet[np.flatnonzero(t >= 2)[0], :, :2]
            drift = np.linalg.norm(feet[t >= 2, :, :2] - initial_feet, axis=-1).max()
            cycles = []
            for cycle in range(round(duration / 16)):
                offset = cycle * 16
                before = (t >= offset + 1) & (t < offset + 2)
                low = (t >= offset + 6.2) & (t < offset + 8.8)
                after = (t >= offset + 14.5) & (t <= offset + 16)
                if not (before.any() and low.any() and after.any()):
                    cycles.append({"completed": False})
                    continue
                stand = float(col("height")[before].mean())
                lowered = float(col("height")[low].mean())
                returned = float(col("height")[after].mean())
                contacts = data[low, 8:10] > 0.5
                cycles.append(
                    dict(
                        completed=True,
                        standing_height_m=stand,
                        low_height_m=lowered,
                        returned_height_m=returned,
                        height_reduction_m=stand - lowered,
                        return_height_error_m=abs(returned - stand),
                        both_contact_fraction=float(contacts.all(-1).mean()),
                    )
                )
            height_rmse = float(
                np.sqrt(np.mean((col("height")[t >= 2] - goal[t >= 2]) ** 2))
            )
            speed = float(
                np.linalg.norm(velocity[t >= duration - 1.5, :2], axis=1).mean()
            )
            m.update(
                cycles=cycles,
                height_rmse_m=height_rmse,
                max_foot_drift_m=float(drift),
                final_planar_speed_m_s=speed,
                max_trunk_tilt_rad=float(col("tilt_rad").max()),
            )
            complete = bool(
                len(cycles) == round(duration / 16)
                and all(c["completed"] for c in cycles)
            )
            checks.update(
                all_cycles_completed=complete,
                each_cycle_lowers_at_least_25_mm=bool(
                    complete and all(c["height_reduction_m"] >= 0.025 for c in cycles)
                ),
                low_height_0_11_to_0_14_m=bool(
                    complete and all(0.11 <= c["low_height_m"] <= 0.14 for c in cycles)
                ),
                standing_height_recovered_within_10_mm=bool(
                    complete and all(c["return_height_error_m"] <= 0.01 for c in cycles)
                ),
                both_feet_contact_at_least_95_percent_of_low_hold=bool(
                    complete and all(c["both_contact_fraction"] >= 0.95 for c in cycles)
                ),
                height_rmse_at_most_10_mm=bool(height_rmse <= 0.01),
                foot_drift_at_most_25_mm=bool(drift <= 0.025),
                max_tilt_at_most_0_70_rad=bool(col("tilt_rad").max() <= 0.70),
                max_heading_at_most_0_15_rad=bool(abs(heading).max() <= 0.15),
                mean_torque_saturation_at_most_5_percent=bool(
                    m["mean_torque_saturation"] <= 0.05
                ),
                final_planar_speed_at_most_0_015_m_s=bool(speed <= 0.015),
            )
        elif case == "demo":
            phases = {}
            for name, lo, hi, sign in [
                ("forward", 5, 13, 1),
                ("backward", 18, 26, -1),
                ("forward_after_posture", 45, 48, 1),
            ]:
                mask = (t >= lo) & (t < hi)
                mean = velocity[mask].mean(0) if mask.any() else np.full(3, np.nan)
                phases[name] = {"mean_velocity": mean.tolist()}
                checks[name + "_speed"] = bool(
                    mask.any() and 0.08 <= sign * mean[0] <= 0.12
                )
            low = (t >= 33.2) & (t < 35.8)
            before = (t >= 27) & (t < 29)
            returned = (t >= 41.5) & (t < 43)
            stop = (t >= 50) & (t <= 52)
            drop = (
                float(col("height")[before].mean() - col("height")[low].mean())
                if low.any()
                else 0.0
            )
            contact = (
                float((data[low, 8:10] > 0.5).all(-1).mean()) if low.any() else 0.0
            )
            stop_speed = (
                float(np.linalg.norm(velocity[stop, :2], axis=1).mean())
                if stop.any()
                else float("inf")
            )
            return_error = (
                float(
                    abs(col("height")[returned].mean() - col("height")[before].mean())
                )
                if returned.any()
                else float("inf")
            )
            m.update(
                phases=phases,
                height_reduction_m=drop,
                low_both_contact_fraction=contact,
                standing_return_error_m=return_error,
                final_planar_speed_m_s=stop_speed,
            )
            checks.update(
                posture_lowers_at_least_25_mm=bool(drop >= 0.025),
                low_both_contact_at_least_95_percent=bool(contact >= 0.95),
                standing_height_recovered_within_10_mm=bool(return_error <= 0.01),
                final_stop_speed_at_most_0_015=bool(stop_speed <= 0.015),
                max_heading_at_most_0_15=bool(abs(heading).max() <= 0.15),
                net_lateral_at_most_0_15=bool(abs(delta[1]) <= 0.15),
            )
        else:
            checks["known_case"] = False
        records.append(
            dict(original, metrics=m, checks=checks, accepted=all(checks.values()))
        )
    summary = {}
    for case in dict.fromkeys(r["command"] for r in records):
        group = [r for r in records if r["command"] == case]
        summary[case] = dict(
            trials=len(group),
            survived=sum(r["survived"] for r in group),
            accepted=sum(r["accepted"] for r in group),
            acceptance_rate=sum(r["accepted"] for r in group) / len(group),
            mean_velocity=np.mean(
                [r["metrics"]["mean_velocity"] for r in group], axis=0
            ).tolist(),
            mean_displacement_xy_m=np.mean(
                [r["metrics"]["world_displacement_xy_m"] for r in group], axis=0
            ).tolist(),
        )
    return dict(
        accepted=all(g["acceptance_rate"] >= 0.9 for g in summary.values()),
        engineering_criteria_not_certification=True,
        scope="Nominal full-mass v2 flat ground. First episode only; no root pose correction or respawns. Locomotion criteria retained from forward verification; posture uses >=25mm lowering and <=10mm height recovery, repeated cycles.",
        seed=evaluation["seed"],
        forward_sha256=evaluation["forward_sha256"],
        summary=summary,
        trials=records,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = assess(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
