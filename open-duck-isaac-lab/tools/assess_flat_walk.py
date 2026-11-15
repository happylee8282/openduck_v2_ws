#!/usr/bin/env python3
"""Assess fixed-command trials without confusing survival or fall speed with walking."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--video-evaluation", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    telemetry = np.load(args.evaluation.parent / "rollout.npz")
    columns = telemetry["columns"].tolist()
    values = telemetry["values"]
    per_trial = []
    for i, trial in enumerate(evaluation["trials"]):
        record = dict(trial)
        if trial["command"] != "forward":
            per_trial.append(record)
            continue
        alive = values[:, i, columns.index("alive")] > 0.5
        lift = []
        for side in ["left", "right"]:
            heights = values[:, i, columns.index("foot_clearance_" + side)]
            raw_contact = values[:, i, columns.index("contact_" + side)] > 0.5
            stable_contact = values[:, i, columns.index("stable_contact_" + side)] > 0.5
            stance = heights[alive & raw_contact]
            swing = heights[alive & ~stable_contact]
            lift.append(
                float(max(0.0, swing.max() - np.median(stance)))
                if len(stance) and len(swing)
                else 0.0
            )
        checks = {
            "survived_full_trial": trial["survived"],
            "xy_rmse_at_most_0_05_m_s": trial["xy_rmse_m_s"] is not None
            and trial["xy_rmse_m_s"] <= 0.05,
            "yaw_mae_at_most_0_2_rad_s": trial["yaw_mae_rad_s"] is not None
            and trial["yaw_mae_rad_s"] <= 0.2,
            "both_feet_at_least_3_stable_landings": min(
                trial.get("stable_foot_landings", [0, 0])
            )
            >= 3,
            "both_feet_lift_at_least_10_mm": min(lift) >= 0.01,
            "forward_displacement_at_least_60_percent_of_target": trial.get(
                "displacement_xy_m", [0]
            )[0]
            >= 0.1 * evaluation["duration_s"] * 0.6,
        }
        record.update(
            checks=checks,
            peak_swing_lift_above_stance_m=lift,
            walking_trial_passed=all(checks.values()),
        )
        per_trial.append(record)
    forward = [r for r in per_trial if r["command"] == "forward"]
    surviving = [r for r in forward if r["survived"]]
    passed = sum(r["walking_trial_passed"] for r in forward)
    result = {
        "checkpoint": evaluation["checkpoint"],
        "profile": {
            key: evaluation.get(key)
            for key in [
                "seed",
                "duration_s",
                "randomized",
                "spawn_height_m",
                "target_height_m",
                "self_collisions",
                "mechanical_collision_filter_distance",
            ]
        },
        "criteria_note": "Initial engineering targets for this experiment, not a certification standard. Foot lift uses link-origin height during debounced non-contact relative to median stance height.",
        "forward_trials": len(forward),
        "forward_walking_passed": passed,
        "forward_walking_pass_rate": passed / len(forward) if forward else 0,
        "forward_survival_rate": len(surviving) / len(forward) if forward else 0,
        "mean_vx_of_surviving_trials_m_s": float(
            np.mean([r["mean_velocity"][0] for r in surviving])
        )
        if surviving
        else None,
        "median_forward_displacement_m": float(
            np.median([r.get("displacement_xy_m", [0])[0] for r in forward])
        )
        if forward
        else None,
        "target_forward_displacement_m": 0.1 * evaluation["duration_s"],
        "initial_forward_criteria_passed": bool(
            forward and passed / len(forward) >= 0.9
        ),
        "command_summary": evaluation["summary"],
        "trials": per_trial,
    }
    if args.baseline:
        result["baseline_same_physics"] = json.loads(args.baseline.read_text())[
            "summary"
        ]
    if args.video_evaluation:
        video = json.loads(args.video_evaluation.read_text())
        result["video_trial"] = video["trials"][0]
        result["video_completed_without_fall"] = video["trials"][0]["survived"]
        result["initial_forward_criteria_passed"] &= video["trials"][0]["survived"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("trials", "command_summary", "baseline_same_physics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
