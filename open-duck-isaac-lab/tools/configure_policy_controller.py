#!/usr/bin/env python3
"""Attach a reproducible command controller to existing weights; does not train."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heading-kp", type=float, default=2.0)
    p.add_argument("--yaw-rate-limit", type=float, default=0.35)
    p.add_argument("--heading-filter-tau", type=float, default=0.25)
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--lateral-trim", type=float, default=0.0)
    p.add_argument("--calibration-evaluation", type=Path, required=True)
    args = p.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        p.error("Keep the source weights; choose a different output")
    import torch
    from open_duck_v2.heading import HeadingHold
    from open_duck_v2.runtime import load_rsl_rl_actor

    load_rsl_rl_actor(args.checkpoint)
    HeadingHold(
        1,
        kp=args.heading_kp,
        max_rate=args.yaw_rate_limit,
        filter_tau=args.heading_filter_tau,
        forward_scale=args.forward_scale,
        lateral_trim=args.lateral_trim,
    )
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    info = source["infos"]
    info["control_profile"].update(
        heading_hold_kp=args.heading_kp,
        heading_hold_max_rate=args.yaw_rate_limit,
        heading_hold_filter_tau=args.heading_filter_tau,
        forward_command_scale=args.forward_scale,
        lateral_command_trim=args.lateral_trim,
    )
    info["post_training_controller"] = {
        "type": "IMU attitude heading hold and command calibration; weights unchanged",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "calibration_evaluation": str(args.calibration_evaluation.resolve()),
        "calibration_evaluation_sha256": hashlib.sha256(
            args.calibration_evaluation.read_bytes()
        ).hexdigest(),
        "additional_sensor": "canonical trunk IMU attitude wxyz",
        "hardware_calibrated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)
    assert all(
        torch.equal(v, reloaded["model_state_dict"][k])
        for k, v in source["model_state_dict"].items()
    )
    report = dict(
        status="controller_configured_not_retrained",
        iteration_unchanged=source.get("iter"),
        weights_identical=True,
        output=str(args.output.resolve()),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        control_profile=info["control_profile"],
        provenance=info["post_training_controller"],
    )
    args.output.with_suffix(".controller.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
