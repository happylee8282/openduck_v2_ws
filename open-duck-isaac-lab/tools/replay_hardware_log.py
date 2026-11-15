#!/usr/bin/env python3
"""Read a canonical sensor JSONL log and produce policy/target traces; no motor I/O."""

import argparse
import json
import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from open_duck_v2.profile import JOINT_NAMES, MOTOR_IDS, CONTRACT_VERSION
from open_duck_v2.control import V2Control, pack_actor_observation
from open_duck_v2.runtime import load_rsl_rl_actor


def replay(records, actor):
    profile = getattr(actor, "control_profile", {})
    control = V2Control(
        1,
        "cpu",
        max_delay=0,
        backlash=0.0,
        **{
            key: profile[key]
            for key in ("head_control_mode", "target_filter_tau_s", "head_target_speed")
            if key in profile
        },
    )
    from open_duck_v2.heading import HeadingHold, validated_imu_heading

    heading = HeadingHold(
        1,
        kp=profile.get("heading_hold_kp", 0.0),
        max_rate=profile.get("heading_hold_max_rate", 0.35),
        filter_tau=profile.get("heading_hold_filter_tau", 0.25),
        lateral_trim=profile.get("lateral_command_trim", 0.0),
        forward_scale=profile.get("forward_command_scale", 1.0),
    )
    last = None
    start = None
    for row in records:
        if (
            row.get("joint_frame") != "canonical-v2"
            or row.get("contract_version") != CONTRACT_VERSION
        ):
            raise ValueError(
                "Sensor log must declare canonical-v2 coordinates and the policy contract version"
            )
        if row.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("Sensor log joint order differs from v2")
        timestamp = float(row["timestamp_s"])
        if not math.isfinite(timestamp) or (
            last is not None and abs(timestamp - last - 0.02) > 0.005
        ):
            raise ValueError(
                "Sensor log violates the 50 Hz policy clock (5 ms tolerance)"
            )
        if start is None:
            start = timestamp

        def tensor(key, width):
            value = torch.tensor(row[key], dtype=torch.float32).reshape(1, -1)
            if value.shape != (1, width):
                raise ValueError(f"{key} shape mismatch")
            return value

        commands = tensor("commands", 7)
        if profile.get("heading_hold_kp", 0):
            measured = validated_imu_heading(tensor("imu_quaternion_wxyz", 4))
            if last is None:
                heading.reset(torch.tensor([0]), measured)
            heading.observe(measured)
            commands = heading.commands(commands)
        phase = 2 * math.pi * (timestamp - start) / actor.phase_period_s
        obs = pack_actor_observation(
            tensor("gyro", 3),
            tensor("accelerometer", 3),
            commands,
            tensor("joint_pos", 14),
            tensor("joint_vel", 14),
            control.history,
            control.motor_targets,
            tensor("contacts", 2),
            torch.tensor([[math.cos(phase), math.sin(phase)]]),
        )
        with torch.inference_mode():
            actions = actor(obs)
        control.process(actions, tensor("commands", 7)[:, 3:7])
        yield {
            "timestamp_s": timestamp,
            "observation": obs[0].tolist(),
            "external_commands": row["commands"],
            "policy_commands": commands[0].tolist(),
            "action": actions[0].tolist(),
            "target_canonical_rad": control.motor_targets[0].tolist(),
            "motor_ids": list(MOTOR_IDS),
            "joint_names": list(JOINT_NAMES),
        }
        last = timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    actor = load_rsl_rl_actor(a.checkpoint, "cpu")
    records = (json.loads(s) for s in a.input.read_text().splitlines() if s.strip())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with a.output.open("w") as f:
        for row in replay(records, actor):
            f.write(json.dumps(row) + "\n")
            count += 1
    if not count:
        raise ValueError("Empty hardware log")
    print(
        f"Validated and replayed {count} sensor frames; no actuator commands were sent"
    )


if __name__ == "__main__":
    main()
