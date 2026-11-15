#!/usr/bin/env python3
"""Retarget a BVH lower-body clip into Open Duck joint reference trajectories.

This is a lightweight first-pass retargeter for imitation learning. It does not
solve full-body IK; it extracts human hip/knee/ankle Euler channels when present,
normalizes the motion, mirrors left/right into the Open Duck joint order, and
clips everything to conservative robot ranges.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_duck_v2.profile import JOINT_NAMES, HOME, LIMITS, POLICY_HZ

CONTROLLED_JOINT_NAMES = JOINT_NAMES
DEFAULT_POSE = np.array(HOME, dtype=np.float32)
JOINT_LIMITS = np.array(LIMITS, dtype=np.float32) + np.array([0.02, -0.02])


@dataclass
class BvhData:
    joint_channels: dict[str, list[str]]
    channel_slices: dict[str, slice]
    frame_time: float
    frames: np.ndarray


def parse_bvh(path: Path) -> BvhData:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    joint_stack: list[str] = []
    joint_channels: dict[str, list[str]] = {}
    channel_slices: dict[str, slice] = {}
    channel_cursor = 0
    motion_idx = None

    for idx, raw in enumerate(text):
        line = raw.strip()
        if line.startswith("MOTION"):
            motion_idx = idx
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"} and len(parts) >= 2:
            joint_stack.append(parts[1])
        elif parts[0] == "End":
            joint_stack.append(f"{joint_stack[-1]}_End")
        elif parts[0] == "}" and joint_stack:
            joint_stack.pop()
        elif parts[0] == "CHANNELS" and joint_stack:
            count = int(parts[1])
            channels = parts[2 : 2 + count]
            joint_name = joint_stack[-1]
            joint_channels[joint_name] = channels
            channel_slices[joint_name] = slice(channel_cursor, channel_cursor + count)
            channel_cursor += count

    if motion_idx is None:
        raise ValueError(f"No MOTION section found in {path}")

    frame_count = int(re.search(r"\d+", text[motion_idx + 1]).group(0))
    frame_time = float(
        re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text[motion_idx + 2]).group(0)
    )
    rows = []
    for raw in text[motion_idx + 3 : motion_idx + 3 + frame_count]:
        if raw.strip():
            rows.append([float(value) for value in raw.split()])
    frames = np.asarray(rows, dtype=np.float32)
    if frames.ndim != 2 or len(frames) != frame_count:
        raise ValueError("Truncated BVH motion")
    if frames.shape[1] != channel_cursor:
        raise ValueError(f"Expected {channel_cursor} channels, got {frames.shape[1]}")

    return BvhData(joint_channels, channel_slices, frame_time, frames)


def find_joint(data: BvhData, *needles: str) -> str | None:
    lower_names = {name.lower(): name for name in data.joint_channels}
    for name_l, original in lower_names.items():
        if all(needle.lower() in name_l for needle in needles) and not (
            needles[-1].lower() == "leg" and "upleg" in name_l
        ):
            return original
    return None


def euler_channel(data: BvhData, joint_name: str | None, axis: str) -> np.ndarray:
    if joint_name is None:
        return np.zeros((data.frames.shape[0],), dtype=np.float32)
    channels = data.joint_channels[joint_name]
    sl = data.channel_slices[joint_name]
    for local_idx, channel in enumerate(channels):
        if channel.lower() == f"{axis.lower()}rotation":
            return np.unwrap(np.deg2rad(data.frames[:, sl.start + local_idx])).astype(
                np.float32
            )
    return np.zeros((data.frames.shape[0],), dtype=np.float32)


def normalize_signal(
    values: np.ndarray, scale: float, center: bool = True
) -> np.ndarray:
    values = values.astype(np.float32)
    if center:
        values = values - np.median(values)
    percentile = np.percentile(np.abs(values), 95)
    if percentile > 1.0e-6:
        values = values / percentile
    return np.clip(values * scale, -abs(scale), abs(scale)).astype(np.float32)


def retarget(
    data: BvhData, target_fps: float
) -> dict[str, np.ndarray | float | list[str]]:
    left_up = (
        find_joint(data, "left", "upleg")
        or find_joint(data, "lhip")
        or find_joint(data, "left", "hip")
    )
    right_up = (
        find_joint(data, "right", "upleg")
        or find_joint(data, "rhip")
        or find_joint(data, "right", "hip")
    )
    left_leg = (
        find_joint(data, "left", "leg")
        or find_joint(data, "lknee")
        or find_joint(data, "left", "knee")
    )
    right_leg = (
        find_joint(data, "right", "leg")
        or find_joint(data, "rknee")
        or find_joint(data, "right", "knee")
    )
    left_foot = (
        find_joint(data, "left", "foot")
        or find_joint(data, "lankle")
        or find_joint(data, "left", "ankle")
    )
    right_foot = (
        find_joint(data, "right", "foot")
        or find_joint(data, "rankle")
        or find_joint(data, "right", "ankle")
    )

    if target_fps <= 0 or data.frame_time <= 0 or len(data.frames) < 2:
        raise ValueError("Invalid motion timing")
    src_fps = 1.0 / data.frame_time
    source_time = np.arange(len(data.frames)) / src_fps
    target_time = (
        np.arange(int(np.floor(source_time[-1] * target_fps)) + 1) / target_fps
    )
    if len(target_time) < 2:
        raise ValueError("Clip too short")

    def interpolate(x):
        return np.interp(target_time, source_time, x).astype(np.float32)

    frame_ids = np.arange(len(target_time))
    joint_pos = np.tile(DEFAULT_POSE, (len(target_time), 1))
    names = {
        "left_hip_pitch": (left_up, "X", 0.25),
        "right_hip_pitch": (right_up, "X", -0.25),
        "left_hip_roll": (left_up, "Z", 0.08),
        "right_hip_roll": (right_up, "Z", -0.08),
        "left_hip_yaw": (left_up, "Y", 0.08),
        "right_hip_yaw": (right_up, "Y", 0.08),
        "left_knee": (left_leg, "X", 0.15),
        "right_knee": (right_leg, "X", 0.15),
        "left_ankle": (left_foot, "X", 0.15),
        "right_ankle": (right_foot, "X", 0.15),
    }
    missing = [name for name, (source, _, _) in names.items() if source is None]
    if missing:
        raise ValueError(f"Missing BVH joints for {missing}")
    for name, (source, axis, scale) in names.items():
        signal = interpolate(
            normalize_signal(euler_channel(data, source, axis), abs(scale))
        )
        joint_pos[:, CONTROLLED_JOINT_NAMES.index(name)] += signal * np.sign(scale)

    # Small expressive head motion makes dance clips visibly distinct while
    # keeping the robot's balance dominated by the legs.
    phase = np.linspace(
        0.0, 2.0 * np.pi, frame_ids.shape[0], endpoint=False, dtype=np.float32
    )
    for name, offset in [
        ("neck_pitch", 0.0),
        ("head_pitch", 0.5),
        ("head_yaw", 0.0),
        ("head_roll", 1.2),
    ]:
        joint_pos[:, CONTROLLED_JOINT_NAMES.index(name)] = 0.03 * np.sin(phase + offset)

    joint_pos = np.clip(joint_pos, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]).astype(
        np.float32
    )
    dt = 1.0 / target_fps
    # Stretch time, rather than clipping reference velocities independently of positions.
    peak = np.max(np.abs(np.diff(joint_pos, axis=0))) / dt
    slowdown = max(1.0, float(peak) / (5.24 * 0.95))
    original_time = np.arange(len(joint_pos)) * dt
    output_time = np.arange(int(np.floor(original_time[-1] * slowdown / dt)) + 1) * dt
    sample_time = output_time / slowdown
    joint_pos = np.stack(
        [np.interp(sample_time, original_time, joint_pos[:, i]) for i in range(14)], -1
    ).astype(np.float32)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    phase = np.linspace(
        0.0, 2.0 * np.pi, len(joint_pos), endpoint=False, dtype=np.float32
    )
    frame_ids = np.rint(sample_time / data.frame_time).astype(np.int32)

    root_height = 0.15 + 0.005 * np.sin(phase + np.pi).astype(np.float32)
    root_lin_vel = np.zeros((frame_ids.shape[0], 3), dtype=np.float32)
    root_lin_vel[:, 0] = 0.06

    return {
        "joint_names": np.array(CONTROLLED_JOINT_NAMES),
        "joint_frame": np.array("canonical-v2"),
        "root_lin_vel_frame": np.array("body"),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_height": root_height.astype(np.float32),
        "root_lin_vel": root_lin_vel,
        "fps": np.float32(1.0 / dt),
        "slowdown_factor": np.float32(slowdown),
        "source_frame_ids": frame_ids.astype(np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bvh", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/motions/open_duck_cmu_imitation.npz"),
    )
    parser.add_argument("--target-fps", type=float, default=float(POLICY_HZ))
    args = parser.parse_args()

    data = parse_bvh(args.input_bvh)
    motion = retarget(data, args.target_fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **motion)
    print(
        f"saved {args.output}: {motion['joint_pos'].shape[0]} frames, "
        f"{float(motion['fps']):.2f} fps"
    )


if __name__ == "__main__":
    main()
