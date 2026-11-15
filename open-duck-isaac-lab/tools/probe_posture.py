#!/usr/bin/env python3
"""Actual Isaac Lab actuator-only squat/stand probe; never moves the base after reset."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "config/posture_reference_v1.json"
    )
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--duration", type=float, default=16.0)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app
    env = None
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv
        from open_duck_complex_terrain.open_duck_env_cfg import OpenDuckStraightV6EnvCfg
        from open_duck_v2.posture import (
            PostureReference,
            PostureControl,
            posture_command,
        )
        from open_duck_v2.profile import JOINT_NAMES, URDF_TO_CANONICAL
        from open_duck_v2.control import quat_rotate_inverse

        ref = PostureReference(args.reference, args.device)
        cfg = OpenDuckStraightV6EnvCfg()
        cfg.scene.num_envs = args.trials
        cfg.sim.device = args.device
        cfg.randomize_domain = cfg.observation_noise = False
        cfg.heading_hold_kp = 0
        cfg.action_delay_min = cfg.action_delay_max = cfg.imu_delay_min = (
            cfg.imu_delay_max
        ) = 0
        cfg.episode_length_s = args.duration + 2
        cfg.command_lin_vel_x = (0, 0)
        cfg.stand_command_probability = 1
        cfg.spawn_height = ref.stand_height + 0.003
        cfg.robot.init_state.pos = (0, 0, cfg.spawn_height)
        cfg.robot.init_state.joint_pos = {
            n: float(q - o)
            for n, q, o in zip(
                JOINT_NAMES, ref.stand_joints.cpu().tolist(), URDF_TO_CANONICAL
            )
        }
        cfg.robot.init_state.rot = tuple(
            Rotation.from_euler("xyz", ref.rpy[-1].cpu().numpy()).as_quat(
                scalar_first=True
            )
        )
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        env = OpenDuckMiniEnv(cfg)
        env.control = PostureControl(env.num_envs, env.device, ref)
        env.actions = env.control.actions
        obs, _ = env.reset()
        env._reset_idx = lambda ids: None
        rows = []
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        falls = torch.full((env.num_envs,), args.duration, device=env.device)
        origins = env.scene.env_origins
        limits = env.robot.root_physx_view.get_dof_max_forces().to(env.device)
        for step in range(round(args.duration / 0.02)):
            time = torch.full((env.num_envs,), step * 0.02, device=env.device)
            h, hdot = posture_command(time, ref.stand_height, ref.low_height)
            q, rpy, x, com = ref.sample(h)
            env.control.reference_target = q
            obs, reward, done, trunc, extra = env.step(
                torch.zeros(env.num_envs, 14, device=env.device)
            )
            fresh = alive & done
            falls[fresh] = (step + 1) * 0.02
            alive &= ~done
            root = env.robot.data.root_state_w
            up = quat_rotate_inverse(
                root[:, 3:7], root.new_tensor([0, 0, 1]).expand(env.num_envs, -1)
            )
            tilt = torch.acos(up[:, 2].clamp(-1, 1))
            joint = (
                env.robot.data.joint_pos[:, env.controlled_joint_ids]
                + env.control.offsets
            )
            contacts = env._get_foot_contacts().float()
            feet = env.robot.data.body_pos_w[:, env.foot_body_ids] - origins[:, None]
            row = torch.cat(
                [
                    h[:, None],
                    env._base_height()[:, None],
                    tilt[:, None],
                    contacts,
                    (root[:, :3] - origins),
                    root[:, 3:7],
                    joint,
                    q,
                    feet.flatten(1),
                    (env.robot.data.applied_torque.abs() > 0.95 * limits)
                    .float()
                    .mean(-1)[:, None],
                    alive.float()[:, None],
                ],
                -1,
            )
            rows.append(row.cpu().numpy())
            if step % 100 == 0:
                print(
                    "step",
                    step,
                    "h",
                    env._base_height().mean().item(),
                    "target",
                    h.mean().item(),
                    "tilt",
                    tilt.mean().item(),
                    "alive",
                    alive.sum().item(),
                    flush=True,
                )
            if not alive.any():
                break
        a = np.stack(rows)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / "rollout.npz", values=a, dt=0.02)
        summary = dict(
            survival_rate=float(alive.float().mean()),
            fall_times=falls.cpu().tolist(),
            minimum_heights=a[:, :, 1].min(0).tolist(),
            max_tilt_rad=a[:, :, 2].max(0).tolist(),
            height_rmse=float(np.sqrt(np.mean((a[:, :, 0] - a[:, :, 1]) ** 2))),
            final_height=a[-1, :, 1].tolist(),
            mean_torque_saturation=float(a[:, :, -2].mean()),
        )
        (args.output / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
