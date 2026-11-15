#!/usr/bin/env python3
"""First-episode evaluation of frozen forward, reverse and residual posture skills."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DURATIONS = {
    "stand": 20.0,
    "forward": 20.0,
    "backward": 20.0,
    "posture": 48.0,
    "demo": 52.0,
}


def scenario_commands(case, time):
    vx = 0.0
    low = False
    if case == "forward":
        vx = 0.1
    elif case == "backward":
        vx = -0.1
    elif case == "posture":
        low = 2 <= time % 16 < 9
    elif case == "demo":
        if 3 <= time < 13 or 43 <= time < 48:
            vx = 0.1
        elif 16 <= time < 26:
            vx = -0.1
        low = 29 <= time < 36
    return vx, low


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "config/multiskill_policy.json"
    )
    p.add_argument(
        "--cases",
        nargs="+",
        choices=list(DURATIONS),
        default=["stand", "forward", "backward", "posture"],
    )
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--eval-seed", type=int, default=62)
    p.add_argument("--duration", type=float)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.video:
        if len(args.cases) != 1 or args.trials != 1:
            p.error("Video requires a single case and trial")
        args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = writer = None
    try:
        import numpy as np
        import torch
        from open_duck_v2.runtime import load_rsl_rl_actor, environment_cfg_for_actor
        from open_duck_v2.posture_runtime import load_posture_actor
        from open_duck_v2.posture import PostureReference
        from open_duck_v2.multiskill import load_manifest, MultiSkillRuntime
        from open_duck_v2.control import quat_rotate_inverse
        from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv

        bundle = load_manifest(args.bundle)
        forward = load_rsl_rl_actor(
            bundle["skills"]["forward"]["resolved_checkpoint"], args.device
        )
        backward = load_rsl_rl_actor(
            bundle["skills"]["backward"]["resolved_checkpoint"], args.device
        )
        posture = load_posture_actor(
            bundle["skills"]["posture"]["resolved_checkpoint"],
            bundle["reference_path"],
            args.device,
        )
        reference = PostureReference(bundle["reference_path"], args.device)
        cfg = environment_cfg_for_actor(forward)
        cfg.seed = args.eval_seed
        cfg.scene.num_envs = len(args.cases) * args.trials
        cfg.sim.device = args.device
        cfg.randomize_domain = cfg.observation_noise = False
        cfg.action_delay_min = cfg.action_delay_max = cfg.imu_delay_min = (
            cfg.imu_delay_max
        ) = 0
        duration = args.duration or max(DURATIONS[case] for case in args.cases)
        cfg.episode_length_s = duration + 2
        cfg.command_resampling_time_s = duration + 5
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        cfg.viewer.resolution = (960, 640)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (0.70, -0.90, 0.45)
        cfg.viewer.lookat = (0.0, 0.0, 0.02)
        env = OpenDuckMiniEnv(cfg, render_mode="rgb_array" if args.video else None)
        env._resample_commands = lambda ids: None
        obs, _ = env.reset()
        env._reset_idx = lambda ids: None
        runtime = MultiSkillRuntime(env, forward, backward, posture, reference)
        labels = [case for case in args.cases for _ in range(args.trials)]
        ends = torch.tensor(
            [min(duration, DURATIONS[c]) for c in labels], device=env.device
        )
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        falls = ends.clone()
        limits = env.robot.root_physx_view.get_dof_max_forces().to(env.device)
        masses = env.robot.root_physx_view.get_masses().to(env.device)
        contact_prev = env._get_foot_contacts()
        contact_count = torch.zeros_like(contact_prev, dtype=torch.long)
        air_count = torch.zeros_like(contact_count)
        stable = contact_prev.clone()
        landings = torch.zeros(env.num_envs, 2, device=env.device)
        rows = []
        command_rows = []
        height_rows = []
        skill_rows = []
        policy_rows = []
        pose_rows = []
        action_rows = []
        target_rows = []
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/bin/ffmpeg")
            writer = imageio.get_writer(
                str(args.output / "playback.mp4"), fps=25, codec="libx264", quality=8
            )
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
            )
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for k in range(-20, 21):
                for axis in range(2):
                    marker = UsdGeom.Cube.Define(
                        stage, f"/World/SkillGrid/g_{axis}_{k + 20}"
                    )
                    marker.CreateSizeAttr(1.0)
                    marker.AddTranslateOp().Set(
                        Gf.Vec3d(
                            k * 0.25 if axis == 0 else 0, k * 0.25 if axis else 0, 0.001
                        )
                    )
                    marker.AddScaleOp().Set(
                        Gf.Vec3d(
                            0.002 if axis == 0 else 10,
                            10 if axis == 0 else 0.002,
                            0.001,
                        )
                    )
                    marker.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.2, 0.24)])
            for _ in range(8):
                env.render()
        for step in range(round(duration / 0.02)):
            t = step * 0.02
            external = torch.zeros(env.num_envs, 7, device=env.device)
            low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for group, case in enumerate(args.cases):
                vx, request_low = scenario_commands(case, t)
                if t >= min(duration, DURATIONS[case]):
                    vx, request_low = 0.0, False
                ids = slice(group * args.trials, (group + 1) * args.trials)
                external[ids, 0] = vx
                low[ids] = request_low
            with torch.no_grad():
                obs, reward, terminated, truncated, extra = runtime.step(external, low)
                within = (step + 1) * 0.02 <= ends + 1e-5
                fresh = active & within & terminated
                falls[fresh] = (step + 1) * 0.02
                active &= ~fresh
                valid = active & within
                root = env.robot.data.root_state_w
                velocity = quat_rotate_inverse(
                    root[:, 3:7], env.robot.data.root_lin_vel_w
                )
                angular = quat_rotate_inverse(
                    root[:, 3:7], env.robot.data.root_ang_vel_w
                )
                up = quat_rotate_inverse(
                    root[:, 3:7], root.new_tensor([0, 0, 1]).expand(env.num_envs, -1)
                )
                tilt = torch.acos(up[:, 2].clamp(-1, 1))
                contacts = env._get_foot_contacts()
                contact_count = torch.where(contacts, contact_count + 1, 0)
                air_count = torch.where(~contacts, air_count + 1, 0)
                new_stable = torch.where(
                    contact_count >= 3, True, torch.where(air_count >= 3, False, stable)
                )
                landings += ((~stable) & new_stable & valid[:, None]).float()
                stable = new_stable
                saturation = (
                    (env.robot.data.applied_torque.abs() >= 0.95 * limits)
                    .float()
                    .mean(-1)
                )
                joints = (
                    env.robot.data.joint_pos[:, env.controlled_joint_ids]
                    + env.control.offsets
                )
                com = (env.robot.data.body_com_pos_w * masses[:, :, None]).sum(
                    1
                ) / masses.sum(1)[:, None]
                feet = (
                    env.robot.data.body_pos_w[:, env.foot_body_ids]
                    - env.scene.env_origins[:, None]
                )
                row = torch.cat(
                    (
                        velocity[:, :2],
                        angular[:, 2:3],
                        root[:, :3],
                        env._base_height()[:, None],
                        tilt[:, None],
                        contacts.float(),
                        saturation[:, None],
                        (env.actions.abs() >= 0.99).float().mean(-1)[:, None],
                        valid.float()[:, None],
                        stable.float(),
                        env._feet_clearance(),
                        joints,
                        root[:, 3:7],
                        env.robot.data.joint_vel[:, env.controlled_joint_ids[5:9]],
                        com,
                        feet.flatten(1),
                        env.robot.data.joint_vel[:, env.controlled_joint_ids],
                    ),
                    -1,
                )
                if not torch.isfinite(row).all():
                    raise RuntimeError("Nonfinite skill rollout")
                rows.append(row.cpu().numpy())
                command_rows.append(
                    runtime.applied_commands[:, :3].cpu().numpy().copy()
                )
                height_rows.append(runtime.height.cpu().numpy().copy())
                skill_rows.append(runtime.active_skill.cpu().numpy().copy())
                policy_rows.append(runtime.last_policy_observation.cpu().numpy().copy())
                pose_rows.append(runtime.last_posture_observation.cpu().numpy().copy())
                action_rows.append(env.actions.cpu().numpy().copy())
                target_rows.append(env.control.motor_targets.cpu().numpy().copy())
            if args.video and step % 2 == 0:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 86), fill=(15, 22, 28))
                name = ["forward/stop", "backward", "posture"][
                    int(runtime.active_skill[0])
                ]
                draw.text(
                    (14, 8),
                    f"{args.cases[0]} | t={(step + 1) * 0.02:.2f}s | skill={name} | command vx={float(runtime.applied_commands[0, 0]):+.3f}m/s",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (14, 37),
                    f"height target={float(runtime.height[0]):.3f} actual={float(row[0, 6]):.3f}m | vx={float(velocity[0, 0]):+.3f} vy={float(velocity[0, 1]):+.3f} | tilt={float(tilt[0]) * 180 / np.pi:.1f}deg",
                    font=font,
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
                if step % 200 == 0:
                    frame.save(args.output / f"frame_{step:04}.png")
                if fresh[0]:
                    frame.save(args.output / "terminal.png")
                    break
            if step % 200 == 0:
                print(
                    "step",
                    step,
                    "alive",
                    int(active.sum()),
                    "vx",
                    velocity[:, 0].mean().item(),
                    "height",
                    row[:, 6].mean().item(),
                    flush=True,
                )
            if not active.any():
                break
        fields = (
            [
                "vx",
                "vy",
                "yaw_rate",
                "x",
                "y",
                "z",
                "height",
                "tilt_rad",
                "contact_left",
                "contact_right",
                "torque_saturation",
                "action_clipping",
                "alive",
                "stable_contact_left",
                "stable_contact_right",
                "foot_clearance_left",
                "foot_clearance_right",
            ]
            + ["joint_" + str(i) for i in range(14)]
            + ["qw", "qx", "qy", "qz"]
            + ["head_velocity_" + str(i) for i in range(4)]
            + ["com_x", "com_y", "com_z"]
            + [
                "foot_" + s + "_" + axis
                for s in ("left", "right")
                for axis in ("x", "y", "z")
            ]
            + ["joint_velocity_" + str(i) for i in range(14)]
        )
        data = np.stack(rows)
        trace = np.stack(command_rows)
        np.savez_compressed(
            args.output / "rollout.npz",
            values=data,
            command_trace=trace,
            height_target=np.stack(height_rows),
            active_skill=np.stack(skill_rows),
            observations=np.stack(policy_rows),
            posture_observations=np.stack(pose_rows),
            actions=np.stack(action_rows),
            motor_targets=np.stack(target_rows),
            labels=np.array(labels),
            commands=trace[0],
            dt=0.02,
            columns=np.array(fields),
        )
        trials = []
        for i, case in enumerate(labels):
            a = data[data[:, i, 12] > 0.5, i]
            target = trace[data[:, i, 12] > 0.5, i]
            trial = dict(
                command=case,
                trial=i % args.trials,
                duration_s=float(ends[i]),
                survived=bool(active[i]),
                survival_s=float(falls[i]),
                stable_foot_landings=landings[i].cpu().tolist(),
            )
            if len(a):
                trial.update(
                    mean_velocity=a[:, :3].mean(0).tolist(),
                    xy_rmse_m_s=float(
                        np.sqrt(
                            np.mean(np.sum((a[:, :2] - target[:, :2]) ** 2, axis=1))
                        )
                    ),
                    yaw_mae_rad_s=float(np.abs(a[:, 2] - target[:, 2]).mean()),
                    displacement_xy_m=(a[-1, 3:5] - a[0, 3:5]).tolist(),
                    mean_height_m=float(a[:, 6].mean()),
                    mean_torque_saturation=float(a[:, 10].mean()),
                )
            trials.append(trial)
        report = dict(
            status="completed",
            bundle=str(args.bundle.resolve()),
            seed=args.eval_seed,
            randomized=False,
            duration_s=duration,
            dt=0.02,
            reference_sha256=reference.sha256,
            total_mass_mean_kg=float(masses.sum(1).mean()),
            no_resets_after_initialization=True,
            base_pose_not_overwritten=True,
            trials=trials,
            forward_sha256=bundle["skills"]["forward"]["sha256"],
            checkpoints={
                k: v["resolved_checkpoint"] for k, v in bundle["skills"].items()
            },
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(trials, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
