#!/usr/bin/env python3
"""Fixed-command, first-episode locomotion evaluation and optional real render."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.10, 0.0, 0.0),
    "backward": (-0.10, 0.0, 0.0),
    "left": (0.0, 0.10, 0.0),
    "right": (0.0, -0.10, 0.0),
    "turn_left": (0.0, 0.0, 0.5),
    "turn_right": (0.0, 0.0, -0.5),
}


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--commands", nargs="+", choices=list(COMMANDS), default=list(COMMANDS)
    )
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--eval-seed", type=int, default=43)
    p.add_argument("--randomized", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument(
        "--forward-command-scale",
        type=float,
        help="Internal forward command calibration (external target unchanged)",
    )
    p.add_argument(
        "--heading-hold-gain",
        type=float,
        help="Diagnostic IMU-attitude heading hold override",
    )
    p.add_argument(
        "--lateral-command-trim",
        type=float,
        help="Diagnostic internal lateral trim (external target unchanged)",
    )
    p.add_argument(
        "--record-policy-io",
        action="store_true",
        help="Record raw 101 inputs/actions/targets for export parity",
    )
    p.add_argument("--sequence", choices=["stop_go_stop"])
    p.add_argument(
        "--include-transition",
        action="store_true",
        help="Evaluate stop/go/stop trials alongside fixed commands",
    )
    p.add_argument(
        "--disable-self-collisions", action="store_true", help="Diagnostic only"
    )
    p.add_argument("--collision-filter-distance", type=int, default=2)
    p.add_argument("--head-control-mode", choices=["policy", "command", "stabilized"])
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    names = (
        [args.sequence]
        if args.sequence
        else args.commands + (["stop_go_stop"] if args.include_transition else [])
    )
    if args.trials < 1 or args.duration <= 0:
        p.error("Positive trials and duration required")
    if args.video and (args.trials != 1 or len(names) != 1):
        p.error("Video requires one command and --trials 1")
    if args.video:
        args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = writer = None
    try:
        import numpy as np
        import torch
        from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv
        from open_duck_v2.control import quat_rotate_inverse
        from open_duck_v2.profile import contract_hash
        from open_duck_v2.runtime import load_rsl_rl_actor, environment_cfg_for_actor

        actor = load_rsl_rl_actor(args.checkpoint, device=args.device)
        cfg = environment_cfg_for_actor(actor)
        if args.forward_command_scale is not None:
            cfg.forward_command_scale = args.forward_command_scale
        if args.heading_hold_gain is not None:
            cfg.heading_hold_kp = args.heading_hold_gain
        if args.lateral_command_trim is not None:
            cfg.lateral_command_trim = args.lateral_command_trim
        cfg.seed = args.eval_seed
        for key, value in getattr(actor, "control_profile", {}).items():
            if key in (
                "head_control_mode",
                "target_filter_tau_s",
                "head_target_speed",
                "reward_profile",
            ):
                setattr(cfg, key, value)
        if args.head_control_mode is not None:
            cfg.head_control_mode = args.head_control_mode
        cfg.mechanical_collision_filter_distance = args.collision_filter_distance
        if args.disable_self_collisions:
            cfg.robot.spawn.articulation_props.enabled_self_collisions = False
        cfg.scene.num_envs = len(names) * args.trials
        cfg.sim.device = args.device
        cfg.episode_length_s = args.duration + 1.0
        cfg.gait_cycle_time_s = actor.phase_period_s
        cfg.command_resampling_time_s = args.duration + 2.0
        cfg.randomize_domain = cfg.observation_noise = args.randomized
        if not args.randomized:
            cfg.action_delay_min = cfg.action_delay_max = 0
            cfg.imu_delay_min = cfg.imu_delay_max = 0
        if cfg.scene.num_envs <= 64:
            # Small evaluations need far fewer contacts than the default 8M-pair buffer.
            # This changes allocation capacity, not collision detection or the solver.
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
        labels = [name for name in names for _ in range(args.trials)]
        commands = torch.tensor(
            [COMMANDS.get(name, (0.0, 0.0, 0.0)) for name in labels], device=env.device
        )
        sequence_ids = [i for i, name in enumerate(labels) if name == "stop_go_stop"]

        def assign_commands(ids):
            env.commands[ids, :3] = commands[ids]
            env.commands[ids, 3:] = 0

        env._resample_commands = assign_commands
        obs, _ = env.reset()
        # Preserve terminal poses. Each trial is the first episode, without respawns.
        env._reset_idx = lambda ids: None
        steps = round(args.duration / env.step_dt)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        first_fall = torch.full(
            (env.num_envs,), steps, device=env.device, dtype=torch.long
        )
        limits = env.robot.root_physx_view.get_dof_max_forces().to(env.device)
        initial_pos = env.robot.data.root_pos_w.clone()
        contacts_previous = env._get_foot_contacts().clone()
        landings = torch.zeros(env.num_envs, 2, device=env.device)
        stable_landings = torch.zeros_like(landings)
        stable_contacts = contacts_previous.clone()
        contact_frames = torch.zeros_like(landings, dtype=torch.long)
        air_frames = torch.zeros_like(contact_frames)
        rows = []
        command_rows = []
        input_rows, action_rows, target_rows = [], [], []
        masses = env.robot.root_physx_view.get_masses().to(env.device)
        head_bodies = env.robot.find_bodies(
            [
                "neck_pitch_assembly",
                "head_pitch_to_yaw",
                "neck_yaw_assembly",
                "head_assembly",
            ],
            preserve_order=True,
        )[0]
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/bin/ffmpeg")
            writer = imageio.get_writer(
                str(args.output / "playback.mp4"), fps=25, codec="libx264", quality=8
            )
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19
            )
            # Visual-only floor markers give a scale reference; no collisions.
            from pxr import UsdGeom, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for i in range(-20, 21):
                for axis in range(2):
                    marker = UsdGeom.Cube.Define(
                        stage, f"/World/EvaluationGrid/g_{axis}_{i + 20}"
                    )
                    marker.CreateSizeAttr(1.0)
                    marker.AddTranslateOp().Set(
                        Gf.Vec3d(
                            i * 0.25 if axis == 0 else 0, i * 0.25 if axis else 0, 0.001
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

        for step in range(steps):
            if sequence_ids:
                if step in (0, round(3 / env.step_dt), round(15 / env.step_dt)):
                    commands[sequence_ids, 0] = (
                        0.1 if 3 <= step * env.step_dt < 15 else 0
                    )
                    env.commands[:, :3] = commands
                    env._obs_stamp = -1
                    obs = env._get_observations()
            with torch.no_grad():
                if args.record_policy_io:
                    input_rows.append(obs["policy"].cpu().numpy().copy())
                actions = actor(obs["policy"])
                if args.record_policy_io:
                    action_rows.append(actions.cpu().numpy().copy())
                obs, reward, terminated, truncated, extras = env.step(actions)
                root = env.robot.data.root_state_w
                velocity = quat_rotate_inverse(
                    root[:, 3:7], env.robot.data.root_lin_vel_w
                )
                angular = quat_rotate_inverse(
                    root[:, 3:7], env.robot.data.root_ang_vel_w
                )
                up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(
                    env.num_envs, -1
                )
                tilt = torch.acos(
                    quat_rotate_inverse(root[:, 3:7], up)[:, 2].clamp(-1, 1)
                )
                contacts = env._get_foot_contacts()
                newly_fallen = active & terminated
                first_fall[newly_fallen] = step + 1
                active &= ~terminated
                landings += ((~contacts_previous) & contacts & active[:, None]).float()
                contacts_previous = contacts.clone()
                contact_frames = torch.where(contacts, contact_frames + 1, 0)
                air_frames = torch.where(~contacts, air_frames + 1, 0)
                new_stable = torch.where(
                    contact_frames >= 3,
                    True,
                    torch.where(air_frames >= 3, False, stable_contacts),
                )
                stable_landings += (
                    (~stable_contacts) & new_stable & active[:, None]
                ).float()
                stable_contacts = new_stable
                saturation = (
                    (env.robot.data.applied_torque.abs() >= 0.95 * limits)
                    .float()
                    .mean(-1)
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
                        (actions.abs() >= 0.99).float().mean(-1)[:, None],
                        active.float()[:, None],
                        stable_contacts.float(),
                        env._feet_clearance(),
                        env.robot.data.joint_pos[:, env.controlled_joint_ids]
                        + env.control.offsets,
                        root[:, 3:7],
                        env.robot.data.joint_vel[:, env.controlled_joint_ids[5:9]],
                        (env.robot.data.body_com_pos_w * masses[:, :, None]).sum(1)
                        / masses.sum(1)[:, None],
                    ),
                    -1,
                )
                if not torch.isfinite(row).all():
                    raise RuntimeError("Non-finite rollout data")
                if args.record_policy_io:
                    target_rows.append(env.control.motor_targets.cpu().numpy().copy())
                rows.append(row.cpu().numpy())
                command_rows.append(commands.cpu().numpy().copy())
            if args.video and (step % 2 == 0 or newly_fallen[0]):
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 85), fill=(15, 22, 28))
                v = velocity[0].cpu().tolist()
                draw.text(
                    (14, 8),
                    f"{names[0]} | t={(step + 1) * env.step_dt:.2f}s | target={tuple(round(x, 3) for x in commands[0].cpu().tolist())}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (14, 36),
                    f"vx={v[0]:+.3f} vy={v[1]:+.3f} m/s | tilt={float(tilt[0]) * 180 / math.pi:.1f} deg"
                    + (" | FALL" if newly_fallen[0] else ""),
                    font=font,
                    fill="white",
                )
                rgb = np.asarray(frame)
                writer.append_data(rgb)
                if step % 100 == 0:
                    frame.save(args.output / f"frame_{step:04d}.png")
                if newly_fallen[0]:
                    frame.save(args.output / "terminal.png")
                    for _ in range(25):
                        writer.append_data(rgb)
                    break
            if not active.any():
                break

        data = np.stack(rows)
        command_trace = np.stack(command_rows)
        io = (
            dict(
                observations=np.stack(input_rows),
                actions=np.stack(action_rows),
                motor_targets=np.stack(target_rows),
            )
            if args.record_policy_io
            else {}
        )
        np.savez_compressed(
            args.output / "rollout.npz",
            **io,
            values=data,
            command_trace=command_trace,
            commands=commands.cpu().numpy(),
            labels=np.array(labels),
            dt=env.step_dt,
            columns=np.array(
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
            ),
        )
        results = []
        falls = first_fall.cpu().numpy()
        for i, name in enumerate(labels):
            valid = data[:, i, 12] > 0.5
            observed = data[valid, i]
            target = command_trace[valid, i]
            if len(observed):
                error = observed[:, :2] - target[:, :2]
                record = dict(
                    xy_rmse_m_s=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    yaw_mae_rad_s=float(np.mean(np.abs(observed[:, 2] - target[:, 2]))),
                    mean_velocity=observed[:, :3].mean(0).tolist(),
                    displacement_xy_m=(
                        observed[-1, 3:5] - initial_pos[i, :2].cpu().numpy()
                    ).tolist(),
                    mean_height_m=float(observed[:, 6].mean()),
                    head_joint_velocity_rms_rad_s=np.sqrt(
                        np.mean(observed[:, 35:39] ** 2, axis=0)
                    ).tolist(),
                    mean_torque_saturation=float(observed[:, 10].mean()),
                    mean_action_clipping=float(observed[:, 11].mean()),
                )
            else:
                record = dict(xy_rmse_m_s=None, yaw_mae_rad_s=None)
            record.update(
                command=name,
                trial=i % args.trials,
                survived=bool(falls[i] == steps and active[i]),
                survival_s=float(falls[i] * env.step_dt),
                foot_landings=landings[i].cpu().tolist(),
                stable_foot_landings=stable_landings[i].cpu().tolist(),
            )
            results.append(record)
        summary = {}
        for name in names:
            group = [r for r in results if r["command"] == name]
            summary[name] = {
                "survival_rate": float(np.mean([r["survived"] for r in group])),
                "mean_survival_s": float(np.mean([r["survival_s"] for r in group])),
                "xy_rmse_m_s": float(
                    np.mean(
                        [
                            r["xy_rmse_m_s"]
                            for r in group
                            if r["xy_rmse_m_s"] is not None
                        ]
                    )
                ),
                "yaw_mae_rad_s": float(
                    np.mean(
                        [
                            r["yaw_mae_rad_s"]
                            for r in group
                            if r["yaw_mae_rad_s"] is not None
                        ]
                    )
                ),
                "mean_velocity": np.mean(
                    [r["mean_velocity"] for r in group if "mean_velocity" in r], axis=0
                ).tolist(),
            }
        phases = {}
        if sequence_ids:
            time = (np.arange(len(data)) + 1) * env.step_dt
            for label, lo, hi in [
                ("initial_stop", 1, 3),
                ("forward", 5, 15),
                ("final_stop", 17, args.duration),
            ]:
                phase_mask = (time >= lo) & (time < hi)
                selected = data[phase_mask][:, sequence_ids]
                valid = selected[:, :, 12] > 0.5
                v = selected[valid]
                phases[label] = (
                    dict(
                        mean_velocity=v[:, :3].mean(0).tolist(),
                        planar_speed_mean_m_s=float(
                            np.linalg.norm(v[:, :2], axis=1).mean()
                        ),
                        yaw_mae_rad_s=float(np.abs(v[:, 2]).mean()),
                    )
                    if len(v)
                    else None
                )
        report = dict(
            status="completed",
            checkpoint=str(args.checkpoint.resolve()),
            contract_hash=contract_hash(),
            head_control_mode=cfg.head_control_mode,
            heading_hold_kp=cfg.heading_hold_kp,
            heading_hold_max_rate=cfg.heading_hold_max_rate,
            heading_hold_filter_tau=cfg.heading_hold_filter_tau,
            heading_imu_noise_rad=cfg.heading_imu_noise if args.randomized else 0.0,
            lateral_command_trim=cfg.lateral_command_trim,
            forward_command_scale=cfg.forward_command_scale,
            heading_sensor="virtual trunk IMU attitude; ideal nominal, biased/noisy/delayed when randomized"
            if cfg.heading_hold_kp
            else None,
            command_sequence="stop_go_stop" if sequence_ids else None,
            phase_metrics=phases,
            total_mass_mean_kg=float(masses.sum(1).mean()),
            head_chain_mass_mean_kg=float(masses[:, head_bodies].sum(1).mean()),
            target_filter_tau_s=cfg.target_filter_tau_s,
            seed=args.eval_seed,
            randomized=args.randomized,
            duration_s=args.duration,
            dt=env.step_dt,
            spawn_height_m=cfg.spawn_height,
            target_height_m=cfg.target_base_height,
            self_collisions=not args.disable_self_collisions,
            mechanical_collision_filter_distance=cfg.mechanical_collision_filter_distance,
            trials_per_command=args.trials,
            summary=summary,
            trials=results,
            protocol="Fixed commands, first episode only, no respawns; errors use samples before termination.",
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
