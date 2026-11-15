#!/usr/bin/env python3
"""Interactive W/S walking and C/R squat/stand with preserved skill policies."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "config/multiskill_policy.json"
    )
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--report", type=Path)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from open_duck_v2.runtime import load_rsl_rl_actor, environment_cfg_for_actor
        from open_duck_v2.posture_runtime import load_posture_actor
        from open_duck_v2.posture import PostureReference
        from open_duck_v2.multiskill import load_manifest, MultiSkillRuntime
        from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv
        from isaac_duck_teleop.scripts.play_open_duck_teleop import (
            TeleopInput,
            CommandLimits,
        )
        from evaluate_multiskill import scenario_commands

        data = load_manifest(args.bundle)
        f = load_rsl_rl_actor(
            data["skills"]["forward"]["resolved_checkpoint"], args.device
        )
        b = load_rsl_rl_actor(
            data["skills"]["backward"]["resolved_checkpoint"], args.device
        )
        pose = load_posture_actor(
            data["skills"]["posture"]["resolved_checkpoint"],
            data["reference_path"],
            args.device,
        )
        reference = PostureReference(data["reference_path"], args.device)
        cfg = environment_cfg_for_actor(f)
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.randomize_domain = cfg.observation_noise = False
        cfg.action_delay_min = cfg.action_delay_max = cfg.imu_delay_min = (
            cfg.imu_delay_max
        ) = 0
        cfg.command_resampling_time_s = 1e9
        cfg.episode_length_s = 3600
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (0.7, -0.9, 0.45)
        cfg.viewer.lookat = (0, 0, 0.02)
        env = OpenDuckMiniEnv(cfg, render_mode="human")
        env._resample_commands = lambda ids: None
        env.reset()
        env._reset_idx = lambda ids: None
        runtime = MultiSkillRuntime(env, f, b, pose, reference)
        teleop = TeleopInput(
            CommandLimits(vx_min=-0.1, vx_max=0.1, vy_max=0, yaw_max=0),
            False,
            use_keyboard=not args.headless,
        )
        print(
            "W forward | S backward | release/Space stop | C low pose | R stand | Esc quit",
            flush=True,
        )
        bound = args.steps if args.steps > 0 else (2600 if args.demo else 0)
        low = False
        steps = 0
        falls = 0
        while (
            app.is_running()
            and not teleop.should_quit()
            and (not bound or steps < bound)
        ):
            tick = time.perf_counter()
            if args.demo:
                vx, low = scenario_commands("demo", steps * 0.02)
                command = torch.tensor(
                    [[vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device
                )
            else:
                command = teleop.update().to(env.device).view(1, 7)
                if "C" in teleop._keys:
                    low = True
                if "R" in teleop._keys or command[0, :2].norm() > 0.02:
                    low = False
            with torch.no_grad():
                obs, reward, done, truncated, extra = runtime.step(
                    command, torch.tensor([low], device=env.device)
                )
            steps += 1
            if done.any():
                falls += int(done.sum())
                print("Playback ended after loss of balance", flush=True)
                break
            if not args.headless:
                time.sleep(max(0, 0.02 - (time.perf_counter() - tick)))
        report = dict(
            status="passed" if not falls else "failed",
            steps=steps,
            falls=falls,
            bundle=str(args.bundle.resolve()),
            forward_sha256=data["skills"]["forward"]["sha256"],
            note="Execution-path check; see independent multiskill assessments for quality criteria.",
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
