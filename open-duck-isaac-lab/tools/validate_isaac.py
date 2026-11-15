#!/usr/bin/env python3
"""Actual USD/PhysX validation. Reports a missing GPU as blocked, never passed."""

import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--output", type=Path, default=ROOT / "validation/isaac.json")
    known, _ = pre.parse_known_args()
    known.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "validation": "actual Isaac Sim / USD / PhysX",
        "status": "blocked",
        "policy_performance": "not_run",
        "hardware_walk": "not_run",
    }
    result["versions"] = {}
    for package in ("torch", "isaacsim", "rsl-rl-lib", "gymnasium"):
        try:
            result["versions"][package] = version(package)
        except PackageNotFoundError:
            result["versions"][package] = None
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if gpu.returncode:
            result["reason"] = gpu.stderr.strip() or gpu.stdout.strip()
            known.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return 2
        result["gpu"] = gpu.stdout.strip()
        result.update(status="running", stage="starting_simulation")
        known.output.write_text(json.dumps(result, indent=2) + "\n")
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
        parser.add_argument("--steps", type=int, default=250)
        parser.add_argument(
            "--task", choices=["flat", "rough", "imitation"], default="flat"
        )
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        app = AppLauncher(args).app
        try:
            import torch
            from pxr import UsdGeom
            import omni.usd
            from open_duck_complex_terrain.open_duck_env_cfg import OpenDuckEnvCfg
            from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv
            from open_duck_v2.profile import HOME

            cfg = OpenDuckEnvCfg()
            env_class = OpenDuckMiniEnv
            if args.task == "rough":
                from open_duck_complex_terrain.open_duck_env_cfg import (
                    OpenDuckRoughEnvCfg,
                )

                cfg = OpenDuckRoughEnvCfg()
            if args.task == "imitation":
                from open_duck_imitation.open_duck_env_cfg import (
                    OpenDuckImitationEnvCfg,
                )
                from open_duck_imitation.open_duck_env import OpenDuckImitationEnv

                cfg = OpenDuckImitationEnvCfg()
                env_class = OpenDuckImitationEnv
                cfg.random_motion_start = False
            cfg.scene.num_envs = 1
            cfg.randomize_domain = False
            cfg.observation_noise = False
            cfg.action_delay_min = cfg.action_delay_max = cfg.imu_delay_min = (
                cfg.imu_delay_max
            ) = 0
            cfg.validate()
            env = env_class(cfg)
            try:
                obs, _ = env.reset()
                assert obs["policy"].shape == (1, 101) and obs["critic"].shape == (
                    1,
                    222,
                )
                stage = omni.usd.get_context().get_stage()
                assert abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-9
                mass = env.robot.root_physx_view.get_masses().sum().item()
                assert abs(mass - 2.107140705) < 1e-4, f"USD mass changed: {mass}"
                model_q = (
                    env.robot.data.joint_pos[:, env.controlled_joint_ids]
                    + env.control.offsets
                )
                reference = (
                    torch.tensor([HOME], device=env.device)
                    if args.task != "imitation"
                    else env.motion_joint_pos[:1]
                )
                torch.testing.assert_close(model_q, reference, atol=1e-4, rtol=1e-4)
                assert torch.allclose(
                    env.robot.root_physx_view.get_dof_max_forces().cpu(),
                    torch.full((1, 14), 3.35),
                )
                for _ in range(args.steps):
                    obs, reward, terminated, truncated, _ = env.step(
                        torch.zeros(1, 14, device=env.device)
                    )
                    assert (
                        all(torch.isfinite(v).all() for v in obs.values())
                        and torch.isfinite(reward).all()
                    )
                cfg.randomize_domain = True
                env.reset()  # real writers and strict PhysX readback checks run here
                result.update(
                    status="passed",
                    smoke_steps=args.steps,
                    usd_mass_kg=mass,
                    task=args.task,
                    stage="completed",
                    policy_shape=[1, 101],
                    critic_shape=[1, 222],
                    randomization_resets=env.randomizer.applied_resets,
                    note="Smoke and readback passed; no learned policy or hardware performance claim",
                )
            finally:
                env.close()
        except Exception as exc:
            result.update(
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        finally:
            known.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            app.close()
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    known.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
