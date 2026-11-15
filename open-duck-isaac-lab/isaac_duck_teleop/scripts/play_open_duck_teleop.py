from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _remove_argparse_dest(parser: argparse.ArgumentParser, dest: str) -> None:
    for action in list(parser._actions):
        if action.dest != dest:
            continue
        parser._remove_action(action)
        for option in action.option_strings:
            parser._option_string_actions.pop(option, None)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate Open Duck Mini policy in Isaac Lab."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--usd-path", type=Path, required=True)
    parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    parser.add_argument("--gamepad", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 runs until closed; positive bounds headless validation",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vx-max", type=float, default=0.15)
    parser.add_argument("--vx-min", type=float, default=-0.15)
    parser.add_argument("--vy-max", type=float, default=0.20)
    parser.add_argument("--yaw-max", type=float, default=1.0)
    _remove_argparse_dest(parser, "device")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


@dataclass
class CommandLimits:
    vx_min: float
    vx_max: float
    vy_max: float
    yaw_max: float


class TeleopInput:
    def __init__(
        self, limits: CommandLimits, use_gamepad: bool, use_keyboard: bool = True
    ):
        self.limits = limits
        self.command = torch.tensor(
            [0.0] * 7,
            dtype=torch.float32,
        )
        self._keys: set[str] = set()
        self._keyboard_sub = None
        self._pygame = None
        self._joystick = None
        if use_keyboard:
            self._init_keyboard()
        if use_gamepad:
            self._init_gamepad()

    def _init_keyboard(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            input_iface = carb.input.acquire_input_interface()
            self._keyboard_sub = input_iface.subscribe_to_keyboard_events(
                keyboard, self._on_keyboard_event
            )
        except Exception as exc:
            print(f"[WARN] keyboard subscription failed: {exc}")

    def _on_keyboard_event(self, event) -> bool:
        try:
            import carb.input

            key_name = str(event.input).split(".")[-1].upper()
            if event.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                self._keys.add(key_name)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._keys.discard(key_name)
        except Exception:
            return True
        return True

    def _init_gamepad(self) -> None:
        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() < 1:
                print("[WARN] --gamepad was set, but no joystick was detected.")
                return
            self._pygame = pygame
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            print(f"[INFO] gamepad connected: {self._joystick.get_name()}")
        except Exception as exc:
            print(f"[WARN] gamepad init failed: {exc}")

    def should_quit(self) -> bool:
        return "ESCAPE" in self._keys or "ESC" in self._keys

    def update(self) -> torch.Tensor:
        vx = 0.0
        vy = 0.0
        yaw = 0.0

        if "W" in self._keys:
            vx += self.limits.vx_max
        if "S" in self._keys:
            vx += self.limits.vx_min
        if "A" in self._keys:
            vy += self.limits.vy_max
        if "D" in self._keys:
            vy -= self.limits.vy_max
        if "Q" in self._keys:
            yaw += self.limits.yaw_max
        if "E" in self._keys:
            yaw -= self.limits.yaw_max
        if "SPACE" in self._keys:
            vx = 0.0
            vy = 0.0
            yaw = 0.0

        if self._pygame is not None and self._joystick is not None:
            self._pygame.event.pump()
            axis_lx = self._deadzone(self._joystick.get_axis(0))
            axis_ly = self._deadzone(self._joystick.get_axis(1))
            axis_rx = self._deadzone(self._joystick.get_axis(2))
            vx = (
                max(vx, -axis_ly * self.limits.vx_max)
                if axis_ly < 0.0
                else min(vx, -axis_ly * abs(self.limits.vx_min))
            )
            vy = -axis_lx * self.limits.vy_max
            yaw = -axis_rx * self.limits.yaw_max

        self.command = torch.tensor(
            [vx, vy, yaw, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        return self.command

    @staticmethod
    def _deadzone(value: float, threshold: float = 0.08) -> float:
        return 0.0 if abs(value) < threshold else float(value)


def _obs_with_command(obs: dict, command: torch.Tensor, device: str) -> dict:
    policy_obs = obs["policy"].clone()
    policy_obs[:, 6:13] = command.to(device).view(1, 7)
    return {**obs, "policy": policy_obs}


def main() -> None:
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from duck_policy_runtime import load_rsl_rl_actor
    from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.usd_path.exists():
        raise FileNotFoundError(args.usd_path)

    from open_duck_v2.runtime import environment_cfg_for_actor

    actor = load_rsl_rl_actor(
        args.checkpoint, device=args.device, obs_dim=101, action_dim=14
    )
    cfg = environment_cfg_for_actor(actor)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = args.device
    cfg.robot.spawn.usd_path = str(args.usd_path)
    cfg.use_rough_terrain = args.terrain == "rough"
    cfg.enable_height_scan = args.terrain == "rough"
    cfg.randomize_domain = False
    cfg.observation_noise = False
    cfg.action_delay_min = cfg.action_delay_max = 0
    cfg.imu_delay_min = cfg.imu_delay_max = 0
    cfg.command_resampling_time_s = 1.0e9
    cfg.episode_length_s = 3600.0

    env = OpenDuckMiniEnv(cfg, render_mode="human")
    env._resample_commands = lambda env_ids: None
    env.command_resampling_steps = 10**9

    limits = CommandLimits(
        vx_min=args.vx_min,
        vx_max=args.vx_max,
        vy_max=args.vy_max,
        yaw_max=args.yaw_max,
    )
    if cfg.reward_profile.startswith("straight_"):
        limits = CommandLimits(
            vx_min=0.0, vx_max=min(0.10, args.vx_max), vy_max=0.0, yaw_max=0.0
        )
        print(
            "[INFO] Straight policy: W/forward axis = 0.10 m/s, release/Space = stop."
        )
    teleop = TeleopInput(
        limits, use_gamepad=args.gamepad, use_keyboard=not args.headless
    )

    obs, _ = env.reset()

    step_count = 0
    falls = 0
    while (
        simulation_app.is_running()
        and not teleop.should_quit()
        and (args.steps <= 0 or step_count < args.steps)
    ):
        command = teleop.update()
        env.commands[:] = command.to(env.device).view(1, 7)
        obs = _obs_with_command(obs, env._effective_commands()[0], env.device)

        # PhysX state caches must remain mutable during subsequent reset calls.
        with torch.no_grad():
            actions = torch.clamp(actor(obs["policy"].to(args.device)), -1.0, 1.0)
            obs, _, dones, _, _ = env.step(actions)

        if torch.any(dones):
            falls += int(torch.count_nonzero(dones))
            obs, _ = env.reset()

        step_count += 1
        if step_count % 30 == 0:
            vx, vy, yaw = command[:3].tolist()
            if args.debug:
                print(f"[CMD] vx={vx:+.3f} vy={vy:+.3f} yaw={yaw:+.3f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "steps": step_count,
                    "falls": falls,
                    "checkpoint": str(args.checkpoint),
                    "note": "Bounded playback smoke, not policy quality certification",
                },
                indent=2,
            )
        )
    print(f"Completed {step_count} policy playback steps", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
