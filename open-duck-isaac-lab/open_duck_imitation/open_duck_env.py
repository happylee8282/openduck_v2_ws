# Open Duck Mini motion-imitation direct RL environment.

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import torch

from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv, quat_rotate_inverse

from .open_duck_env_cfg import OpenDuckImitationEnvCfg


class OpenDuckImitationEnv(OpenDuckMiniEnv):
    cfg: OpenDuckImitationEnvCfg

    def __init__(
        self, cfg: OpenDuckImitationEnvCfg, render_mode: str | None = None, **kwargs
    ):
        self._motion_npz_path = Path(cfg.motion_file)
        self._motion_arrays: dict[str, np.ndarray] | None = None
        self.motion_joint_pos: torch.Tensor | None = None
        self.motion_joint_vel: torch.Tensor | None = None
        self.motion_root_height: torch.Tensor | None = None
        self.motion_root_lin_vel: torch.Tensor | None = None
        self.motion_start_steps: torch.Tensor | None = None
        from open_duck_v2.motion import load_motion

        self._motion_arrays = load_motion(
            self._motion_npz_path, 1.0 / (cfg.sim.dt * cfg.decimation)
        )
        cfg.gait_cycle_time_s = (
            len(self._motion_arrays["joint_pos"]) * cfg.sim.dt * cfg.decimation
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_motion_library()

    def _ensure_motion_library(self) -> None:
        if self.motion_joint_pos is not None:
            return
        joint_pos = torch.as_tensor(
            self._motion_arrays["joint_pos"], device=self.device
        )
        joint_vel = torch.as_tensor(
            self._motion_arrays["joint_vel"], device=self.device
        )
        root_height = torch.as_tensor(
            self._motion_arrays["root_height"], device=self.device
        )
        root_lin_vel = torch.as_tensor(
            self._motion_arrays["root_lin_vel"], device=self.device
        )
        if joint_pos.shape[1] != self.cfg.num_actions:
            raise RuntimeError(
                f"Motion has {joint_pos.shape[1]} joints, expected {self.cfg.num_actions}"
            )
        self.motion_joint_pos = joint_pos
        self.motion_joint_vel = joint_vel
        self.motion_root_height = root_height
        self.motion_root_lin_vel = root_lin_vel
        self.motion_start_steps = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

    def _motion_indices(self) -> torch.Tensor:
        self._ensure_motion_library()
        assert self.motion_joint_pos is not None
        assert self.motion_start_steps is not None
        frame_count = self.motion_joint_pos.shape[0]
        indices = self.motion_start_steps + self.episode_length_buf.to(dtype=torch.long)
        if self.cfg.motion_loop:
            return torch.remainder(indices, frame_count)
        return torch.clamp(indices, max=frame_count - 1)

    def _reference(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_motion_library()
        assert self.motion_joint_pos is not None
        assert self.motion_joint_vel is not None
        assert self.motion_root_height is not None
        assert self.motion_root_lin_vel is not None
        indices = self._motion_indices()
        phase = indices.float() / float(max(self.motion_joint_pos.shape[0] - 1, 1))
        return (
            self.motion_joint_pos[indices],
            self.motion_joint_vel[indices],
            self.motion_root_height[indices],
            self.motion_root_lin_vel[indices],
            phase.unsqueeze(1),
        )

    def _phase(self):
        self._ensure_motion_library()
        phase = (
            self._motion_indices().float() / len(self.motion_joint_pos) * 2 * math.pi
        )
        return torch.stack((torch.cos(phase), torch.sin(phase)), -1)

    def _get_observations(self):
        # Reference is a periodic teacher signal. Actor remains the hardware 101 contract.
        return super()._get_observations()

    def _get_rewards(self) -> torch.Tensor:
        ref_joint_pos, ref_joint_vel, ref_height, ref_lin_vel_b, phase = (
            self._reference()
        )
        root_quat = self.robot.data.root_state_w[:, 3:7]
        root_lin_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )
        projected_gravity = quat_rotate_inverse(root_quat, up_vec)

        joint_pos = (
            self.robot.data.joint_pos[:, self.controlled_joint_ids]
            + self.control.offsets
        )
        joint_vel = self.robot.data.joint_vel[:, self.controlled_joint_ids]
        pos_error = torch.mean(torch.square(joint_pos - ref_joint_pos), dim=1)
        vel_error = torch.mean(torch.square(joint_vel - ref_joint_vel), dim=1)
        height_error = torch.square(self._base_height() - ref_height)
        lin_vel_error = torch.sum(torch.square(root_lin_vel_b - ref_lin_vel_b), dim=1)

        joint_pos_reward = torch.exp(-pos_error / self.cfg.imitation_joint_sigma)
        joint_vel_reward = torch.exp(-vel_error / self.cfg.imitation_joint_vel_sigma)
        height_reward = torch.exp(-height_error / self.cfg.imitation_height_sigma)
        lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.imitation_lin_vel_sigma)
        upright_reward = torch.clamp(projected_gravity[:, 2], min=0.0) ** 2

        energy_penalty = self.cfg.energy_penalty_scale * torch.sum(
            torch.square(self.robot.data.applied_torque), dim=1
        )
        action_rate_penalty = torch.zeros(self.num_envs, device=self.device)
        if self.actions is not None and self.previous_actions is not None:
            action_rate_penalty = self.cfg.action_rate_penalty_scale * torch.sum(
                torch.square(self.actions - self.previous_actions), dim=1
            )
        joint_deviation = torch.sum(
            torch.square(self.robot.data.joint_pos - self.robot.data.default_joint_pos),
            dim=1,
        )
        joint_deviation_penalty = (
            self.cfg.joint_deviation_penalty_scale * joint_deviation
        )
        fall_penalty = (
            (self._base_height() < self.cfg.termination_height)
            | (projected_gravity[:, 2] < math.cos(self.cfg.max_root_angle))
        ).float()

        reward = (
            self.cfg.rew_scale_imitation_joint_pos * joint_pos_reward
            + self.cfg.rew_scale_imitation_joint_vel * joint_vel_reward
            + self.cfg.rew_scale_imitation_height * height_reward
            + self.cfg.rew_scale_imitation_lin_vel * lin_vel_reward
            + self.cfg.rew_scale_upright * upright_reward
            + self.cfg.rew_scale_alive
            - energy_penalty
            - action_rate_penalty
            - joint_deviation_penalty
            - self.cfg.fall_penalty_scale * fall_penalty
        )

        self.extras["log"] = {
            "Reward/imitation_joint_pos": (
                self.cfg.rew_scale_imitation_joint_pos * joint_pos_reward
            ).mean(),
            "Reward/imitation_joint_vel": (
                self.cfg.rew_scale_imitation_joint_vel * joint_vel_reward
            ).mean(),
            "Reward/imitation_height": (
                self.cfg.rew_scale_imitation_height * height_reward
            ).mean(),
            "Reward/imitation_lin_vel": (
                self.cfg.rew_scale_imitation_lin_vel * lin_vel_reward
            ).mean(),
            "Reward/upright": (self.cfg.rew_scale_upright * upright_reward).mean(),
            "Penalty/energy": energy_penalty.mean(),
            "Penalty/action_rate": action_rate_penalty.mean(),
            "Stats/motion_phase": phase.mean(),
            "Stats/ref_height": ref_height.mean(),
        }
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._ensure_motion_library()
        if self.motion_start_steps is None or self.motion_joint_pos is None:
            return
        if self.cfg.random_motion_start and self.motion_joint_pos.shape[0] > 1:
            self.motion_start_steps[env_ids] = torch.randint(
                0,
                self.motion_joint_pos.shape[0],
                (env_ids.numel(),),
                device=self.device,
            )
        else:
            self.motion_start_steps[env_ids] = 0

        # Align reset state and commanded target to the chosen reference frame.
        indices = self._motion_indices()[env_ids]
        q = self.motion_joint_pos[indices]
        vel = self.motion_joint_vel[indices]
        native = self.robot.data.default_joint_pos[env_ids].clone()
        native[:, self.controlled_joint_ids] = q - self.control.offsets
        speed = torch.zeros_like(native)
        speed[:, self.controlled_joint_ids] = vel
        self.robot.write_joint_state_to_sim(native, speed, env_ids=env_ids)
        self.control.motor_targets[env_ids] = q
        self.control.actuator_targets[env_ids] = q
        self._obs_stamp = self._imu_stamp = -1
