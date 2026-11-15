"""Isaac Lab squat/stand residual learning over checked full-mass v2 IK targets."""

from __future__ import annotations

import math
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from isaaclab.utils import configclass

from open_duck_complex_terrain.open_duck_env import OpenDuckMiniEnv
from open_duck_complex_terrain.open_duck_env_cfg import (
    OpenDuckStraightV6EnvCfg,
    PROJECT_ROOT,
)
from open_duck_v2.control import quat_rotate_inverse
from open_duck_v2.profile import JOINT_NAMES, URDF_TO_CANONICAL
from open_duck_v2.posture import (
    PostureReference,
    PostureControl,
    posture_command,
    posture_phase,
    POSTURE_OBS_DIM,
    POSTURE_CRITIC_DIM,
)


@configclass
class OpenDuckPostureEnvCfg(OpenDuckStraightV6EnvCfg):
    motion_skill: str = "posture"
    num_observations: int = POSTURE_OBS_DIM
    num_states: int = POSTURE_CRITIC_DIM
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(POSTURE_OBS_DIM,), dtype=np.float32
    )
    state_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(POSTURE_CRITIC_DIM,), dtype=np.float32
    )
    posture_reference_file: str = str(PROJECT_ROOT / "config/posture_reference_v1.json")
    posture_residual_scale: float = 0.12
    episode_length_s: float = 18.0
    gait_cycle_time_s: float = 16.0
    critic_context: str = "posture"
    command_lin_vel_x: tuple[float, float] = (0.0, 0.0)
    command_lin_vel_y: tuple[float, float] = (0.0, 0.0)
    command_yaw_rate: tuple[float, float] = (0.0, 0.0)
    stand_command_probability: float = 1.0
    reset_root_ang_vel_range: float = 0.05
    randomize_domain: bool = False
    observation_noise: bool = False
    action_delay_min: int = 0
    action_delay_max: int = 0
    imu_delay_min: int = 0
    imu_delay_max: int = 0
    heading_hold_kp: float = 0.0
    forward_command_scale: float = 1.0


class OpenDuckPostureEnv(OpenDuckMiniEnv):
    cfg: OpenDuckPostureEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.reference = PostureReference(cfg.posture_reference_file, cfg.sim.device)
        cfg.spawn_height = self.reference.stand_height + 0.003
        cfg.robot.init_state.pos = (0, 0, cfg.spawn_height)
        cfg.robot.init_state.rot = tuple(
            float(v)
            for v in Rotation.from_euler(
                "xyz", self.reference.rpy[-1].cpu().numpy()
            ).as_quat(scalar_first=True)
        )
        cfg.robot.init_state.joint_pos = {
            n: float(q - o)
            for n, q, o in zip(
                JOINT_NAMES,
                self.reference.stand_joints.cpu().tolist(),
                URDF_TO_CANONICAL,
            )
        }
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**17
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        super().__init__(cfg, render_mode, **kwargs)
        self.control = PostureControl(
            self.num_envs,
            self.device,
            self.reference,
            cfg.posture_residual_scale,
            self.step_dt,
            cfg.backlash_half_width,
        )
        self.actions = self.control.actions
        self.pose_time = torch.zeros(self.num_envs, device=self.device)
        self.pose_low_height = torch.full_like(
            self.pose_time, self.reference.low_height
        )
        self.pose_height = torch.full_like(self.pose_time, self.reference.stand_height)
        self.pose_velocity = torch.zeros_like(self.pose_time)
        self.external_posture_command = False
        self.reference_joints = self.reference.stand_joints.expand(
            self.num_envs, -1
        ).clone()
        self.reference_rpy = self.reference.rpy[-1].expand(self.num_envs, -1).clone()
        self.reference_com = self.reference.com[-1].expand(self.num_envs, -1).clone()

    def set_posture_command(self, height, velocity):
        self.external_posture_command = True
        self.pose_height.copy_(
            torch.as_tensor(height, device=self.device).expand(self.num_envs)
        )
        self.pose_velocity.copy_(
            torch.as_tensor(velocity, device=self.device).expand(self.num_envs)
        )
        self._update_reference()
        self._obs_stamp = -1

    def _update_reference(self):
        self.reference_joints, self.reference_rpy, _, self.reference_com = (
            self.reference.sample(self.pose_height)
        )
        self.control.reference_target = self.reference_joints

    def _pre_physics_step(self, actions):
        if not self.external_posture_command:
            self.pose_height, self.pose_velocity = posture_command(
                self.pose_time, self.reference.stand_height, self.pose_low_height
            )
        self._update_reference()
        self.previous_actions = self.actions.clone()
        self._native_targets = self.control.process(actions)
        self.actions = self.control.actions
        self.pose_time += self.step_dt

    def _phase(self):
        return posture_phase(self.pose_height)

    def _get_observations(self):
        base = super()._get_observations()
        policy = torch.cat(
            (base["policy"], self.pose_height[:, None], self.pose_velocity[:, None]), -1
        )
        q = self.robot.data.root_quat_w
        velocity = quat_rotate_inverse(q, self.robot.data.root_lin_vel_w)
        up = quat_rotate_inverse(
            q, q.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1)
        )
        masses = self.robot.root_physx_view.get_masses().to(self.device)
        com = (self.robot.data.body_com_pos_w * masses[:, :, None]).sum(1) / masses.sum(
            1
        )[:, None]
        feet = self.robot.data.body_pos_w[:, self.foot_body_ids, :2].mean(1)
        critic = torch.cat(
            (
                policy,
                velocity,
                self._base_height()[:, None],
                up,
                com[:, :2] - feet,
                self.reference_joints,
            ),
            -1,
        )
        if (
            policy.shape[-1] != POSTURE_OBS_DIM
            or critic.shape[-1] != POSTURE_CRITIC_DIM
        ):
            raise RuntimeError("Posture observation contract mismatch")
        return {"policy": policy, "critic": critic}

    def _get_rewards(self):
        height = self._base_height()
        q = self.robot.data.root_quat_w
        up = quat_rotate_inverse(
            q, q.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1)
        )
        roll, pitch = self.reference_rpy[:, 0], self.reference_rpy[:, 1]
        target_up = torch.stack(
            (
                -torch.sin(pitch),
                torch.sin(roll) * torch.cos(pitch),
                torch.cos(roll) * torch.cos(pitch),
            ),
            -1,
        )
        velocity = quat_rotate_inverse(q, self.robot.data.root_lin_vel_w)
        joints = (
            self.robot.data.joint_pos[:, self.controlled_joint_ids]
            + self.control.offsets
        )
        contacts = self._get_foot_contacts()
        foot_velocity = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        slip = (foot_velocity.square().sum(-1) * contacts).sum(-1).clamp_max(0.5)
        masses = self.robot.root_physx_view.get_masses().to(self.device)
        com = (self.robot.data.body_com_pos_w * masses[:, :, None]).sum(1) / masses.sum(
            1
        )[:, None]
        target_com = self.reference_com[:, :2] + self.scene.env_origins[:, :2]
        h_reward = torch.exp(-((height - self.pose_height) / 0.008).square())
        orientation_reward = torch.exp(-(up - target_up).square().sum(-1) / 0.15**2)
        pose_reward = torch.exp(
            -(joints - self.reference_joints).square().mean(-1) / 0.12**2
        )
        com_error = ((com[:, :2] - target_com) / 0.025).square().sum(-1).clamp_max(4)
        planar = ((velocity[:, :2] / 0.06).square().sum(-1)).clamp_max(4)
        vertical = (
            ((self.robot.data.root_lin_vel_w[:, 2] - self.pose_velocity) / 0.05)
            .square()
            .clamp_max(4)
        )
        falling = (
            (height < self.cfg.termination_height)
            | (up[:, 2] < math.cos(self.cfg.max_root_angle))
        ).float()
        reward = (
            4 * h_reward
            + 2 * orientation_reward
            + pose_reward
            + contacts.float().mean(-1)
            - 0.5 * com_error
            - 0.4 * planar
            - 0.2 * vertical
            - 10 * slip
            - 0.3 * self.actions.square().mean(-1)
            - 0.05 * (self.actions - self.previous_actions).square().mean(-1)
            - 0.0001 * self.robot.data.applied_torque.square().sum(-1)
            - 12 * falling
        )
        self.extras["log"] = {
            "Reward/height": (4 * h_reward).mean(),
            "Reward/orientation": (2 * orientation_reward).mean(),
            "Reward/pose": pose_reward.mean(),
            "Stats/base_height": height.mean(),
            "Stats/target_height": self.pose_height.mean(),
            "Stats/height_error_abs_m": (height - self.pose_height).abs().mean(),
            "Stats/both_feet_contact": contacts.all(-1).float().mean(),
            "Stats/tilt_error_rad": torch.acos(
                (up * target_up).sum(-1).clamp(-1, 1)
            ).mean(),
            "Stats/residual_rms": self.actions.square().mean().sqrt(),
        }
        return reward * self.step_dt

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device).long()
        super()._reset_idx(env_ids)
        self.pose_time[env_ids] = 0
        self.pose_low_height[env_ids] = (
            self.reference.low_height
            + torch.rand(len(env_ids), device=self.device) * 0.010
        )
        self.pose_height[env_ids] = self.reference.stand_height
        self.pose_velocity[env_ids] = 0
        self.reference_joints[env_ids] = self.reference.stand_joints
        self.reference_rpy[env_ids] = self.reference.rpy[-1]
        self.reference_com[env_ids] = self.reference.com[-1]
        self.control.reference_target[env_ids] = self.reference.stand_joints
