# Open Duck Mini Isaac Lab direct RL environment.

from __future__ import annotations

import math
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshRandomGridTerrainCfg
from isaaclab.utils.math import sample_uniform

from .open_duck_env_cfg import OpenDuckEnvCfg

class OpenDuckMiniEnv(DirectRLEnv):
    cfg: OpenDuckEnvCfg

    def __init__(self, cfg: OpenDuckEnvCfg, render_mode: str | None = None, **kwargs):
        self.actions: torch.Tensor | None = None
        self.previous_actions: torch.Tensor | None = None
        self.commands: torch.Tensor | None = None
        self.command_resampling_steps = 1
        self.controlled_joint_ids: torch.Tensor | None = None
        self.controlled_action_scales: torch.Tensor | None = None
        self.foot_body_ids: torch.Tensor | None = None
        self.feet_air_time: torch.Tensor | None = None
        self.terrain = None

        super().__init__(cfg, render_mode, **kwargs)

        controlled_joint_names, self.controlled_joint_ids = self._resolve_controlled_joint_ids()
        self.controlled_action_scales = self._make_controlled_action_scales(controlled_joint_names)
        self.foot_body_ids = self._resolve_foot_body_ids()
        actuator_names = list(getattr(self.robot, "actuators", {}).keys())
        print(
            "[OpenDuckMiniEnv] controlled joints="
            f"{controlled_joint_names}, action_scales={self.controlled_action_scales.tolist()}, "
            f"num_actions={self.cfg.num_actions}, robot_instances={self.robot.data.default_root_state.shape[0]}, "
            f"envs={self.num_envs}, actuators={actuator_names}"
        )
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device)
        self.command_resampling_steps = max(1, int(self.cfg.command_resampling_time_s / (self.cfg.sim.dt * self.cfg.decimation)))
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        usd_path = Path(self.cfg.robot.spawn.usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Training USD not found: {usd_path}. Run tools/convert_urdf_to_training_usd.py first."
            )

        self.robot = Articulation(self.cfg.robot)

        if self.cfg.use_rough_terrain:
            self._setup_terrains()
        else:
            ground_cfg = GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            )
            spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.feet_contact_sensor = ContactSensor(self.cfg.feet_contact)
        self.scene.sensors["feet_contact"] = self.feet_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_terrains(self):
        root_spawn_patch = FlatPatchSamplingCfg(
            num_patches=64,
            patch_radius=0.35,
            max_height_diff=0.02,
            x_range=(-1e6, 1e6),
            y_range=(-1e6, 1e6),
            z_range=(-1e6, 1e6)
        )
        terrain_gen_cfg = TerrainGeneratorCfg(
            size=(5.0, 5.0),
            border_width=0.5,
            num_rows=1,
            num_cols=2,
            horizontal_scale=0.08,
            vertical_scale=0.005,
            slope_threshold=0.755,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                "packed_sand": MeshRandomGridTerrainCfg(
                    proportion=0.5,
                    grid_width=0.35,
                    grid_height_range=(0.004, 0.012),
                    platform_width=1.5,
                    flat_patch_sampling={
                        "root_spawn":root_spawn_patch
                    },
                ),
                "loose_sand": MeshRandomGridTerrainCfg(
                    proportion=0.5,
                    grid_width=0.28,
                    grid_height_range=(0.010, 0.025),
                    platform_width=1.5,
                    flat_patch_sampling={
                        "root_spawn":root_spawn_patch
                    },
                )
            }
        )
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen_cfg,
            max_init_terrain_level=None,
            # collision_group=-1,
            debug_vis=False
        )
        
        self.terrain = TerrainImporter(terrain_cfg)
        # self.scene.terrain["terrain"] = self.terrain
    
    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------
    def _resolve_controlled_joint_ids(self) -> tuple[list[str], torch.Tensor]:
        if not hasattr(self.robot, "joint_names"):
            raise RuntimeError("Robot joint names are unavailable after scene setup.")

        controlled_joint_names = list(self.cfg.controlled_joint_names)
        if len(controlled_joint_names) != self.cfg.num_actions:
            raise RuntimeError(
                f"Expected {self.cfg.num_actions} configured joints, got {len(controlled_joint_names)}: "
                f"{controlled_joint_names}"
            )
        missing_joint_names = [name for name in controlled_joint_names if name not in self.robot.joint_names]
        if missing_joint_names:
            raise RuntimeError(f"Configured controlled joints are missing from the robot: {missing_joint_names}")

        joint_ids = [self.robot.joint_names.index(name) for name in controlled_joint_names]
        return controlled_joint_names, torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    def _make_controlled_action_scales(self, joint_names: list[str]) -> torch.Tensor:
        scales = []
        for joint_name in joint_names:
            if joint_name == "head_pitch":
                scales.append(self.cfg.action_scale_head_pitch)
            elif joint_name == "head_yaw":
                scales.append(self.cfg.action_scale_head_yaw)
            elif joint_name == "head_roll":
                scales.append(self.cfg.action_scale_head_roll)
            elif "neck" in joint_name:
                scales.append(self.cfg.action_scale_neck)
            elif "yaw" in joint_name:
                scales.append(self.cfg.action_scale_yaw)
            elif "roll" in joint_name:
                scales.append(self.cfg.action_scale_roll)
            elif "hip_pitch" in joint_name:
                scales.append(self.cfg.action_scale_pitch)
            elif "knee" in joint_name:
                scales.append(self.cfg.action_scale_knee)
            elif "ankle" in joint_name:
                scales.append(self.cfg.action_scale_ankle)
            else:
                scales.append(self.cfg.action_scale)
        return torch.tensor(scales, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _resolve_foot_body_ids(self) -> torch.Tensor:
        if not hasattr(self.robot, "body_names"):
            raise RuntimeError("Robot body names are unavailable after scene setup.")

        missing_body_names = [name for name in self.cfg.foot_body_names if name not in self.robot.body_names]
        if missing_body_names:
            raise RuntimeError(f"Configured foot bodies are missing from the robot: {missing_body_names}")

        body_ids = [self.robot.body_names.index(name) for name in self.cfg.foot_body_names]
        return torch.tensor(body_ids, dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _pre_physics_step!")
        if self.actions is None:
            self.actions = torch.zeros_like(actions)
        self.previous_actions = self.actions.clone()
        self.actions = torch.clamp(actions, -1.0, 1.0)

        if self.commands is not None and self.command_resampling_steps > 0:
            resample = (self.episode_length_buf % self.command_resampling_steps) == 0
            env_ids = resample.nonzero(as_tuple=False).flatten()
            if env_ids.numel() > 0:
                self._resample_commands(env_ids)

    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------
    def _apply_action(self):
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _apply_action!")
        if self.actions is None:
            return
        if self.controlled_joint_ids is None or self.controlled_action_scales is None:
            raise RuntimeError("Controlled joint ids were not initialized.")

        targets = self.robot.data.default_joint_pos.clone()
        targets[:, self.controlled_joint_ids] = (
            self.robot.data.default_joint_pos[:, self.controlled_joint_ids] + self.actions * self.controlled_action_scales
        )

        self.robot.set_joint_position_target(targets)
        
    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        if self.robot is None:
            raise RuntimeError("Robot not initialized in _get_observations!")

        if self.commands is None:
            self.commands = torch.zeros((self.num_envs, 3), device=self.device)

        root_lin_vel_b, imu_ang_vel_b, imu_projected_gravity = self._get_base_imu_observation()
        obs = torch.cat(
            [
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                root_lin_vel_b,
                imu_ang_vel_b,
                imu_projected_gravity,
                self.commands,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_base_imu_observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return root linear velocity plus IMU-like gyro and gravity observations in the base frame."""
        root_quat = self.robot.data.root_state_w[:, 3:7]
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        root_lin_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        imu_ang_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)
        imu_projected_gravity = quat_rotate_inverse(root_quat, up_vec)
        return root_lin_vel_b, imu_ang_vel_b, imu_projected_gravity

    # ---------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------
    # def _get_rewards(self) -> torch.Tensor:
    #     forward_vel = self.robot.data.root_lin_vel_w[:, 0]
    #     quat_w = self.robot.data.root_state_w[:, 6]
    #     upright = quat_w
    #     energy = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
    #     reward = 1.0 + 1.0 * forward_vel + 1.5 * upright - 0.0005 * energy
    #     return reward
    
    def _get_rewards(self) -> torch.Tensor:
        if self.commands is None:
            self.commands = torch.zeros((self.num_envs, 3), device=self.device)

        root_quat = self.robot.data.root_state_w[:, 3:7]
        root_lin_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.vel_sigma)

        yaw_rate_error = torch.square(self.commands[:, 2] - root_ang_vel_b[:, 2])
        yaw_rate_reward = torch.exp(-yaw_rate_error / self.cfg.yaw_rate_sigma)

        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity = quat_rotate_inverse(root_quat, up_vec)
        upright_reward = torch.clamp(projected_gravity[:, 2], min=0.0) ** 2

        base_height = self.robot.data.root_state_w[:, 2]
        height_error = base_height - self.cfg.target_base_height
        height_reward = torch.exp(-torch.square(height_error) / self.cfg.height_sigma)

        energy = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        energy_penalty = self.cfg.energy_penalty_scale * energy

        lin_vel_z_penalty = torch.abs(root_lin_vel_b[:, 2])

        ang_vel_xy_penalty = torch.norm(root_ang_vel_b[:, :2], dim=1)
        commanded_speed = torch.norm(self.commands[:, :2], dim=1)
        actual_speed = torch.norm(root_lin_vel_b[:, :2], dim=1)
        moving_command = commanded_speed > self.cfg.min_move_command_speed
        turning_command = torch.abs(self.commands[:, 2]) > self.cfg.min_move_command_speed
        command_dir = self.commands[:, :2] / torch.clamp(commanded_speed, min=1e-6).unsqueeze(1)
        velocity_along_command = torch.sum(root_lin_vel_b[:, :2] * command_dir, dim=1)
        lin_vel_progress_reward = torch.clamp(
            velocity_along_command / torch.clamp(commanded_speed, min=1e-6),
            min=0.0,
            max=1.0,
        ) * moving_command.float()
        underspeed_penalty = torch.clamp(commanded_speed - velocity_along_command, min=0.0) * moving_command.float()
        no_motion_penalty = (commanded_speed > self.cfg.min_move_command_speed).float() * torch.exp(
            -actual_speed / self.cfg.no_motion_sigma
        )
        yaw_progress_reward = torch.clamp(
            root_ang_vel_b[:, 2] * torch.sign(self.commands[:, 2]) / torch.clamp(torch.abs(self.commands[:, 2]), min=1e-6),
            min=0.0,
            max=1.0,
        ) * turning_command.float()
        yaw_no_motion_penalty = turning_command.float() * torch.exp(
            -torch.abs(root_ang_vel_b[:, 2]) / self.cfg.no_motion_sigma
        )
        gait_contact_reward, feet_air_time_reward, step_length_reward, foot_clearance_reward = self._get_gait_rewards(
            root_quat, commanded_speed
        )
        in_place_stepping_penalty = no_motion_penalty * gait_contact_reward
        fall_penalty = (
            (base_height < self.cfg.termination_height)
            | (projected_gravity[:, 2] < math.cos(self.cfg.max_root_angle))
        ).float()

        if self.actions is None or self.previous_actions is None:
            action_rate_penalty = torch.zeros(self.num_envs, device=self.device)
        else:
            action_rate_penalty = self.cfg.action_rate_penalty_scale * torch.sum(
                torch.square(self.actions - self.previous_actions), dim=1
            )

        joint_deviation = torch.sum(torch.square(self.robot.data.joint_pos - self.robot.data.default_joint_pos), dim=1)
        joint_deviation_penalty = self.cfg.joint_deviation_penalty_scale * joint_deviation

        reward = (
            self.cfg.rew_scale_lin_vel_xy * lin_vel_reward +
            self.cfg.rew_scale_lin_vel_progress * lin_vel_progress_reward +
            self.cfg.rew_scale_yaw_rate * yaw_rate_reward +
            self.cfg.rew_scale_yaw_progress * yaw_progress_reward +
            self.cfg.rew_scale_upright * upright_reward +
            self.cfg.rew_scale_height * height_reward +
            self.cfg.rew_scale_gait_contact * gait_contact_reward +
            self.cfg.rew_scale_feet_air_time * feet_air_time_reward +
            self.cfg.rew_scale_step_length * step_length_reward +
            self.cfg.rew_scale_foot_clearance * foot_clearance_reward -
            energy_penalty -
            self.cfg.penalty_scale_lin_z * lin_vel_z_penalty -
            self.cfg.penalty_scale_ang_xy * ang_vel_xy_penalty -
            self.cfg.no_motion_penalty_scale * no_motion_penalty -
            self.cfg.underspeed_penalty_scale * underspeed_penalty -
            self.cfg.yaw_no_motion_penalty_scale * yaw_no_motion_penalty -
            self.cfg.in_place_stepping_penalty_scale * in_place_stepping_penalty -
            self.cfg.fall_penalty_scale * fall_penalty -
            action_rate_penalty -
            joint_deviation_penalty
        )

        reward += self.cfg.rew_scale_alive

        self.extras["log"] = {
            "Reward/lin_vel_track": (self.cfg.rew_scale_lin_vel_xy * lin_vel_reward).mean(),
            "Reward/lin_vel_progress": (self.cfg.rew_scale_lin_vel_progress * lin_vel_progress_reward).mean(),
            "Reward/yaw_rate_track": (self.cfg.rew_scale_yaw_rate * yaw_rate_reward).mean(),
            "Reward/yaw_progress": (self.cfg.rew_scale_yaw_progress * yaw_progress_reward).mean(),
            "Reward/upright": (self.cfg.rew_scale_upright * upright_reward).mean(),
            "Reward/height": (self.cfg.rew_scale_height * height_reward).mean(),
            "Penalty/no_motion": (self.cfg.no_motion_penalty_scale * no_motion_penalty).mean(),
            "Penalty/underspeed": (self.cfg.underspeed_penalty_scale * underspeed_penalty).mean(),
            "Penalty/yaw_no_motion": (self.cfg.yaw_no_motion_penalty_scale * yaw_no_motion_penalty).mean(),
            "Stats/commanded_speed": commanded_speed.mean(),
            "Stats/actual_speed": actual_speed.mean(),
            "Stats/velocity_along_command": velocity_along_command.mean(),
            "Stats/commanded_yaw_rate": torch.abs(self.commands[:, 2]).mean(),
            "Stats/actual_yaw_rate": torch.abs(root_ang_vel_b[:, 2]).mean(),
        }

        return reward

    def _get_gait_rewards(
        self, root_quat: torch.Tensor, commanded_speed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.foot_body_ids is None:
            zero = torch.zeros(self.num_envs, device=self.device)
            return zero, zero, zero, zero

        moving = commanded_speed > self.cfg.min_move_command_speed
        desired_contact = self._get_desired_foot_contacts()
        foot_contact = self._get_foot_contacts()
        if self.feet_air_time is None:
            self.feet_air_time = torch.zeros((self.num_envs, 2), device=self.device)
        step_dt = self.cfg.sim.dt * self.cfg.decimation
        self.feet_air_time = torch.where(
            foot_contact,
            torch.zeros_like(self.feet_air_time),
            self.feet_air_time + step_dt,
        )

        contact_match = (foot_contact == desired_contact).float()
        gait_contact_reward = torch.mean(contact_match, dim=1) * moving.float()

        desired_swing = ~desired_contact
        feet_air_time_reward = torch.mean(
            torch.clamp(self.feet_air_time - self.cfg.min_air_time, min=0.0) * desired_swing.float(),
            dim=1,
        ) * moving.float()

        root_pos_w = self.robot.data.root_state_w[:, :3]
        foot_pos_w = self.robot.data.body_pos_w[:, self.foot_body_ids, :]
        foot_pos_b = quat_rotate_inverse(
            root_quat.repeat_interleave(2, dim=0),
            (foot_pos_w - root_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).reshape(self.num_envs, 2, 3)

        command_dir = self.commands[:, :2] / torch.clamp(commanded_speed, min=1e-6).unsqueeze(1)
        swing_step = torch.sum(foot_pos_b[:, :, :2] * command_dir.unsqueeze(1), dim=2)
        step_length_target = self.cfg.target_step_length * torch.clamp(
            commanded_speed / max(self.cfg.command_lin_vel_x[1], self.cfg.command_lin_vel_y[1], 1e-6),
            min=0.5,
            max=1.5,
        )
        step_length_error = torch.square(torch.clamp(step_length_target.unsqueeze(1) - swing_step, min=0.0))
        step_length_reward = torch.mean(
            torch.exp(-step_length_error / 0.004) * desired_swing.float(),
            dim=1,
        ) * moving.float()

        clearance_error = torch.square(foot_pos_w[:, :, 2] - self.cfg.foot_clearance_target)
        foot_clearance_reward = torch.mean(
            torch.exp(-clearance_error / 0.0004) * desired_swing.float(),
            dim=1,
        ) * moving.float()

        return gait_contact_reward, feet_air_time_reward, step_length_reward, foot_clearance_reward

    def _get_desired_foot_contacts(self) -> torch.Tensor:
        cycle_steps = max(1, int(self.cfg.gait_cycle_time_s / (self.cfg.sim.dt * self.cfg.decimation)))
        phase = (self.episode_length_buf % cycle_steps).float() / float(cycle_steps)
        left_phase = phase
        right_phase = torch.remainder(phase + 0.5, 1.0)
        return torch.stack(
            [
                left_phase < self.cfg.gait_duty_factor,
                right_phase < self.cfg.gait_duty_factor,
            ],
            dim=1,
        )

    def _get_foot_contacts(self) -> torch.Tensor:
        contact_forces = self.feet_contact_sensor.data.net_forces_w[:, :, 2]
        if contact_forces.shape[1] < 2:
            return torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        return contact_forces[:, :2].abs() > self.cfg.foot_contact_force_threshold
    # ---------------------------------------------------------------------
    # Dones
    # ---------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        height = self.robot.data.root_state_w[:, 2]
        died = height < self.cfg.termination_height

        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        root_quat = self.robot.data.root_state_w[:, 3:7]
        projected_up = quat_rotate_inverse(root_quat, up_vec)
        tilted = projected_up[:, 2] < math.cos(self.cfg.max_root_angle)

        terminated = died | tilted
        return terminated, timeout

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        num = env_ids.numel()
        
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 2] = self.cfg.spawn_height

        if self.terrain is not None and "root_spawn" in self.terrain.flat_patches:
            patches = self.terrain.flat_patches["root_spawn"]
            rows = self.terrain.terrain_levels[env_ids]
            cols = self.terrain.terrain_types[env_ids]

            num_patches = patches.shape[2]
            pick = torch.randint(0, num_patches, (env_ids.numel(),), device=self.device)
            local_spawn = patches[rows, cols, pick]
            
            world_spawn = self.scene.env_origins[env_ids] + local_spawn
            world_spawn[:, 0] += sample_uniform(-0.25, 0.25, (env_ids.numel(),),device=self.device)
            world_spawn[:, 1] += sample_uniform(-0.25, 0.25, (env_ids.numel(),),device=self.device)
            world_spawn[:, 2] += self.cfg.spawn_height
            
            root_state[:, 0:3] = world_spawn
        else:
            root_state[:, :3] += self.scene.env_origins[env_ids]
            root_state[:, 0] += sample_uniform(-0.1, 0.1, (num,), device=self.device)
            root_state[:, 1] += sample_uniform(-0.1, 0.1, (num,), device=self.device)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.controlled_joint_ids is not None:
            joint_pos[:, self.controlled_joint_ids] += sample_uniform(
                -0.02, 0.02, (num, self.controlled_joint_ids.numel()), device=self.device
            )
            joint_vel[:, self.controlled_joint_ids] += sample_uniform(
                -0.01, 0.01, (num, self.controlled_joint_ids.numel()), device=self.device
            )
        root_state[:, 10:13] = sample_uniform(
            -self.cfg.reset_root_ang_vel_range,
            self.cfg.reset_root_ang_vel_range,
            (num, 3),
            device=self.device,
        )

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.actions is not None:
            self.actions[env_ids] = 0.0
        if self.previous_actions is not None:
            self.previous_actions[env_ids] = 0.0
        if self.feet_air_time is not None:
            self.feet_air_time[env_ids] = 0.0
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.commands is None:
            self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        num = env_ids.numel()
        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.command_lin_vel_x[0], self.cfg.command_lin_vel_x[1], (env_ids.numel(),), device=self.device
        )
        self.commands[env_ids, 1] = sample_uniform(
            self.cfg.command_lin_vel_y[0], self.cfg.command_lin_vel_y[1], (env_ids.numel(),), device=self.device
        )
        self.commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_rate[0], self.cfg.command_yaw_rate[1], (env_ids.numel(),), device=self.device
        )

        cardinal_mask = torch.rand((num,), device=self.device) < self.cfg.command_cardinal_probability
        if cardinal_mask.any():
            cardinal_env_ids = env_ids[cardinal_mask]
            cardinal_count = cardinal_env_ids.numel()
            modes = torch.randint(0, 6, (cardinal_count,), device=self.device)
            self.commands[cardinal_env_ids, :] = 0.0

            forward = modes == 0
            backward = modes == 1
            left = modes == 2
            right = modes == 3
            turn_left = modes == 4
            turn_right = modes == 5

            if forward.any():
                self.commands[cardinal_env_ids[forward], 0] = sample_uniform(
                    self.cfg.command_min_cardinal_speed,
                    self.cfg.command_lin_vel_x[1],
                    (forward.sum().item(),),
                    device=self.device,
                )
            if backward.any():
                self.commands[cardinal_env_ids[backward], 0] = sample_uniform(
                    self.cfg.command_lin_vel_x[0],
                    -self.cfg.command_min_cardinal_speed,
                    (backward.sum().item(),),
                    device=self.device,
                )
            if left.any():
                self.commands[cardinal_env_ids[left], 1] = sample_uniform(
                    self.cfg.command_min_cardinal_speed,
                    self.cfg.command_lin_vel_y[1],
                    (left.sum().item(),),
                    device=self.device,
                )
            if right.any():
                self.commands[cardinal_env_ids[right], 1] = sample_uniform(
                    self.cfg.command_lin_vel_y[0],
                    -self.cfg.command_min_cardinal_speed,
                    (right.sum().item(),),
                    device=self.device,
                )
            if turn_left.any():
                self.commands[cardinal_env_ids[turn_left], 2] = sample_uniform(
                    self.cfg.command_min_cardinal_speed,
                    self.cfg.command_yaw_rate[1],
                    (turn_left.sum().item(),),
                    device=self.device,
                )
            if turn_right.any():
                self.commands[cardinal_env_ids[turn_right], 2] = sample_uniform(
                    self.cfg.command_yaw_rate[0],
                    -self.cfg.command_min_cardinal_speed,
                    (turn_right.sum().item(),),
                    device=self.device,
                )

        stand_mask = torch.rand((num,), device=self.device) < self.cfg.stand_command_probability
        if stand_mask.any():
            self.commands[env_ids[stand_mask], :] = 0.0

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
