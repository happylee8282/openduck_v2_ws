"""Open Duck v2 direct RL task; deployable 101-value actor, privileged critic."""

from __future__ import annotations

import math
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPitTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
from isaaclab.utils.math import sample_uniform

from open_duck_v2.profile import CRITIC_DIM
from open_duck_v2.control import (
    V2Control,
    EpisodeDelay,
    pack_actor_observation,
    quat_rotate_inverse,
)
from open_duck_v2.randomization import RobotRandomizer
from .open_duck_env_cfg import OpenDuckEnvCfg


class OpenDuckMiniEnv(DirectRLEnv):
    cfg: OpenDuckEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.terrain = self.height_scanner = None
        super().__init__(cfg, render_mode, **kwargs)
        ids, names = self.robot.find_joints(
            list(cfg.controlled_joint_names), preserve_order=True
        )
        if tuple(names) != tuple(cfg.controlled_joint_names):
            raise RuntimeError("Controlled joint order mismatch")
        self.controlled_joint_ids = torch.tensor(ids, device=self.device)
        self.foot_body_ids = torch.tensor(
            self.robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)[0],
            device=self.device,
        )
        self.contact_body_ids = self.feet_contact_sensor.find_bodies(
            list(cfg.foot_body_names), preserve_order=True
        )[0]
        if len(self.contact_body_ids) != 2:
            raise RuntimeError("Both foot contact sensors are required")
        self.root_body_id = self.robot.find_bodies("trunk_assembly")[0][0]
        self.control = V2Control(
            self.num_envs,
            self.device,
            self.step_dt,
            cfg.action_delay_max,
            cfg.backlash_half_width,
            head_control_mode=cfg.head_control_mode,
            target_filter_tau_s=cfg.target_filter_tau_s,
            head_target_speed=cfg.head_target_speed,
        )
        self.actions = self.control.actions
        self.previous_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros(self.num_envs, 7, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.imu_delay = EpisodeDelay(self.num_envs, 6, cfg.imu_delay_max, self.device)
        self.imu_bias = torch.zeros(self.num_envs, 6, device=self.device)
        self._imu_cache = torch.zeros_like(self.imu_bias)
        self._imu_stamp = -1
        self._obs_cache = None
        self._obs_stamp = -1
        self.command_resampling_steps = max(
            1, round(cfg.command_resampling_time_s / self.step_dt)
        )
        self.randomizer = RobotRandomizer(self.robot, cfg, self.root_body_id)
        self.reward_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.reward_angular = torch.zeros_like(self.reward_velocity)
        self.path_origin = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_origin = torch.zeros(self.num_envs, device=self.device)
        self.previous_heading = torch.zeros_like(self.heading_origin)
        from open_duck_v2.heading import HeadingHold

        self.heading_hold = HeadingHold(
            self.num_envs,
            self.device,
            cfg.heading_hold_kp,
            cfg.heading_hold_max_rate,
            cfg.heading_hold_filter_tau,
            self.step_dt,
            cfg.lateral_command_trim,
            cfg.forward_command_scale,
        )
        self.heading_delay = EpisodeDelay(
            self.num_envs, 1, cfg.imu_delay_max, self.device
        )
        self.heading_bias = torch.zeros_like(self.heading_origin)
        self._heading_stamp = -1

    def _effective_commands(self):
        if not self.cfg.heading_hold_kp:
            return self.commands
        if self._heading_stamp != int(self.common_step_counter):
            from open_duck_v2.straight_rewards import heading_from_quaternion

            measured = (
                heading_from_quaternion(self.robot.data.root_quat_w) + self.heading_bias
            )
            if self.cfg.observation_noise:
                measured += (
                    2 * torch.rand_like(measured) - 1
                ) * self.cfg.heading_imu_noise
            measured = self.heading_delay.push(measured[:, None])[:, 0]
            self.heading_hold.observe(measured)
            self._heading_stamp = int(self.common_step_counter)
        return self.heading_hold.commands(self.commands)

    def _pre_physics_step(self, actions):
        self.previous_actions = self.actions.clone()
        self._native_targets = self.control.process(actions, self.commands[:, 3:7])
        self.actions = self.control.actions
        ids = (
            (self.episode_length_buf % self.command_resampling_steps == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(ids):
            self._resample_commands(ids)

    def _apply_action(self):
        self.robot.set_joint_position_target(
            self._native_targets, joint_ids=self.controlled_joint_ids
        )

    def _raw_imu(self):
        data = self.robot.data
        q = data.root_state_w[:, 3:7]
        gyro = quat_rotate_inverse(q, data.root_ang_vel_w)
        angular_acc = quat_rotate_inverse(q, data.body_ang_acc_w[:, self.root_body_id])
        gravity = torch.tensor(self.cfg.sim.gravity, device=self.device).expand(
            self.num_envs, -1
        )
        accel = quat_rotate_inverse(
            q, data.body_lin_acc_w[:, self.root_body_id] - gravity
        )
        offset = (
            torch.tensor(self.cfg.imu_position_b, device=self.device)
            - data.body_com_pos_b[:, self.root_body_id]
        )
        accel = (
            accel
            + torch.cross(angular_acc, offset, dim=-1)
            + torch.cross(gyro, torch.cross(gyro, offset, dim=-1), dim=-1)
        )
        return torch.cat((gyro, accel), -1)

    def _imu_observation(self):
        stamp = int(self.common_step_counter)
        if self._imu_stamp != stamp:
            value = self._raw_imu()
            if self.cfg.observation_noise:
                noise = value.new_tensor(
                    [self.cfg.gyro_noise] * 3 + [self.cfg.accelerometer_noise] * 3
                )
                value = value + self.imu_bias + (2 * torch.rand_like(value) - 1) * noise
            self._imu_cache = self.imu_delay.push(value)
            self._imu_stamp = stamp
        return self._imu_cache

    def _phase(self):
        phase = (
            self.episode_length_buf.float()
            * self.step_dt
            / self.cfg.gait_cycle_time_s
            * 2
            * math.pi
        )
        return torch.stack((torch.cos(phase), torch.sin(phase)), -1)

    def _get_observations(self):
        stamp = int(self.common_step_counter)
        if self._obs_stamp == stamp and self._obs_cache is not None:
            return self._obs_cache
        imu = self._imu_observation()
        joint_pos = (
            self.robot.data.joint_pos[:, self.controlled_joint_ids]
            + self.control.offsets
        )
        joint_vel = self.robot.data.joint_vel[:, self.controlled_joint_ids]
        if self.cfg.observation_noise:
            joint_pos = (
                joint_pos
                + (2 * torch.rand_like(joint_pos) - 1) * self.cfg.joint_pos_noise
            )
            joint_vel = (
                joint_vel
                + (2 * torch.rand_like(joint_vel) - 1) * self.cfg.joint_vel_noise
            )
        policy = pack_actor_observation(
            imu[:, :3],
            imu[:, 3:],
            self._effective_commands(),
            joint_pos,
            joint_vel,
            self.control.history,
            self.control.motor_targets,
            self._get_foot_contacts(),
            self._phase(),
        )
        velocity = quat_rotate_inverse(
            self.robot.data.root_state_w[:, 3:7], self.robot.data.root_lin_vel_w
        )
        context = self._height_scan()
        if self.cfg.critic_context == "flat_path":
            if self.cfg.enable_height_scan or self.cfg.use_rough_terrain:
                raise ValueError(
                    "flat_path critic is reserved for flat straight walking"
                )
            from open_duck_v2.straight_rewards import (
                heading_from_quaternion,
                wrap_angle,
            )

            context = torch.zeros(self.num_envs, 117, device=self.device)
            context[:, 0] = wrap_angle(
                heading_from_quaternion(self.robot.data.root_quat_w)
                - self.heading_origin
            )
            offset = self.robot.data.root_pos_w[:, :2] - self.path_origin
            context[:, 1] = (
                -torch.sin(self.heading_origin) * offset[:, 0]
                + torch.cos(self.heading_origin) * offset[:, 1]
            )
        critic = torch.cat(
            (policy, velocity, self._base_height()[:, None], context), -1
        )
        if critic.shape[1] != CRITIC_DIM:
            raise RuntimeError("Critic contract mismatch")
        self._obs_cache = {"policy": policy, "critic": critic}
        self._obs_stamp = stamp
        return self._obs_cache

    def _ground_at(self, xy):
        if self.height_scanner is None:
            return torch.zeros(xy.shape[:-1], device=self.device)
        hits = self.height_scanner.data.ray_hits_w
        valid = torch.isfinite(hits).all(-1)
        if not valid.any(-1).all():
            raise RuntimeError("No valid ground rays for an environment")
        distance = (hits[:, None, :, :2] - xy[:, :, None, :]).square().sum(-1)
        distance = distance.masked_fill(~valid[:, None, :], float("inf"))
        indices = distance.argmin(-1)
        return torch.gather(hits[:, :, 2], 1, indices)

    def _base_height(self):
        return (
            self.robot.data.root_state_w[:, 2]
            - self._ground_at(self.robot.data.root_state_w[:, None, :2])[:, 0]
        )

    def _feet_clearance(self):
        feet = self.robot.data.body_pos_w[:, self.foot_body_ids]
        return feet[:, :, 2] - self._ground_at(feet[:, :, :2])

    def _height_scan(self):
        if not self.cfg.enable_height_scan or self.height_scanner is None:
            return torch.zeros(self.num_envs, 117, device=self.device)
        hits = self.height_scanner.data.ray_hits_w[:, :, 2]
        if hits.shape[1] != 117:
            raise RuntimeError(f"Expected 117 terrain rays, got {hits.shape[1]}")
        h = self.robot.data.root_state_w[:, 2:3] - hits - self.cfg.target_base_height
        return torch.nan_to_num(h, nan=0.0, posinf=0.25, neginf=-0.25).clamp(
            -0.25, 0.25
        )

    def _get_foot_contacts(self):
        force = self.feet_contact_sensor.data.net_forces_w[:, self.contact_body_ids, 2]
        return force.abs() > self.cfg.foot_contact_force_threshold

    def _get_dones(self):
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        tilted = quat_rotate_inverse(self.robot.data.root_state_w[:, 3:7], up)[
            :, 2
        ] < math.cos(self.cfg.max_root_angle)
        return (
            self._base_height() < self.cfg.termination_height
        ) | tilted, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device).long()
        super()._reset_idx(env_ids)
        self.randomizer.reset(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        origins = (
            self.terrain.env_origins
            if self.terrain is not None
            else self.scene.env_origins
        )
        root[:, :3] += origins[env_ids]
        root[:, 2] = origins[env_ids, 2] + self.cfg.spawn_height
        root[:, 10:13] = sample_uniform(
            -self.cfg.reset_root_ang_vel_range,
            self.cfg.reset_root_ang_vel_range,
            (len(env_ids), 3),
            self.device,
        )
        joints = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            joints, torch.zeros_like(joints), env_ids=env_ids
        )
        self.control.reset(
            env_ids, self.cfg.action_delay_min, self.cfg.action_delay_max
        )
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.reward_velocity[env_ids] = 0.0
        self.reward_angular[env_ids] = 0.0
        from open_duck_v2.straight_rewards import heading_from_quaternion

        self.path_origin[env_ids] = root[:, :2]
        self.heading_origin[env_ids] = heading_from_quaternion(root[:, 3:7])
        self.previous_heading[env_ids] = self.heading_origin[env_ids]
        if self.cfg.heading_hold_kp:
            self.heading_bias[env_ids] = (
                sample_uniform(-0.02, 0.02, (len(env_ids),), self.device)
                if self.cfg.observation_noise
                else 0.0
            )
            measured = self.heading_origin[env_ids] + self.heading_bias[env_ids]
            self.heading_hold.reset(env_ids, measured)
            self.heading_delay.reset(
                env_ids,
                measured[:, None],
                self.cfg.imu_delay_min,
                self.cfg.imu_delay_max,
            )
        self.feet_air_time[env_ids] = 0.0
        self._resample_commands(env_ids)
        nominal = torch.zeros(len(env_ids), 6, device=self.device)
        nominal[:, 5] = -self.cfg.sim.gravity[2]
        self.imu_delay.reset(
            env_ids, nominal, self.cfg.imu_delay_min, self.cfg.imu_delay_max
        )
        gyro = sample_uniform(
            -self.cfg.gyro_bias_range,
            self.cfg.gyro_bias_range,
            (len(env_ids), 3),
            self.device,
        )
        accel = sample_uniform(
            -self.cfg.accelerometer_bias_range,
            self.cfg.accelerometer_bias_range,
            (len(env_ids), 3),
            self.device,
        )
        self.imu_bias[env_ids] = torch.cat((gyro, accel), -1)
        self._imu_stamp = self._obs_stamp = -1

    def _resample_commands(self, env_ids):
        for i, bounds in enumerate(
            (
                self.cfg.command_lin_vel_x,
                self.cfg.command_lin_vel_y,
                self.cfg.command_yaw_rate,
            )
        ):
            self.commands[env_ids, i] = sample_uniform(
                *bounds, (len(env_ids),), self.device
            )
        self.commands[env_ids, 3:] = 0.0
        stand = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.stand_command_probability
        )
        self.commands[env_ids[stand], :3] = 0.0

    def _setup_scene(self):
        usd_path = Path(self.cfg.robot.spawn.usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Training USD not found: {usd_path}. Run tools/convert_urdf_to_training_usd.py first."
            )

        self.robot = Articulation(self.cfg.robot)

        if (
            self.cfg.robot.spawn.articulation_props.enabled_self_collisions
            and self.cfg.mechanical_collision_filter_distance
        ):
            import omni.usd
            from open_duck_v2.collision import apply_mechanical_neighbor_filters

            self.mechanical_collision_pairs = apply_mechanical_neighbor_filters(
                omni.usd.get_context().get_stage(),
                "/World/envs/env_0/Robot",
                Path(__file__).resolve().parents[1] / "Robots/open_duck_mini.urdf",
                self.cfg.mechanical_collision_filter_distance,
            )
            print(
                f"Filtered {len(self.mechanical_collision_pairs)} close mechanical ancestor pairs; cross-limb collisions remain enabled"
            )

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
        if self.cfg.use_rough_terrain:
            self.height_scanner = RayCaster(
                RayCasterCfg(
                    prim_path="/World/envs/env_.*/Robot/trunk_assembly",
                    offset=RayCasterCfg.OffsetCfg(pos=self.cfg.height_scan_offset),
                    ray_alignment="yaw",
                    pattern_cfg=GridPatternCfg(
                        resolution=self.cfg.height_scan_resolution,
                        size=self.cfg.height_scan_size,
                    ),
                    mesh_prim_paths=["/World/ground"],
                    max_distance=2.0,
                    debug_vis=False,
                )
            )
            self.scene.sensors["height_scanner"] = self.height_scanner

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_terrains(self):
        root_spawn_patch = FlatPatchSamplingCfg(
            num_patches=64,
            patch_radius=0.35,
            max_height_diff=0.02,
            x_range=(-1e6, 1e6),
            y_range=(-1e6, 1e6),
            z_range=(-1e6, 1e6),
        )
        terrain_gen_cfg = TerrainGeneratorCfg(
            size=(5.0, 5.0),
            border_width=0.5,
            num_rows=5,
            num_cols=6,
            horizontal_scale=0.08,
            vertical_scale=0.005,
            slope_threshold=0.755,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                "low_rigid_grid": MeshRandomGridTerrainCfg(
                    proportion=0.24,
                    grid_width=0.35,
                    grid_height_range=(0.004, 0.020),
                    platform_width=1.5,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
                "high_rigid_grid": MeshRandomGridTerrainCfg(
                    proportion=0.24,
                    grid_width=0.28,
                    grid_height_range=(0.010, 0.040),
                    platform_width=1.5,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
                "broken_ground": MeshRandomGridTerrainCfg(
                    proportion=0.18,
                    grid_width=0.24,
                    grid_height_range=(0.010, 0.045),
                    platform_width=1.4,
                    holes=True,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
                "pits": MeshPitTerrainCfg(
                    proportion=0.14,
                    pit_depth_range=(0.015, 0.045),
                    platform_width=1.2,
                    double_pit=True,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
                "stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.08,
                    step_height_range=(0.008, 0.030),
                    step_width=0.28,
                    platform_width=1.2,
                    border_width=0.1,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
                "down_stairs": MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.04,
                    step_height_range=(0.008, 0.028),
                    step_width=0.30,
                    platform_width=1.5,
                    border_width=0.1,
                    flat_patch_sampling={"root_spawn": root_spawn_patch},
                ),
            },
        )
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen_cfg,
            max_init_terrain_level=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            # collision_group=-1,
            debug_vis=False,
        )

        self.terrain = TerrainImporter(terrain_cfg)

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.reward_profile in (
            "straight_v2",
            "straight_v3",
            "straight_v4",
            "straight_v5",
            "straight_v6",
        ):
            return self._get_straight_rewards()
        if self.commands is None:
            self.commands = torch.zeros((self.num_envs, 7), device=self.device)
            self.commands[:, 3:] = 0.0

        root_quat = self.robot.data.root_state_w[:, 3:7]
        root_lin_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)

        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.vel_sigma)

        yaw_rate_error = torch.square(self.commands[:, 2] - root_ang_vel_b[:, 2])
        yaw_rate_reward = torch.exp(-yaw_rate_error / self.cfg.yaw_rate_sigma)

        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )
        projected_gravity = quat_rotate_inverse(root_quat, up_vec)
        upright_reward = torch.clamp(projected_gravity[:, 2], min=0.0) ** 2

        base_height = self._base_height()
        target_height = torch.full_like(base_height, self.cfg.target_base_height)
        height_error = base_height - target_height
        height_reward = torch.exp(-torch.square(height_error) / self.cfg.height_sigma)

        energy = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        energy_penalty = self.cfg.energy_penalty_scale * energy

        lin_vel_z_penalty = torch.abs(root_lin_vel_b[:, 2])

        ang_vel_xy_penalty = torch.norm(root_ang_vel_b[:, :2], dim=1)
        commanded_speed = torch.norm(self.commands[:, :2], dim=1)
        actual_speed = torch.norm(root_lin_vel_b[:, :2], dim=1)
        moving_command = commanded_speed > self.cfg.min_move_command_speed
        turning_command = (
            torch.abs(self.commands[:, 2]) > self.cfg.min_move_command_speed
        )
        command_dir = self.commands[:, :2] / torch.clamp(
            commanded_speed, min=1e-6
        ).unsqueeze(1)
        velocity_along_command = torch.sum(root_lin_vel_b[:, :2] * command_dir, dim=1)
        lin_vel_progress_reward = (
            torch.clamp(
                velocity_along_command / torch.clamp(commanded_speed, min=1e-6),
                min=0.0,
                max=1.0,
            )
            * moving_command.float()
        )
        underspeed_penalty = (
            torch.clamp(commanded_speed - velocity_along_command, min=0.0)
            * moving_command.float()
        )
        no_motion_penalty = (
            commanded_speed > self.cfg.min_move_command_speed
        ).float() * torch.exp(-actual_speed / self.cfg.no_motion_sigma)
        yaw_progress_reward = (
            torch.clamp(
                root_ang_vel_b[:, 2]
                * torch.sign(self.commands[:, 2])
                / torch.clamp(torch.abs(self.commands[:, 2]), min=1e-6),
                min=0.0,
                max=1.0,
            )
            * turning_command.float()
        )
        yaw_no_motion_penalty = turning_command.float() * torch.exp(
            -torch.abs(root_ang_vel_b[:, 2]) / self.cfg.no_motion_sigma
        )
        (
            gait_contact_reward,
            feet_air_time_reward,
            step_length_reward,
            foot_clearance_reward,
        ) = self._get_gait_rewards(root_quat, commanded_speed)
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

        joint_deviation = torch.sum(
            torch.square(self.robot.data.joint_pos - self.robot.data.default_joint_pos),
            dim=1,
        )
        joint_deviation_penalty = (
            self.cfg.joint_deviation_penalty_scale * joint_deviation
        )

        reward = (
            self.cfg.rew_scale_lin_vel_xy * lin_vel_reward
            + self.cfg.rew_scale_lin_vel_progress * lin_vel_progress_reward
            + self.cfg.rew_scale_yaw_rate * yaw_rate_reward
            + self.cfg.rew_scale_yaw_progress * yaw_progress_reward
            + self.cfg.rew_scale_upright * upright_reward
            + self.cfg.rew_scale_height * height_reward
            + self.cfg.rew_scale_gait_contact * gait_contact_reward
            + self.cfg.rew_scale_feet_air_time * feet_air_time_reward
            + self.cfg.rew_scale_step_length * step_length_reward
            + self.cfg.rew_scale_foot_clearance * foot_clearance_reward
            - energy_penalty
            - self.cfg.penalty_scale_lin_z * lin_vel_z_penalty
            - self.cfg.penalty_scale_ang_xy * ang_vel_xy_penalty
            - self.cfg.no_motion_penalty_scale * no_motion_penalty
            - self.cfg.underspeed_penalty_scale * underspeed_penalty
            - self.cfg.yaw_no_motion_penalty_scale * yaw_no_motion_penalty
            - self.cfg.in_place_stepping_penalty_scale * in_place_stepping_penalty
            - self.cfg.fall_penalty_scale * fall_penalty
            - action_rate_penalty
            - joint_deviation_penalty
        )

        reward += self.cfg.rew_scale_alive

        height_scan = self._height_scan()
        self.extras["log"] = {
            "DomainRandomization/applied_resets": float(self.randomizer.applied_resets),
            "Reward/lin_vel_track": (
                self.cfg.rew_scale_lin_vel_xy * lin_vel_reward
            ).mean(),
            "Reward/lin_vel_progress": (
                self.cfg.rew_scale_lin_vel_progress * lin_vel_progress_reward
            ).mean(),
            "Reward/yaw_rate_track": (
                self.cfg.rew_scale_yaw_rate * yaw_rate_reward
            ).mean(),
            "Reward/yaw_progress": (
                self.cfg.rew_scale_yaw_progress * yaw_progress_reward
            ).mean(),
            "Reward/upright": (self.cfg.rew_scale_upright * upright_reward).mean(),
            "Reward/height": (self.cfg.rew_scale_height * height_reward).mean(),
            "Reward/feet_air_time": (
                self.cfg.rew_scale_feet_air_time * feet_air_time_reward
            ).mean(),
            "Reward/step_length": (
                self.cfg.rew_scale_step_length * step_length_reward
            ).mean(),
            "Reward/foot_clearance": (
                self.cfg.rew_scale_foot_clearance * foot_clearance_reward
            ).mean(),
            "Penalty/no_motion": (
                self.cfg.no_motion_penalty_scale * no_motion_penalty
            ).mean(),
            "Penalty/underspeed": (
                self.cfg.underspeed_penalty_scale * underspeed_penalty
            ).mean(),
            "Penalty/yaw_no_motion": (
                self.cfg.yaw_no_motion_penalty_scale * yaw_no_motion_penalty
            ).mean(),
            "Stats/commanded_speed": commanded_speed.mean(),
            "Stats/actual_speed": actual_speed.mean(),
            "Stats/velocity_along_command": velocity_along_command.mean(),
            "Stats/commanded_yaw_rate": torch.abs(self.commands[:, 2]).mean(),
            "Stats/actual_yaw_rate": torch.abs(root_ang_vel_b[:, 2]).mean(),
            "Stats/target_height": target_height.mean(),
            "Stats/base_height": base_height.mean(),
            "HeightScan/abs_mean": torch.abs(height_scan).mean(),
            "HeightScan/std": height_scan.std(),
            "HeightScan/min": height_scan.min(),
            "HeightScan/max": height_scan.max(),
        }

        return reward

    def _get_straight_rewards(self):
        from open_duck_v2.straight_rewards import (
            tracking_terms,
            heading_from_quaternion,
            wrap_angle,
        )

        q = self.robot.data.root_quat_w
        velocity = quat_rotate_inverse(q, self.robot.data.root_lin_vel_w)
        angular = quat_rotate_inverse(q, self.robot.data.root_ang_vel_w)
        alpha = 1 - math.exp(-self.step_dt / self.cfg.reward_velocity_filter_s)
        self.reward_velocity += alpha * (velocity - self.reward_velocity)
        tracked_angular = angular
        if self.cfg.reward_profile in ("straight_v4", "straight_v5", "straight_v6"):
            heading = heading_from_quaternion(q)
            heading_rate = wrap_angle(heading - self.previous_heading) / self.step_dt
            self.previous_heading.copy_(heading)
            tracked_angular = angular.clone()
            tracked_angular[:, 2] = heading_rate
        self.reward_angular += alpha * (tracked_angular - self.reward_angular)
        tracking_commands = self.commands.clone()
        tracking_commands[:, 2] = self._effective_commands()[:, 2]
        terms = tracking_terms(
            tracking_commands, self.reward_velocity, self.reward_angular
        )
        if self.cfg.reward_profile in ("straight_v4", "straight_v5", "straight_v6"):
            terms["track_yaw"] = torch.exp(
                -((self.reward_angular[:, 2] - tracking_commands[:, 2]) / 0.05).square()
            )
        up = quat_rotate_inverse(
            q, q.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1)
        )[:, 2].clamp(0, 1)
        height = self._base_height()
        height_reward = torch.exp(
            -((height - self.cfg.target_base_height) / 0.03).square()
        )
        gait, air, step, clearance = self._get_gait_rewards(
            q, self.commands[:, :2].norm(dim=-1)
        )
        contacts = self._get_foot_contacts()
        foot_velocity = self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2]
        slip = (foot_velocity.square().sum(-1) * contacts).sum(-1).clamp_max(0.5)
        head_velocity = self.robot.data.joint_vel[:, self.controlled_joint_ids[5:9]]
        delta = self.actions - self.previous_actions
        jerk = (
            self.actions - 2 * self.control.history[:, 1] + self.control.history[:, 2]
        )
        action_cost = (
            0.05 * delta.square().mean(-1)
            + 0.02 * jerk.square().mean(-1)
            + 0.03 * self.actions.square().mean(-1)
        )
        tilt_cost = (1 - up).square() * 2
        fall = (
            (height < self.cfg.termination_height)
            | (up < math.cos(self.cfg.max_root_angle))
        ).float()
        reward = (
            3 * terms["track_x"]
            + terms["track_y"]
            + terms["track_yaw"]
            + 2 * terms["progress"]
            + up.square()
            + 0.4 * height_reward
            + 2 * terms["still"]
            + 0.05
            + self.cfg.rew_scale_gait_contact * gait
            + self.cfg.rew_scale_feet_air_time * air
            + self.cfg.rew_scale_step_length * step
            + self.cfg.rew_scale_foot_clearance * clearance
            - 0.6 * terms["lateral_cost"]
            - 0.4 * terms["yaw_cost"]
            - 0.3 * terms["idle_cost"]
            - terms["no_progress"]
            - 10 * slip
            - action_cost
            - tilt_cost
            - 0.2 * velocity[:, 2].square()
            - 0.05 * angular[:, :2].square().sum(-1)
            - 0.01 * head_velocity.square().mean(-1)
            - 0.00003 * self.robot.data.applied_torque.square().sum(-1)
            - self.cfg.fall_penalty_scale * fall
        )
        if self.cfg.reward_profile in (
            "straight_v3",
            "straight_v4",
            "straight_v5",
            "straight_v6",
        ):
            idle = terms["idle"].float()
            idle_dynamic = (
                (velocity[:, :2] / 0.04).square().sum(-1)
                + 0.2 * (angular[:, 2] / 0.2).square()
            ).clamp_max(10)
            reward -= 0.3 * idle * idle_dynamic
            reward -= 0.02 * angular[:, 2].square()
        if self.cfg.reward_profile in ("straight_v4", "straight_v5", "straight_v6"):
            # World position is reward/critic-only. The 101-input actor receives
            # corrected rate commands when optional IMU attitude heading hold is on.
            heading_error = wrap_angle(heading - self.heading_origin)
            offset = self.robot.data.root_pos_w[:, :2] - self.path_origin
            lateral = (
                -torch.sin(self.heading_origin) * offset[:, 0]
                + torch.cos(self.heading_origin) * offset[:, 1]
            )
            reward += 2 * torch.exp(-(heading_error / 0.08).square())
            reward -= 0.5 * (heading_error / 0.1).square().clamp_max(4)
            reward += torch.exp(-(lateral / 0.03).square())
            reward -= 0.5 * (lateral / 0.05).square().clamp_max(4)
        if self.cfg.reward_time_scale:
            reward = reward * self.step_dt
        self.extras["log"] = {
            "Reward/lin_vel_track": (3 * terms["track_x"]).mean(),
            "Reward/lin_vel_progress": (2 * terms["progress"]).mean(),
            "Reward/yaw_rate_track": terms["track_yaw"].mean(),
            "Reward/gait": (self.cfg.rew_scale_gait_contact * gait).mean(),
            "Reward/foot_clearance": (
                self.cfg.rew_scale_foot_clearance * clearance
            ).mean(),
            "Penalty/lateral": (0.6 * terms["lateral_cost"]).mean(),
            "Penalty/yaw": (0.4 * terms["yaw_cost"]).mean(),
            "Penalty/idle_motion": (0.3 * terms["idle_cost"]).mean(),
            "Penalty/no_motion": terms["no_progress"].mean(),
            "Penalty/foot_slip": (10 * slip).mean(),
            "Stats/commanded_speed": self.commands[:, :2].norm(dim=-1).mean(),
            "Stats/actual_speed": self.reward_velocity[:, :2].norm(dim=-1).mean(),
            "Stats/velocity_along_command": terms["along"].mean(),
            "Stats/actual_yaw_rate": self.reward_angular[:, 2].abs().mean(),
            "Stats/head_velocity_rms": head_velocity.square().mean().sqrt(),
            "Stats/base_height": height.mean(),
        }
        if self.cfg.reward_profile in ("straight_v4", "straight_v5", "straight_v6"):
            self.extras["log"].update(
                {
                    "Stats/heading_error_abs_rad": heading_error.abs().mean(),
                    "Stats/lateral_path_error_abs_m": lateral.abs().mean(),
                }
            )
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
        feet_air_time_reward = (
            torch.mean(
                torch.clamp(self.feet_air_time - self.cfg.min_air_time, min=0.0)
                * desired_swing.float(),
                dim=1,
            )
            * moving.float()
        )

        root_pos_w = self.robot.data.root_state_w[:, :3]
        foot_pos_w = self.robot.data.body_pos_w[:, self.foot_body_ids, :]
        foot_pos_b = quat_rotate_inverse(
            root_quat.repeat_interleave(2, dim=0),
            (foot_pos_w - root_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).reshape(self.num_envs, 2, 3)

        command_dir = self.commands[:, :2] / torch.clamp(
            commanded_speed, min=1e-6
        ).unsqueeze(1)
        swing_step = torch.sum(foot_pos_b[:, :, :2] * command_dir.unsqueeze(1), dim=2)
        step_length_target = self.cfg.target_step_length * torch.clamp(
            commanded_speed
            / max(
                *(
                    abs(v)
                    for v in (*self.cfg.command_lin_vel_x, *self.cfg.command_lin_vel_y)
                ),
                1e-6,
            ),
            min=0.5,
            max=1.5,
        )
        step_length_error = torch.square(
            torch.clamp(step_length_target.unsqueeze(1) - swing_step, min=0.0)
        )
        step_length_reward = (
            torch.mean(
                torch.exp(-step_length_error / 0.004) * desired_swing.float(),
                dim=1,
            )
            * moving.float()
        )

        clearance_error = torch.square(
            self._feet_clearance() - self.cfg.foot_clearance_target
        )
        foot_clearance_reward = (
            torch.mean(
                torch.exp(-clearance_error / 0.0004) * desired_swing.float(),
                dim=1,
            )
            * moving.float()
        )

        return (
            gait_contact_reward,
            feet_air_time_reward,
            step_length_reward,
            foot_clearance_reward,
        )

    def _get_desired_foot_contacts(self) -> torch.Tensor:
        cycle_steps = max(
            1, int(self.cfg.gait_cycle_time_s / (self.cfg.sim.dt * self.cfg.decimation))
        )
        phase = (self.episode_length_buf % cycle_steps).float() / float(cycle_steps)
        from open_duck_v2.straight_rewards import desired_contacts

        return desired_contacts(
            phase, self.cfg.gait_duty_factor, self.cfg.gait_reward_phase_offset
        )
