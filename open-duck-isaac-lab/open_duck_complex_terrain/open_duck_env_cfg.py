# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab.envs import ViewerCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# from isaaclab_tasks.direct.open_duck.open_duck_mini import OPEN_DUCK_MINI_CFG

import isaaclab.sim as sim_utils

from open_duck_v2.profile import (
    JOINT_NAMES,
    OBS_DIM,
    CRITIC_DIM,
    native_home,
    PHYSICS_DT,
    DECIMATION,
    SPAWN_HEIGHT,
    STANCE_HEIGHT,
)
from open_duck_v2.actuators import V2ServoActuatorCfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_DUCK_TRAINING_USD = PROJECT_ROOT / "Robots" / "open_duck_mini_training.usd"


OPEN_DUCK_MINI_CFG = ArticulationCfg(
    prim_path="/World/envs/env_0/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(OPEN_DUCK_TRAINING_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SPAWN_HEIGHT), joint_pos=native_home(), joint_vel={".*": 0.0}
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "servos": V2ServoActuatorCfg(
            joint_names_expr=list(JOINT_NAMES),
            effort_limit=3.35,
            effort_limit_sim=3.35,
            velocity_limit_sim=5.24,
            stiffness=17.8,
            damping=0.60,
            armature=0.028,
            friction=0.0,
            coulomb_friction=0.052,
        )
    },
)


@configclass
class OpenDuckEnvCfg(DirectRLEnvCfg):
    seed: int = 42
    # =========================
    # 基础
    # =========================
    decimation: int = DECIMATION
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=12,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        ),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1280, 720),
    )

    # =========================
    # Scene
    # =========================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # =========================
    # Robot
    # =========================
    robot: ArticulationCfg = OPEN_DUCK_MINI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    feet_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*foot_assembly.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )

    # =========================
    # Action / Observation
    # =========================
    num_actions: int = 14
    num_observations: int = OBS_DIM
    num_states: int = CRITIC_DIM
    controlled_joint_names: tuple[str, ...] = JOINT_NAMES
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    state_space = gym.spaces.Box(-np.inf, np.inf, shape=(CRITIC_DIM,), dtype=np.float32)
    imu_position_b: tuple[float, float, float] = (-0.08711, 0.0, 0.0422909)
    observation_noise: bool = True
    action_delay_min: int = 0
    action_delay_max: int = 3
    imu_delay_min: int = 0
    imu_delay_max: int = 3
    gyro_noise: float = 0.1
    accelerometer_noise: float = 0.05
    joint_pos_noise: float = 0.03
    joint_vel_noise: float = 0.5
    gyro_bias_range: float = 0.02
    accelerometer_bias_range: float = 0.05
    backlash_half_width: float = 0.0087266463
    verify_randomization: bool = True
    head_control_mode: str = "policy"
    target_filter_tau_s: float = 0.0
    head_target_speed: float = 1.0
    reward_profile: str = "legacy"
    motion_skill: str = "forward"
    reward_time_scale: bool = False
    critic_context: str = "terrain"
    heading_hold_kp: float = 0.0
    heading_hold_max_rate: float = 0.35
    heading_hold_filter_tau: float = 0.25
    heading_imu_noise: float = 0.01
    lateral_command_trim: float = 0.0
    forward_command_scale: float = 1.0
    joint_gain_scale_range: tuple[float, float] = (0.9, 1.1)
    armature_scale_range: tuple[float, float] = (1.0, 1.05)

    # =========================
    # Reward 参数
    # =========================
    action_scale: float = 0.25
    lin_vel_scale: float = 1.0
    energy_cost_scale: float = 0.05
    alive_reward_scale: float = 1.0
    termination_height: float = 0.08
    max_root_angle: float = 0.75  # radians

    use_rough_terrain: bool = False
    spawn_height: float = SPAWN_HEIGHT
    target_base_height: float = STANCE_HEIGHT
    reset_root_ang_vel_range: float = 0.15

    enable_height_scan: bool = False
    height_scan_resolution: float = 0.1
    height_scan_size: tuple[float, float] = (1.2, 0.8)
    height_scan_offset: tuple[float, float, float] = (0.15, 0.0, 0.35)
    height_scan_num_points: int = 117
    height_scan_clip: float = 0.25

    randomize_domain: bool = True
    material_buckets: int = 32
    mechanical_collision_filter_distance: int = 2
    ground_static_friction_range: tuple[float, float] = (0.75, 1.45)
    ground_dynamic_friction_range: tuple[float, float] = (0.65, 1.20)
    ground_restitution_range: tuple[float, float] = (0.0, 0.04)
    body_mass_scale_range: tuple[float, float] = (0.94, 1.08)
    base_com_offset_range: tuple[float, float] = (-0.005, 0.005)
    motor_effort_scale_range: tuple[float, float] = (0.90, 1.08)
    joint_friction_range: tuple[float, float] = (0.0468, 0.0572)

    command_resampling_time_s: float = 3.0
    stand_command_probability: float = 0.02
    command_lin_vel_x: tuple[float, float] = (-0.15, 0.15)
    command_lin_vel_y: tuple[float, float] = (-0.20, 0.20)
    command_yaw_rate: tuple[float, float] = (-1.0, 1.0)

    vel_sigma: float = 0.045
    yaw_rate_sigma: float = 0.08
    height_sigma: float = 0.015
    energy_penalty_scale: float = 0.00008
    rew_scale_lin_vel_xy: float = 4.0
    rew_scale_lin_vel_progress: float = 7.0
    rew_scale_yaw_rate: float = 0.8
    rew_scale_yaw_progress: float = 2.0
    rew_scale_upright: float = 1.6
    rew_scale_height: float = 0.6
    penalty_scale_lin_z: float = 0.35
    penalty_scale_ang_xy: float = 0.8
    no_motion_penalty_scale: float = 9.0
    underspeed_penalty_scale: float = 5.5
    yaw_no_motion_penalty_scale: float = 3.0
    in_place_stepping_penalty_scale: float = 0.3
    fall_penalty_scale: float = 8.0
    min_move_command_speed: float = 0.04
    no_motion_sigma: float = 0.03
    action_rate_penalty_scale: float = 0.015
    joint_deviation_penalty_scale: float = 0.0008
    rew_scale_alive: float = 0.05

    foot_body_names: tuple[str, str] = ("foot_assembly", "foot_assembly_2")
    gait_cycle_time_s: float = 0.8
    gait_duty_factor: float = 0.58
    gait_reward_phase_offset: float = 0.0
    foot_contact_force_threshold: float = 1.0
    min_air_time: float = 0.10
    target_step_length: float = 0.080
    foot_clearance_target: float = 0.060
    rew_scale_gait_contact: float = 0.35
    rew_scale_feet_air_time: float = 0.25
    rew_scale_step_length: float = 0.90
    rew_scale_foot_clearance: float = 0.60


@configclass
class OpenDuckRoughEnvCfg(OpenDuckEnvCfg):
    """Same deployable actor; ideal height scans are critic-only."""

    use_rough_terrain: bool = True
    enable_height_scan: bool = True


@configclass
class OpenDuckStraightEnvCfg(OpenDuckEnvCfg):
    """Full v2 mass, command-held head, and explicit stop/straight tracking."""

    head_control_mode: str = "command"
    target_filter_tau_s: float = 0.04
    reward_profile: str = "straight_v2"
    stand_command_probability: float = 0.25
    command_lin_vel_x: tuple[float, float] = (0.04, 0.12)
    command_lin_vel_y: tuple[float, float] = (0.0, 0.0)
    command_yaw_rate: tuple[float, float] = (0.0, 0.0)
    min_move_command_speed: float = 0.02
    reward_velocity_filter_s: float = 0.10
    foot_clearance_target: float = 0.06
    target_step_length: float = 0.04
    rew_scale_gait_contact: float = 1.2
    rew_scale_feet_air_time: float = 2.0
    rew_scale_step_length: float = 0.2
    rew_scale_foot_clearance: float = 1.0
    min_air_time: float = 0.06
    fall_penalty_scale: float = 12.0


@configclass
class OpenDuckStraightV3EnvCfg(OpenDuckStraightEnvCfg):
    """Preserve learned neck balance; constrain head speed and penalize net drift."""

    head_control_mode: str = "stabilized"
    target_filter_tau_s: float = 0.0
    head_target_speed: float = 1.0
    reward_profile: str = "straight_v3"
    reward_time_scale: bool = True
    reward_velocity_filter_s: float = 0.4
    gait_cycle_time_s: float = 0.54


@configclass
class OpenDuckStraightV4EnvCfg(OpenDuckStraightV3EnvCfg):
    """Penalize accumulated world-heading and lateral path error, with full v2 mass."""

    reward_profile: str = "straight_v4"
    command_resampling_time_s: float = 6.0


@configclass
class OpenDuckStraightV5EnvCfg(OpenDuckStraightV4EnvCfg):
    """Make path-shaped returns observable to the critic, preserving actor 101."""

    reward_profile: str = "straight_v5"
    critic_context: str = "flat_path"
    foot_clearance_target: float = 0.065
    rew_scale_foot_clearance: float = 3.0


@configclass
class OpenDuckStraightV6EnvCfg(OpenDuckStraightV5EnvCfg):
    """Align reward swing/contact timing with the official 0.54s actor gait."""

    reward_profile: str = "straight_v6"
    gait_reward_phase_offset: float = 0.25


@configclass
class OpenDuckBackwardEnvCfg(OpenDuckStraightV6EnvCfg):
    """Separate reverse-walking learner; forward checkpoint remains immutable."""

    motion_skill: str = "backward"
    command_lin_vel_x: tuple[float, float] = (-0.12, -0.04)
    stand_command_probability: float = 0.15
    heading_hold_kp: float = 2.0
    forward_command_scale: float = 1.0
