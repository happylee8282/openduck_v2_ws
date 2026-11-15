# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
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
"""Configuration for the Open Duck Mini robot in Isaac Lab."""

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPEN_DUCK_TRAINING_USD = PROJECT_ROOT / "Robots" / "open_duck_mini_training.usd"



OPEN_DUCK_MINI_CFG = ArticulationCfg(
    prim_path="/World/envs/env_0/Robot",  # 多环境时自动替换
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(OPEN_DUCK_TRAINING_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            linear_damping=0.02,
            angular_damping=0.02,
            max_linear_velocity=10.0,
            max_angular_velocity=20.0,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22),
        joint_pos={
            "right_hip_yaw":   0.0,
            "right_hip_roll":  0.0,
            "right_hip_pitch": 0.10,
            "right_knee":      -0.32,
            "right_ankle":     0.16,

            "left_hip_yaw":    0.0,
            "left_hip_roll":   0.0,
            "left_hip_pitch":   0.10,
            "left_knee":       -0.32,
            "left_ankle":       0.16,

            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw":   0.0,
            "head_roll":  0.0,
        },
        joint_vel={jname: 0.0 for jname in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll"
        ]},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee", ".*_ankle"],
            effort_limit_sim=55.0,
            velocity_limit_sim=16.0,
            stiffness=160.0,
            damping=10.0,
            armature=0.01,
            friction=0.02,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["neck_.*", "head_.*"],
            effort_limit_sim=18.0,
            velocity_limit_sim=8.0,
            stiffness=90.0,
            damping=6.0,
            armature=0.005,
            friction=0.03,
        )
    },
)

@ configclass
class OpenDuckEnvCfg(DirectRLEnvCfg):
    # =========================
    # 基础
    # =========================
    decimation: int = 4
    episode_length_s: float = 20.0
    is_finite_horizon: bool = False
    
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        dt=1.0 / 120.0,
        render_interval=4,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=12,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        )
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
        num_envs=128,
        env_spacing=3.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # =========================
    # Robot
    # =========================
    robot: ArticulationCfg = OPEN_DUCK_MINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
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
    num_observations: int = 40
    controlled_joint_names: tuple[str, ...] = (
        "left_hip_yaw",
        "right_hip_yaw",
        "left_hip_roll",
        "right_hip_roll",
        "left_hip_pitch",
        "right_hip_pitch",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
    )

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(14,),
        dtype=np.float32,
    )
    
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(40,),
        dtype=np.float32,
    )
    
    state_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(40,),
        dtype=np.float32,
    )

    # =========================
    # Reward 参数
    # =========================
    action_scale: float = 1.0
    action_scale_yaw: float = 0.22
    action_scale_roll: float = 0.28
    action_scale_pitch: float = 0.55
    action_scale_knee: float = 0.65
    action_scale_ankle: float = 0.55
    action_scale_neck: float = 0.12
    action_scale_head_pitch: float = 0.15
    action_scale_head_yaw: float = 0.09
    action_scale_head_roll: float = 0.07
    lin_vel_scale: float = 1.0
    energy_cost_scale: float = 0.05
    alive_reward_scale: float = 1.0
    termination_height: float = 0.11
    max_root_angle: float = 0.75  # radians
    
    use_rough_terrain: bool = True
    spawn_height: float = 0.22
    target_base_height: float = 0.22
    reset_root_ang_vel_range: float = 0.15

    enable_height_scan: bool = False
    height_scan_resolution: float = 0.1
    height_scan_size: tuple[float, float] = (1.2, 0.8)
    height_scan_offset: tuple[float, float, float] = (0.15, 0.0, 0.35)
    height_scan_num_points: int = 117
    height_scan_clip: float = 0.25

    randomize_domain: bool = False
    ground_static_friction_range: tuple[float, float] = (0.65, 1.8)
    ground_dynamic_friction_range: tuple[float, float] = (0.55, 1.45)
    ground_restitution_range: tuple[float, float] = (0.0, 0.04)
    body_mass_scale_range: tuple[float, float] = (0.90, 1.12)
    base_com_offset_range: tuple[float, float] = (-0.012, 0.012)
    motor_effort_scale_range: tuple[float, float] = (0.82, 1.12)
    joint_friction_range: tuple[float, float] = (0.01, 0.06)

    command_resampling_time_s: float = 3.0
    stand_command_probability: float = 0.02
    command_lin_vel_x: tuple[float, float] = (-0.35, 0.45)
    command_lin_vel_y: tuple[float, float] = (-0.28, 0.28)
    command_yaw_rate: tuple[float, float] = (-0.60, 0.60)
    command_cardinal_probability: float = 0.90
    command_min_cardinal_speed: float = 0.16
    
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
    penalty_scale_ang_xy: float = 0.6
    no_motion_penalty_scale: float = 7.0
    underspeed_penalty_scale: float = 4.0
    yaw_no_motion_penalty_scale: float = 2.0
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
    foot_contact_force_threshold: float = 1.0
    min_air_time: float = 0.08
    target_step_length: float = 0.055
    foot_clearance_target: float = 0.024
    rew_scale_gait_contact: float = 0.35
    rew_scale_feet_air_time: float = 0.25
    rew_scale_step_length: float = 0.6
    rew_scale_foot_clearance: float = 0.25
