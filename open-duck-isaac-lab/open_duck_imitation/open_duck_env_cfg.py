# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from open_duck_complex_terrain.open_duck_env_cfg import (
    OPEN_DUCK_MINI_CFG,
    OpenDuckEnvCfg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMITATION_OPEN_DUCK_MINI_CFG = deepcopy(OPEN_DUCK_MINI_CFG)
IMITATION_OPEN_DUCK_MINI_CFG.spawn.usd_path = str(
    PROJECT_ROOT / "Robots" / "open_duck_mini_training.usd"
)


@configclass
class OpenDuckImitationEnvCfg(OpenDuckEnvCfg):
    """Open Duck motion imitation task driven by a retargeted BVH reference."""

    episode_length_s: float = 6.0
    use_rough_terrain: bool = False
    enable_height_scan: bool = False
    robot: ArticulationCfg = IMITATION_OPEN_DUCK_MINI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    num_observations: int = 101
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(101,),
        dtype=np.float32,
    )
    motion_file: str = str(
        PROJECT_ROOT / "data" / "motions" / "open_duck_cmu_imitation.npz"
    )
    random_motion_start: bool = True
    motion_loop: bool = True

    imitation_joint_sigma: float = 0.06
    imitation_joint_vel_sigma: float = 1.25
    imitation_height_sigma: float = 0.006
    imitation_lin_vel_sigma: float = 0.05
    rew_scale_imitation_joint_pos: float = 6.0
    rew_scale_imitation_joint_vel: float = 1.0
    rew_scale_imitation_height: float = 1.4
    rew_scale_imitation_lin_vel: float = 0.8
    rew_scale_upright: float = 1.5
    rew_scale_alive: float = 0.05
    energy_penalty_scale: float = 0.00006
    action_rate_penalty_scale: float = 0.012
    fall_penalty_scale: float = 8.0
    joint_deviation_penalty_scale: float = 0.0004
