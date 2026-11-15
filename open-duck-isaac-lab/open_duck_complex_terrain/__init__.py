"""Explicitly import this package (or use tools/launch.py) to register tasks."""

import gymnasium as gym

for task, cfg in [
    ("Isaac-Open-Duck-Direct-v0", "OpenDuckEnvCfg"),
    ("Isaac-Open-Duck-Straight-Direct-v0", "OpenDuckStraightEnvCfg"),
    ("Isaac-Open-Duck-StraightV3-Direct-v0", "OpenDuckStraightV3EnvCfg"),
    ("Isaac-Open-Duck-StraightV4-Direct-v0", "OpenDuckStraightV4EnvCfg"),
    ("Isaac-Open-Duck-StraightV5-Direct-v0", "OpenDuckStraightV5EnvCfg"),
    ("Isaac-Open-Duck-StraightV6-Direct-v0", "OpenDuckStraightV6EnvCfg"),
    ("Isaac-Open-Duck-Backward-Direct-v0", "OpenDuckBackwardEnvCfg"),
    ("Isaac-Open-Duck-Rough-Direct-v0", "OpenDuckRoughEnvCfg"),
]:
    gym.register(
        id=task,
        entry_point=f"{__name__}.open_duck_env:OpenDuckMiniEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.open_duck_env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )


gym.register(
    id="Isaac-Open-Duck-Posture-Direct-v0",
    entry_point=f"{__name__}.posture_env:OpenDuckPostureEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.posture_env:OpenDuckPostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
