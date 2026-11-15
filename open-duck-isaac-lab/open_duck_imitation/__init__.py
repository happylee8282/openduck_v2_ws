"""Register the versioned v2 motion-reference experiment."""

import gymnasium as gym

gym.register(
    id="Isaac-Open-Duck-Imitation-Direct-v0",
    entry_point=f"{__name__}.open_duck_env:OpenDuckImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_duck_env_cfg:OpenDuckImitationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
