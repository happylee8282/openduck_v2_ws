# Open Duck Imitation Task

This package contains a direct RL imitation-learning task for Open Duck Mini.
It is intended as a starting point for motion-reference experiments, not a
complete high-fidelity whole-body retargeting pipeline.

## Task ID

```text
Isaac-Open-Duck-Imitation-Direct-v0
```

## Contents

```text
open_duck_imitation/
├── __init__.py                  # Gym task registration
├── open_duck_env.py             # Direct RL imitation environment
├── open_duck_env_cfg.py         # Robot, scene, reward, and motion config
└── agents/
    └── rsl_rl_ppo_cfg.py        # RSL-RL PPO defaults
```

Related tools:

- `tools/download_cmu_mocap.py`: downloads BVH clips from the CMU mocap mirror.
- `tools/retarget_bvh_to_open_duck.py`: maps BVH lower-body motion to the 14
  controlled Open Duck joints and writes an `.npz` motion reference.

## Prepare Motion Data

Run these commands through the Isaac Lab Python launcher:

```powershell
.\isaaclab.bat -p D:\path\to\open-duck-isaac-lab\tools\download_cmu_mocap.py `
  --clip cmu_05_20 `
  --output D:\path\to\open-duck-isaac-lab\data\motions\cmu_05_20.bvh

.\isaaclab.bat -p D:\path\to\open-duck-isaac-lab\tools\retarget_bvh_to_open_duck.py `
  D:\path\to\open-duck-isaac-lab\data\motions\cmu_05_20.bvh `
  --output D:\path\to\open-duck-isaac-lab\data\motions\open_duck_cmu_imitation.npz
```

`data/motions/` is ignored by git because BVH clips and generated motion
references are local experiment artifacts.

## Train

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Open-Duck-Imitation-Direct-v0 `
  --num_envs 1024 `
  --headless
```

For debugging, reduce `--num_envs` to `1`, `64`, or `512`.

## Notes

The current retargeter is intentionally simple: it performs a coarse mapping
from BVH joint angles into Open Duck's controlled joints, then the environment
adds joint-position, joint-velocity, base-height, velocity, and upright rewards.
For dance-quality or contact-rich skills, replace this with an IK-based
retargeting pipeline that enforces foot contacts and base motion consistency.
