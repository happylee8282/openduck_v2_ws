# Open Duck Teleoperation and Policy Playback

This module runs a trained Open Duck Mini policy in Isaac Lab with keyboard or
optional gamepad commands.

## Contents

```text
isaac_duck_teleop/
├── README.md
└── scripts/
    ├── play_open_duck_teleop.py       # Isaac Lab keyboard/gamepad playback
    ├── duck_policy_runtime.py         # RSL-RL actor loading helper
    └── export_open_duck_policy.py     # Optional checkpoint export helper
```

## Prerequisites

- `Robots/open_duck_mini_training.usd` has been generated from the root README
  instructions.
- A compatible RSL-RL checkpoint exists under the Isaac Lab log directory or a
  release model bundle.
- The repository root is included in `PYTHONPATH`.

## Run

From the Isaac Lab root directory:

```powershell
.\isaaclab.bat -p D:\path\to\open-duck-isaac-lab\isaac_duck_teleop\scripts\play_open_duck_teleop.py `
  --project-root D:\path\to\open-duck-isaac-lab `
  --checkpoint C:\path\to\IsaacLab\logs\rsl_rl\open_duck_mini_velocity\<run_name>\model_30100.pt `
  --usd-path D:\path\to\open-duck-isaac-lab\Robots\open_duck_mini_training.usd `
  --terrain flat
```

Use rough terrain and height scan:

```powershell
.\isaaclab.bat -p D:\path\to\open-duck-isaac-lab\isaac_duck_teleop\scripts\play_open_duck_teleop.py `
  --project-root D:\path\to\open-duck-isaac-lab `
  --checkpoint C:\path\to\IsaacLab\logs\rsl_rl\open_duck_mini_velocity\<run_name>\model_30100.pt `
  --usd-path D:\path\to\open-duck-isaac-lab\Robots\open_duck_mini_training.usd `
  --terrain rough
```

## Keyboard Controls

| Key | Command |
| --- | --- |
| `W` | Forward |
| `S` | Backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `Q` | Turn left |
| `E` | Turn right |
| `R` | Raise target height |
| `F` | Lower target height |
| `Space` | Clear velocity command |
| `Esc` | Exit |

Default command limits match the training configuration:

```text
vx:       -0.35 to 0.45 m/s
vy:       -0.28 to 0.28 m/s
yaw_rate: -0.35 to 0.35 rad/s
height:    0.155 to 0.235 m
```

## Gamepad Controls

Install the optional dependency through Isaac Lab:

```powershell
.\isaaclab.bat -p -m pip install pygame
```

Add `--gamepad` to the playback command.

```text
left stick Y: forward/backward
left stick X: strafe left/right
right stick X: yaw left/right
```
