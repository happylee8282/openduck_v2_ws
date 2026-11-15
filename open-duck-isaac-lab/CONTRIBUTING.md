# Contributing

Thanks for helping improve Open Duck Isaac Lab. Keep changes focused and
reproducible so training behavior can be compared across machines.

## Development Workflow

1. Create a branch for each change.
2. Keep generated artifacts out of git: USD files, checkpoints, logs, mocap
   caches, and raw screen recordings.
3. Run formatting or validation scripts that are relevant to the files you
   changed.
4. Include the Isaac Sim, Isaac Lab, CUDA, and GPU versions when reporting
   training or simulation issues.

## Pull Request Checklist

- The README or task documentation is updated for user-facing workflow changes.
- New Isaac Lab tasks register a stable Gym ID in the package `__init__.py`.
- Checkpoint changes are published through Releases, not committed to the repo.
- Robot asset changes include enough context to regenerate or inspect the USD.
- Reward, observation, or action-order changes are called out explicitly.

## Generated Files

The following files are intentionally ignored:

- `Robots/open_duck_mini_training.usd`
- `logs/`, `runs/`, and experiment outputs
- `*.pt`, `*.onnx`, and other model exports
- local mocap data under `data/motions/`
- raw video captures such as `*.mp4` and `*.mov`
