#!/usr/bin/env bash
# Run this project with its Isaac Lab interpreter, including from conda (base).
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(dirname -- "$SCRIPT_PATH")"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/happy/miniconda3/envs/env_isaaclab/bin/python}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/home/happy/IsaacLab}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

mode="${1:-help}"
if (( $# > 0 )); then shift; fi
case "$mode" in
    help|-h|--help)
        cat <<'HELP'
Open Duck Mini v2 / Isaac Lab
Usage: ./duck.sh COMMAND [arguments]   (or ./run.sh in the project directory)

  check   --headless --steps 32       Check the actual GPU/USD/PhysX environment
  test                               Run the CPU regression tests
  train   --headless --num_envs 32 --max_iterations 3 --seed 42
                                     Run a short PPO pipeline check
  train-straight --headless           Fine-tune the validated straight/stop actor
  play-skills                       W/S walking, C/R low pose and stand
  play-skills --demo                 Replay the full multiskill sequence
  train-skill backward|posture --headless
                                     Train a separate skill
  play    --headless --steps 128      Replay the validated policy (latest if none)
  play    --checkpoint /path/model.pt Replay a selected checkpoint
  list    --headless                  List the registered Isaac Lab tasks
  convert --headless                  Regenerate the USD from the URDF
  python  tools/example.py [args]     Run another tool with the Isaac interpreter

Without --headless, playback opens a window on the remote PC's graphical session.
Override ISAAC_PYTHON, ISAACLAB_PATH, or DUCK_CHECKPOINT when needed.
Three training iterations check the pipeline; they do not produce a trained walker.
HELP
        exit 0
        ;;
esac

if [[ ! -x "$ISAAC_PYTHON" ]]; then
    printf 'Isaac Python not found: %s\nSet ISAAC_PYTHON to the Isaac Lab interpreter.\n' "$ISAAC_PYTHON" >&2
    exit 1
fi

case "$mode" in
    check) exec "$ISAAC_PYTHON" tools/validate_isaac.py "$@" ;;
    test) exec "$ISAAC_PYTHON" -m unittest discover -s tests -v "$@" ;;
    play-skills) exec "$ISAAC_PYTHON" tools/play_multiskill.py "$@" ;;
    train-skill) exec "$ISAAC_PYTHON" tools/train_skill.py "$@" ;;
    train-straight) exec "$ISAAC_PYTHON" tools/train_straight.py "$@" ;;
    train|list) exec "$ISAAC_PYTHON" tools/launch.py "$mode" "$@" ;;
    convert) exec "$ISAAC_PYTHON" tools/convert_urdf_to_training_usd.py "$@" ;;
    python) exec "$ISAAC_PYTHON" "$@" ;;
    play)
        play_args=(--project-root "$PROJECT_ROOT" --usd-path "$PROJECT_ROOT/Robots/open_duck_mini_training.usd")
        has_checkpoint=false
        for arg in "$@"; do
            case "$arg" in --checkpoint|--checkpoint=*) has_checkpoint=true ;; esac
        done
        if [[ "$has_checkpoint" == false ]]; then
            checkpoint="${DUCK_CHECKPOINT:-}"
            if [[ -z "$checkpoint" ]]; then
                checkpoint="$("$ISAAC_PYTHON" - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
manifest = root / "config/validated_flat_policy.json"
if manifest.exists():
    record = json.loads(manifest.read_text())
    selected = root / record["checkpoint"]
    if not selected.is_file() or hashlib.sha256(selected.read_bytes()).hexdigest() != record["sha256"]:
        sys.exit("Validated checkpoint missing or changed; restore it or pass --checkpoint /path/model.pt")
    print(selected)
    sys.exit(0)
files = list((root / "logs/rsl_rl").rglob("model_*.pt"))
if not files:
    sys.exit("No checkpoint found. Train first or pass --checkpoint /path/model.pt")
print(max(files, key=lambda p: p.stat().st_mtime_ns))
PY
)"
            fi
            play_args+=(--checkpoint "$checkpoint")
        fi
        exec "$ISAAC_PYTHON" isaac_duck_teleop/scripts/play_open_duck_teleop.py "${play_args[@]}" "$@"
        ;;
    *) printf 'Unknown command: %s\nRun ./duck.sh help for usage.\n' "$mode" >&2; exit 2 ;;
esac
