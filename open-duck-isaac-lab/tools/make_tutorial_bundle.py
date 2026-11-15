#!/usr/bin/env python3
"""Connect one training candidate to a separate, hash-checked playback manifest.

This writes JSON only. It never trains, changes checkpoint weights, or activates
an experimental policy as the default. Run names resolve inside logs/rsl_rl.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENTS = {
    "forward": "open_duck_v2_straight",
    "backward": "open_duck_v2_backward",
    "posture": "open_duck_v2_posture",
}
PROTECTED = {"multiskill_policy.json", "validated_flat_policy.json"}


def rooted(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def select_checkpoint(skill, run_name, iteration=None):
    """Select by run-name suffix, then numeric iteration; never by quality."""
    if not run_name or any(c in run_name for c in ("/", "\\", "\n")):
        raise ValueError("run-name must be the label passed to --run_name")
    folder = ROOT / "logs/rsl_rl" / EXPERIMENTS[skill]
    runs = sorted(
        p for p in folder.glob("*") if p.is_dir() and p.name.endswith("_" + run_name)
    )
    if not runs:
        raise ValueError(f"No run named {run_name!r} in {folder}; train it first")
    run = runs[-1]
    files = []
    for path in run.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and (iteration is None or int(match[1]) == iteration):
            files.append((int(match[1]), path))
    if not files:
        raise ValueError(f"No saved checkpoint for iteration {iteration} in {run}")
    return max(files, key=lambda item: item[0])[1].resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", required=True, choices=list(EXPERIMENTS))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-name", help="Same label as training --run_name")
    source.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--iteration", type=int, help="Explicit saved iteration within a run"
    )
    p.add_argument(
        "--base-bundle", type=Path, default=ROOT / "config/multiskill_policy.json"
    )
    p.add_argument("--output", type=Path, help="New candidate JSON under config/")
    p.add_argument(
        "--print-checkpoint",
        action="store_true",
        help="Print the selected path only; no writes",
    )
    args = p.parse_args()
    try:
        if args.iteration is not None and not args.run_name:
            raise ValueError("--iteration is used with --run-name")
        if args.iteration is not None and args.iteration < 0:
            raise ValueError("--iteration must be nonnegative")
        checkpoint = (
            select_checkpoint(args.skill, args.run_name, args.iteration)
            if args.run_name
            else rooted(args.checkpoint)
        )
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint missing: {checkpoint}")
        if args.print_checkpoint:
            if args.output:
                raise ValueError("--print-checkpoint does not write --output")
            print(checkpoint)
            return
        if not args.output:
            raise ValueError("Provide --output config/<new_candidate>.json")
        output = rooted(args.output)
        if output.parent != (ROOT / "config").resolve() or output.suffix != ".json":
            raise ValueError(
                "Write the new candidate directly inside this repository's config/"
            )
        if output.name in PROTECTED:
            raise ValueError(
                "The validated default manifests are protected; choose a candidate name"
            )
        if output.exists():
            raise ValueError(
                f"Candidate already exists: {output.name}; choose a new name"
            )
        base_path = rooted(args.base_bundle)
        from open_duck_v2.multiskill import load_manifest
        from open_duck_v2.runtime import load_rsl_rl_actor
        from open_duck_v2.posture_runtime import load_posture_actor

        checked = load_manifest(base_path)
        if args.skill == "posture":
            actor = load_posture_actor(checkpoint, checked["reference_path"])
        else:
            actor = load_rsl_rl_actor(checkpoint)
        actual_skill = actor.control_profile.get("motion_skill", "forward")
        if actual_skill != args.skill:
            raise ValueError(
                f"Checkpoint declares {actual_skill!r}, requested {args.skill!r}"
            )
        base = json.loads(base_path.read_text())
        try:
            checkpoint_record = str(checkpoint.relative_to(ROOT))
        except ValueError:
            checkpoint_record = str(checkpoint)
        candidate = {
            "status": "candidate_pending_validation",
            "skills": copy.deepcopy(base["skills"]),
            "posture_reference": copy.deepcopy(base["posture_reference"]),
            "validation_scope": "Candidate only; run independent evaluation and assessment before selection.",
            "tutorial_candidate": {
                "created_by": "tools/make_tutorial_bundle.py",
                "changed_skill": args.skill,
                "base_bundle": str(base_path.relative_to(ROOT)),
                "base_bundle_sha256": hashlib.sha256(
                    base_path.read_bytes()
                ).hexdigest(),
                "selection": "explicit checkpoint"
                if args.checkpoint
                else "explicit iteration"
                if args.iteration is not None
                else "latest matching run, highest saved iteration; not ranked by performance",
                "control_profile": actor.control_profile,
                "weights_modified": False,
            },
        }
        candidate["skills"][args.skill] = {
            "checkpoint": checkpoint_record,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        # Exclusive creation protects previous candidates even if another process
        # writes the same filename after the existence check.
        with output.open("x") as stream:
            stream.write(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": "candidate_created_not_validated",
                    "bundle": str(output.relative_to(ROOT)),
                    "skill": args.skill,
                    "checkpoint": checkpoint_record,
                    "control_profile": actor.control_profile,
                    "play": f"./run.sh play-skills --bundle {output.relative_to(ROOT)} --demo",
                    "note": "Original checkpoints and default manifests are unchanged. Latest does not mean best.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
