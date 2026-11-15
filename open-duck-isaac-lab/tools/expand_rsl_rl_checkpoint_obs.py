"""Legacy observation padding is incompatible with the v2 semantic contract."""

import argparse
from pathlib import Path


def expand_checkpoint(input_path: Path, output_path: Path, old_obs: int, new_obs: int):
    raise ValueError(
        "The v2 observation semantics changed. Retrain using tools/launch.py; weight padding cannot migrate this policy."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-obs", type=int, default=158)
    parser.add_argument("--new-obs", type=int, default=101)
    args = parser.parse_args()
    try:
        expand_checkpoint(args.input, args.output, args.old_obs, args.new_obs)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
