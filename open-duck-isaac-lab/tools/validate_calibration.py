#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_duck_v2.calibration import calibration_issues

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Check a measured hardware record; the template intentionally remains incomplete."
    )
    p.add_argument("record", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    issues = calibration_issues(json.loads(a.record.read_text()))
    report = {
        "status": "incomplete" if issues else "record_valid",
        "issues": issues,
        "note": "Record validity does not establish walking performance or replace fitting the simulator to these measurements.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(1 if issues else 0)
