#!/usr/bin/env python3
"""Restore the published three-policy bundle using only the Python standard library."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = "v0.2.0-isaac-multiskill"
ARCHIVE_NAME = "open-duck-v2-multiskill-policies.zip"
ARCHIVE_SHA256 = "a45d93f30365e267bce56c639ba81b709dda3334c3a2f3465feebb12afdb7f84"
URL = (
    "https://github.com/happylee8282/openduck_v2_ws/releases/download/"
    + RELEASE_TAG
    + "/"
    + ARCHIVE_NAME
)
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def policy_records():
    manifest = json.loads((ROOT / "config/multiskill_policy.json").read_text())
    reference = manifest["posture_reference"]
    if sha256((ROOT / reference["path"]).read_bytes()) != reference["sha256"]:
        raise ValueError("Posture reference differs from the published manifest")
    records = []
    for skill in ("forward", "backward", "posture"):
        record = manifest["skills"][skill]
        relative = Path(record["checkpoint"])
        if relative.parent != Path("policies") or relative.suffix != ".pt":
            raise ValueError("Unexpected policy path in the published manifest")
        records.append((relative, record["sha256"]))
    return records


def check_installed(records):
    for relative, expected in records:
        target = ROOT / relative
        if not target.is_file() or sha256(target.read_bytes()) != expected:
            raise ValueError(f"Published policy missing or changed: {relative}")
        print(f"SHA-256 OK: {relative}")


def install_archive(archive, records):
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive exceeds the published bundle size limit")
    if sha256(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("ZIP SHA-256 mismatch; the published bundle was not installed")
    pending = []
    with zipfile.ZipFile(archive) as bundle:
        for relative, expected in records:
            member = "open-duck-isaac-lab/" + relative.as_posix()
            matches = [info for info in bundle.infolist() if info.filename == member]
            if len(matches) != 1 or matches[0].file_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f"Unexpected archive member: {member}")
            data = bundle.read(matches[0])
            if sha256(data) != expected:
                raise ValueError(f"Policy SHA-256 mismatch: {relative}")
            target = ROOT / relative
            # Preflight all files before installing any model.
            if target.is_symlink():
                raise ValueError(f"Refusing a policy symlink: {relative}")
            if target.exists():
                if not target.is_file() or sha256(target.read_bytes()) != expected:
                    raise ValueError(
                        f"Existing file differs; move it aside first: {relative}"
                    )
            else:
                pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        try:
            # Link atomically without overwriting a concurrently created policy.
            os.link(temporary_path, target)
            target.chmod(0o644)
        finally:
            temporary_path.unlink(missing_ok=True)
    check_installed(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--archive", type=Path, help="Use the exact published ZIP offline"
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Check installed models without network or writes",
    )
    args = parser.parse_args()
    try:
        records = policy_records()
        if args.check_only:
            check_installed(records)
            return
        if args.archive:
            install_archive(args.archive.expanduser().resolve(), records)
            return
        try:
            check_installed(records)
            print("All published policies are already installed.")
            return
        except ValueError:
            pass
        with tempfile.TemporaryDirectory(
            prefix="openduck-policy-download-"
        ) as directory:
            archive = Path(directory) / ARCHIVE_NAME
            print(f"Downloading {URL}", flush=True)
            request = urllib.request.Request(
                URL, headers={"User-Agent": "OpenDuck-policy-downloader"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                archive.open("wb") as output,
            ):
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError(
                            "Download exceeds the published bundle size limit"
                        )
                    output.write(chunk)
            install_archive(archive, records)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Policy restore failed: {error}\n")


if __name__ == "__main__":
    main()
