#!/usr/bin/env python3
"""Download a public CMU MoCap BVH clip for Open Duck imitation training.

The default clip is intentionally configurable. The CMU database and the
community BVH mirrors use stable subject/clip IDs, but dance clips are not all
equally suitable for a small two-legged robot. Start with slow dance or march
clips, then retarget with ``tools/retarget_bvh_to_open_duck.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


import hashlib
import io
import json
import zipfile

# CMU BVH conversion mirror, pinned commit; source terms: mocap.cs.cmu.edu.
DEFAULT_CLIPS = {
    "cmu_05_01": {
        "url": "https://raw.githubusercontent.com/Shriinivas/cmubvh/3bc9c75c481ac6e622f3c378369b293d8de9be45/Sequence-001-009/05/Data/05_01.zip",
        "sha256": "91a7b15b7f02cb19af15cb82b0fd8ab6f12e9230bf2903b353ebcb413f77d1ea",
    },
    "cmu_05_20": {
        "url": "https://raw.githubusercontent.com/Shriinivas/cmubvh/3bc9c75c481ac6e622f3c378369b293d8de9be45/Sequence-001-009/05/Data/05_20.zip",
        "sha256": "ea1526e9486ab906b569ae780cd33b47ce6eb28e02bdcbbb0150f9eeeea799b3",
    },
}
DEFAULT_URLS = {name: entry["url"] for name, entry in DEFAULT_CLIPS.items()}


def download(url: str, output: Path, expected_sha256=None):
    with urlopen(url, timeout=30) as response:
        raw = response.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Motion download exceeds 32 MiB")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Motion SHA-256 mismatch")
    if raw.startswith(b"PK"):
        archive = zipfile.ZipFile(io.BytesIO(raw))
        files = [x for x in archive.infolist() if x.filename.lower().endswith(".bvh")]
        if len(files) != 1 or files[0].file_size > 32 * 1024 * 1024:
            raise ValueError("Expected one bounded BVH member")
        raw = archive.read(files[0])  # Read bytes; never extract archive paths.
    if not raw.decode("utf-8-sig").lstrip().startswith("HIERARCHY"):
        raise ValueError("Downloaded file is not BVH")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    output.with_suffix(output.suffix + ".source.json").write_text(
        json.dumps(
            {
                "url": url,
                "download_sha256": digest,
                "bvh_sha256": hashlib.sha256(raw).hexdigest(),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="cmu_05_20", choices=sorted(DEFAULT_URLS))
    parser.add_argument("--url", default=None, help="Direct BVH URL. Overrides --clip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/motions/cmu_05_20.bvh"),
        help="Where to save the BVH file.",
    )
    args = parser.parse_args()

    url = args.url or DEFAULT_URLS[args.clip]
    download(url, args.output, None if args.url else DEFAULT_CLIPS[args.clip]["sha256"])
    print(f"saved {args.output} from {url}")


if __name__ == "__main__":
    main()
