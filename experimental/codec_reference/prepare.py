#!/usr/bin/env python3
"""Verify preserved bytes and prepare one self-contained decoder fixture."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--tools", type=Path, help="Built loom/src/loom/tools directory"
    )
    parser.add_argument(
        "--output", type=Path, help="New directory outside the source tree"
    )
    parser.add_argument(
        "--case",
        choices=(
            "small-r3-w1",
            "small-r3-w128-invalid",
            "small-r3-w128-invalid-padding",
        ),
        default="small-r3-w1",
    )
    parser.add_argument("--variant", choices=("baseline", "polling"), default="polling")
    arguments = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Archive content changed: {entry['path']}")
    snapshot = manifest["source_archive"]
    archive_path = ROOT / snapshot["path"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != snapshot["sha256"]:
        raise RuntimeError("Source archive content changed")
    with tarfile.open(archive_path, "r:gz") as archive:
        for entry in snapshot["entries"]:
            data = archive.extractfile(entry["member"]).read()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Source snapshot changed: {entry['member']}")
    count = len(manifest["entries"]) + len(snapshot["entries"])
    print(f"Verified {count} preserved files.", flush=True)
    if arguments.verify_only:
        return
    if arguments.tools is None or arguments.output is None:
        parser.error("preparation requires --tools and --output")
    output = arguments.output.resolve()
    if output.is_relative_to(ROOT.parents[1]):
        parser.error("--output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / "fixtures" / arguments.case
    for path in fixture.iterdir():
        shutil.copyfile(path, output / path.name)
    source = ROOT / "kernels/resize" / (arguments.variant + ".cxx")
    tools = arguments.tools.resolve()
    commands = [
        [
            str(tools / "loom-import-cxx/loom-import-cxx"),
            str(source),
            "--output=" + str(output / "kernels.loom"),
        ],
        [
            str(tools / "loom-link/loom-link"),
            str(output / "kernels.loom"),
            str(output / "command.loom"),
            "--to=bc",
            "--output=" + str(output / "summary.loombc"),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    print(output / "summary.loombc")


if __name__ == "__main__":
    main()
