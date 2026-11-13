# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises one cached Windows image under concurrent process creation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tools.ci import windows_diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_failed_image(executable: Path, artifact_dir: Path) -> list[str]:
    preserved = []
    for path in (executable, executable.with_suffix(".pdb")):
        if not path.is_file():
            continue
        destination = artifact_dir / path.name
        shutil.copy2(path, destination)
        preserved.append(destination.name)
    return preserved


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launches a cached Windows Loom image concurrently while checking "
            "that its on-disk bytes remain equal to the seeded digest."
        )
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=40)
    parser.add_argument("--waves", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    executable = args.executable.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    expected_sha256 = args.expected_sha256.lower()
    if args.parallelism < 1 or args.waves < 1:
        raise ValueError("parallelism and waves must both be positive")

    manifest = {
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "executable": str(executable),
        "input": str(input_path),
        "expected_sha256": expected_sha256,
        "parallelism": args.parallelism,
        "waves_requested": args.waves,
        "waves_completed": 0,
        "launches_completed": 0,
        "failures": [],
    }
    result = 0
    started_at = time.monotonic()
    dump_state = {}
    try:
        initial_sha256 = sha256_file(executable)
        manifest["initial_sha256"] = initial_sha256
        manifest["size"] = executable.stat().st_size
        if initial_sha256 != expected_sha256:
            raise RuntimeError(
                f"materialized image digest {initial_sha256} does not match "
                f"seeded digest {expected_sha256}"
            )

        with windows_diagnostics.local_dump_collection(
            artifact_dir / "crash-dumps"
        ) as dump_state:
            if not dump_state.get("enabled"):
                raise RuntimeError(
                    "Windows Error Reporting dump collection is unavailable: "
                    + json.dumps(dump_state, sort_keys=True)
                )
            for wave in range(args.waves):
                before_sha256 = sha256_file(executable)
                if before_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"image digest changed before wave {wave}: "
                        f"{before_sha256} != {expected_sha256}"
                    )
                command = [
                    str(executable),
                    "--mode=merge",
                    "--to=bc",
                    "--output=-",
                    str(input_path),
                ]
                processes = [
                    subprocess.Popen(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    for _ in range(args.parallelism)
                ]
                failures = []
                for index, process in enumerate(processes):
                    _, stderr = process.communicate()
                    if process.returncode:
                        failures.append(
                            {
                                "wave": wave,
                                "index": index,
                                "returncode": process.returncode,
                                "stderr": stderr.decode("utf-8", errors="replace")[
                                    -65536:
                                ],
                            }
                        )
                manifest["launches_completed"] += len(processes)
                if failures:
                    manifest["failures"].extend(failures)
                    raise RuntimeError(
                        f"{len(failures)} of {len(processes)} launches failed "
                        f"in wave {wave}"
                    )
                after_sha256 = sha256_file(executable)
                if after_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"image digest changed after wave {wave}: "
                        f"{after_sha256} != {expected_sha256}"
                    )
                manifest["waves_completed"] = wave + 1
                if (wave + 1) % 10 == 0 or wave + 1 == args.waves:
                    print(
                        f"[cache-canary] {wave + 1}/{args.waves} waves, "
                        f"{manifest['launches_completed']} launches",
                        flush=True,
                    )
    except Exception as error:
        result = 1
        manifest["error"] = str(error)
        manifest["preserved_images"] = preserve_failed_image(executable, artifact_dir)
        print(f"[cache-canary] {error}", file=sys.stderr, flush=True)
    finally:
        manifest["dump_collection"] = dump_state
        manifest["completed_at"] = dt.datetime.now(dt.UTC).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started_at
        manifest["final_sha256"] = sha256_file(executable)
        (artifact_dir / "stress-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
