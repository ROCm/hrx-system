#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the HRX core build in TheRock's manylinux build container."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import DEFAULT_BUILD_IMAGE, REPO_ROOT, run


def do_build(args: argparse.Namespace, rest_args: list[str]) -> int:
    if args.pull:
        run([args.docker, "pull", args.image], cwd=REPO_ROOT)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [args.docker, "run", "--rm"]
    if sys.stdin.isatty():
        cmd.extend(["-i", "-t"])
    cmd.extend(
        [
            "--mount",
            f"type=bind,src={REPO_ROOT},dst=/hrx/src",
            "--mount",
            f"type=bind,src={output_dir},dst=/hrx/output",
            "-e",
            "HRX_OUTPUT_DIR=/hrx/output",
            "-e",
            f"HRX_RELEASE_TYPE={args.release_type}",
            "-e",
            f"HRX_ARTIFACT_SET={args.artifact_set}",
        ]
    )
    if args.run_id:
        cmd.extend(["-e", f"HRX_RUN_ID={args.run_id}"])
    if args.gpu:
        cmd.extend(["-e", "HRX_TEST_GPU=1"])
    for env_name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        if env_name in os.environ:
            cmd.extend(["-e", f"{env_name}={os.environ[env_name]}"])

    if args.exec:
        cmd.append(args.image)
        cmd.extend(rest_args)
    elif args.interactive:
        cmd.extend([args.image, "/bin/bash"])
    else:
        cmd.extend(
            [
                args.image,
                "/bin/bash",
                "/hrx/src/build_tools/detail/linux_core_build_in_container.sh",
            ]
        )
        cmd.extend(rest_args)

    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        rest_pos = argv.index("--")
    except ValueError:
        rest_args: list[str] = []
    else:
        rest_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default=DEFAULT_BUILD_IMAGE)
    parser.add_argument("--pull", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT.parent / "build" / "linux",
    )
    parser.add_argument(
        "--release-type", default="nightly", choices=["dev", "nightly", "prerelease"]
    )
    parser.add_argument("--artifact-set", default="core")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--exec", action=argparse.BooleanOptionalAction)
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction)
    args = parser.parse_args(argv)
    return do_build(args, rest_args)


if __name__ == "__main__":
    raise SystemExit(main())
