# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for GitHub dependency snapshot generation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from build_tools.dependencies.github_snapshot import build_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project an HRX dependency inventory into GitHub JSON."
    )
    parser.add_argument(
        "--inventory",
        default="-",
        help="Derived inventory JSON, or - for stdin.",
    )
    parser.add_argument("--sha", required=True)
    parser.add_argument("--ref", default="refs/heads/main")
    parser.add_argument("--scanned")
    parser.add_argument("--job-id")
    parser.add_argument("--job-url")
    args = parser.parse_args()

    if args.inventory == "-":
        inventory = json.load(sys.stdin)
    else:
        with Path(args.inventory).open(encoding="utf-8") as file:
            inventory = json.load(file)
    scanned = (
        args.scanned
        or datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    )
    job_id = args.job_id or os.environ.get("GITHUB_RUN_ID") or f"local-{scanned}"
    job_url = args.job_url or _github_job_url()
    snapshot = build_snapshot(
        inventory,
        sha=args.sha,
        ref=args.ref,
        scanned=scanned,
        job_id=job_id,
        job_url=job_url,
    )
    json.dump(snapshot, sys.stdout, indent=2)
    print()
    return 0


def _github_job_url() -> str:
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repository and run_id:
        return f"https://github.com/{repository}/actions/runs/{run_id}"
    return "https://github.com/ROCm/hrx-system"


if __name__ == "__main__":
    raise SystemExit(main())
