#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stages a failed tagged SDK run for exact prerelease publication retry."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path, PurePath

from build_tools.ci import github_api

SOURCE_WORKFLOW = ".github/workflows/release_loom_sdk.yml"
REQUIRED_SUCCESSFUL_JOBS = frozenset(
    {
        "Validate / Release identity",
        "Security / Native production / Analyze",
    }
)
TERMINAL_CONCLUSIONS = {
    "Bundle / Verify artifact set": frozenset({"success", "failure"}),
    "Publish / GitHub assets": frozenset({"failure", "skipped"}),
    "Publish / PyPI wheels": frozenset({"skipped"}),
    "Complete / Advance main development": frozenset({"skipped"}),
}
RECOVERABLE_PUBLICATION_STATES = frozenset(
    {
        ("failure", "skipped"),
        ("success", "failure"),
    }
)
LEGACY_STANDALONE_DIRECTORIES = frozenset({"amdf-info-artifact", "server-artifact"})


def verify_source_run(
    client: github_api.RepositoryClient,
    run_id: int,
    repository: str,
    revision: str,
    tag: str,
) -> None:
    """Requires a failed tag run whose publication prerequisites all passed."""
    if run_id <= 0:
        raise ValueError("Source run ID must be positive")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Source revision must be a full lowercase Git commit")
    document = client.json(f"/actions/runs/{run_id}")
    expected = {
        "id": run_id,
        "path": SOURCE_WORKFLOW,
        "event": "push",
        "status": "completed",
        "conclusion": "failure",
        "head_sha": revision,
        "head_branch": tag,
    }
    if not isinstance(document, dict) or any(
        document.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("Source run is not the failed workflow for this exact tag")
    for field in ("repository", "head_repository"):
        record = document.get(field)
        if not isinstance(record, dict) or record.get("full_name") != repository:
            raise ValueError("Source run does not belong to this repository")

    records = client.pages(f"/actions/runs/{run_id}/jobs?filter=latest", "jobs")
    jobs = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("name"), str)
            or not isinstance(record.get("status"), str)
            or not isinstance(record.get("conclusion"), str)
        ):
            raise ValueError("GitHub returned an invalid source job record")
        name = record["name"]
        if name in jobs:
            raise ValueError(f"Source run has duplicate job name {name!r}")
        jobs[name] = record

    required = REQUIRED_SUCCESSFUL_JOBS | TERMINAL_CONCLUSIONS.keys()
    missing = required - jobs.keys()
    if missing:
        raise ValueError(f"Source run is missing required jobs: {sorted(missing)}")
    for name, record in jobs.items():
        if record["status"] != "completed":
            raise ValueError(f"Source job {name!r} has not completed")
        allowed = TERMINAL_CONCLUSIONS.get(name, frozenset({"success"}))
        if record["conclusion"] not in allowed:
            raise ValueError(
                f"Source job {name!r} has unusable conclusion {record['conclusion']!r}"
            )

    publication_state = (
        jobs["Bundle / Verify artifact set"]["conclusion"],
        jobs["Publish / GitHub assets"]["conclusion"],
    )
    if publication_state not in RECOVERABLE_PUBLICATION_STATES:
        raise ValueError(
            "Source run did not fail at bundle verification or GitHub publication"
        )


def _release_asset_name(relative_path: PurePath) -> str:
    parts = relative_path.parts
    if len(parts) == 2 and parts[0] == "wheels" and relative_path.suffix == ".whl":
        return relative_path.name
    if parts[0:1] == ("standalones",) and relative_path.suffix == ".zip":
        if len(parts) == 2:
            return relative_path.name
        if len(parts) == 3 and parts[1] in LEGACY_STANDALONE_DIRECTORIES:
            return relative_path.name
    raise ValueError(f"Unexpected recovered artifact path: {relative_path}")


def stage_release_assets(inputs: Path, output: Path) -> list[Path]:
    """Flattens qualified action artifacts without interpreting release bytes."""
    if not inputs.is_dir():
        raise ValueError(f"Recovered artifact directory does not exist: {inputs}")
    sources = {}
    for path in sorted(inputs.rglob("*")):
        relative_path = PurePath(path.relative_to(inputs))
        if path.is_symlink():
            raise ValueError(f"Recovered artifact is a symbolic link: {relative_path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"Recovered artifact is not a regular file: {relative_path}"
            )
        name = _release_asset_name(relative_path)
        if name in sources:
            raise ValueError(f"Recovered artifact has duplicate filename: {name}")
        sources[name] = path
    if not any(name.endswith(".whl") for name in sources):
        raise ValueError("Recovered artifact set contains no SDK wheels")
    if not any(name.endswith(".zip") for name in sources):
        raise ValueError("Recovered artifact set contains no standalone archives")

    output.mkdir(parents=True, exist_ok=False)
    staged = []
    for name, source in sorted(sources.items()):
        destination = output / name
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        staged.append(destination)
        print(f"{source.relative_to(inputs)} -> {destination}")
    return staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify-run", help="verify the exact failed tagged workflow run"
    )
    verify.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    verify.add_argument("--run-id", required=True, type=int)
    verify.add_argument("--revision", required=True)
    verify.add_argument("--tag", required=True)
    stage = commands.add_parser(
        "stage", help="flatten downloaded action artifacts for tag-owned verification"
    )
    stage.add_argument("--inputs", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify-run" and not args.repository:
        parser.error("verify-run requires GITHUB_REPOSITORY or --repository")
    try:
        if args.command == "verify-run":
            verify_source_run(
                github_api.RepositoryClient(
                    args.repository, github_api.resolve_token()
                ),
                args.run_id,
                args.repository,
                args.revision,
                args.tag,
            )
        else:
            stage_release_assets(args.inputs, args.output)
    except (OSError, ValueError, github_api.GitHubAPIError) as error:
        parser.exit(1, f"SDK release recovery failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
