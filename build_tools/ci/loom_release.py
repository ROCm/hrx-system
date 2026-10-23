#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Publishes verified SDK assets and prepares immutable PyPI upload retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zipfile import BadZipFile

from build_tools.ci import github_api
from loom.packaging.python import release


def verified_release(
    directory: Path, identity: release.Identity, description: str, tag: str
) -> list[release.Wheel]:
    identity.check_tag(tag)
    wheels = release.release_wheels(directory, identity, description)
    expected = release.checksum_document(wheels).encode("utf-8")
    if (directory / "SHA256SUMS").read_bytes() != expected:
        raise ValueError("SHA256SUMS differs from the verified release artifacts")
    return wheels


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _gh(*arguments: str) -> None:
    subprocess.run(["gh", *arguments], check=True)


def publish_github(
    repository: str,
    tag: str,
    identity: release.Identity,
    wheels: list[release.Wheel],
    checksums: Path,
) -> None:
    """Uploads absent assets after every existing counterpart matches."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("GitHub repository must be OWNER/REPOSITORY")
    api = f"https://api.github.com/repos/{repository}"
    token = os.environ.get("GH_TOKEN", "")
    try:
        existing = github_api.get_json(
            f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token
        )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        existing = None

    assets = {wheel.path.name: (wheel.path, wheel.digest) for wheel in wheels}
    assets[checksums.name] = (checksums, _digest(checksums))
    prerelease = re.fullmatch(r"\d+\.\d+\.\d+", identity.version) is None
    if existing is None:
        options = ["--prerelease", "--latest=false"] if prerelease else []
        _gh(
            "release",
            "create",
            tag,
            *[str(path) for path, _ in assets.values()],
            "--repo",
            repository,
            "--verify-tag",
            "--title",
            f"Loom {tag}",
            "--generate-notes",
            *options,
        )
        return
    if (
        existing["tag_name"] != tag
        or existing["draft"]
        or existing["prerelease"] != prerelease
    ):
        raise ValueError(
            "Existing GitHub release has conflicting tag or draft/prerelease status"
        )

    names = set()
    page = 1
    while True:
        records = github_api.get_json(
            f"{api}/releases/{existing['id']}/assets?per_page=100&page={page}", token
        )
        for record in records:
            name = record["name"]
            if name in names:
                raise ValueError(f"Duplicate GitHub release asset: {name}")
            names.add(name)
            if name.endswith(".whl") and name not in assets:
                raise ValueError(f"Unexpected GitHub wheel: {name}; use a new version")
            if name in assets and record["state"] != "uploaded":
                raise ValueError(
                    f"Incomplete GitHub asset: {name}; review the failed upload"
                )
        if len(records) < 100:
            break
        page += 1

    # Finish every comparison before the first write. Stage one downloaded
    # counterpart at a time so retry verification needs one extra asset's space.
    for name in sorted(names & assets.keys()):
        with tempfile.TemporaryDirectory(prefix="loom-release-") as temporary:
            _gh(
                "release",
                "download",
                tag,
                "--repo",
                repository,
                "--pattern",
                name,
                "--dir",
                temporary,
            )
            if _digest(Path(temporary) / name) != assets[name][1]:
                raise ValueError(
                    f"Existing GitHub asset differs: {name}; use a new version"
                )
    missing = [str(path) for name, (path, _) in assets.items() if name not in names]
    if missing:
        _gh("release", "upload", tag, *missing, "--repo", repository)
    else:
        print("GitHub release already contains identical assets.")


def stage_pypi(
    wheels: list[release.Wheel], identity: release.Identity, output: Path
) -> bool:
    """Copies only missing normal wheels into a fresh upload directory."""
    normal = {wheel.path.name: wheel for wheel in wheels if wheel.profile == "release"}
    request = urllib.request.Request(
        f"https://pypi.org/pypi/hrx-loom/{identity.version}/json",
        headers={"Accept": "application/json", "User-Agent": "hrx-loom-release"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            records = json.load(response)["urls"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        records = []
    existing = set()
    for record in records:
        name = record["filename"]
        if name not in normal or name in existing:
            raise ValueError(
                f"Unexpected or duplicate PyPI asset: {name}; use a new version"
            )
        if record["yanked"] or record["digests"]["sha256"] != normal[name].digest:
            raise ValueError(
                f"Existing PyPI asset differs or is yanked: {name}; use a new version"
            )
        existing.add(name)

    missing = [wheel for name, wheel in normal.items() if name not in existing]
    output.mkdir(parents=True, exist_ok=False)
    for wheel in missing:
        shutil.copyfile(wheel.path, output / wheel.path.name)
    return bool(missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "github", "pypi-stage"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", type=Path, default=Path("release-assets"))
    parser.add_argument(
        "--identity", type=Path, default=Path("loom/packaging/python/version.json")
    )
    parser.add_argument(
        "--description", type=Path, default=Path("loom/packaging/python/PYPI_README.md")
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--output", type=Path, help="fresh PyPI upload directory")
    args = parser.parse_args(argv)
    if args.operation == "github" and not args.repository:
        parser.error("github requires --repository or GITHUB_REPOSITORY")
    if args.operation == "pypi-stage" and args.output is None:
        parser.error("pypi-stage requires --output")
    try:
        identity = release.read_identity(args.identity)
        wheels = verified_release(
            args.directory,
            identity,
            args.description.read_text(encoding="utf-8"),
            args.tag,
        )
        if args.operation == "github":
            publish_github(
                args.repository,
                args.tag,
                identity,
                wheels,
                args.directory / "SHA256SUMS",
            )
        elif args.operation == "pypi-stage":
            pending = stage_pypi(wheels, identity, args.output)
            result = f"pending={str(pending).lower()}\n"
            print(result, end="")
            if os.environ.get("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(result)
        else:
            print(release.checksum_document(wheels), end="")
    except (ValueError, OSError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDK publication failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
