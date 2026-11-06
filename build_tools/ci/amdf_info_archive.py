#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualifies and archives amdf-info independently of its SDK wheel companions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from loom.packaging.python import release


def package_tool(
    wheel_path: Path, output: Path, identity: release.Identity, description: str
) -> release.StandaloneArchive:
    """Exports only a Release executable that runs detached on its native OS."""
    wheel = release.inspect_wheel(wheel_path, identity, description)
    if wheel.profile != "release":
        raise ValueError("Standalone amdf-info requires the Release wheel")
    native = {
        "linux": release.LINUX_X86_64_PLATFORM,
        "win32": release.WINDOWS_X86_64_PLATFORM,
    }.get(sys.platform)
    if wheel.platform != native:
        raise ValueError(
            f"Qualify {wheel.platform} on its native OS, not {sys.platform}"
        )

    binary_name = release.standalone_binary_name("amdf-info", wheel.platform)
    member = f"hrx_loom-{identity.version}.data/scripts/{binary_name}"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH"}
    }
    with tempfile.TemporaryDirectory(prefix="amdf-info-") as temporary:
        directory = Path(temporary)
        binary = directory / binary_name
        with ZipFile(wheel.path) as archive:
            try:
                payload = archive.read(member)
            except KeyError as error:
                raise ValueError(f"Release wheel is missing {member}") from error
        binary.write_bytes(payload)
        binary.chmod(0o755)
        # The isolated directory contains only this executable. In particular,
        # Windows must not resolve a wheel companion through the parent's PATH.
        environment["PATH"] = ""
        for arguments in (["--help"], [], ["--format=json"]):
            result = subprocess.run(
                [os.fspath(binary), *arguments],
                cwd=directory,
                env=environment,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            if not result.stdout.strip():
                raise ValueError(f"amdf-info {arguments} produced no output")
            if arguments == ["--format=json"]:
                document = json.loads(result.stdout)
                if (
                    not isinstance(document, dict)
                    or document.get("schema_version") != 1
                    or not isinstance(document.get("result"), dict)
                    or document["result"].get("exit_code") != 0
                ):
                    raise ValueError("amdf-info returned an invalid inventory result")

        filename = release.standalone_archive_name(
            identity, "amdf-info", wheel.platform
        )
        candidate = directory / filename
        release.write_standalone_archive(binary, candidate)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        output.mkdir(parents=True, exist_ok=True)
        destination = output / filename
        with candidate.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    artifact = release.StandaloneArchive(
        destination, "amdf-info", wheel.platform, digest
    )
    print(
        f"{artifact.digest}  {artifact.path} (archive={artifact.path.stat().st_size})"
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path("loom/packaging/python")
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity", type=Path, default=package / "version.json")
    parser.add_argument("--description", type=Path, default=package / "PYPI_README.md")
    args = parser.parse_args(argv)
    try:
        package_tool(
            args.wheel,
            args.output,
            release.read_identity(args.identity),
            args.description.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Standalone amdf-info qualification failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
