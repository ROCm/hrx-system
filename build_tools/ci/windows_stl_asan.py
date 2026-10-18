# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolves the x64 STL ASAN companion for a selected Visual C++ installation.

Clang supplies the sanitizer runtime, but instrumented Microsoft STL headers
also require stl_asan.lib from Visual Studio's optional ASAN component. Publish
the selected installation's exact companion into build-owned storage, fetching
and verifying its package when it is not installed. The installed toolchain is
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def _publish_library(destination: Path, contents: bytes, source: str) -> None:
    if not contents:
        raise ValueError(f"Empty STL ASAN companion: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as directory:
        staged = Path(directory) / "stl_asan.lib"
        staged.write_bytes(contents)
        staged.replace(destination / "stl_asan.lib")


def resolve_stl_asan(tools_dir: Path, catalog_path: Path, output_dir: Path) -> Path:
    """Publishes the exact companion and returns its build-owned directory."""
    tools_version = tools_dir.name
    if not re.fullmatch(r"\d+\.\d+\.\d+", tools_version):
        raise ValueError(f"Invalid Visual C++ tools version: {tools_version}")
    destination = output_dir / tools_version / "x64"

    installed = tools_dir / "lib" / "x64" / "stl_asan.lib"
    if installed.is_file():
        _publish_library(destination, installed.read_bytes(), str(installed))
        return destination

    family = ".".join(tools_version.split(".")[:2])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    packages = [
        package
        for package in catalog["packages"]
        if re.fullmatch(
            rf"Microsoft\.VC\.{re.escape(family)}\.\d+\.\d+\.ASAN\.X64\.base",
            package["id"],
        )
    ]
    if len(packages) != 1:
        raise ValueError(
            f"Expected one x64 ASAN package for MSVC {family} in {catalog_path}; "
            f"found {len(packages)}. Install the matching Visual Studio "
            "C++ AddressSanitizer component."
        )
    package = packages[0]
    (payload,) = package["payloads"]
    url = urllib.parse.urlparse(payload["url"])
    if url.scheme != "https" or url.netloc != "download.visualstudio.microsoft.com":
        raise ValueError(f"Unexpected Visual Studio payload URL: {payload['url']}")
    print(f"Fetching {package['id']} {package['version']}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as directory:
        archive_path = Path(directory) / "asan.vsix"
        with urllib.request.urlopen(payload["url"], timeout=60) as response:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive_path.open("rb") as archive:
            digest = hashlib.file_digest(archive, "sha256").hexdigest()
        if digest.lower() != payload["sha256"].lower():
            raise ValueError(f"SHA256 mismatch for {package['id']}: {digest}")
        member = f"Contents/VC/Tools/MSVC/{tools_version}/lib/x64/stl_asan.lib"
        with zipfile.ZipFile(archive_path) as archive:
            # Read one exact member; no archive-controlled path is extracted.
            library = archive.read(member)
    _publish_library(destination, library, member)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(resolve_stl_asan(args.tools_dir, args.catalog, args.output_dir))


if __name__ == "__main__":
    main()
