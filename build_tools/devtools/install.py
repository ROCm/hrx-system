#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Installs pinned standalone developer tools.

Required shared Python-packaged tools belong in requirements-dev.in. Optional
static-analysis providers belong in requirements-analysis.in so their
transitive environments are locked separately from the core developer tool
environment. Standalone release binaries belong in this manifest so local setup
and CI use the same versions, download URLs, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ToolAsset:
    url: str
    sha256: str
    binary_name: str


@dataclass(frozen=True)
class Tool:
    version: str
    install_names: tuple[str, ...]
    assets: dict[str, ToolAsset]
    groups: tuple[str, ...]
    default: bool = True


TOOLS = {
    "bazelisk": Tool(
        version="1.29.0",
        install_names=("bazelisk", "bazel"),
        groups=("bazel",),
        assets={
            "darwin-amd64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-darwin-amd64",
                sha256="16c3d7aa15323a9fb69f56c7ec5733ed18bedb786680d0ba13bb12a3c8083007",
                binary_name="bazelisk",
            ),
            "darwin-arm64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-darwin-arm64",
                sha256="cee851f726789227d5561004e9904a52be45c3efb56f8b38b6993d6adbaa0409",
                binary_name="bazelisk",
            ),
            "linux-amd64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64",
                sha256="5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992",
                binary_name="bazelisk",
            ),
            "linux-arm64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-arm64",
                sha256="e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010",
                binary_name="bazelisk",
            ),
        },
    ),
    "buildifier": Tool(
        version="8.5.1",
        install_names=("buildifier",),
        groups=("bazel",),
        assets={
            "darwin-amd64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-darwin-amd64",
                sha256="31de189e1a3fe53aa9e8c8f74a0309c325274ad19793393919e1ca65163ca1a4",
                binary_name="buildifier",
            ),
            "darwin-arm64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-darwin-arm64",
                sha256="62836a9667fa0db309b0d91e840f0a3f2813a9c8ea3e44b9cd58187c90bc88ba",
                binary_name="buildifier",
            ),
            "linux-amd64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-linux-amd64",
                sha256="887377fc64d23a850f4d18a077b5db05b19913f4b99b270d193f3c7334b5a9a7",
                binary_name="buildifier",
            ),
            "linux-arm64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-linux-arm64",
                sha256="947bf6700d708026b2057b09bea09abbc3cafc15d9ecea35bb3885c4b09ccd04",
                binary_name="buildifier",
            ),
        },
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install pinned developer tools.")
    parser.add_argument(
        "tools",
        nargs="*",
        help="Tool names to install. Defaults to required tools. Use 'all' for every known tool.",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=("bazel", "cmake"),
        default=[],
        help="Install tools required by a developer command lane.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        help="Install directory. Defaults to VIRTUAL_ENV/bin or .venv/bin.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that selected tools are installed with the expected hash.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known tools and versions without installing.",
    )
    return parser.parse_args()


def host_platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system not in ("darwin", "linux"):
        raise RuntimeError(f"unsupported host operating system: {platform.system()}")

    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        raise RuntimeError(f"unsupported host architecture: {platform.machine()}")
    return f"{system}-{arch}"


def default_bin_dir() -> Path:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env) / "bin"
    return REPO_ROOT / ".venv" / "bin"


def selected_tools(args: argparse.Namespace) -> dict[str, Tool]:
    if args.list:
        return TOOLS
    if args.group:
        groups = set(args.group)
        return {
            name: tool
            for name, tool in TOOLS.items()
            if groups.intersection(tool.groups)
        }
    if not args.tools:
        return {name: tool for name, tool in TOOLS.items() if tool.default}
    if "all" in args.tools:
        return TOOLS
    unknown_tools = sorted(set(args.tools) - set(TOOLS))
    if unknown_tools:
        raise RuntimeError("unknown tool(s): " + ", ".join(unknown_tools))
    return {name: TOOLS[name] for name in args.tools}


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset: ToolAsset, destination: Path) -> None:
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "iree-devtools-installer"},
    )
    digest = hashlib.sha256()
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != asset.sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{asset.url} sha256 mismatch: expected {asset.sha256}, got {actual_sha256}"
        )
    mode = temporary_path.stat().st_mode
    temporary_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(temporary_path, destination)


def install_alias(primary_path: Path, alias_path: Path, expected_sha256: str) -> None:
    if alias_path == primary_path:
        return
    alias_sha256 = file_sha256(alias_path)
    if alias_path.is_symlink() or alias_sha256 != expected_sha256:
        alias_path.unlink(missing_ok=True)
        try:
            alias_path.symlink_to(primary_path.name)
        except OSError:
            shutil.copy2(primary_path, alias_path)


def check_alias(primary_path: Path, alias_path: Path, expected_sha256: str) -> bool:
    if alias_path == primary_path:
        return True
    if alias_path.is_symlink():
        return alias_path.resolve() == primary_path.resolve()
    return file_sha256(alias_path) == expected_sha256


def install_tool(name: str, tool: Tool, asset: ToolAsset, bin_dir: Path) -> bool:
    primary_path = bin_dir / asset.binary_name
    if file_sha256(primary_path) == asset.sha256:
        print(f"{name} {tool.version}: already installed at {primary_path}")
    else:
        print(f"{name} {tool.version}: installing {asset.url}")
        download_asset(asset, primary_path)
    for install_name in tool.install_names:
        install_alias(primary_path, bin_dir / install_name, asset.sha256)
    return True


def check_tool(name: str, tool: Tool, asset: ToolAsset, bin_dir: Path) -> bool:
    primary_path = bin_dir / asset.binary_name
    ok = True
    if file_sha256(primary_path) != asset.sha256:
        print(f"{name} {tool.version}: missing or wrong hash at {primary_path}")
        ok = False
    for install_name in tool.install_names:
        alias_path = bin_dir / install_name
        if not check_alias(primary_path, alias_path, asset.sha256):
            print(f"{name} {tool.version}: missing alias {alias_path}")
            ok = False
    if ok:
        print(f"{name} {tool.version}: ok")
    return ok


def main() -> int:
    args = parse_arguments()
    tools = selected_tools(args)
    if args.list:
        for name, tool in sorted(tools.items()):
            marker = "default" if tool.default else "optional"
            groups = ",".join(tool.groups) if tool.groups else "none"
            print(f"{name} {tool.version} ({marker}; groups: {groups})")
        return 0

    if not tools:
        print("devtools: no standalone tools selected")
        return 0

    platform_key = host_platform_key()
    bin_dir = args.bin_dir or default_bin_dir()
    if not bin_dir.is_dir():
        raise RuntimeError(
            f"install directory does not exist: {bin_dir}. Create the venv first."
        )

    ok = True
    for name, tool in tools.items():
        asset = tool.assets.get(platform_key)
        if not asset:
            print(f"{name} {tool.version}: no asset for {platform_key}")
            ok = False
            continue
        if args.check:
            ok = check_tool(name, tool, asset, bin_dir) and ok
        else:
            ok = install_tool(name, tool, asset, bin_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"devtools: {error}", file=sys.stderr)
        sys.exit(1)
