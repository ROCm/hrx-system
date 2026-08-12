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
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ToolAsset:
    url: str
    sha256: str
    binary_name: str | None = None
    archive_format: str | None = None
    archive_files: tuple["ArchiveFile", ...] = ()


@dataclass(frozen=True)
class ArchiveFile:
    member_name: str
    install_name: str
    sha256: str
    executable: bool = False


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
            "windows-amd64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-windows-amd64.exe",
                sha256="092a8738d5b41aae7a85c42cc961b1034e3389aba43ffc20c0fabda7b43e095b",
                binary_name="bazelisk.exe",
            ),
            "windows-arm64": ToolAsset(
                url="https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-windows-arm64.exe",
                sha256="8bc42bd5d7857f18a21440b906469bb6c7cf91a7c72364d4b1e5ec56a76fe94f",
                binary_name="bazelisk.exe",
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
            "windows-amd64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-windows-amd64.exe",
                sha256="f4ecb9c73de2bc38b845d4ee27668f6248c4813a6647db4b4931a7556052e4e1",
                binary_name="buildifier.exe",
            ),
            "windows-arm64": ToolAsset(
                url="https://github.com/bazelbuild/buildtools/releases/download/v8.5.1/buildifier-windows-arm64.exe",
                sha256="55a276ad8b1ff46be48bf64e432264034ea69a45aa3914e89c1d1936f5c2d85c",
                binary_name="buildifier.exe",
            ),
        },
    ),
    "doxygen": Tool(
        version="1.17.0",
        install_names=("doxygen",),
        groups=("docs",),
        default=False,
        assets={
            "darwin-amd64": ToolAsset(
                url="https://github.com/doxygen/doxygen/releases/download/Release_1_17_0/doxygen-1.17.0-mac-intel.zip",
                sha256="27a057d6084e45750f0f99c3d00d19e9fdbe6cd49598add1cb7160462796d195",
                archive_format="zip",
                archive_files=(
                    ArchiveFile(
                        member_name="doxygen-1.17.0/doxygen",
                        install_name="doxygen",
                        sha256="5af04fe7781294c99f2b25d10b7b6bff0c17f6a7aab7a3b362501e22cfddec99",
                        executable=True,
                    ),
                ),
            ),
            "darwin-arm64": ToolAsset(
                url="https://github.com/doxygen/doxygen/releases/download/Release_1_17_0/doxygen-1.17.0-mac-arm.zip",
                sha256="e05a7f647f894d2d82ef26a1d231cda079d2d1d85cf1e1c6fd8072430d80d614",
                archive_format="zip",
                archive_files=(
                    ArchiveFile(
                        member_name="doxygen-1.17.0/doxygen",
                        install_name="doxygen",
                        sha256="1ccc13d7cebaeb6965e6616fe82d313646f3ab32be1273163b31623c6947cda8",
                        executable=True,
                    ),
                ),
            ),
            "linux-amd64": ToolAsset(
                url="https://github.com/doxygen/doxygen/releases/download/Release_1_17_0/doxygen-1.17.0.linux.bin.tar.gz",
                sha256="75419ef4f446fc1c24ef12514b574e66e898ee6f527c6ae2ad84f91a905823c2",
                archive_format="tar.gz",
                archive_files=(
                    ArchiveFile(
                        member_name="doxygen-1.17.0/bin/doxygen",
                        install_name="doxygen",
                        sha256="9c46e7fb9b6a842503a299f6b9b085f112081c4fc8c73b70a6ee69afac093073",
                        executable=True,
                    ),
                ),
            ),
            "windows-amd64": ToolAsset(
                url="https://github.com/doxygen/doxygen/releases/download/Release_1_17_0/doxygen-1.17.0.windows.x64.bin.zip",
                sha256="94594407c4cbca3049d76aacbb05d4a6f7d0f4e93c0de410b825d25ca5621c83",
                archive_format="zip",
                archive_files=(
                    ArchiveFile(
                        member_name="doxygen.exe",
                        install_name="doxygen.exe",
                        sha256="45a5e0b6d2fbca7487affe5cb1445b6fb202a3e4dba5fe39549bc846c5693f1a",
                        executable=True,
                    ),
                    ArchiveFile(
                        member_name="libclang.dll",
                        install_name="libclang.dll",
                        sha256="f60c38ff9600416a95357c0d19bea2fd5bf651b7072ee3e30d25f93546f1f709",
                    ),
                ),
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
        choices=("bazel", "cmake", "docs"),
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
    if system not in ("darwin", "linux", "windows"):
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
        venv_root = Path(virtual_env)
    else:
        venv_root = REPO_ROOT / ".venv"
    return venv_bin_dir(venv_root)


def venv_bin_dir(venv_root: Path, system: str | None = None) -> Path:
    system = platform.system() if system is None else system
    return venv_root / ("Scripts" if system.lower() == "windows" else "bin")


def install_name(name: str, system: str | None = None) -> str:
    system = platform.system() if system is None else system
    if system.lower() == "windows" and not name.lower().endswith(".exe"):
        return name + ".exe"
    return name


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


def primary_file(asset: ToolAsset) -> tuple[str, str]:
    if asset.archive_files:
        archive_file = asset.archive_files[0]
        return archive_file.install_name, archive_file.sha256
    if asset.binary_name is None:
        raise RuntimeError(f"tool asset has no installed binary: {asset.url}")
    return asset.binary_name, asset.sha256


def installed_files(asset: ToolAsset) -> tuple[tuple[str, str], ...]:
    if asset.archive_files:
        return tuple(
            (archive_file.install_name, archive_file.sha256)
            for archive_file in asset.archive_files
        )
    return (primary_file(asset),)


def _copy_archive_file(
    source: BinaryIO,
    archive_file: ArchiveFile,
    bin_dir: Path,
) -> None:
    if Path(archive_file.install_name).name != archive_file.install_name:
        raise RuntimeError(
            f"archive install name must be a file name: {archive_file.install_name}"
        )
    destination = bin_dir / archive_file.install_name
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()
    with temporary_path.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != archive_file.sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{archive_file.member_name} sha256 mismatch: expected "
            f"{archive_file.sha256}, got {actual_sha256}"
        )
    if archive_file.executable:
        mode = temporary_path.stat().st_mode
        temporary_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(temporary_path, destination)


def extract_archive(asset: ToolAsset, archive_path: Path, bin_dir: Path) -> None:
    if not asset.archive_files or asset.binary_name is not None:
        raise RuntimeError(f"tool asset is not an archive: {asset.url}")
    if asset.archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for archive_file in asset.archive_files:
                try:
                    source = archive.open(archive_file.member_name)
                except KeyError as exc:
                    raise RuntimeError(
                        f"{asset.url} does not contain {archive_file.member_name}"
                    ) from exc
                with source:
                    _copy_archive_file(source, archive_file, bin_dir)
        return
    if asset.archive_format == "tar.gz":
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for archive_file in asset.archive_files:
                try:
                    member = archive.getmember(archive_file.member_name)
                except KeyError as exc:
                    raise RuntimeError(
                        f"{asset.url} does not contain {archive_file.member_name}"
                    ) from exc
                if not member.isfile():
                    raise RuntimeError(
                        f"{asset.url} member is not a file: {archive_file.member_name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        f"{asset.url} member cannot be read: {archive_file.member_name}"
                    )
                with source:
                    _copy_archive_file(source, archive_file, bin_dir)
        return
    raise RuntimeError(
        f"unsupported archive format {asset.archive_format!r} for {asset.url}"
    )


def install_alias(
    primary_path: Path,
    alias_path: Path,
    expected_sha256: str,
    system: str | None = None,
) -> None:
    if alias_path == primary_path:
        return
    alias_sha256 = file_sha256(alias_path)
    if alias_path.is_symlink() or alias_sha256 != expected_sha256:
        alias_path.unlink(missing_ok=True)
        system = platform.system() if system is None else system
        if system.lower() == "windows":
            shutil.copy2(primary_path, alias_path)
        else:
            try:
                alias_path.symlink_to(primary_path.name)
            except OSError:
                shutil.copy2(primary_path, alias_path)


def check_alias(primary_path: Path, alias_path: Path, expected_sha256: str) -> bool:
    if alias_path == primary_path:
        return True
    if alias_path.is_symlink():
        try:
            return alias_path.resolve() == primary_path.resolve()
        except OSError:
            return False
    return file_sha256(alias_path) == expected_sha256


def install_tool(name: str, tool: Tool, asset: ToolAsset, bin_dir: Path) -> bool:
    primary_name, primary_sha256 = primary_file(asset)
    primary_path = bin_dir / primary_name
    files_ok = all(
        file_sha256(bin_dir / install_name) == expected_sha256
        for install_name, expected_sha256 in installed_files(asset)
    )
    if files_ok:
        print(f"{name} {tool.version}: already installed at {primary_path}")
    else:
        print(f"{name} {tool.version}: installing {asset.url}")
        if asset.archive_files:
            archive_path = bin_dir / f".{name}-{tool.version}.archive"
            try:
                download_asset(asset, archive_path)
                extract_archive(asset, archive_path, bin_dir)
            finally:
                archive_path.unlink(missing_ok=True)
        else:
            download_asset(asset, primary_path)
    for alias_name in tool.install_names:
        install_alias(
            primary_path,
            bin_dir / install_name(alias_name),
            primary_sha256,
        )
    return True


def check_tool(name: str, tool: Tool, asset: ToolAsset, bin_dir: Path) -> bool:
    primary_name, primary_sha256 = primary_file(asset)
    primary_path = bin_dir / primary_name
    ok = True
    for installed_name, expected_sha256 in installed_files(asset):
        installed_path = bin_dir / installed_name
        if file_sha256(installed_path) != expected_sha256:
            print(f"{name} {tool.version}: missing or wrong hash at {installed_path}")
            ok = False
    for alias_name in tool.install_names:
        alias_path = bin_dir / install_name(alias_name)
        if not check_alias(primary_path, alias_path, primary_sha256):
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
