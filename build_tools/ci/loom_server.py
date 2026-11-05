#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build, qualify, and archive standalone loom-serve release executables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from build_tools.ci import native_artifact_security
from loom.packaging.python import release

SERVER = "//loom/src/loom/tools/loom-serve"
QUALIFIER = "build_tools/ci/loom_server_qualify.py"


@dataclass(frozen=True)
class ServerPlatform:
    binary_name: str
    bazel_options: tuple[str, ...]


PLATFORMS = {
    release.LINUX_X86_64_PLATFORM: ServerPlatform(
        release.SERVER_BINARIES[release.LINUX_X86_64_PLATFORM],
        # ThinLTO replays preprocessor options during backend compilation.
        # Scope the intentional undefinition to suppress that one false warning.
        (
            "--copt=--start-no-unused-arguments",
            "--copt=-U_FORTIFY_SOURCE",
            "--copt=--end-no-unused-arguments",
            "--copt=-D_FORTIFY_SOURCE=2",
        ),
    ),
    release.WINDOWS_X86_64_PLATFORM: ServerPlatform(
        release.SERVER_BINARIES[release.WINDOWS_X86_64_PLATFORM],
        ("--config=windows-x86_64", "--features=static_link_msvcrt_release"),
    ),
}


@dataclass(frozen=True)
class ServerArchive:
    path: Path
    platform: str
    digest: str
    binary_size: int


def _check_commit(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Commit must be a lowercase 40-character Git identity")


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def _declared_server_output(
    root: Path,
    bazel: str,
    flags: list[str],
    environment: dict[str, str],
) -> Path:
    result = subprocess.run(
        [
            bazel,
            "cquery",
            *flags,
            SERVER,
            "--output=starlark",
            "--starlark:expr=json.encode([f.path for f in target.files.to_list()])",
        ],
        cwd=root,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        outputs = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Bazel returned invalid loom-serve output JSON") from error
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError(f"Expected one declared loom-serve output, found {outputs!r}")
    output = outputs[0]
    if not isinstance(output, str):
        raise ValueError(f"Expected a declared output path, found {output!r}")
    return root / output


def _linux_dependencies(readobj: Path, binary: Path) -> set[str]:
    result = subprocess.run(
        [
            os.fspath(readobj),
            "--elf-output-style=JSON",
            "--needed-libs",
            os.fspath(binary),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        document = json.loads(result.stdout)
        libraries = document[0]["NeededLibraries"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("llvm-readobj returned invalid ELF dependencies") from error
    if not isinstance(libraries, list) or not all(
        isinstance(name, str) for name in libraries
    ):
        raise ValueError("llvm-readobj returned invalid ELF dependency names")
    dependencies = set(libraries)
    allowed = {"ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6"}
    if "libc.so.6" not in dependencies or not dependencies <= allowed:
        raise ValueError(
            f"Standalone Linux server has unapproved dependencies: {sorted(dependencies)}"
        )
    return dependencies


def _windows_dependencies(readobj: Path, binary: Path) -> set[str]:
    result = subprocess.run(
        [os.fspath(readobj), "--coff-imports", os.fspath(binary)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    dependencies = {
        line.removeprefix("  Name: ")
        for line in result.stdout.splitlines()
        if line.startswith("  Name: ")
    }
    named = {"bcrypt.dll", "kernel32.dll", "ws2_32.dll"}
    unapproved = {
        name
        for name in dependencies
        if name.lower() not in named and not name.lower().startswith("api-ms-win-core-")
    }
    if not dependencies or "kernel32.dll" not in {
        name.lower() for name in dependencies
    }:
        raise ValueError("llvm-readobj returned no Windows system import closure")
    if unapproved:
        raise ValueError(
            f"Standalone Windows server has unapproved dependencies: {sorted(unapproved)}"
        )
    return dependencies


def build_server(
    root: Path,
    output: Path,
    platform_name: str,
    commit: str,
    bazel: str,
    llvm_bin: Path,
    options: list[str],
) -> Path:
    _check_commit(commit)
    head = _git_head(root)
    if head != commit:
        raise ValueError(f"Requested commit {commit} does not match checkout {head}")
    platform = PLATFORMS[platform_name]
    identity = release.read_identity(root / "loom/packaging/python/version.json")
    flags = [
        "--config=loom-sdk",
        "--config=opt",
        *options,
        "--stamp",
        f"--embed_label={identity.version}@{commit}",
        "--//loom/config:shared_tools=false",
        *platform.bazel_options,
    ]
    environment = dict(os.environ)
    subprocess.run(
        [bazel, "build", *flags, SERVER],
        cwd=root,
        env=environment,
        check=True,
    )
    source = _declared_server_output(root, bazel, flags, environment)
    if source.name != platform.binary_name:
        raise ValueError(
            f"Expected {platform.binary_name}, Bazel declared {source.name}"
        )

    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / platform.binary_name
        shutil.copyfile(source, candidate)
        candidate.chmod(0o755)
        subprocess.run(
            [os.fspath(llvm_bin / "llvm-strip"), "--strip-all", os.fspath(candidate)],
            check=True,
        )
        readobj = llvm_bin / "llvm-readobj"
        dependencies = (
            _windows_dependencies(readobj, candidate)
            if platform_name == release.WINDOWS_X86_64_PLATFORM
            else _linux_dependencies(readobj, candidate)
        )
        native_artifact_security.inspect_file(
            readobj,
            candidate,
            executable=True,
            runpath_policy=native_artifact_security.ElfRunpathPolicy.ABSENT,
        )
        output.mkdir(parents=True, exist_ok=False)
        destination = output / platform.binary_name
        shutil.copyfile(candidate, destination)
        destination.chmod(0o755)
    print(f"dependencies[{platform_name}]={','.join(sorted(dependencies))}")
    return destination


def _native_platform() -> str | None:
    if os.name == "nt":
        return release.WINDOWS_X86_64_PLATFORM
    if sys.platform == "linux":
        return release.LINUX_X86_64_PLATFORM
    return None


def _write_archive(binary: Path, archive: Path) -> None:
    information = ZipInfo(binary.name, (1980, 1, 1, 0, 0, 0))
    information.create_system = 3
    information.external_attr = (stat.S_IFREG | 0o755) << 16
    information.compress_type = ZIP_DEFLATED
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=9) as output:
        output.writestr(information, binary.read_bytes())


def qualify_server(
    root: Path,
    inputs: Path,
    output: Path,
    platform_name: str,
    commit: str,
) -> ServerArchive:
    _check_commit(commit)
    native = _native_platform()
    if native != platform_name:
        raise ValueError(
            f"Qualify {platform_name} on its native OS, not {native or sys.platform}"
        )
    platform = PLATFORMS[platform_name]
    files = sorted(
        path.relative_to(inputs) for path in inputs.rglob("*") if path.is_file()
    )
    if files != [PurePath(platform.binary_name)]:
        raise ValueError(
            f"Standalone server inputs require only {platform.binary_name}; found {files}"
        )
    binary = (inputs / platform.binary_name).resolve()
    if platform_name != release.WINDOWS_X86_64_PLATFORM:
        binary.chmod(0o755)
    identity = release.read_identity(root / "loom/packaging/python/version.json")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            os.fspath(root / "loom/src/loom/serve/testing"),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    subprocess.run(
        [
            sys.executable,
            os.fspath(root / QUALIFIER),
            os.fspath(binary),
            identity.version,
            commit,
        ],
        cwd=root,
        env=environment,
        check=True,
    )

    filename = release.server_archive_name(identity, platform_name)
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / filename
        _write_archive(binary, candidate)
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        output.mkdir(parents=True, exist_ok=False)
        destination = output / filename
        shutil.copyfile(candidate, destination)
    archive = ServerArchive(destination, platform_name, digest, binary.stat().st_size)
    print(
        f"{archive.digest}  {archive.path} "
        f"(binary={archive.binary_size} archive={archive.path.stat().st_size})"
    )
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build an intermediate executable")
    build.add_argument("--platform", required=True, choices=PLATFORMS)
    build.add_argument("--commit", required=True)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--llvm-bin", required=True, type=Path)
    build.add_argument("--bazel", default="bazelisk")
    build.add_argument("options", nargs=argparse.REMAINDER, help="Bazel flags after --")
    qualify = commands.add_parser(
        "qualify", help="Run native checks and export the deterministic ZIP"
    )
    qualify.add_argument("--platform", required=True, choices=PLATFORMS)
    qualify.add_argument("--commit", required=True)
    qualify.add_argument("--inputs", required=True, type=Path)
    qualify.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            options = args.options[1:] if args.options[:1] == ["--"] else args.options
            bazel = shutil.which(args.bazel)
            if bazel is None:
                raise ValueError(f"Bazel executable not found: {args.bazel}")
            build_server(
                Path.cwd(),
                args.output,
                args.platform,
                args.commit,
                bazel,
                args.llvm_bin,
                options,
            )
        else:
            qualify_server(
                Path.cwd(), args.inputs, args.output, args.platform, args.commit
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Standalone server {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
