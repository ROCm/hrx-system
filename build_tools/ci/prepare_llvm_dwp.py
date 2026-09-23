#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Builds llvm-dwp from the source revision of an installed LLVM toolchain."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

_SOURCE_REPOSITORIES = {
    "https://github.com/ROCm/llvm-project": "ROCm/llvm-project",
    "https://github.com/llvm/llvm-project": "llvm/llvm-project",
}
_SOURCE_PATHS = (
    "llvm/tools/llvm-dwp/llvm-dwp.cpp",
    "llvm/tools/llvm-dwp/Opts.td",
)
_CACHE_VERSION = 1


@dataclasses.dataclass(frozen=True)
class LlvmSourceIdentity:
    repository: str
    github_repository: str
    commit: str


def parse_llvm_source_identity(version_output: str) -> LlvmSourceIdentity:
    """Returns the immutable source revision reported by clang --version."""
    match = re.search(
        r"\((https://github\.com/[^ )]+?)(?:\.git)? ([0-9a-f]{40})\)",
        version_output,
    )
    if match is None:
        raise ValueError("clang --version does not report a GitHub source revision")
    repository = match.group(1).removesuffix(".git")
    github_repository = _SOURCE_REPOSITORIES.get(repository)
    if github_repository is None:
        raise ValueError(f"unsupported LLVM source repository: {repository}")
    return LlvmSourceIdentity(
        repository=repository,
        github_repository=github_repository,
        commit=match.group(2),
    )


def source_url(identity: LlvmSourceIdentity, source_path: str) -> str:
    if source_path not in _SOURCE_PATHS:
        raise ValueError(f"unsupported llvm-dwp source path: {source_path}")
    return (
        "https://raw.githubusercontent.com/"
        f"{identity.github_repository}/{identity.commit}/{source_path}"
    )


def _fetch_sources(
    repository: str,
    commit: str,
    source_paths: tuple[str, ...],
    destination: Path,
) -> dict[str, bytes]:
    source_repository = destination / "source"
    subprocess.run(
        ["git", "init", "--quiet", source_repository],
        check=True,
    )
    # Fetch the exact commit once and materialize only the requested blobs. This
    # avoids the raw-content endpoint's per-file rate limits.
    subprocess.run(
        [
            "git",
            "-C",
            source_repository,
            "-c",
            "protocol.version=2",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            repository,
            commit,
        ],
        check=True,
    )
    fetched_commit = subprocess.check_output(
        ["git", "-C", source_repository, "rev-parse", "FETCH_HEAD"],
        text=True,
    ).strip()
    if fetched_commit != commit:
        raise ValueError(f"fetched LLVM revision {fetched_commit}, expected {commit}")
    return {
        source_path: subprocess.check_output(
            ["git", "-C", source_repository, "show", f"FETCH_HEAD:{source_path}"]
        )
        for source_path in source_paths
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _toolchain_identity(llvm_root: Path) -> tuple[LlvmSourceIdentity, str]:
    clang = llvm_root / "bin" / "clang++"
    version_output = subprocess.check_output(
        [clang, "--version"], text=True, stderr=subprocess.STDOUT
    )
    return parse_llvm_source_identity(version_output), version_output.splitlines()[0]


def _cached_tool_matches(
    output: Path,
    manifest_path: Path,
    identity: LlvmSourceIdentity,
    llvm_root: Path,
) -> bool:
    if (
        not output.is_file()
        or not os.access(output, os.X_OK)
        or not manifest_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("cache_version") == _CACHE_VERSION
        and manifest.get("source_repository") == identity.repository
        and manifest.get("source_commit") == identity.commit
        and manifest.get("llvm_root") == os.fspath(llvm_root)
        and manifest.get("output_sha256") == _sha256(output)
    )


def prepare_llvm_dwp(*, llvm_root: Path, output: Path) -> dict[str, object]:
    """Builds a matching llvm-dwp or returns the verified cached output."""
    llvm_root = llvm_root.resolve(strict=True)
    output = output.resolve()
    manifest_path = output.with_name(output.name + ".json")
    identity, compiler_version = _toolchain_identity(llvm_root)
    if _cached_tool_matches(output, manifest_path, identity, llvm_root):
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result

    tools = {
        name: llvm_root / "bin" / name
        for name in ("clang++", "llvm-config", "llvm-tblgen")
    }
    missing_tools = [name for name, path in tools.items() if not path.is_file()]
    if missing_tools:
        raise FileNotFoundError(
            f"{llvm_root} is missing required LLVM tools: {', '.join(missing_tools)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=".llvm-dwp-build-", dir=output.parent
    ) as temporary_directory:
        build_directory = Path(temporary_directory)
        source_contents = _fetch_sources(
            identity.repository,
            identity.commit,
            _SOURCE_PATHS,
            build_directory,
        )
        source_records = []
        for source_path, content in source_contents.items():
            url = source_url(identity, source_path)
            destination = build_directory / Path(source_path).name
            destination.write_bytes(content)
            source_records.append(
                {
                    "path": source_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "url": url,
                }
            )

        llvm_include_directory = llvm_root / "include"
        subprocess.run(
            [
                tools["llvm-tblgen"],
                "-I",
                llvm_include_directory,
                "-gen-opt-parser-defs",
                build_directory / "Opts.td",
                "-o",
                build_directory / "Opts.inc",
            ],
            check=True,
        )

        llvm_config = tools["llvm-config"]
        cmake_directory = Path(
            subprocess.check_output([llvm_config, "--cmakedir"], text=True).strip()
        )
        driver_template = cmake_directory / "llvm-driver-template.cpp.in"
        driver_source = driver_template.read_text(encoding="utf-8")
        substitutions = {
            "@TOOL_NAME@": "llvm_dwp",
            "@INITLLVM_ARGS@": "",
        }
        for marker, replacement in substitutions.items():
            if marker not in driver_source:
                raise ValueError(f"unsupported LLVM driver template: {driver_template}")
            driver_source = driver_source.replace(marker, replacement)
        driver_path = build_directory / "driver.cpp"
        driver_path.write_text(driver_source, encoding="utf-8")

        llvm_flags = shlex.split(
            subprocess.check_output(
                [
                    llvm_config,
                    "--cxxflags",
                    "--ldflags",
                    "--system-libs",
                    "--libs",
                    "dwp",
                ],
                text=True,
            )
        )
        llvm_library_directory = subprocess.check_output(
            [llvm_config, "--libdir"], text=True
        ).strip()
        built_output = build_directory / "llvm-dwp"
        subprocess.run(
            [
                tools["clang++"],
                build_directory / "llvm-dwp.cpp",
                driver_path,
                f"-I{build_directory}",
                f"-ffile-prefix-map={build_directory}=.",
                f"-fmacro-prefix-map={build_directory}=.",
                *llvm_flags,
                f"-Wl,-rpath,{llvm_library_directory}",
                "-o",
                built_output,
            ],
            check=True,
        )
        subprocess.run(
            [built_output, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        os.replace(built_output, output)

    result: dict[str, object] = {
        "cache_version": _CACHE_VERSION,
        "cached": False,
        "compiler_version": compiler_version,
        "elapsed_seconds": time.monotonic() - started,
        "llvm_root": os.fspath(llvm_root),
        "output": os.fspath(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "source_commit": identity.commit,
        "source_files": source_records,
        "source_repository": identity.repository,
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_llvm_dwp(llvm_root=args.llvm_root, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
