# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source selection shared by local maintenance and declared hygiene actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

SOURCE_SUFFIXES = frozenset({".loom", ".loom-test"})
REPOSITORY_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".h"})
REPOSITORY_SOURCE_ROOT = "loom/src/loom/"
AUTHORING_CORPUS_ROOT = REPOSITORY_SOURCE_ROOT + "test/corpus/authoring/"
# Syntax fixtures retain their exact parser/printer contract. Standalone syntax
# and semantic error fixtures cannot pass canonical-source verification.
FORMAT_EXCLUDED_PREFIXES = (REPOSITORY_SOURCE_ROOT + "test/corpus/text/",)
FORMAT_EXCLUDED_PATHS = frozenset(
    {
        "loom/src/loom/serve/runtime/program/testdata/compilation/invalid.loom",
        "loom/src/loom/tooling/target/amdgpu/test/amdgpu_bad_return.loom",
        "loom/src/loom/tools/iree-benchmark-loom/testdata/duplicate_symbol.loom",
    }
)


def is_lint_source_path(path: str) -> bool:
    source_path = PurePosixPath(path)
    return (
        "\\" not in path
        and source_path.as_posix() == path
        and ".." not in source_path.parts
        and path.startswith("loom/")
        and source_path.suffix in SOURCE_SUFFIXES
    )


def is_format_source_path(path: str) -> bool:
    return (
        is_lint_source_path(path)
        and path not in FORMAT_EXCLUDED_PATHS
        and not path.startswith(FORMAT_EXCLUDED_PREFIXES)
    )


def is_repository_policy_path(path: str) -> bool:
    source_path = PurePosixPath(path)
    return path.startswith(REPOSITORY_SOURCE_ROOT) and (
        source_path.suffix in REPOSITORY_SOURCE_SUFFIXES
        or source_path.name in ("BUILD.bazel", "CMakeLists.txt")
        or (path.startswith(AUTHORING_CORPUS_ROOT) and source_path.suffix == ".loom")
    )


def inventory(repository_root: Path) -> dict:
    directories = {}
    sources = []
    policy_sources = []
    pending = [repository_root / "loom"]
    while pending:
        directory = pending.pop()
        entries = {}
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if entry.name.startswith(".") or entry.name == "__pycache__":
                    continue
                is_directory = entry.is_dir()
                entries[entry.name] = is_directory
                path = Path(entry.path)
                relative_path = path.relative_to(repository_root).as_posix()
                if is_directory:
                    if entry.is_symlink():
                        raise ValueError(
                            f"source directory must not be a symlink: {relative_path}"
                        )
                    pending.append(path)
                elif is_lint_source_path(relative_path) or is_repository_policy_path(
                    relative_path
                ):
                    if not entry.is_file():
                        raise ValueError(
                            f"source must be a regular file: {relative_path}"
                        )
                    if is_lint_source_path(relative_path):
                        sources.append(relative_path)
                    if is_repository_policy_path(relative_path):
                        policy_sources.append(relative_path)
        directories[directory.relative_to(repository_root).as_posix()] = entries
    return {
        "directories": directories,
        "sources": sorted(sources),
        "format_sources": sorted(filter(is_format_source_path, sources)),
        "policy_sources": sorted(policy_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository_root", type=Path)
    arguments = parser.parse_args()
    sys.stdout.write(
        json.dumps(inventory(arguments.repository_root), sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
