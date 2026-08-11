# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical checked-in generated-file maintenance."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

GENERATED_FILE_MARKER = "GENERATED FILE: DO NOT EDIT."
type GeneratedFileMaintenanceMode = Literal["check", "update"]


@dataclass(frozen=True, slots=True)
class GeneratedFile:
    """Expected contents of one repo-relative checked-in generated file."""

    path: str
    contents: str


@dataclass(frozen=True, slots=True)
class GeneratedFileSet:
    """Complete ownership surface for one checked-in artifact family."""

    files: tuple[GeneratedFile, ...]
    obsolete_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        output_paths = tuple(_validate_repo_relative_path(generated_file.path) for generated_file in self.files)
        if output_paths != tuple(sorted(set(output_paths))):
            raise ValueError("generated file paths must be unique and sorted")
        validated_obsolete_paths = tuple(_validate_repo_relative_path(path) for path in self.obsolete_paths)
        if validated_obsolete_paths != tuple(sorted(set(validated_obsolete_paths))):
            raise ValueError("obsolete generated file paths must be unique and sorted")
        overlap = set(output_paths).intersection(validated_obsolete_paths)
        if overlap:
            raise ValueError("generated and obsolete paths overlap: " + ", ".join(sorted(overlap)))

    @classmethod
    def from_mapping(
        cls,
        files: Mapping[str, str],
        *,
        obsolete_paths: Sequence[str] = (),
    ) -> GeneratedFileSet:
        """Creates a deterministic file set and validates its ownership paths."""
        normalized_files = tuple(GeneratedFile(_validate_repo_relative_path(path), contents) for path, contents in sorted(files.items()))
        normalized_obsolete_paths = tuple(sorted({_validate_repo_relative_path(path) for path in obsolete_paths}))
        return cls(normalized_files, normalized_obsolete_paths)

    @property
    def output_paths(self) -> tuple[str, ...]:
        """Returns the deterministic expected output path list."""
        return tuple(generated_file.path for generated_file in self.files)


@dataclass(frozen=True, slots=True)
class GeneratedFileIssue:
    """One checked-in generated-file ownership violation."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class GeneratedFileMaintenanceResult:
    """Result of checking or updating one generated artifact family."""

    ok: bool
    changed_paths: tuple[str, ...] = ()


def _validate_repo_relative_path(path: str) -> str:
    if not path:
        raise ValueError("generated file path must not be empty")
    if "\\" in path:
        raise ValueError(f"generated file path is not canonical: {path!r}")
    normalized_path = PurePosixPath(path)
    if normalized_path.is_absolute() or ".." in normalized_path.parts:
        raise ValueError(f"generated file path must be repo-relative: {path!r}")
    normalized = normalized_path.as_posix()
    if normalized != path or normalized in ("", "."):
        raise ValueError(f"generated file path is not canonical: {path!r}")
    return normalized


def inspect_generated_file_set(
    repo_root: Path,
    generated_file_set: GeneratedFileSet,
) -> tuple[GeneratedFileIssue, ...]:
    """Returns every missing, stale, or obsolete file in an ownership set."""
    issues: list[GeneratedFileIssue] = []
    for generated_file in generated_file_set.files:
        path = repo_root / generated_file.path
        if path.is_symlink():
            issues.append(GeneratedFileIssue(generated_file.path, "generated file must not be a symlink"))
        elif not path.is_file():
            issues.append(GeneratedFileIssue(generated_file.path, "missing generated file"))
        elif path.read_text(encoding="utf-8") != generated_file.contents:
            issues.append(GeneratedFileIssue(generated_file.path, "stale generated file"))
    issues.extend(
        GeneratedFileIssue(relative_path, "obsolete generated file")
        for relative_path in generated_file_set.obsolete_paths
        if (repo_root / relative_path).exists() or (repo_root / relative_path).is_symlink()
    )
    return tuple(issues)


def update_generated_file_set(
    repo_root: Path,
    generated_file_set: GeneratedFileSet,
) -> tuple[str, ...]:
    """Updates one ownership set and returns every changed repo-relative path."""
    changed_paths: list[str] = []
    for generated_file in generated_file_set.files:
        path = repo_root / generated_file.path
        if path.is_symlink():
            path.unlink()
        if path.is_file() and path.read_text(encoding="utf-8") == generated_file.contents:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated_file.contents, encoding="utf-8")
        changed_paths.append(generated_file.path)
    for relative_path in generated_file_set.obsolete_paths:
        path = repo_root / relative_path
        if not path.exists() and not path.is_symlink():
            continue
        path.unlink()
        changed_paths.append(relative_path)
    return tuple(sorted(changed_paths))


def maintain_generated_file_set(
    repo_root: Path,
    generated_file_set: GeneratedFileSet,
    *,
    mode: GeneratedFileMaintenanceMode,
    description: str,
    regenerate_command: str,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates one generated artifact family with uniform reporting."""
    if mode == "update":
        changed_paths = update_generated_file_set(repo_root, generated_file_set)
        print(f"{description}: updated {len(changed_paths)} of {len(generated_file_set.files)} generated files")
        return GeneratedFileMaintenanceResult(True, changed_paths)

    if mode != "check":
        raise ValueError(f"unknown generated-file maintenance mode: {mode}")

    issues = inspect_generated_file_set(repo_root, generated_file_set)
    if issues:
        print(f"{description}: generated-file check failed", file=sys.stderr)
        for issue in issues:
            print(f"  {issue.path}: {issue.reason}", file=sys.stderr)
        print(f"regenerate with {regenerate_command}", file=sys.stderr)
        return GeneratedFileMaintenanceResult(False)
    print(f"{description}: checked {len(generated_file_set.files)} generated files")
    return GeneratedFileMaintenanceResult(True)


def line_comment_header(
    comment_prefix: str,
    *,
    generator: str,
    regenerate: str | None = None,
) -> list[str]:
    """Returns canonical generated-file header lines for line-comment syntax."""
    lines = [
        f"{comment_prefix} {GENERATED_FILE_MARKER}",
        f"{comment_prefix} Generator: {generator}.",
    ]
    if regenerate is not None:
        lines.append(f"{comment_prefix} Regenerate: {regenerate}")
    return lines


def generated_comment(*, generator: str, regenerate: str | None = None) -> str:
    """Returns the canonical generated-file marker for metadata-only formats."""
    parts = [GENERATED_FILE_MARKER, f"Generator: {generator}."]
    if regenerate is not None:
        parts.append(f"Regenerate: {regenerate}")
    return " ".join(parts)
