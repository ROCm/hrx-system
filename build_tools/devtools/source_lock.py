# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Worktree-scoped source mutation coordination and integrity guards."""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


def git_worktree_dir(repo_root: Path) -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=repo_root,
            text=True,
        ).strip()
    )


def git_tracked_paths(repo_root: Path) -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
    )
    if not output:
        return ()
    return tuple(path.decode("utf-8") for path in output.split(b"\0") if path)


@dataclass(frozen=True)
class NonEmptyTrackedFileSnapshot:
    files: tuple[str, ...]

    @classmethod
    def capture_paths(
        cls,
        repo_root: Path,
        paths: tuple[str, ...],
    ) -> NonEmptyTrackedFileSnapshot:
        non_empty_paths = []
        for path in paths:
            try:
                contents = (repo_root / path).read_bytes()
            except OSError:
                continue
            if contents:
                non_empty_paths.append(path)
        return cls(tuple(sorted(non_empty_paths)))

    @classmethod
    def capture_tracked_package_initializers(
        cls,
        repo_root: Path,
    ) -> NonEmptyTrackedFileSnapshot:
        paths = tuple(
            path
            for path in git_tracked_paths(repo_root)
            if Path(path).name == "__init__.py"
        )
        return cls.capture_paths(repo_root, paths)

    def verify(self, repo_root: Path) -> bool:
        ok = True
        for path in self.files:
            try:
                contents = (repo_root / path).read_bytes()
            except OSError as exc:
                print(
                    f"{path}: tracked file could not be read after source mutation: {exc}"
                )
                ok = False
                continue
            if contents:
                continue
            print(f"{path}: tracked package initializer became empty")
            ok = False
        return ok


@contextlib.contextmanager
def source_mutation_lock(repo_root: Path, owner: str) -> Iterator[None]:
    if fcntl is None:
        yield
        return

    lock_path = git_worktree_dir(repo_root) / "iree-source-mutation.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"pid={os.getpid()} owner={owner}\n")
            lock_file.flush()
            yield
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
