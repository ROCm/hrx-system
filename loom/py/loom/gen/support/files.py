# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Filesystem helpers shared by Loom generators."""

from __future__ import annotations

from pathlib import Path


def write_text_file(path: Path, contents: str) -> None:
    """Writes UTF-8 text, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def read_optional_text_file(path: Path | None) -> str | None:
    """Reads UTF-8 text when a path is present."""
    return path.read_text(encoding="utf-8") if path is not None else None
