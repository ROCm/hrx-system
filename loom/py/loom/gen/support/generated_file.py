# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical generated-file header helpers."""

from __future__ import annotations

GENERATED_FILE_MARKER = "GENERATED FILE: DO NOT EDIT."


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
