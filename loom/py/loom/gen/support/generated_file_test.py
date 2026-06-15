# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.gen.support.generated_file import (
    GENERATED_FILE_MARKER,
    generated_comment,
    line_comment_header,
)


def test_line_comment_header() -> None:
    assert line_comment_header(
        "//",
        generator="loom.gen.example",
        regenerate="python3 run.py",
    ) == [
        f"// {GENERATED_FILE_MARKER}",
        "// Generator: loom.gen.example.",
        "// Regenerate: python3 run.py",
    ]


def test_generated_comment() -> None:
    assert generated_comment(generator="loom.gen.example") == (f"{GENERATED_FILE_MARKER} Generator: loom.gen.example.")
