# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.gen.support.files import read_optional_text_file, write_text_file


def test_write_text_file_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "output.txt"
    write_text_file(output_path, "contents")

    assert output_path.read_text(encoding="utf-8") == "contents"


def test_read_optional_text_file(tmp_path: Path) -> None:
    output_path = tmp_path / "input.txt"
    output_path.write_text("contents", encoding="utf-8")

    assert read_optional_text_file(output_path) == "contents"
    assert read_optional_text_file(None) is None
