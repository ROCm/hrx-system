# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path

from loom.importers.check.import_test import (
    _expand_args,
    _remove_script_directory_from_sys_path,
)


def test_expand_args_splits_cmake_locations_argument() -> None:
    assert _expand_args(["tilelang", "a.py b.py", "--filter=copy"]) == [
        "tilelang",
        "a.py",
        "b.py",
        "--filter=copy",
    ]


def test_script_directory_is_removed_from_sys_path(
    tmp_path: Path,
) -> None:
    script_directory = tmp_path / "check"
    script_directory.mkdir()
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    original_path = list(sys.path)
    try:
        sys.path[:] = [str(script_directory), str(other_directory)]

        _remove_script_directory_from_sys_path(script_directory)

        assert sys.path == [str(other_directory)]
    finally:
        sys.path[:] = original_path
