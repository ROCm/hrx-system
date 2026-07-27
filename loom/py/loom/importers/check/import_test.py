# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build-system driven import-check test entry point."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    _remove_script_directory_from_sys_path(Path(__file__).resolve().parent)
    from loom.importers.check.main import main as import_check_main

    return import_check_main(_expand_args(sys.argv[1:] if argv is None else argv))


def _expand_args(argv: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for arg in argv:
        expanded.extend(shlex.split(arg))
    return expanded


def _remove_script_directory_from_sys_path(script_directory: Path) -> None:
    """Prevents this helper package from shadowing frontend distributions."""

    normalized_script_directory = script_directory.resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != normalized_script_directory
    ]


if __name__ == "__main__":
    raise SystemExit(main())
