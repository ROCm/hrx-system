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

from loom.importers.check.main import main as import_check_main


def main(argv: Sequence[str] | None = None) -> int:
    return import_check_main(_expand_args(sys.argv[1:] if argv is None else argv))


def _expand_args(argv: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for arg in argv:
        expanded.extend(shlex.split(arg))
    return expanded


if __name__ == "__main__":
    raise SystemExit(main())
