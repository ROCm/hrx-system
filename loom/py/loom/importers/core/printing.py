# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom text printing helpers shared by importer CLIs and tests."""

from __future__ import annotations

from collections.abc import Sequence

import loom
from loom.dsl import Op
from loom.format.text.printer import Printer
from loom.ir import Module


def print_loom_module(
    module: Module,
    *,
    ops: Sequence[Op] | None = None,
    print_locations: bool = False,
) -> str:
    printer = Printer(print_locations=print_locations)
    printer.register_ops(tuple(ops) if ops is not None else loom.default_ops())
    printer.register_types(loom.default_types())
    return printer.print_module(module)
