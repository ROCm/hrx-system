# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Func dialect: program structure operations.

Provides runtime function definitions, declarations, calls, and returns. See
defs.py for the full operation declarations.
"""

from loom.dialect.func.defs import (
    ALL_FUNC_OPS,
    CallingConv,
    Visibility,
    func_call,
    func_decl,
    func_def,
    func_ops,
    func_return,
)

__all__ = [
    "func_ops",
    "func_def",
    "func_decl",
    "func_call",
    "func_return",
    "Visibility",
    "CallingConv",
    "ALL_FUNC_OPS",
]
