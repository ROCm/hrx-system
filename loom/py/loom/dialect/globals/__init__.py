# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Global dialect: module-level state operations.

Provides global.constant, global.variable, global.load, and global.store
for managing immutable and mutable module-level state.
"""

from loom.dialect.globals.defs import (
    ALL_GLOBAL_OPS,
    global_constant,
    global_load,
    global_ops,
    global_store,
    global_variable,
)

__all__ = [
    "global_ops",
    "global_constant",
    "global_variable",
    "global_load",
    "global_store",
    "ALL_GLOBAL_OPS",
]
