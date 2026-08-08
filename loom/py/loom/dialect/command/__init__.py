# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command dialect: reusable command-program construction."""

from loom.dialect.command.defs import (
    ALL_COMMAND_OPS,
    command_concurrent,
    command_ops,
    command_parameter,
    command_program_decl,
    command_program_def,
    command_program_launch,
    command_return,
    command_serial,
    command_yield,
)

__all__ = [
    "command_ops",
    "command_parameter",
    "command_program_def",
    "command_program_decl",
    "command_program_launch",
    "command_return",
    "command_yield",
    "command_serial",
    "command_concurrent",
    "ALL_COMMAND_OPS",
]
