# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM IR target dialect operations."""

from loom.dialect.llvmir.defs import (
    ALL_LLVMIR_OPS,
    AsmFlags,
    LlvmirTargetKind,
    llvmir_inline_asm,
    llvmir_intrinsic,
    llvmir_ops,
    llvmir_target,
)

__all__ = [
    "llvmir_ops",
    "AsmFlags",
    "LlvmirTargetKind",
    "llvmir_target",
    "llvmir_inline_asm",
    "llvmir_intrinsic",
    "ALL_LLVMIR_OPS",
]
