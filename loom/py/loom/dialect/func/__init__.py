# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Func dialect: program structure operations.

Provides runtime function definitions, declarations, calls, returns, and captured
source provenance. See defs.py for the full operation declarations.
"""

from loom.dialect.func.defs import (
    ALL_FUNC_OPS,
    ALL_FUNC_PARAMETERIZED_ATTRS,
    CAbiFloatFormat,
    CAbiTypeKind,
    CallingConv,
    Visibility,
    func_abi_c_field_attr,
    func_abi_c_signature_attr,
    func_abi_c_type_attr,
    func_call,
    func_decl,
    func_def,
    func_location,
    func_ops,
    func_return,
)

__all__ = [
    "func_ops",
    "func_def",
    "func_decl",
    "func_call",
    "func_return",
    "func_location",
    "Visibility",
    "CallingConv",
    "CAbiTypeKind",
    "CAbiFloatFormat",
    "func_abi_c_field_attr",
    "func_abi_c_type_attr",
    "func_abi_c_signature_attr",
    "ALL_FUNC_OPS",
    "ALL_FUNC_PARAMETERIZED_ATTRS",
]
