# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check dialect: production testbench operations."""

from loom.dialect.check.defs import (
    ALL_CHECK_OPS,
    FileWriteMode,
    NanPolicy,
    RangePolicy,
    Visibility,
    check_benchmark,
    check_case,
    check_expect,
    check_expect_bitwise,
    check_expect_close,
    check_expect_equal,
    check_expect_shape,
    check_file_read_npy,
    check_file_write_npy,
    check_generate_fill,
    check_generate_iota,
    check_generate_random_uniform,
    check_literal,
    check_ops,
    check_oracle_call,
    check_param_choice,
    check_param_range,
    check_param_seed,
    check_requires,
    check_return,
    check_skip_if,
)

__all__ = [
    "ALL_CHECK_OPS",
    "FileWriteMode",
    "NanPolicy",
    "RangePolicy",
    "Visibility",
    "check_ops",
    "check_benchmark",
    "check_case",
    "check_expect",
    "check_expect_bitwise",
    "check_expect_close",
    "check_expect_equal",
    "check_expect_shape",
    "check_file_read_npy",
    "check_file_write_npy",
    "check_generate_fill",
    "check_generate_iota",
    "check_generate_random_uniform",
    "check_literal",
    "check_oracle_call",
    "check_param_choice",
    "check_param_range",
    "check_param_seed",
    "check_requires",
    "check_return",
    "check_skip_if",
]
