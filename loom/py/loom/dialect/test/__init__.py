# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test dialect: ops exercising every DSL and format feature.

These ops exist solely to validate the DSL, format elements, printer,
parser, and builder infrastructure. They are not part of the loom
compiler — they are the acceptance tests for the toolchain.

Every format element type appears in at least one test op. Every
DSL feature (variadic, optional, tied results, regions, enums,
constraints, traits) is exercised. If the test ops round-trip
through printer and parser, real dialect ops will too.
"""

from loom.dialect.test.defs import (
    ALL_TEST_OPS,
    cmp_predicates,
    test_addi,
    test_assume,
    test_attr_table,
    test_attrs,
    test_block_args,
    test_branch,
    test_cast,
    test_clause_constant,
    test_clause_copy,
    test_cmp,
    test_constant,
    test_convert,
    test_counter,
    test_deflate,
    test_dim,
    test_fact_buffer_memory_space,
    test_fact_buffer_min_alignment,
    test_fact_divisor,
    test_fact_non_negative,
    test_fact_non_zero,
    test_fact_positive,
    test_fact_power_of_two,
    test_fact_range_hi,
    test_fact_range_lo,
    test_fact_view_memory_space,
    test_fact_view_root_min_alignment,
    test_func,
    test_implicit_yield,
    test_invoke,
    test_loop,
    test_map,
    test_neg,
    test_operand_dict,
    test_ops,
    test_record,
    test_region_table,
    test_slice,
    test_split_func,
    test_update,
    test_use,
    test_yield,
)

__all__ = [
    "test_ops",
    "ALL_TEST_OPS",
    "test_addi",
    "test_neg",
    "test_cast",
    "test_constant",
    "test_clause_constant",
    "test_clause_copy",
    "test_use",
    "test_cmp",
    "test_map",
    "test_update",
    "test_invoke",
    "test_slice",
    "test_loop",
    "test_block_args",
    "test_branch",
    "test_implicit_yield",
    "test_yield",
    "test_func",
    "test_split_func",
    "test_attrs",
    "test_operand_dict",
    "test_attr_table",
    "test_region_table",
    "test_record",
    "test_deflate",
    "test_assume",
    "test_convert",
    "test_counter",
    "test_dim",
    "test_fact_range_lo",
    "test_fact_range_hi",
    "test_fact_divisor",
    "test_fact_buffer_memory_space",
    "test_fact_buffer_min_alignment",
    "test_fact_non_negative",
    "test_fact_non_zero",
    "test_fact_positive",
    "test_fact_power_of_two",
    "test_fact_view_memory_space",
    "test_fact_view_root_min_alignment",
    "cmp_predicates",
]
