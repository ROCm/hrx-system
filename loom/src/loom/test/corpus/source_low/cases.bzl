# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical classification of the shared source-Low corpus."""

SOURCE_LOW_CORPUS_CASES = [
    "async_empty.loom-test",
    "async_gather.loom-test",
    "config_decl_facts.loom-test",
    "encoding_facts.loom-test",
    "index_addressing.loom-test",
    "lookup_table.loom-test",
    "memory_cache_policy_non_temporal.loom-test",
    "memory_cache_policy_scopes.loom-test",
    "memory_generic_atomic.loom-test",
    "memory_global.loom-test",
    "memory_global_atomic.loom-test",
    "memory_global_atomic_ordering.loom-test",
    "memory_prefetch.loom-test",
    "memory_scalar_addressing.loom-test",
    "memory_vector_baseline.loom-test",
    "memory_workgroup_atomic.loom-test",
    "memory_workgroup_b128.loom-test",
    "memory_workgroup_indexed.loom-test",
    "memory_workgroup_packed16.loom-test",
    "memory_workgroup_scalar_addressing.loom-test",
    "ml_attention.loom-test",
    "ml_contraction.loom-test",
    "numeric_conversion.loom-test",
    "numeric_conversion_baseline.loom-test",
    "numeric_f32_memory.loom-test",
    "numeric_i32_memory.loom-test",
    "numeric_index.loom-test",
    "numeric_index_conversion.loom-test",
    "numeric_minmax.loom-test",
    "numeric_reduce.loom-test",
    "numeric_scalar_baseline.loom-test",
    "numeric_scalar_f32.loom-test",
    "numeric_scalar_i32.loom-test",
    "numeric_scalar_i64.loom-test",
    "numeric_scalar_literals.loom-test",
    "ordinary_vector_structure.loom-test",
    "packed_bitfield.loom-test",
    "packed_bitpack.loom-test",
    "packed_bitunpack.loom-test",
    "packed_dot_float8.loom-test",
    "packed_dot_i4_mixed.loom-test",
    "packed_dot_integer.loom-test",
    "packed_dot_wide.loom-test",
    "packed_i8_arithmetic.loom-test",
    "packed_integer_arithmetic.loom-test",
    "packed_payload_offsets.loom-test",
    "packed_q8_dequant_dot.loom-test",
    "sync_barrier.loom-test",
    "vector_bitcast.loom-test",
    "vector_carrier_structure.loom-test",
    "vector_dot.loom-test",
    "vector_mask_select.loom-test",
    "vector_scalarization_diagnostics.loom-test",
    "vector_slice.loom-test",
    "vector_v16.loom-test",
    "vector_v4.loom-test",
]

# These cases exercise source facts or diagnostics without targeting a machine.
SOURCE_LOW_SOURCE_ONLY_CASES = [
    "config_decl_facts.loom-test",
    "encoding_facts.loom-test",
    "vector_scalarization_diagnostics.loom-test",
]

# Every target overlay must include or explicitly classify each of these cases.
SOURCE_LOW_TARGET_CASES = [
    case
    for case in SOURCE_LOW_CORPUS_CASES
    if case not in SOURCE_LOW_SOURCE_ONLY_CASES
]
