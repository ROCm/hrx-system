// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/memory_access.h"

#include "iree/base/api.h"

loom_low_memory_space_t loom_low_memory_access_normalize_space(
    loom_low_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return memory_space;
    case LOOM_LOW_MEMORY_SPACE_NONE:
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

bool loom_low_memory_access_spaces_may_alias(loom_low_memory_space_t left,
                                             loom_low_memory_space_t right) {
  left = loom_low_memory_access_normalize_space(left);
  right = loom_low_memory_access_normalize_space(right);
  return left == right || left == LOOM_LOW_MEMORY_SPACE_GENERIC ||
         right == LOOM_LOW_MEMORY_SPACE_GENERIC;
}

const loom_low_memory_access_summary_t*
loom_low_memory_access_summary_for_space(loom_low_memory_space_t memory_space) {
  static const loom_low_memory_access_summary_t generic = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_GENERIC,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  static const loom_low_memory_access_summary_t global = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t workgroup = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t stack = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_STACK,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  static const loom_low_memory_access_summary_t wasm_memory = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_WASM_MEMORY,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE,
  };
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
      return &global;
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return &workgroup;
    case LOOM_LOW_MEMORY_SPACE_STACK:
      return &stack;
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return &wasm_memory;
    default:
      return &generic;
  }
}

static bool loom_low_byte_intervals_have_range(
    const loom_low_byte_interval_t* interval) {
  const loom_low_byte_interval_precision_flags_t required_flags =
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;
  return iree_all_bits_set(interval->precision_flags, required_flags);
}

static bool loom_low_byte_interval_envelopes_are_disjoint(
    const loom_low_byte_interval_t* left,
    const loom_low_byte_interval_t* right) {
  if (!loom_low_byte_intervals_have_range(left) ||
      !loom_low_byte_intervals_have_range(right)) {
    return false;
  }
  return left->end_facts.range_hi <= right->begin_facts.range_lo ||
         right->end_facts.range_hi <= left->begin_facts.range_lo;
}

static bool loom_low_strided_byte_intervals_are_disjoint(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right) {
  const loom_low_memory_access_precision_flags_t required_precision =
      LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL;
  if (!iree_all_bits_set(left->precision_flags, required_precision) ||
      !iree_all_bits_set(right->precision_flags, required_precision) ||
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  const loom_low_strided_byte_interval_t* left_interval =
      &left->strided_interval;
  const loom_low_strided_byte_interval_t* right_interval =
      &right->strided_interval;
  if (left_interval->stride_bytes == 0 ||
      left_interval->stride_bytes != right_interval->stride_bytes ||
      left_interval->begin_bytes >= left_interval->end_bytes ||
      left_interval->end_bytes > left_interval->stride_bytes ||
      right_interval->begin_bytes >= right_interval->end_bytes ||
      right_interval->end_bytes > right_interval->stride_bytes) {
    return false;
  }
  return left_interval->end_bytes <= right_interval->begin_bytes ||
         right_interval->end_bytes <= left_interval->begin_bytes;
}

bool loom_low_memory_access_summaries_may_alias(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right) {
  if (!loom_low_memory_access_spaces_may_alias(left->memory_space,
                                               right->memory_space)) {
    return false;
  }
  if (iree_all_bits_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      iree_all_bits_set(right->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  if (iree_all_bits_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      iree_all_bits_set(right->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      left->alias_group_id != right->alias_group_id) {
    return false;
  }
  if (loom_low_strided_byte_intervals_are_disjoint(left, right)) {
    return false;
  }
  const loom_low_memory_access_precision_flags_t interval_precision =
      LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL;
  if (iree_all_bits_set(left->precision_flags, interval_precision) &&
      iree_all_bits_set(right->precision_flags, interval_precision) &&
      left->alias_root_id == right->alias_root_id && left->byte_interval &&
      right->byte_interval &&
      loom_low_byte_interval_envelopes_are_disjoint(left->byte_interval,
                                                    right->byte_interval)) {
    return false;
  }
  return true;
}

bool loom_low_memory_access_summaries_equal(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right) {
  if (left == right) return true;
  if (left->memory_space != right->memory_space ||
      left->precision_flags != right->precision_flags) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
      left->alias_root_id != right->alias_root_id) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP) &&
      left->alias_group_id != right->alias_group_id) {
    return false;
  }
  if (iree_any_bit_set(left->precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL) &&
      (left->strided_interval.stride_bytes !=
           right->strided_interval.stride_bytes ||
       left->strided_interval.begin_bytes !=
           right->strided_interval.begin_bytes ||
       left->strided_interval.end_bytes != right->strided_interval.end_bytes)) {
    return false;
  }
  if (!iree_any_bit_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL) ||
      left->byte_interval == right->byte_interval) {
    return true;
  }
  if (left->byte_interval == NULL || right->byte_interval == NULL) return false;
  const loom_low_byte_interval_t* left_interval = left->byte_interval;
  const loom_low_byte_interval_t* right_interval = right->byte_interval;
  return left_interval->precision_flags == right_interval->precision_flags &&
         left_interval->begin_expr_id == right_interval->begin_expr_id &&
         left_interval->end_expr_id == right_interval->end_expr_id &&
         loom_value_facts_equal(left_interval->begin_facts,
                                right_interval->begin_facts) &&
         loom_value_facts_equal(left_interval->end_facts,
                                right_interval->end_facts);
}
