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
    case LOOM_LOW_MEMORY_SPACE_VM_REF:
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

loom_low_memory_access_summary_t loom_low_memory_access_summary_from_effect(
    const loom_low_effect_t* effect) {
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(effect->memory_space);
  loom_low_memory_access_precision_flags_t precision_flags = 0;
  if (memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  return (loom_low_memory_access_summary_t){
      .memory_space = memory_space,
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = precision_flags,
      .byte_interval = NULL,
  };
}

loom_low_memory_access_summary_t loom_low_memory_access_summary_synthetic(
    loom_low_memory_space_t memory_space) {
  loom_low_effect_t effect = {
      .memory_space = memory_space,
  };
  return loom_low_memory_access_summary_from_effect(&effect);
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
  if (iree_all_bits_set(left->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL) &&
      iree_all_bits_set(right->precision_flags,
                        LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL) &&
      left->byte_interval && right->byte_interval &&
      loom_low_byte_interval_envelopes_are_disjoint(left->byte_interval,
                                                    right->byte_interval)) {
    return false;
  }
  return true;
}

bool loom_low_memory_access_write_subsumes_read(
    const loom_low_memory_access_summary_t* write_summary,
    const loom_low_memory_access_summary_t* read_summary) {
  if (!loom_low_memory_access_summaries_may_alias(write_summary,
                                                  read_summary)) {
    return false;
  }
  const loom_low_memory_access_precision_flags_t identity_precision =
      LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL |
      LOOM_LOW_MEMORY_ACCESS_PRECISION_EXACT_LANES;
  if (iree_any_bit_set(write_summary->precision_flags, identity_precision) ||
      iree_any_bit_set(read_summary->precision_flags, identity_precision)) {
    return false;
  }
  return write_summary->memory_space == LOOM_LOW_MEMORY_SPACE_GENERIC ||
         write_summary->memory_space == read_summary->memory_space;
}
