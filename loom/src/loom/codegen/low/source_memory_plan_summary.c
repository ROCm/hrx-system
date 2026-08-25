// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/facts.h"

static loom_low_memory_space_t loom_low_source_memory_access_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return LOOM_LOW_MEMORY_SPACE_STACK;
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

bool loom_low_source_memory_access_plan_lane_byte_envelope(
    const loom_low_source_memory_access_plan_t* plan, int64_t* out_begin_offset,
    int64_t* out_end_offset) {
  *out_begin_offset = 0;
  *out_end_offset = 0;
  if (plan->vector_lane_count == 0 || plan->element_byte_count == 0) {
    return false;
  }

  int64_t last_lane_offset = 0;
  if (!iree_checked_mul_i64((int64_t)(plan->vector_lane_count - 1),
                            plan->vector_lane_byte_stride, &last_lane_offset)) {
    return false;
  }
  int64_t begin_offset = iree_min(0, last_lane_offset);
  int64_t end_offset = 0;
  if (!iree_checked_add_i64(iree_max(0, last_lane_offset),
                            (int64_t)plan->element_byte_count, &end_offset)) {
    return false;
  }
  *out_begin_offset = begin_offset;
  *out_end_offset = end_offset;
  return true;
}

static bool loom_low_source_memory_access_plan_strided_interval(
    const loom_low_source_memory_access_plan_t* plan, int64_t lane_begin_offset,
    int64_t lane_end_offset, loom_low_strided_byte_interval_t* out_interval) {
  *out_interval = (loom_low_strided_byte_interval_t){0};
  if (plan->dynamic_term_count == 0) {
    return false;
  }
  uint64_t stride_bytes = 0;
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &plan->dynamic_terms[i];
    const int64_t signed_stride = term->byte_stride;
    if (term->stride_value_count != 0 || signed_stride == 0 ||
        signed_stride == INT64_MIN) {
      return false;
    }
    const uint64_t term_stride_bytes =
        (uint64_t)(signed_stride < 0 ? -signed_stride : signed_stride);
    stride_bytes = stride_bytes == 0
                       ? term_stride_bytes
                       : iree_math_gcd_u64(stride_bytes, term_stride_bytes);
  }
  int64_t access_begin = 0;
  int64_t access_end = 0;
  if (!iree_checked_add_i64(plan->static_byte_offset, lane_begin_offset,
                            &access_begin) ||
      !iree_checked_add_i64(plan->static_byte_offset, lane_end_offset,
                            &access_end) ||
      access_end <= access_begin) {
    return false;
  }
  int64_t signed_length_bytes = 0;
  if (!iree_checked_sub_i64(access_end, access_begin, &signed_length_bytes) ||
      signed_length_bytes <= 0) {
    return false;
  }
  const uint64_t length_bytes = (uint64_t)signed_length_bytes;
  if (length_bytes > stride_bytes) {
    return false;
  }
  int64_t signed_begin_residue = access_begin % (int64_t)stride_bytes;
  if (signed_begin_residue < 0) {
    signed_begin_residue += (int64_t)stride_bytes;
  }
  const uint64_t begin_bytes = (uint64_t)signed_begin_residue;
  if (begin_bytes > stride_bytes - length_bytes) {
    return false;
  }
  *out_interval = (loom_low_strided_byte_interval_t){
      .stride_bytes = stride_bytes,
      .begin_bytes = begin_bytes,
      .end_bytes = begin_bytes + length_bytes,
  };
  return true;
}

void loom_low_source_memory_access_plan_make_summary(
    const loom_low_source_memory_access_plan_t* plan,
    loom_low_byte_interval_t* out_interval,
    loom_low_memory_access_summary_t* out_summary) {
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(
          loom_low_source_memory_access_space(plan->memory_space));
  loom_low_memory_access_precision_flags_t precision_flags = 0;
  if (memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  uint32_t alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  if (plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    alias_root_id = plan->alias_scope_id;
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }

  *out_interval = (loom_low_byte_interval_t){0};
  const loom_low_byte_interval_t* interval = NULL;
  loom_low_strided_byte_interval_t strided_interval = {0};
  int64_t lane_begin_offset = 0;
  int64_t lane_end_offset = 0;
  if (loom_low_source_memory_access_plan_lane_byte_envelope(
          plan, &lane_begin_offset, &lane_end_offset)) {
    loom_value_facts_t begin_facts =
        loom_low_source_memory_dynamic_offset_facts(plan,
                                                    plan->static_byte_offset);
    loom_value_facts_t end_facts = begin_facts;
    const loom_value_facts_t begin_adjustment =
        loom_value_facts_exact_i64(lane_begin_offset);
    const loom_value_facts_t end_adjustment =
        loom_value_facts_exact_i64(lane_end_offset);
    loom_value_facts_addi(&begin_facts, &begin_adjustment, &begin_facts);
    loom_value_facts_addi(&end_facts, &end_adjustment, &end_facts);
    *out_interval = (loom_low_byte_interval_t){
        .begin_facts = begin_facts,
        .end_facts = end_facts,
        .begin_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .end_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE,
        .precision_flags = LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
                           LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE |
                           LOOM_LOW_BYTE_INTERVAL_PRECISION_EXACT_LENGTH,
    };
    precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL;
    interval = out_interval;
    if (iree_any_bit_set(precision_flags,
                         LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT) &&
        loom_low_source_memory_access_plan_strided_interval(
            plan, lane_begin_offset, lane_end_offset, &strided_interval)) {
      precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL;
    }
  }

  *out_summary = (loom_low_memory_access_summary_t){
      .memory_space = memory_space,
      .alias_root_id = alias_root_id,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .precision_flags = precision_flags,
      .strided_interval = strided_interval,
      .byte_interval = interval,
  };
}
