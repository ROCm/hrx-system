// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/context.h"

iree_status_t loom_low_lower_record_memory_effect(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    uint16_t effect_ordinal, const loom_low_memory_access_summary_t* summary) {
  if (context->result->memory_accesses == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_memory_access_map_create(
        &context->module->arena, &context->result->memory_accesses));
  }
  return loom_low_memory_access_map_insert(context->result->memory_accesses,
                                           low_op, effect_ordinal, summary);
}

// Reduce participant-varying contributions to an envelope while retaining
// correlations only between workgroup-uniform values. A lane's SSA identity
// never licenses canceling the value observed by another lane.
iree_status_t loom_low_lower_record_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_source_memory_access_plan_t* source_plan,
    loom_value_facts_t additional_offset) {
  if (source_plan->root_value_id == LOOM_VALUE_ID_INVALID ||
      source_plan->root_uniform_scope <
          LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
      source_plan->vector_offset_kind ==
          LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_OTHER) {
    return iree_ok_status();
  }
  int64_t lane_begin = 0, lane_end = 0;
  if (!loom_low_source_memory_access_plan_lane_byte_envelope(
          source_plan, &lane_begin, &lane_end)) {
    return iree_ok_status();
  }
  int64_t lane_bytes = 0;
  if (!iree_checked_sub_i64(lane_end, lane_begin, &lane_bytes)) {
    return iree_ok_status();
  }
  loom_low_memory_relative_interval_t relative = {
      .scope = context->source_function.op,
      .storage_id = source_plan->root_value_id,
      .disjoint_storage_ordinal =
          source_plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE
              ? source_plan->alias_scope_id + 1u
              : 0,
  };
  int64_t root_relative_byte_offset = 0;
  if (!iree_checked_sub_i64(source_plan->static_byte_offset,
                            source_plan->physical_root_byte_offset,
                            &root_relative_byte_offset)) {
    return iree_ok_status();
  }
  loom_symbolic_expr_constant(root_relative_byte_offset, &relative.origin);
  loom_value_facts_t varying = additional_offset;
  loom_symbolic_expr_context_t* expressions =
      loom_low_lower_context_symbolic_expr_context(context);
  for (uint8_t i = 0; i < source_plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_plan->dynamic_terms[i];
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(context->lowering.fact_table, term->index);
    if (term->stride_value_count == 0 &&
        loom_value_facts_is_workgroup_uniform(facts)) {
      loom_symbolic_expr_t value;
      IREE_RETURN_IF_ERROR(
          loom_symbolic_expr_from_value(expressions, term->index, &value));
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_mul_i64(
          expressions, &value, term->byte_stride, &value));
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_add(expressions, &relative.origin,
                                                  &value, &relative.origin));
    } else {
      loom_value_facts_addi(&varying, &term->byte_facts, &varying);
    }
  }
  if (!iree_checked_add_i64(varying.range_lo, lane_begin, &relative.lower) ||
      !iree_checked_add_i64(varying.range_hi, lane_end, &relative.upper)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &context->descriptor_set->effects[descriptor->effect_start + i];
    if (!loom_low_effect_is_memory_access(effect) ||
        !iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    // Unknown descriptor width cannot license a narrower footprint. Packet
    // selectors publish a geometry at least as wide as the issued effect.
    if (effect->width_bits == 0 || effect->width_bits % 8 != 0 ||
        effect->width_bits / 8 > lane_bytes) {
      continue;
    }
    const loom_low_memory_access_summary_t summary = {
        .memory_space = effect->memory_space,
        .relative_interval = &relative,
    };
    IREE_RETURN_IF_ERROR(
        loom_low_lower_record_memory_effect(context, low_op, i, &summary));
  }
  return iree_ok_status();
}
