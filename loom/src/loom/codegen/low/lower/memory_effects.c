// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/memory_effects.h"

#include "loom/codegen/low/lower/context.h"

struct loom_low_lower_memory_origin_t {
  // Common workgroup translation with independently retained expression
  // storage.
  loom_symbolic_expr_t uniform;
  // Envelope for terms that cannot cancel across different participants.
  loom_value_facts_t varying;
};

iree_status_t loom_low_lower_memory_origin_plan(
    loom_symbolic_expr_context_t* expressions,
    const loom_low_source_memory_access_plan_t* source_plan,
    iree_arena_allocator_t* arena,
    const loom_low_lower_memory_origin_t** out_origin) {
  *out_origin = NULL;
  if (source_plan->materialized_base_region != NULL ||
      source_plan->dynamic_term_count == 0 ||
      source_plan->root_value_id == LOOM_VALUE_ID_INVALID ||
      source_plan->root_uniform_scope <
          LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
      source_plan->vector_offset_kind ==
          LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_OTHER) {
    return iree_ok_status();
  }
  loom_symbolic_expr_t uniform;
  loom_symbolic_expr_constant(0, &uniform);
  loom_value_facts_t varying = loom_value_facts_exact_i64(0);
  for (uint8_t i = 0; i < source_plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_plan->dynamic_terms[i];
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(expressions->fact_table, term->index);
    if (term->stride_value_count == 0 &&
        loom_value_facts_is_workgroup_uniform(facts)) {
      loom_symbolic_expr_t value;
      IREE_RETURN_IF_ERROR(
          loom_symbolic_expr_from_value(expressions, term->index, &value));
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_mul_i64(
          expressions, &value, term->byte_stride, &value));
      IREE_RETURN_IF_ERROR(
          loom_symbolic_expr_add(expressions, &uniform, &value, &uniform));
    } else {
      loom_value_facts_addi(&varying, &term->byte_facts, &varying);
    }
  }
  loom_low_lower_memory_origin_t* origin = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*origin), (void**)&origin));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_clone(&uniform, arena, &origin->uniform));
  origin->varying = varying;
  *out_origin = origin;
  return iree_ok_status();
}

bool loom_low_lower_memory_packet_interval(
    const loom_low_lower_memory_origin_t* origin,
    const loom_low_source_memory_access_plan_t* source_plan,
    const loom_low_lower_memory_packet_geometry_t* geometry,
    loom_value_facts_t additional_offset,
    loom_low_memory_relative_interval_t* out_interval,
    int64_t* out_lane_byte_count) {
  *out_interval = (loom_low_memory_relative_interval_t){0};
  *out_lane_byte_count = 0;
  if (source_plan->materialized_base_region != NULL ||
      source_plan->root_value_id == LOOM_VALUE_ID_INVALID ||
      source_plan->root_uniform_scope <
          LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP ||
      source_plan->vector_offset_kind ==
          LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_OTHER ||
      (source_plan->dynamic_term_count != 0 && origin == NULL)) {
    return false;
  }
  int64_t lane_begin = 0, lane_end = 0;
  int64_t lane_bytes = 0;
  if (geometry->vector_lane_count == 0 || geometry->element_byte_count == 0) {
    return false;
  }
  int64_t last_lane_offset = 0;
  if (!iree_checked_mul_i64((int64_t)(geometry->vector_lane_count - 1),
                            geometry->vector_lane_byte_stride,
                            &last_lane_offset)) {
    return false;
  }
  lane_begin = iree_min(0, last_lane_offset);
  if (!iree_checked_add_i64(iree_max(0, last_lane_offset),
                            (int64_t)geometry->element_byte_count, &lane_end) ||
      !iree_checked_sub_i64(lane_end, lane_begin, &lane_bytes)) {
    return false;
  }
  int64_t root_relative_byte_offset = 0;
  if (!iree_checked_sub_i64(geometry->static_byte_offset,
                            source_plan->physical_root_byte_offset,
                            &root_relative_byte_offset)) {
    return false;
  }
  loom_low_memory_relative_interval_t relative = {
      .storage_id = source_plan->root_value_id,
      .disjoint_storage_ordinal =
          source_plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE
              ? source_plan->alias_scope_id + 1u
              : 0,
  };
  loom_value_facts_t varying = additional_offset;
  if (source_plan->dynamic_term_count != 0) {
    relative.origin = origin->uniform;
    loom_value_facts_addi(&varying, &origin->varying, &varying);
  } else {
    loom_symbolic_expr_constant(0, &relative.origin);
  }
  // Static instruction coordinates translate the envelope, leaving the
  // canonical origin and any periodic relationship shared by all packets.
  if (!iree_checked_add_i64(varying.range_lo, lane_begin, &relative.lower) ||
      !iree_checked_add_i64(varying.range_hi, lane_end, &relative.upper) ||
      !iree_checked_add_i64(relative.lower, root_relative_byte_offset,
                            &relative.lower) ||
      !iree_checked_add_i64(relative.upper, root_relative_byte_offset,
                            &relative.upper)) {
    return false;
  }
  *out_interval = relative;
  *out_lane_byte_count = lane_bytes;
  return true;
}

iree_status_t loom_low_lower_record_memory_effect(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    uint16_t effect_ordinal, const loom_low_memory_access_summary_t* summary) {
  if (context->result->memory_accesses == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_memory_access_map_allocate(
        &context->module->arena, &context->result->memory_accesses));
  }
  return loom_low_memory_access_map_insert(context->result->memory_accesses,
                                           low_op, effect_ordinal, summary);
}

static iree_status_t loom_low_lower_record_memory_packet_effects(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_descriptor_t* descriptor,
    loom_low_memory_access_source_flags_t source_flags,
    const loom_low_memory_relative_interval_t* relative_interval,
    int64_t lane_byte_count) {
  if (relative_interval == NULL && source_flags == 0) {
    return iree_ok_status();
  }
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(context->descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &context->descriptor_set->effects[descriptor_layout->effect_start + i];
    if (!loom_low_effect_is_memory_access(effect) ||
        !iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    // Unknown descriptor width cannot license a narrower footprint. Packet
    // selectors publish a geometry at least as wide as the issued effect.
    const bool has_relative_interval =
        relative_interval != NULL && effect->width_bits != 0 &&
        effect->width_bits % 8 == 0 &&
        effect->width_bits / 8 <= lane_byte_count;
    if (!has_relative_interval && source_flags == 0) {
      continue;
    }
    loom_low_memory_access_summary_t summary = {0};
    if (has_relative_interval) {
      summary.memory_space = effect->memory_space;
      summary.source_flags = source_flags;
      summary.relative_interval = relative_interval;
    } else {
      summary = *loom_low_memory_access_summary_for_space(effect->memory_space);
      summary.source_flags = source_flags;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_record_memory_effect(context, low_op, i, &summary));
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_record_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_lower_memory_origin_t* origin,
    const loom_low_source_memory_access_plan_t* source_plan,
    const loom_low_lower_memory_packet_geometry_t* geometry,
    loom_value_facts_t additional_offset) {
  loom_low_memory_relative_interval_t relative;
  int64_t lane_byte_count = 0;
  const bool has_interval = loom_low_lower_memory_packet_interval(
      origin, source_plan, geometry, additional_offset, &relative,
      &lane_byte_count);
  relative.scope = context->source_function.op;
  return loom_low_lower_record_memory_packet_effects(
      context, low_op, descriptor,
      loom_low_source_memory_access_plan_source_flags(source_plan),
      has_interval ? &relative : NULL, lane_byte_count);
}
