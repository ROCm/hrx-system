// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/codegen/low/lower/source_memory.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/address_realization.h"
#include "loom/target/arch/amdgpu/lower/buffer_descriptor.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/source_value_analysis.h"

static iree_status_t loom_amdgpu_memory_access_plan_select_from_context(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_selection_t* out_selection, bool* out_selected) {
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_context(context, &analysis));
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  const loom_low_source_memory_access_plan_t* retained_source =
      loom_low_lower_source_memory_access(context, source_op,
                                          &source_diagnostic);
  if (retained_source == NULL) {
    return iree_ok_status();
  }
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context));
  IREE_ASSERT(target_facts != NULL);
  if (!loom_amdgpu_memory_access_plan_select(
          module, loom_low_lower_context_fact_table(context),
          loom_low_lower_context_descriptor_set(context), view_regions,
          analysis, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context),
          target_facts->properties.instruction_constraints, alloca_layout,
          loom_low_lower_context_read_visibility_scope(context), source_op,
          retained_source, out_selection, &diagnostic)) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_memory_access_plan_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_selection_t* selection) {
  for (uint32_t i = 0; i < selection->packet_count; ++i) {
    if (!loom_amdgpu_memory_cache_policy_can_lower(
            descriptor_set, &selection->packets[i].access)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_plan_memory_store_payload(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    loom_amdgpu_memory_packet_plan_t* packet) {
  packet->constant_words = NULL;
  const uint32_t count = packet->access.payload_register_count;
  if (packet->access.payload_register_class !=
          LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR ||
      count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* facts =
      loom_low_lower_context_fact_table(context);
  uint32_t words[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  const uint32_t first_word = packet->payload_byte_offset / 4u;
  for (uint32_t i = 0; i < count; ++i) {
    if (!loom_amdgpu_source_lane_as_u32_bits(facts, module, source_value,
                                             first_word + i, &words[i])) {
      return iree_ok_status();
    }
  }
  uint32_t* retained_words = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, count * sizeof(*retained_words), (void**)&retained_words));
  memcpy(retained_words, words, count * sizeof(*retained_words));
  packet->constant_words = retained_words;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_memory_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_amdgpu_memory_access_selection_t selection;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_plan_select_from_context(
      context, source_op, &selection, &selected));
  if (!selected) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_memory_access_plan_cache_policy_can_lower(
          loom_low_lower_context_descriptor_set(context), &selection)) {
    return iree_ok_status();
  }

  const loom_module_t* source_module = loom_low_lower_context_module(context);
  const loom_memory_access_t source_access =
      loom_memory_access_cast(source_module, source_op);
  const loom_value_id_t mask = loom_memory_access_mask(source_access);
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (mask != LOOM_VALUE_ID_INVALID &&
      !loom_value_fact_table_query_uniform_element_origin(
          loom_low_lower_context_fact_table(context), source_module, mask,
          &condition)) {
    return iree_ok_status();
  }

  IREE_ASSERT_GT(selection.packet_count, 0u);
  iree_host_size_t plan_byte_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_amdgpu_memory_access_plan_t), &plan_byte_length,
      IREE_STRUCT_FIELD_FAM(selection.packet_count,
                            loom_amdgpu_memory_packet_plan_t)));
  const bool has_mask = condition != LOOM_VALUE_ID_INVALID;
  if (has_mask) {
    plan_byte_length += sizeof(loom_amdgpu_memory_mask_plan_t);
  }
  loom_amdgpu_memory_access_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, plan_byte_length, (void**)&retained_plan));
  retained_plan->source = selection.source;
  retained_plan->packet_count = selection.packet_count;
  retained_plan->has_mask = has_mask;
  if (has_mask) {
    loom_amdgpu_memory_mask_plan_t* retained_mask =
        (loom_amdgpu_memory_mask_plan_t*)(retained_plan->packets +
                                          retained_plan->packet_count);
    *retained_mask = (loom_amdgpu_memory_mask_plan_t){
        .condition = condition,
        .passthrough = loom_memory_access_passthrough(source_access),
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_native_i1_mask(
        context, condition, &retained_mask->condition_plan));
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_memory_access_t memory_access =
      loom_memory_access_cast(module, source_op);
  const loom_value_id_t stored_value = loom_memory_access_value(memory_access);
  if (source_op->result_count != 0) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, loom_op_const_results(source_op)[0], &result_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_op_const_results(source_op)[0], result_type));
  }
  const loom_amdgpu_memory_dynamic_term_plan_t* dynamic_term_plans = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_plan_memory_dynamic_terms(
      context, &retained_plan->source, &dynamic_term_plans));
  for (uint32_t i = 0; i < selection.packet_count; ++i) {
    retained_plan->packets[i] = selection.packets[i];
    retained_plan->packets[i].access.source = &retained_plan->source;
    retained_plan->packets[i].access.buffer_extent = NULL;
    retained_plan->packets[i].access.dynamic_term_plans = dynamic_term_plans;
    retained_plan->packets[i].constant_words = NULL;
    if (stored_value != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_plan_memory_store_payload(
          context, stored_value, &retained_plan->packets[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_memory_address_realizations(
        context, source_op, &retained_plan->packets[i].access));
  }
  *out_plan = loom_low_lower_plan_make(source_op->kind, retained_plan);
  out_plan->access_flags = retained_plan->source.access_flags;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_memory_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  return loom_amdgpu_select_memory_plan(context, source_op, out_plan);
}

iree_status_t loom_amdgpu_select_memory_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  return loom_amdgpu_select_memory_plan(context, source_op, out_plan);
}

iree_status_t loom_amdgpu_finalize_memory_plan(
    loom_low_lower_context_t* context, loom_amdgpu_memory_access_plan_t* plan) {
  const loom_amdgpu_buffer_extent_plan_t* buffer_extent = NULL;
  const loom_amdgpu_memory_access_t* first = &plan->packets[0].access;
  if ((first->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT ||
       first->address_form ==
           LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) &&
      first->source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      first->source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_plan_buffer_extent(context, first->source, &buffer_extent));
  }
  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    plan->packets[i].access.buffer_extent = buffer_extent;
  }
  return iree_ok_status();
}
