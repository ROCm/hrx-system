// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_string_view_t loom_amdgpu_prefetch_intent_name(uint8_t intent) {
  switch ((loom_view_prefetch_intent_t)intent) {
    case LOOM_VIEW_PREFETCH_INTENT_READ:
      return IREE_SV("read");
    case LOOM_VIEW_PREFETCH_INTENT_WRITE:
      return IREE_SV("write");
    case LOOM_VIEW_PREFETCH_INTENT_COUNT_:
      break;
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_prefetch_locality_name(uint8_t locality) {
  switch ((loom_view_prefetch_locality_t)locality) {
    case LOOM_VIEW_PREFETCH_LOCALITY_NONE:
      return IREE_SV("none");
    case LOOM_VIEW_PREFETCH_LOCALITY_L1:
      return IREE_SV("l1");
    case LOOM_VIEW_PREFETCH_LOCALITY_L2:
      return IREE_SV("l2");
    case LOOM_VIEW_PREFETCH_LOCALITY_L3:
      return IREE_SV("l3");
    case LOOM_VIEW_PREFETCH_LOCALITY_COUNT_:
      break;
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_prefetch_dynamic_index_kind_name(
    loom_amdgpu_memory_dynamic_index_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
      return IREE_SV("vaddr");
    case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
      return IREE_SV("soffset");
  }
  return IREE_SV("invalid");
}

static bool loom_amdgpu_prefetch_static_offset_split(
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_prefetch_plan_t* plan) {
  if (plan->source.static_byte_offset < 0 ||
      offset_info->unit_byte_count != 1) {
    return false;
  }
  const uint64_t static_byte_offset = (uint64_t)plan->source.static_byte_offset;
  const uint64_t immediate_offset =
      iree_min(static_byte_offset, offset_info->unsigned_max);
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    return false;
  }
  plan->immediate_offset = (int64_t)immediate_offset;
  plan->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  return loom_amdgpu_source_memory_offset_fits_u32(
      &plan->source, plan->source.static_byte_offset);
}

static bool loom_amdgpu_prefetch_select_dynamic_index(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_prefetch_plan_t* plan) {
  if (!loom_low_source_memory_access_is_dynamic(&plan->source)) {
    return true;
  }
  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(&plan->source);
  if (!term) {
    return false;
  }
  if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX) {
    return false;
  }
  if (!loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term, 32)) {
    return false;
  }
  if (term->byte_stride != 1 &&
      term->byte_shift == LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
    return false;
  }

  switch (term->source) {
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
      plan->dynamic_term_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
      return true;
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
      if (loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                view_regions, term->index)) {
        return false;
      }
      plan->dynamic_term_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
      return true;
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_view_prefetch_count(const loom_op_t* source_op,
                                            uint32_t* out_count) {
  *out_count = 0;
  switch (
      (loom_view_prefetch_locality_t)loom_view_prefetch_locality(source_op)) {
    case LOOM_VIEW_PREFETCH_LOCALITY_NONE:
    case LOOM_VIEW_PREFETCH_LOCALITY_L1:
    case LOOM_VIEW_PREFETCH_LOCALITY_L2:
    case LOOM_VIEW_PREFETCH_LOCALITY_L3:
      // AMDGPU data-prefetch packets expose a span count, not a portable cache
      // locality selector. The source locality chooses whether to prefetch; the
      // packet span stays at the minimal representable value.
      *out_count = 1;
      return true;
    case LOOM_VIEW_PREFETCH_LOCALITY_COUNT_:
      break;
  }
  return false;
}

static bool loom_amdgpu_prefetch_memory_space_is_buffer_backed(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return false;
  }
  return false;
}

static bool loom_amdgpu_prefetch_select_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op,
    loom_amdgpu_prefetch_plan_t* out_plan,
    const loom_low_descriptor_t** out_descriptor,
    iree_string_view_t* out_decision_key) {
  *out_plan = (loom_amdgpu_prefetch_plan_t){
      .descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
  };
  *out_descriptor = NULL;
  *out_decision_key = IREE_SV("prefetch.unhandled");
  if (!loom_view_prefetch_isa(source_op)) {
    return false;
  }
  if (loom_view_prefetch_intent(source_op) != LOOM_VIEW_PREFETCH_INTENT_READ) {
    *out_decision_key = IREE_SV("prefetch.intent");
    return false;
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_BUFFER_PREFETCH_DATA);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    *out_decision_key = IREE_SV("prefetch.descriptor_missing");
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info = {0};
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    *out_decision_key = IREE_SV("prefetch.offset_immediate");
    return false;
  }

  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &out_plan->source,
                                                &source_diagnostic)) {
    *out_decision_key = source_diagnostic.rejection_bits != 0
                            ? loom_low_source_memory_access_rejection_key(
                                  source_diagnostic.rejection_bits)
                            : IREE_SV("prefetch.source_memory_access");
    return false;
  }
  if (!loom_amdgpu_prefetch_memory_space_is_buffer_backed(
          out_plan->source.memory_space)) {
    *out_decision_key = IREE_SV("prefetch.memory_space");
    return false;
  }
  if (!loom_amdgpu_prefetch_select_dynamic_index(module, fact_table,
                                                 view_regions, out_plan)) {
    *out_decision_key = IREE_SV("prefetch.dynamic_index");
    return false;
  }
  if (!loom_amdgpu_prefetch_static_offset_split(&offset_info, out_plan)) {
    *out_decision_key = IREE_SV("prefetch.offset_range");
    return false;
  }
  if (!loom_amdgpu_view_prefetch_count(source_op, &out_plan->count)) {
    *out_decision_key = IREE_SV("prefetch.locality");
    return false;
  }

  out_plan->descriptor_ordinal = descriptor_ordinal;
  *out_descriptor = descriptor;
  *out_decision_key = IREE_SV("prefetch.s_buffer_prefetch_data");
  return true;
}

static iree_status_t loom_amdgpu_prefetch_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_prefetch_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor = NULL;
  iree_string_view_t unused_decision_key = iree_string_view_empty();
  if (!loom_amdgpu_prefetch_select_source(module, fact_table, descriptor_set,
                                          view_regions, source_op, out_plan,
                                          &descriptor, &unused_decision_key)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
      context, descriptor, &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("offset"),
                                          &out_plan->offset_attr_name_id));
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("count"),
                                          &out_plan->count_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_record_view_prefetch_diagnostic(
    loom_target_low_legality_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_set_t* descriptor_set) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      !loom_view_prefetch_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  loom_amdgpu_prefetch_plan_t plan = {0};
  const loom_low_descriptor_t* descriptor = NULL;
  iree_string_view_t decision_key = iree_string_view_empty();
  const bool selected = loom_amdgpu_prefetch_select_source(
      loom_target_low_legality_module(context),
      loom_target_low_legality_fact_table(context), descriptor_set,
      view_regions, source_op, &plan, &descriptor, &decision_key);
  const iree_string_view_t packet_key =
      descriptor ? loom_low_descriptor_set_string(descriptor_set,
                                                  descriptor->key_string_offset)
                 : IREE_SV("<none>");
  return loom_target_low_legality_record_memory_prefetch(
      context, source_op,
      loom_amdgpu_memory_space_name(plan.source.memory_space),
      loom_amdgpu_prefetch_intent_name(loom_view_prefetch_intent(source_op)),
      loom_amdgpu_prefetch_locality_name(
          loom_view_prefetch_locality(source_op)),
      decision_key, selected ? IREE_SV("selected") : IREE_SV("dropped"),
      packet_key, plan.immediate_offset, plan.scalar_byte_offset,
      loom_amdgpu_prefetch_dynamic_index_kind_name(plan.dynamic_term_kind),
      plan.count);
}

iree_status_t loom_amdgpu_select_view_prefetch_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_prefetch_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_prefetch_select(context, source_op, out_plan,
                                     out_selected);
}

iree_status_t loom_amdgpu_lower_view_prefetch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_prefetch_plan_t* plan) {
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_view_prefetch_view(source_op), &low_resource));

  const loom_value_id_t dynamic_index =
      plan->dynamic_term_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET
          ? plan->source.dynamic_terms[0].index
          : LOOM_VALUE_ID_INVALID;
  const int64_t dynamic_index_byte_stride =
      dynamic_index == LOOM_VALUE_ID_INVALID
          ? 1
          : plan->source.dynamic_terms[0].byte_stride;
  const uint32_t dynamic_index_byte_shift =
      dynamic_index == LOOM_VALUE_ID_INVALID
          ? LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE
          : plan->source.dynamic_terms[0].byte_shift;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset(
      context, source_op, dynamic_index, dynamic_index_byte_stride,
      dynamic_index_byte_shift, plan->scalar_byte_offset, &low_soffset));

  loom_named_attr_t attrs[] = {
      {
          .name_id = plan->offset_attr_name_id,
          .value = loom_attr_i64(plan->immediate_offset),
      },
      {
          .name_id = plan->count_attr_name_id,
          .value = loom_attr_i64(plan->count),
      },
  };
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_resource, &plan->source, &low_descriptor));
  loom_value_id_t operands[] = {
      low_descriptor,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_op);
}
