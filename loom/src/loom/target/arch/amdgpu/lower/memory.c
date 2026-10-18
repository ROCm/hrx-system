// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory_ordering.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/source_value_analysis.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static loom_amdgpu_memory_access_rejection_flags_t
loom_amdgpu_memory_access_alloca_root_rejection_bit(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PRIVATE_ROOT;
    default:
      return LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
  }
}

bool loom_amdgpu_memory_access_include_alloca_root_byte_offset(
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (source->memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      break;
    default:
      return true;
  }

  uint64_t root_byte_offset = 0;
  if (!loom_amdgpu_source_alloca_layout_lookup_byte_offset(
          alloca_layout, source->memory_space, source->root_value_id,
          &root_byte_offset) ||
      root_byte_offset > INT64_MAX) {
    diagnostic->rejection_bits |=
        loom_amdgpu_memory_access_alloca_root_rejection_bit(
            source->memory_space);
    return false;
  }

  if (!loom_low_source_memory_access_plan_include_root_byte_offset(
          source, root_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_register_footprint(
    loom_type_t vector_type, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  access->payload_format = LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_GENERIC;
  uint32_t register_count = loom_amdgpu_vector_32bit_lane_count(vector_type);
  if (register_count != 0) {
    access->payload_register_class =
        LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR;
    access->payload_register_count = register_count;
    access->packet_byte_count = register_count * 4u;
    return true;
  }

  loom_amdgpu_vector_storage_t storage = {0};
  if (loom_type_rank(vector_type) != 1 ||
      !loom_amdgpu_type_vector_storage(vector_type, &storage)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  const uint32_t payload_bit_count =
      storage.element_count * storage.element_bit_count;
  register_count = storage.register_count;
  const uint32_t register_bit_count = register_count * 32u;
  if (payload_bit_count != register_bit_count &&
      !iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage.kind),
                        LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT;
    diagnostic->payload_type = vector_type;
    diagnostic->payload_bit_count = payload_bit_count;
    diagnostic->register_bit_count = register_bit_count;
    return false;
  }
  access->payload_register_class =
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR;
  access->payload_register_count = register_count;
  access->packet_byte_count = payload_bit_count / 8u;
  return true;
}

static uint32_t loom_amdgpu_memory_access_vector_lane_limit(
    uint32_t element_byte_count, uint32_t max_32bit_lane_count) {
  if (element_byte_count == 0) {
    return 0;
  }
  const uint64_t max_byte_count = (uint64_t)max_32bit_lane_count * 4u;
  return (uint32_t)(max_byte_count / element_byte_count);
}

static void loom_amdgpu_memory_access_try_record_vector_width_diagnostic(
    const loom_low_source_memory_access_plan_t* source, loom_type_t vector_type,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_type_is_vector(vector_type) || source->vector_lane_count == 0 ||
      source->element_byte_count == 0) {
    return;
  }
  const uint64_t payload_byte_count =
      (uint64_t)source->vector_lane_count * source->element_byte_count;
  if (payload_byte_count == 0 ||
      payload_byte_count > (uint64_t)UINT32_MAX * 4u) {
    return;
  }
  const uint32_t required_32bit_lane_count =
      (uint32_t)((payload_byte_count + 3u) / 4u);
  if (required_32bit_lane_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return;
  }
  diagnostic->vector_type = vector_type;
  diagnostic->vector_lane_count = source->vector_lane_count;
  diagnostic->element_byte_count = source->element_byte_count;
  diagnostic->required_32bit_lane_count = required_32bit_lane_count;
  diagnostic->native_max_vector_lane_count =
      loom_amdgpu_memory_access_vector_lane_limit(
          source->element_byte_count, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES);
  diagnostic->scalarized_max_vector_lane_count =
      loom_amdgpu_memory_access_vector_lane_limit(
          source->element_byte_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

// Canonical address expressions can retain integer payloads after a
// value-preserving address cast. Narrow payloads keep their signed numeric
// domain, and the selected materializer normalizes their native carrier before
// byte arithmetic. Predicates project to zero/one.
static bool loom_amdgpu_memory_dynamic_index_can_materialize_vaddr(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_amdgpu_type_is_address_scalar(type) ||
         (loom_type_is_scalar(type) &&
          loom_scalar_type_is_integer(loom_type_element_type(type)));
}

static bool loom_amdgpu_memory_dynamic_index_can_materialize_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, loom_value_id_t value_id) {
  if (!loom_amdgpu_memory_dynamic_index_can_materialize_vaddr(module,
                                                              value_id)) {
    return false;
  }
  return !loom_amdgpu_analyzed_source_value_is_native_i1_mask(
             module, fact_table, view_regions, analysis, value_id) &&
         !loom_amdgpu_analyzed_source_value_prefers_vgpr(
             module, fact_table, view_regions, analysis, value_id);
}

static bool loom_amdgpu_memory_dynamic_index_can_materialize_u32_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, loom_value_id_t value_id) {
  if (!loom_amdgpu_memory_dynamic_index_can_materialize_soffset(
          module, fact_table, view_regions, analysis, value_id)) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_amdgpu_type_is_i64(type) &&
      (!loom_type_is_scalar(type) ||
       loom_type_element_type(type) != LOOM_SCALAR_TYPE_OFFSET)) {
    return true;
  }
  return fact_table != NULL &&
         loom_value_facts_fit_unsigned_bit_count(
             loom_value_fact_table_lookup(fact_table, value_id), 32);
}

static bool loom_amdgpu_memory_dynamic_term_needs_scaled_materialization(
    const loom_low_source_memory_dynamic_term_t* term) {
  return term->stride_value_count != 0 ||
         (term->byte_stride != 1 &&
          term->byte_shift == LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE);
}

static bool loom_amdgpu_memory_dynamic_term_can_materialize_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_low_source_memory_dynamic_term_t* term) {
  if (term->source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID &&
      !loom_amdgpu_memory_dynamic_index_can_materialize_soffset(
          module, fact_table, view_regions, analysis, term->index)) {
    return false;
  }
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_index_can_materialize_u32_soffset(
            module, fact_table, view_regions, analysis,
            term->stride_values[i])) {
      return false;
    }
  }
  return true;
}

uint32_t loom_amdgpu_source_memory_scalar_term_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_low_source_memory_access_plan_t* source) {
  uint32_t mask = 0;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (loom_amdgpu_memory_dynamic_term_can_materialize_soffset(
            module, fact_table, view_regions, analysis,
            &source->dynamic_terms[i])) {
      mask |= UINT32_C(1) << i;
    }
  }
  return mask;
}

bool loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(
    const loom_module_t* module,
    const loom_low_source_memory_dynamic_term_t* term) {
  if (!loom_amdgpu_memory_dynamic_index_can_materialize_vaddr(module,
                                                              term->index)) {
    return false;
  }
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_index_can_materialize_vaddr(
            module, term->stride_values[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_memory_access_dynamic_view_base_can_materialize_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_low_source_memory_access_plan_t* source) {
  return source->dynamic_view_base_term_count != 0 &&
         source->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID &&
         loom_amdgpu_memory_dynamic_index_can_materialize_u32_soffset(
             module, fact_table, view_regions, analysis,
             source->dynamic_view_base_value_id);
}

static void loom_amdgpu_memory_dynamic_term_materialization_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_dynamic_term_materialization_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_memory_dynamic_term_materialization_plan_t){
      .dynamic_view_base_can_materialize_soffset =
          loom_amdgpu_memory_access_dynamic_view_base_can_materialize_soffset(
              module, fact_table, view_regions, analysis, source),
  };
  for (uint8_t term_index = 0; term_index < source->dynamic_term_count;
       ++term_index) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[term_index];
    out_plan->terms[term_index] =
        (loom_amdgpu_memory_dynamic_term_materialization_t){
            .needs_scaled_materialization =
                loom_amdgpu_memory_dynamic_term_needs_scaled_materialization(
                    term),
            .can_materialize_soffset =
                loom_amdgpu_memory_dynamic_term_can_materialize_soffset(
                    module, fact_table, view_regions, analysis, term),
            .can_materialize_vaddr =
                loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(module,
                                                                      term),
        };
  }
}

static bool loom_amdgpu_memory_dynamic_term_select_value_kind(
    const loom_amdgpu_memory_dynamic_term_materialization_t* materialization,
    loom_amdgpu_memory_dynamic_index_kind_t* out_dynamic_index_kind) {
  if (materialization->can_materialize_soffset) {
    *out_dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
    return true;
  }
  if (materialization->can_materialize_vaddr) {
    *out_dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    return true;
  }
  return false;
}

static bool loom_amdgpu_memory_dynamic_term_select_preferred_kind(
    loom_amdgpu_memory_dynamic_index_kind_t preferred_kind,
    const loom_amdgpu_memory_dynamic_term_materialization_t* materialization,
    loom_amdgpu_memory_dynamic_index_kind_t* out_dynamic_index_kind) {
  *out_dynamic_index_kind = preferred_kind;
  if (!materialization->needs_scaled_materialization) {
    return preferred_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET
               ? materialization->can_materialize_soffset
               : materialization->can_materialize_vaddr;
  }
  if (preferred_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET &&
      materialization->can_materialize_soffset) {
    *out_dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
    return true;
  }
  if (materialization->can_materialize_vaddr) {
    *out_dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    return true;
  }
  return false;
}

void loom_amdgpu_mark_source_memory_dynamic_terms_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_dynamic_term_t* terms, uint8_t term_count) {
  for (uint8_t i = 0; i < term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &terms[i];
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t j = 0; j < term->stride_value_count; ++j) {
      loom_low_lower_require_source_value_storage(context,
                                                  term->stride_values[j]);
    }
  }
}

void loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  loom_amdgpu_mark_source_memory_dynamic_terms_storage_demands(
      context, source->dynamic_terms, source->dynamic_term_count);
}

void loom_amdgpu_mark_source_memory_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  loom_low_lower_require_source_value_storage(
      context, loom_low_source_memory_access_base_view_value_id(source));
  if (source->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID) {
    loom_low_lower_require_source_value_storage(
        context, source->dynamic_view_base_value_id);
  }
  loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(context, source);
}

void loom_amdgpu_mark_source_memory_plan_root_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  loom_low_lower_require_source_value_storage(context, source->root_value_id);
  loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(context, source);
}

static loom_value_id_t loom_amdgpu_memory_access_payload_value(
    const loom_module_t* module, const loom_op_t* source_op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  IREE_ASSERT(loom_memory_access_isa(access));
  return loom_memory_access_value(access);
}

void loom_amdgpu_mark_memory_access_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0u);
  const loom_amdgpu_memory_mask_plan_t* mask =
      loom_amdgpu_memory_access_mask_plan(plan);
  if (mask) {
    loom_low_lower_require_source_value_storage(context, mask->condition);
    loom_low_lower_require_source_value_storage(context, mask->passthrough);
  }
  for (uint32_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_memory_access_t* access =
        &plan->packets[packet_index].access;
    if (access->realization.vaddr) {
      continue;
    }
    loom_amdgpu_mark_source_memory_plan_storage_demands(context,
                                                        access->source);
    if (access->retained_component_kind ==
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE) {
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        access->source->retained_component.term;
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t i = 0; i < term->stride_value_count; ++i) {
      loom_low_lower_require_source_value_storage(context,
                                                  term->stride_values[i]);
    }
  }

  const loom_value_id_t value = loom_amdgpu_memory_access_payload_value(
      loom_low_lower_context_module(context), source_op);
  if (value != LOOM_VALUE_ID_INVALID) {
    loom_low_lower_require_source_value_storage(context, value);
  }
}

static bool loom_amdgpu_memory_dynamic_term_select_source_kind(
    const loom_amdgpu_memory_access_t* access,
    const loom_low_source_memory_dynamic_term_t* term,
    const loom_amdgpu_memory_dynamic_term_materialization_t* materialization,
    loom_amdgpu_memory_dynamic_index_kind_t* out_dynamic_index_kind,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (term->source) {
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
      if (access->source->memory_space ==
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE;
        return false;
      }
      if (!loom_amdgpu_memory_dynamic_term_select_preferred_kind(
              LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET, materialization,
              out_dynamic_index_kind)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE;
        return false;
      }
      return true;
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
      return loom_amdgpu_memory_dynamic_term_select_preferred_kind(
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR, materialization,
          out_dynamic_index_kind);
    case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
      if (!loom_amdgpu_memory_dynamic_term_select_value_kind(
              materialization, out_dynamic_index_kind)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
        return false;
      }
      return true;
    default:
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      return false;
  }
}

bool loom_amdgpu_memory_access_select_dynamic_term_kinds_from_plan(
    const loom_amdgpu_memory_dynamic_term_materialization_plan_t*
        materialization_plan,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  bool has_workgroup_coordinate = false;
  bool has_vaddr_only_term = false;
  for (uint8_t term_index = 0; term_index < access->source->dynamic_term_count;
       ++term_index) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source->dynamic_terms[term_index];
    has_workgroup_coordinate |=
        term->source ==
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
    has_vaddr_only_term |=
        materialization_plan->terms[term_index].can_materialize_vaddr &&
        !materialization_plan->terms[term_index].can_materialize_soffset;
  }
  // Grouping narrow terms can retain a cheaper whole-address realization.
  // A wide uniform contribution instead belongs in the scalar base; routing
  // it through VADDR would force full-width vector address construction.
  const bool route_value_terms_through_vaddr =
      has_vaddr_only_term && !has_workgroup_coordinate &&
      loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
          access->source, 0, 32);

  for (uint8_t term_index = 0; term_index < access->source->dynamic_term_count;
       ++term_index) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source->dynamic_terms[term_index];
    if (term_index < access->source->dynamic_view_base_term_count &&
        materialization_plan->dynamic_view_base_can_materialize_soffset) {
      access->dynamic_term_kinds[term_index] =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
      continue;
    }
    if (route_value_terms_through_vaddr &&
        term->source == LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE &&
        materialization_plan->terms[term_index].can_materialize_vaddr) {
      access->dynamic_term_kinds[term_index] =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
      continue;
    }
    loom_amdgpu_memory_dynamic_index_kind_t dynamic_index_kind =
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE;
    if (!loom_amdgpu_memory_dynamic_term_select_source_kind(
            access, term, &materialization_plan->terms[term_index],
            &dynamic_index_kind, diagnostic)) {
      return false;
    }
    access->dynamic_term_kinds[term_index] = dynamic_index_kind;
  }
  return true;
}

bool loom_amdgpu_memory_access_select_dynamic_term_kinds(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  loom_amdgpu_memory_dynamic_term_materialization_plan_t materialization_plan;
  loom_amdgpu_memory_dynamic_term_materialization_plan_build(
      module, fact_table, view_regions, analysis, access->source,
      &materialization_plan);
  return loom_amdgpu_memory_access_select_dynamic_term_kinds_from_plan(
      &materialization_plan, access, diagnostic);
}

bool loom_amdgpu_memory_access_select_vaddr_dynamic_terms(
    const loom_module_t* module, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source->dynamic_terms[i];
    if (!loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(module, term)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      diagnostic->dynamic_term_index = i;
      return false;
    }
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
  return true;
}

bool loom_amdgpu_source_memory_offset_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    int64_t static_byte_offset) {
  return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      source, static_byte_offset, 32);
}

static loom_type_t loom_amdgpu_memory_access_value_vector_type(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return loom_type_none();
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (loom_type_is_vector(value_type)) {
    return value_type;
  }
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                             loom_type_element_type(value_type),
                             loom_dim_pack_static(1), /*encoding_id=*/0);
}

static loom_type_t loom_amdgpu_memory_access_source_vector_type(
    const loom_module_t* module, const loom_op_t* source_op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  IREE_ASSERT(loom_memory_access_isa(access));
  if (loom_memory_access_byte_offset(access) != LOOM_VALUE_ID_INVALID) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                               loom_dim_pack_static(1), /*encoding_id=*/0);
  }
  const loom_value_id_t value = loom_memory_access_value(access);
  if (value != LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_memory_access_value_vector_type(module, value);
  }
  if (source_op->result_count != 1) {
    IREE_ASSERT_UNREACHABLE("expected load-like memory access result");
    return loom_type_none();
  }
  return loom_amdgpu_memory_access_value_vector_type(
      module, loom_op_const_results(source_op)[0]);
}

static bool loom_amdgpu_memory_low16_float_use_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_use_t* use) {
  const loom_op_t* user_op = loom_use_user_op(*use);
  const uint16_t operand_index = loom_use_operand_index(*use);
  const loom_operand_role_t operand_role = loom_op_operand_role_at(
      loom_op_vtable(module, user_op), user_op, operand_index);
  if (operand_role == LOOM_OPERAND_ROLE_FLOAT_EXTENSION_SOURCE) {
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    const loom_scalar_type_t element_type = loom_type_element_type(value_type);
    return loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                         element_type);
  }
  const loom_memory_access_t store_access =
      loom_memory_access_cast(module, user_op);
  if (loom_memory_access_isa(store_access) &&
      loom_memory_access_operation_kind(store_access) ==
          LOOM_MEMORY_ACCESS_OPERATION_STORE &&
      loom_memory_access_operand_index_is_payload(store_access,
                                                  operand_index)) {
    return true;
  }
  if (operand_role == LOOM_OPERAND_ROLE_COMPOSITE_ELEMENT) {
    uint32_t payload_bit_count = 0;
    uint32_t register_count = 0;
    if (user_op->result_count != 1 ||
        !loom_amdgpu_type_packed_16bit_float_storage(
            loom_module_value_type(module, loom_op_results(user_op)[0]),
            &payload_bit_count, &register_count) ||
        payload_bit_count == 0 || register_count == 0) {
      return false;
    }
    return true;
  }
  return false;
}

static bool loom_amdgpu_memory_access_selects_low16_float_descriptor(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t kind, loom_type_t vector_type) {
  if (kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      source_op->result_count != 1) {
    return false;
  }

  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          vector_type, &payload_bit_count, &register_count) ||
      payload_bit_count != 16u || register_count != 1u) {
    return false;
  }

  const loom_value_id_t result_id = loom_op_const_results(source_op)[0];
  const loom_value_t* result_value = loom_module_value(module, result_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result_value, use) {
    if (!loom_amdgpu_memory_low16_float_use_is_supported(module, result_id,
                                                         use)) {
      return false;
    }
  }
  return !loom_value_has_no_uses(result_value);
}

static bool loom_amdgpu_memory_access_selects_signed_i8_descriptor(
    const loom_op_t* source_op, loom_low_source_memory_operation_kind_t kind,
    loom_type_t vector_type) {
  return kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
         source_op->result_count == 1 &&
         loom_amdgpu_static_vector_lane_count(vector_type, LOOM_SCALAR_TYPE_I8,
                                              1) == 1;
}

static bool loom_amdgpu_memory_access_selects_signed_i16_descriptor(
    const loom_op_t* source_op, loom_low_source_memory_operation_kind_t kind,
    loom_type_t vector_type) {
  return kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
         source_op->result_count == 1 &&
         loom_amdgpu_static_vector_lane_count(vector_type, LOOM_SCALAR_TYPE_I16,
                                              1) == 1;
}

static bool loom_amdgpu_memory_access_selects_unsigned_i8_descriptor(
    const loom_op_t* source_op, loom_low_source_memory_operation_kind_t kind) {
  return kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
         source_op->kind == LOOM_OP_BUFFER_LOAD_I8_U;
}

static bool loom_amdgpu_memory_access_make_packet(
    const loom_amdgpu_memory_access_t* logical_access,
    int64_t source_byte_offset, uint32_t packet_byte_count,
    uint32_t payload_register_count, loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  const loom_low_source_memory_access_plan_t* source = logical_access->source;
  if (source->element_byte_count == 0 ||
      packet_byte_count % source->element_byte_count != 0) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  int64_t static_byte_offset = 0;
  if (source_byte_offset < 0 ||
      !iree_checked_add_i64(logical_access->static_byte_offset,
                            source_byte_offset, &static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  *out_access = *logical_access;
  out_access->vector_lane_count =
      (uint32_t)(packet_byte_count / source->element_byte_count);
  out_access->static_byte_offset = static_byte_offset;
  out_access->payload_register_class =
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR;
  out_access->payload_register_count = payload_register_count;
  out_access->packet_byte_count = packet_byte_count;
  // Byte packets are assembled into packed words with shifts and ORs. Their
  // unused bits must be zero so a negative byte cannot overwrite its neighbors.
  if (packet_byte_count == 1u && source->vector_lane_count > 1u &&
      source->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    out_access->payload_format =
        LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_UNSIGNED_8BIT_INTEGER;
  }
  return true;
}

static bool loom_amdgpu_memory_access_plan_push_packet(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    uint32_t payload_byte_offset, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_selection_t* out_selection,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  loom_amdgpu_memory_access_diagnostic_t packet_diagnostic = *out_diagnostic;
  if (!loom_amdgpu_memory_access_select_packet(selection_context, kind,
                                               allow_global_smem, access,
                                               &packet_diagnostic)) {
    if (!selection_context->requires_scalar_payload ||
        access->packet_byte_count <= 4u ||
        packet_diagnostic.rejection_bits !=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING) {
      *out_diagnostic = packet_diagnostic;
      return false;
    }
    // Keep the retained scalar carrier when the target lacks this wide packet.
    // Power-of-two prefixes cover the payload without reading beyond its end.
    const uint32_t prefix_register_count =
        UINT32_C(1) << iree_math_floor_log2_u64(access->payload_register_count -
                                                1u);
    const uint32_t prefix_byte_count = prefix_register_count * 4u;
    loom_amdgpu_memory_access_t prefix = {0};
    loom_amdgpu_memory_access_t suffix = {0};
    if (!loom_amdgpu_memory_access_make_packet(access, 0, prefix_byte_count,
                                               prefix_register_count, &prefix,
                                               out_diagnostic) ||
        !loom_amdgpu_memory_access_make_packet(
            access, prefix_byte_count,
            access->packet_byte_count - prefix_byte_count,
            access->payload_register_count - prefix_register_count, &suffix,
            out_diagnostic)) {
      return false;
    }
    return loom_amdgpu_memory_access_plan_push_packet(
               selection_context, kind, allow_global_smem, payload_byte_offset,
               &prefix, out_selection, out_diagnostic) &&
           loom_amdgpu_memory_access_plan_push_packet(
               selection_context, kind, allow_global_smem,
               payload_byte_offset + prefix_byte_count, &suffix, out_selection,
               out_diagnostic);
  }
  *out_diagnostic = packet_diagnostic;
  if (out_selection->packet_count >= IREE_ARRAYSIZE(out_selection->packets)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  loom_amdgpu_memory_access_select_vaddr_realizations(access);
  out_selection->packets[out_selection->packet_count++] =
      (loom_amdgpu_memory_packet_plan_t){
          .access = *access,
          .payload_byte_offset = payload_byte_offset,
      };
  return true;
}

static bool loom_amdgpu_memory_access_plan_push_chunk_packet(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    const loom_amdgpu_memory_access_t* logical_access,
    uint32_t payload_byte_offset, uint32_t payload_register_count,
    uint32_t packet_byte_count,
    loom_amdgpu_memory_access_selection_t* out_selection,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  loom_amdgpu_memory_access_t packet_access = {0};
  if (!loom_amdgpu_memory_access_make_packet(
          logical_access, payload_byte_offset, packet_byte_count,
          payload_register_count, &packet_access, out_diagnostic)) {
    return false;
  }
  return loom_amdgpu_memory_access_plan_push_packet(
      selection_context, kind, allow_global_smem, payload_byte_offset,
      &packet_access, out_selection, out_diagnostic);
}

static bool loom_amdgpu_memory_access_make_strided_packet(
    const loom_amdgpu_memory_access_t* logical_access, uint32_t lane_index,
    uint32_t lane_count, uint32_t* out_payload_byte_offset,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  const loom_low_source_memory_access_plan_t* source = logical_access->source;
  const int64_t source_byte_offset =
      (int64_t)lane_index * source->vector_lane_byte_stride;
  *out_payload_byte_offset = lane_index * source->element_byte_count;
  const uint32_t packet_byte_count = lane_count * source->element_byte_count;
  const uint32_t payload_register_count = (packet_byte_count + 3u) / 4u;
  return loom_amdgpu_memory_access_make_packet(
      logical_access, source_byte_offset, packet_byte_count,
      payload_register_count, out_access, out_diagnostic);
}

static bool loom_amdgpu_memory_access_plan_push_strided_packet(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind,
    const loom_amdgpu_memory_access_t* logical_access, uint32_t lane_index,
    uint32_t lane_count, loom_amdgpu_memory_access_selection_t* out_selection,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  uint32_t payload_byte_offset = 0;
  loom_amdgpu_memory_access_t packet_access = {0};
  if (!loom_amdgpu_memory_access_make_strided_packet(
          logical_access, lane_index, lane_count, &payload_byte_offset,
          &packet_access, out_diagnostic)) {
    return false;
  }
  const bool allow_global_smem = packet_access.packet_byte_count ==
                                 packet_access.payload_register_count * 4u;
  return loom_amdgpu_memory_access_plan_push_packet(
      selection_context, kind, allow_global_smem, payload_byte_offset,
      &packet_access, out_selection, out_diagnostic);
}

static bool loom_amdgpu_memory_access_requires_ds_pair_split(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    const loom_amdgpu_memory_access_t* access) {
  return iree_any_bit_set(
             selection_context->instruction_constraints,
             LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT) &&
         loom_amdgpu_memory_access_is_ds_pair(access) &&
         access->source->vector_lane_byte_stride > 0 &&
         access->source->vector_lane_byte_stride <= UINT32_MAX &&
         !loom_amdgpu_memory_access_has_contiguous_vector_lanes(access);
}

bool loom_amdgpu_memory_access_plan_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    loom_amdgpu_instruction_constraint_bits_t instruction_constraints,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    uint8_t read_visibility_scope, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_selection_t* out_selection,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  out_selection->packet_count = 0;
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  out_selection->source = *source;
  loom_low_source_memory_access_plan_t* selected_source =
      &out_selection->source;

  loom_low_source_memory_operation_kind_t kind =
      selected_source->operation_kind;
  if (kind == LOOM_MEMORY_ACCESS_OPERATION_LOAD &&
      (selected_source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
       selected_source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC)) {
    selected_source->read_visibility_scope = read_visibility_scope;
  }
  const bool is_atomic = kind == LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_LOAD ||
                         kind == LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_STORE;
  if (is_atomic) {
    out_diagnostic->atomic_constraint = loom_amdgpu_atomic_memory_rejection_key(
        descriptor_set, selected_source);
    if (!iree_string_view_is_empty(out_diagnostic->atomic_constraint)) {
      return false;
    }
    kind = kind == LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_LOAD
               ? LOOM_MEMORY_ACCESS_OPERATION_LOAD
               : LOOM_MEMORY_ACCESS_OPERATION_STORE;
    // The selected naturally aligned packet supplies atomicity. Observable
    // execution prevents Low cleanup from merging or deleting observations.
    selected_source->access_flags |= LOOM_MEMORY_ACCESS_FLAG_VOLATILE;
  }
  if (!loom_amdgpu_memory_access_include_alloca_root_byte_offset(
          alloca_layout, selected_source, out_diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_dynamic_term_materialization_plan_t materialization_plan;
  loom_amdgpu_memory_dynamic_term_materialization_plan_build(
      module, fact_table, view_regions, analysis, selected_source,
      &materialization_plan);
  const bool preserve_inactive =
      loom_memory_access_mask(loom_memory_access_cast(module, source_op)) !=
      LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_memory_packet_selection_context_t selection_context = {
      .module = module,
      .fact_table = fact_table,
      .descriptor_set = descriptor_set,
      .view_regions = view_regions,
      .source_function = source_function,
      .bundle = bundle,
      .instruction_constraints = instruction_constraints,
      .materialization_plan = &materialization_plan,
      .root_prefers_vgpr = loom_amdgpu_analyzed_source_value_prefers_vgpr(
          module, fact_table, view_regions, analysis,
          selected_source->root_value_id),
      .requires_scalar_payload =
          !preserve_inactive && source_op->result_count != 0 &&
          !loom_amdgpu_analyzed_source_value_prefers_vgpr(
              module, fact_table, view_regions, analysis,
              loom_op_const_results(source_op)[0]),
  };

  loom_amdgpu_memory_access_t access =
      loom_amdgpu_memory_access_from_source(selected_source);
  access.preserve_inactive = preserve_inactive;
  const loom_type_t vector_type =
      loom_amdgpu_memory_access_source_vector_type(module, source_op);
  loom_amdgpu_memory_access_try_record_vector_width_diagnostic(
      selected_source, vector_type, out_diagnostic);
  if (!loom_amdgpu_memory_access_register_footprint(vector_type, &access,
                                                    out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          access.static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (access.source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      loom_amdgpu_memory_access_selects_low16_float_descriptor(
          module, source_op, kind, vector_type)) {
    access.payload_format = LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_LOW_16BIT_FLOAT;
  }
  if (loom_amdgpu_memory_access_selects_signed_i8_descriptor(source_op, kind,
                                                             vector_type)) {
    access.payload_format =
        LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_8BIT_INTEGER;
  }
  if (loom_amdgpu_memory_access_selects_signed_i16_descriptor(source_op, kind,
                                                              vector_type)) {
    access.payload_format =
        LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
  }
  if (loom_amdgpu_memory_access_selects_unsigned_i8_descriptor(source_op,
                                                               kind)) {
    access.payload_format =
        LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_UNSIGNED_8BIT_INTEGER;
  }
  const uint32_t whole_register_byte_count = access.payload_register_count * 4u;
  const bool whole_register_payload =
      access.packet_byte_count == whole_register_byte_count;
  if (!is_atomic &&
      !loom_amdgpu_memory_access_has_contiguous_vector_lanes(&access)) {
    if (selected_source->vector_lane_byte_stride <= 0 ||
        selected_source->vector_lane_byte_stride > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
      return false;
    }
    const bool native_packet_shape =
        access.payload_register_count <= LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES &&
        (whole_register_payload || access.packet_byte_count <= 2);
    if (native_packet_shape &&
        !loom_amdgpu_memory_access_requires_ds_pair_split(&selection_context,
                                                          &access)) {
      loom_amdgpu_memory_access_t native_access = access;
      loom_amdgpu_memory_access_diagnostic_t native_diagnostic =
          *out_diagnostic;
      const bool allow_global_smem = whole_register_payload;
      if (loom_amdgpu_memory_access_plan_push_packet(
              &selection_context, kind, allow_global_smem,
              /*payload_byte_offset=*/0, &native_access, out_selection,
              &native_diagnostic)) {
        return true;
      }
      out_selection->packet_count = 0;
      if (native_diagnostic.rejection_bits !=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE) {
        *out_diagnostic = native_diagnostic;
        return false;
      }
    }
    IREE_ASSERT_LE(selected_source->vector_lane_count,
                   IREE_ARRAYSIZE(out_selection->packets));
    uint32_t lane_index = 0;
    while (lane_index < selected_source->vector_lane_count) {
      const uint32_t remaining_lane_count =
          selected_source->vector_lane_count - lane_index;
      const bool can_try_ds_pair = remaining_lane_count >= 2u &&
                                   selected_source->memory_space ==
                                       LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
                                   (selected_source->element_byte_count == 4u ||
                                    selected_source->element_byte_count == 8u);
      if (can_try_ds_pair) {
        loom_amdgpu_memory_access_diagnostic_t pair_diagnostic =
            *out_diagnostic;
        uint32_t pair_payload_byte_offset = 0;
        loom_amdgpu_memory_access_t pair_access = {0};
        if (loom_amdgpu_memory_access_make_strided_packet(
                &access, lane_index, /*lane_count=*/2,
                &pair_payload_byte_offset, &pair_access, &pair_diagnostic) &&
            !loom_amdgpu_memory_access_requires_ds_pair_split(
                &selection_context, &pair_access) &&
            loom_amdgpu_memory_access_plan_push_packet(
                &selection_context, kind, /*allow_global_smem=*/false,
                pair_payload_byte_offset, &pair_access, out_selection,
                &pair_diagnostic)) {
          lane_index += 2u;
          continue;
        }
      }
      if (!loom_amdgpu_memory_access_plan_push_strided_packet(
              &selection_context, kind, &access, lane_index,
              /*lane_count=*/1, out_selection, out_diagnostic)) {
        return false;
      }
      ++lane_index;
    }
    return true;
  }
  if (access.payload_register_count <= LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES &&
      (whole_register_payload || access.packet_byte_count <= 2)) {
    const bool allow_global_smem = !is_atomic && whole_register_payload;
    return loom_amdgpu_memory_access_plan_push_packet(
        &selection_context, kind, allow_global_smem, 0, &access, out_selection,
        out_diagnostic);
  }
  if (access.payload_register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  if (selected_source->vector_lane_byte_stride <= 0 ||
      selected_source->vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint32_t payload_byte_offset = 0;
  while (payload_byte_offset < access.packet_byte_count) {
    const uint32_t remaining = access.packet_byte_count - payload_byte_offset;
    // Full registers use the widest packet. Sub-register tails use a halfword
    // and then a byte, retaining both positions within the packed SSA value.
    const uint32_t packet_byte_count =
        remaining >= 4u
            ? iree_min(remaining / 4u, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) * 4u
            : iree_min(remaining, 2u);
    if (!loom_amdgpu_memory_access_plan_push_chunk_packet(
            &selection_context, kind,
            /*allow_global_smem=*/whole_register_payload, &access,
            payload_byte_offset, (packet_byte_count + 3u) / 4u,
            packet_byte_count, out_selection, out_diagnostic)) {
      return false;
    }
    payload_byte_offset += packet_byte_count;
  }
  return true;
}
