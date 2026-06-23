// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_is_single_byte_payload(
    loom_type_t vector_type) {
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type) ||
      loom_type_dim_static_size_at(vector_type, 0) != 1) {
    return false;
  }
  return loom_scalar_type_bitwidth(loom_type_element_type(vector_type)) == 8;
}

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
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (access->source.memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      break;
    default:
      return true;
  }

  uint64_t root_byte_offset = 0;
  if (!loom_amdgpu_source_alloca_layout_lookup_root(
          fact_table, source_function, access->source.memory_space,
          access->source.root_value_id, &root_byte_offset) ||
      root_byte_offset > INT64_MAX) {
    diagnostic->rejection_bits |=
        loom_amdgpu_memory_access_alloca_root_rejection_bit(
            access->source.memory_space);
    return false;
  }

  int64_t static_byte_offset = 0;
  if (!iree_checked_add_i64(access->source.static_byte_offset,
                            (int64_t)root_byte_offset, &static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET_RANGE;
    return false;
  }
  access->source.static_byte_offset = static_byte_offset;
  return true;
}

static bool loom_amdgpu_memory_access_register_footprint(
    loom_type_t vector_type, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  access->payload_format = LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_GENERIC;
  if (loom_amdgpu_memory_access_is_single_byte_payload(vector_type)) {
    access->payload_register_class =
        LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR;
    access->payload_register_count = 1;
    access->packet_byte_count = 1;
    return true;
  }

  uint32_t register_count = loom_amdgpu_vector_32bit_lane_count(vector_type);
  if (register_count != 0) {
    access->payload_register_class =
        LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR;
    access->payload_register_count = register_count;
    access->packet_byte_count = register_count * 4u;
    return true;
  }

  uint32_t payload_bit_count = 0;
  bool packed_16bit_float = false;
  if (!loom_amdgpu_type_packed_integer_storage(vector_type, &payload_bit_count,
                                               &register_count)) {
    packed_16bit_float = loom_amdgpu_type_packed_16bit_float_storage(
        vector_type, &payload_bit_count, &register_count);
  }
  if (payload_bit_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  const uint32_t register_bit_count = register_count * 32u;
  const bool packed_i8_pair = payload_bit_count == 16u &&
                              register_count == 1u &&
                              loom_amdgpu_static_vector_lane_count(
                                  vector_type, LOOM_SCALAR_TYPE_I8, 2) == 2;
  const bool packed_i16_scalar = payload_bit_count == 16u &&
                                 register_count == 1u &&
                                 loom_amdgpu_static_vector_lane_count(
                                     vector_type, LOOM_SCALAR_TYPE_I16, 1) == 1;
  if (payload_bit_count != register_bit_count && !packed_16bit_float &&
      !packed_i8_pair && !packed_i16_scalar) {
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

static bool loom_amdgpu_memory_access_is_packed_16bit_float_tail(
    loom_type_t vector_type, const loom_amdgpu_memory_access_t* access) {
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          vector_type, &payload_bit_count, &register_count)) {
    return false;
  }
  return register_count == access->payload_register_count &&
         payload_bit_count / 8u == access->packet_byte_count &&
         access->packet_byte_count % 4u == 2u;
}

static bool loom_amdgpu_memory_access_has_32bit_lanes(
    const loom_amdgpu_memory_access_t* access) {
  return access->packet_byte_count ==
         access->source.element_byte_count *
             iree_max(access->payload_register_count, 1u);
}

static bool loom_amdgpu_memory_access_has_contiguous_vector_lanes(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.vector_lane_count <= 1 ||
         access->source.vector_lane_byte_stride ==
             access->source.element_byte_count;
}

static bool loom_amdgpu_memory_dynamic_index_can_materialize_vaddr(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_amdgpu_type_is_address_scalar(type) ||
         loom_amdgpu_type_is_i32(type);
}

static bool loom_amdgpu_memory_dynamic_index_can_materialize_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return (loom_amdgpu_type_is_address_scalar(type) ||
          loom_amdgpu_type_is_i32(type)) &&
         !loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                view_regions, value_id);
}

static bool loom_amdgpu_memory_dynamic_index_can_materialize_u32_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, loom_value_id_t value_id) {
  if (!loom_amdgpu_memory_dynamic_index_can_materialize_soffset(
          module, fact_table, view_regions, value_id)) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type) ||
      loom_type_element_type(type) != LOOM_SCALAR_TYPE_OFFSET) {
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
    const loom_low_source_memory_dynamic_term_t* term) {
  if (term->index >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, term->index);
  if (!loom_amdgpu_type_is_address_scalar(type) &&
      !loom_amdgpu_type_is_i32(type)) {
    return false;
  }
  if (term->source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID &&
      loom_amdgpu_source_value_prefers_vgpr(module, fact_table, view_regions,
                                            term->index)) {
    return false;
  }
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_index_can_materialize_u32_soffset(
            module, fact_table, view_regions, term->stride_values[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(
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

typedef struct loom_amdgpu_memory_dynamic_term_materialization_t {
  // Term requires materializing scaled address arithmetic.
  bool needs_scaled_materialization;
  // Term and stride inputs can be materialized through a scalar offset operand.
  bool can_materialize_soffset;
  // Term and stride inputs can be materialized through a vector address
  // operand.
  bool can_materialize_vaddr;
} loom_amdgpu_memory_dynamic_term_materialization_t;

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

static void loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t j = 0; j < term->stride_value_count; ++j) {
      loom_low_lower_require_source_value_storage(context,
                                                  term->stride_values[j]);
    }
  }
}

void loom_amdgpu_mark_source_memory_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  loom_low_lower_require_source_value_storage(context, source->view_value_id);
  loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(context, source);
}

void loom_amdgpu_mark_source_memory_plan_root_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source) {
  loom_low_lower_require_source_value_storage(context, source->root_value_id);
  loom_amdgpu_mark_source_memory_plan_dynamic_storage_demands(context, source);
}

static loom_value_id_t loom_amdgpu_memory_access_payload_value(
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VIEW_LOAD:
    case LOOM_OP_VECTOR_LOAD:
      return LOOM_VALUE_ID_INVALID;
    case LOOM_OP_VIEW_STORE:
      return loom_view_store_value(source_op);
    case LOOM_OP_VECTOR_STORE:
      return loom_vector_store_value(source_op);
    default:
      IREE_ASSERT(false);
      return LOOM_VALUE_ID_INVALID;
  }
}

void loom_amdgpu_mark_memory_access_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0u);
  // Split packets only change static offsets and register slices; dynamic
  // address storage is shared by every packet in the selected access plan.
  loom_amdgpu_mark_source_memory_plan_storage_demands(
      context, &plan->packets[0].access.source);

  const loom_value_id_t value =
      loom_amdgpu_memory_access_payload_value(source_op);
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
      if (access->source.memory_space ==
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

bool loom_amdgpu_memory_access_select_dynamic_term_kinds(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t term_index = 0; term_index < access->source.dynamic_term_count;
       ++term_index) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source.dynamic_terms[term_index];
    if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    const loom_amdgpu_memory_dynamic_term_materialization_t materialization = {
        .needs_scaled_materialization =
            loom_amdgpu_memory_dynamic_term_needs_scaled_materialization(term),
        .can_materialize_soffset =
            loom_amdgpu_memory_dynamic_term_can_materialize_soffset(
                module, fact_table, view_regions, term),
        .can_materialize_vaddr =
            loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(module, term),
    };
    loom_amdgpu_memory_dynamic_index_kind_t dynamic_index_kind =
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE;
    if (!loom_amdgpu_memory_dynamic_term_select_source_kind(
            access, term, &materialization, &dynamic_index_kind, diagnostic)) {
      return false;
    }
    access->dynamic_term_kinds[term_index] = dynamic_index_kind;
  }
  return true;
}

bool loom_amdgpu_source_memory_offset_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    int64_t static_byte_offset) {
  return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      source, static_byte_offset, 32);
}

static bool loom_amdgpu_memory_vaddr_offset_fits_u32(
    const loom_amdgpu_memory_access_t* access, int64_t static_byte_offset) {
  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64(static_byte_offset);
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    loom_value_facts_addi(&offset_facts,
                          &access->source.dynamic_terms[i].byte_facts,
                          &offset_facts);
  }
  return loom_value_facts_fit_unsigned_bit_count(offset_facts, 32);
}

typedef enum loom_amdgpu_memory_descriptor_domain_e {
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE = 0,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR = 1,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS = 2,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_FLAT = 3,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SMEM = 4,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH = 5,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_,
} loom_amdgpu_memory_descriptor_domain_t;

typedef enum loom_amdgpu_memory_address_attempt_kind_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR = 4,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_FLAT = 5,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM = 6,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR = 7,
} loom_amdgpu_memory_address_attempt_kind_t;

typedef enum loom_amdgpu_memory_address_attempt_result_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED = 2,
} loom_amdgpu_memory_address_attempt_result_t;

typedef struct loom_amdgpu_memory_address_attempt_t {
  // Address-form selector attempted by this row.
  loom_amdgpu_memory_address_attempt_kind_t kind;
} loom_amdgpu_memory_address_attempt_t;

static const loom_amdgpu_memory_address_attempt_t kAmdgpuLdsAddressAttempts[] =
    {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuGlobalAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_FLAT,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuDescriptorAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuScratchAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR,
        },
};

static const uint8_t kAmdgpuMemorySpaceDescriptorDomainMap[] = {
    [LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
    [LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH,
    [LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_HOST] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_,
    [LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuMemorySpaceDescriptorDomainMap) ==
                  LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC + 1u,
              "AMDGPU memory-space map must cover every memory-space fact");

static bool loom_amdgpu_memory_descriptor_domain_from_memory_space(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_memory_descriptor_domain_t* out_descriptor_domain) {
  *out_descriptor_domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE;
  const uint32_t memory_space_ordinal = (uint32_t)memory_space;
  if (memory_space_ordinal >=
      IREE_ARRAYSIZE(kAmdgpuMemorySpaceDescriptorDomainMap)) {
    return false;
  }
  const uint8_t descriptor_domain =
      kAmdgpuMemorySpaceDescriptorDomainMap[memory_space_ordinal];
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_) {
    return false;
  }
  *out_descriptor_domain =
      (loom_amdgpu_memory_descriptor_domain_t)descriptor_domain;
  return true;
}

typedef uint32_t loom_amdgpu_memory_descriptor_candidate_key_t;

enum {
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS = 3u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS = 3u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_BITS = 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS = 5u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS = 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS = 2u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS = 3u,

  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT = 0u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS,

  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS) -
      1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_MAX =
      (1u
       << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS) -
      1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX =
      (1u
       << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS) -
      1u,
};

static_assert(LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_MAX,
              "memory descriptor domain key field is too small");
static_assert(LOOM_AMDGPU_MEMORY_ADDRESS_FORM_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_MAX,
              "memory address form key field is too small");
static_assert(LOOM_AMDGPU_MEMORY_OPERATION_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_MAX,
              "memory operation key field is too small");
static_assert(4u * LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX,
              "memory packet byte count key field is too small");
static_assert(
    LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_COUNT_ - 1u <=
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_MAX,
    "memory payload register class key field is too small");
static_assert(LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_MAX,
              "memory payload format key field is too small");
static_assert(
    LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES <=
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX,
    "memory payload register count key field is too small");
static_assert(
    LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT +
            LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS <=
        8u * sizeof(loom_amdgpu_memory_descriptor_candidate_key_t),
    "memory descriptor candidate key storage is too small");

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                           \
    domain, address_form, kind, packet_byte_count, payload_register_class,     \
    payload_format, payload_register_count)                                    \
  ((((loom_amdgpu_memory_descriptor_candidate_key_t)(domain))                  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT) |                 \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(address_form))            \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT) |           \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(kind))                    \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KIND_SHIFT) |                   \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(packet_byte_count))       \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT) |      \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_register_class))  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT) | \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_format))          \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT) |         \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_register_count))  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT))

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE(                             \
    domain, address_form, kind, packet_byte_count, payload_register_class,   \
    payload_format, payload_register_count, descriptor_ref_value)            \
  {                                                                          \
      .key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                    \
          domain, address_form, kind, (packet_byte_count),                   \
          payload_register_class, payload_format, (payload_register_count)), \
      .descriptor_ref = descriptor_ref_value,                                \
  },

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE(                       \
    domain, address_form, kind, packet_byte_count, payload_register_class,   \
    payload_format, payload_register_count, first_candidate_index,           \
    candidate_count_value)                                                   \
  {                                                                          \
      .key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                    \
          domain, address_form, kind, (packet_byte_count),                   \
          payload_register_class, payload_format, (payload_register_count)), \
      .first_candidate = (first_candidate_index),                            \
      .candidate_count = (candidate_count_value),                            \
  },

static bool loom_amdgpu_memory_descriptor_candidate_key(
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint32_t packet_byte_count,
    loom_amdgpu_memory_payload_register_class_t payload_register_class,
    loom_amdgpu_memory_payload_format_t payload_format,
    uint32_t payload_register_count,
    loom_amdgpu_memory_descriptor_candidate_key_t* out_key) {
  if (packet_byte_count >
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX ||
      payload_register_count >
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX) {
    *out_key = 0;
    return false;
  }
  *out_key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(
      domain, address_form, kind, packet_byte_count, payload_register_class,
      payload_format, payload_register_count);
  return true;
}

typedef struct loom_amdgpu_memory_descriptor_candidate_t {
  // Packed selector key for memory operation shape and payload format.
  loom_amdgpu_memory_descriptor_candidate_key_t key;
  // Stable descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_memory_descriptor_candidate_t;
static_assert(sizeof(loom_amdgpu_memory_descriptor_candidate_t) == 8,
              "memory descriptor candidate rows must stay compact");

typedef struct loom_amdgpu_memory_descriptor_candidate_range_t {
  // Packed selector key shared by a contiguous fallback candidate range.
  loom_amdgpu_memory_descriptor_candidate_key_t key;
  // First candidate row for the selector key.
  uint16_t first_candidate;
  // Number of candidate rows for the selector key.
  uint16_t candidate_count;
} loom_amdgpu_memory_descriptor_candidate_range_t;
static_assert(sizeof(loom_amdgpu_memory_descriptor_candidate_range_t) == 8,
              "memory descriptor candidate ranges must stay compact");

static const loom_amdgpu_memory_descriptor_candidate_t
    kAmdgpuMemoryDescriptorCandidates[] = {
#include "loom/target/arch/amdgpu/lower/memory_descriptor_candidates.inl"
};

static const loom_amdgpu_memory_descriptor_candidate_range_t
    kAmdgpuMemoryDescriptorCandidateRanges[] = {
#include "loom/target/arch/amdgpu/lower/memory_descriptor_candidate_ranges.inl"
};

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY

static const loom_amdgpu_memory_descriptor_candidate_range_t*
loom_amdgpu_find_memory_descriptor_candidate_range(
    loom_amdgpu_memory_descriptor_candidate_key_t candidate_key) {
  iree_host_size_t lower_bound = 0;
  iree_host_size_t upper_bound =
      IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidateRanges);
  while (lower_bound < upper_bound) {
    const iree_host_size_t middle =
        lower_bound + ((upper_bound - lower_bound) / 2);
    const loom_amdgpu_memory_descriptor_candidate_range_t* range =
        &kAmdgpuMemoryDescriptorCandidateRanges[middle];
    if (candidate_key < range->key) {
      upper_bound = middle;
    } else if (candidate_key > range->key) {
      lower_bound = middle + 1;
    } else {
      return range;
    }
  }
  return NULL;
}

static bool loom_amdgpu_select_memory_descriptor_candidate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint32_t packet_byte_count,
    loom_amdgpu_memory_payload_register_class_t payload_register_class,
    loom_amdgpu_memory_payload_format_t payload_format,
    uint32_t payload_register_count,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  *out_descriptor = NULL;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  loom_amdgpu_memory_descriptor_candidate_key_t candidate_key = 0;
  if (!loom_amdgpu_memory_descriptor_candidate_key(
          domain, address_form, kind, packet_byte_count, payload_register_class,
          payload_format, payload_register_count, &candidate_key)) {
    return false;
  }
  const loom_amdgpu_memory_descriptor_candidate_range_t* range =
      loom_amdgpu_find_memory_descriptor_candidate_range(candidate_key);
  if (range == NULL) {
    return false;
  }
  IREE_ASSERT((iree_host_size_t)range->first_candidate +
                  range->candidate_count <=
              IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates));
  const iree_host_size_t candidate_end =
      (iree_host_size_t)range->first_candidate + range->candidate_count;
  for (iree_host_size_t i = range->first_candidate; i < candidate_end; ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, candidate->descriptor_ref);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              descriptor_ordinal);
    IREE_ASSERT(descriptor != NULL);
    *out_descriptor = descriptor;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_buffer_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
      address_form, kind, access->packet_byte_count,
      access->payload_register_class, access->payload_format,
      access->payload_register_count, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_saddr_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count, out_descriptor,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_smem_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SMEM,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count, out_descriptor,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_flat_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_FLAT,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT, kind, access->packet_byte_count,
      access->payload_register_class, access->payload_format,
      access->payload_register_count, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_scratch_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count, out_descriptor,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_ds_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind, access->packet_byte_count,
      access->payload_register_class, access->payload_format,
      access->payload_register_count, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          access->source.memory_space, &descriptor_domain)) {
    return false;
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    return loom_amdgpu_select_ds_memory_descriptor(
        descriptor_set, access, kind, out_descriptor, out_descriptor_ordinal);
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    return loom_amdgpu_select_scratch_memory_descriptor(
        descriptor_set, access, kind, out_descriptor, out_descriptor_ordinal);
  }
  return loom_amdgpu_select_buffer_memory_descriptor(
      descriptor_set, access, access->address_form, kind, out_descriptor,
      out_descriptor_ordinal);
}

typedef struct loom_amdgpu_offset_immediate_encoding_t {
  // Target immediate encoding ID.
  uint16_t encoding_id;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_offset_immediate_encoding_t;

static const loom_amdgpu_offset_immediate_encoding_t
    kAmdgpuOffsetImmediateEncodings[] = {
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE,
            .unit_byte_count = 1,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD,
            .unit_byte_count = 4,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD,
            .unit_byte_count = 8,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64,
            .unit_byte_count = 4 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64,
            .unit_byte_count = 8 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16,
            .unit_byte_count = 1,
        },
};

static bool loom_amdgpu_immediate_encoding_address_unit_byte_count(
    uint16_t encoding_id, uint32_t* out_unit_byte_count) {
  *out_unit_byte_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuOffsetImmediateEncodings); ++i) {
    const loom_amdgpu_offset_immediate_encoding_t* encoding =
        &kAmdgpuOffsetImmediateEncodings[i];
    if (encoding->encoding_id == encoding_id) {
      *out_unit_byte_count = encoding->unit_byte_count;
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_descriptor_offset_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint16_t expected_offset_immediate_count,
    loom_low_immediate_kind_t expected_kind,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  *out_info = (loom_amdgpu_descriptor_offset_immediate_info_t){
      .kind = expected_kind,
      .signed_min = INT64_MIN,
      .unsigned_max = UINT64_MAX,
  };
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  if (descriptor->immediate_count != 0 &&
      descriptor->immediate_start >= descriptor_set->immediate_count) {
    return false;
  }
  uint16_t offset_immediate_count = 0;
  uint64_t unsigned_max = UINT64_MAX;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor->immediate_start + i;
    if (immediate_index >= descriptor_set->immediate_count) {
      return false;
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    uint32_t unit_byte_count = 0;
    if (!loom_amdgpu_immediate_encoding_address_unit_byte_count(
            immediate->encoding_id, &unit_byte_count)) {
      continue;
    }
    if (immediate->kind != expected_kind) {
      return false;
    }
    if (offset_immediate_count == 0) {
      out_info->unit_byte_count = unit_byte_count;
    } else if (out_info->unit_byte_count != unit_byte_count) {
      return false;
    }
    if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
      out_info->signed_min =
          iree_max(out_info->signed_min, immediate->signed_min);
    }
    unsigned_max = iree_min(unsigned_max, immediate->unsigned_max);
    ++offset_immediate_count;
  }
  if (offset_immediate_count != expected_offset_immediate_count) {
    return false;
  }
  out_info->unsigned_max = unsigned_max;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_buffer_off_zero(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access) {
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_low_source_memory_access_is_dynamic(&access->source) ||
      access->scalar_byte_offset != 0) {
    return false;
  }
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO, kind, &descriptor,
          &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info) ||
      offset_info.unit_byte_count != 1 ||
      (uint64_t)access->immediate_offset > offset_info.unsigned_max) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_split_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  const uint64_t static_byte_offset =
      (uint64_t)access->source.static_byte_offset;
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_unit_byte_count) != 0) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    const uint64_t encoded_offset = static_byte_offset / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    access->immediate_offset = (uint32_t)encoded_offset;
    access->scalar_byte_offset = 0;
    if (!loom_amdgpu_source_memory_offset_fits_u32(
            &access->source, access->source.static_byte_offset)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if (offset_unit_byte_count != 1) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  uint64_t immediate_offset = iree_min(static_byte_offset, offset_unsigned_max);
  if (access->payload_register_count == 4) {
    immediate_offset &= ~UINT64_C(15);
  }
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = (uint32_t)immediate_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &access->source, access->source.static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_split_lds_default_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &access->source, access->source.static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }

  const uint64_t static_byte_offset =
      (uint64_t)access->source.static_byte_offset;
  const bool static_fits_immediate =
      (static_byte_offset % offset_unit_byte_count) == 0 &&
      (static_byte_offset / offset_unit_byte_count) <= offset_unsigned_max;
  access->immediate_offset =
      static_fits_immediate
          ? (int64_t)(static_byte_offset / offset_unit_byte_count)
          : 0;
  access->secondary_immediate_offset = 0;
  access->vaddr_static_byte_offset =
      static_fits_immediate ? 0 : (uint32_t)static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_global_saddr_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  const int64_t static_byte_offset = access->source.static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  if (static_byte_offset < offset_info->signed_min ||
      static_byte_offset > signed_max) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE;
    return false;
  }
  if (!loom_amdgpu_memory_vaddr_offset_fits_u32(access,
                                                /*static_byte_offset=*/0)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_global_smem_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (access->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }

  const uint64_t static_byte_offset =
      (uint64_t)access->source.static_byte_offset;
  const uint64_t immediate_offset =
      iree_min(static_byte_offset, offset_info->unsigned_max);
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    access->immediate_offset = 0;
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = static_byte_offset;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
    return true;
  }
  access->immediate_offset = (int64_t)immediate_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  if (!loom_amdgpu_source_memory_offset_fits_u32(&access->source,
                                                 access->scalar_byte_offset)) {
    access->immediate_offset = 0;
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = static_byte_offset;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
    return true;
  }
  access->scalar_base_byte_offset = 0;
  access->scalar_offset_placement =
      LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_SOFFSET;
  return true;
}

static bool loom_amdgpu_memory_access_split_global_flat_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  const int64_t static_byte_offset = access->source.static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  if (static_byte_offset < offset_info->signed_min) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  if (static_byte_offset <= signed_max) {
    access->immediate_offset = static_byte_offset;
    access->vaddr_static_byte_offset = 0;
  } else {
    access->immediate_offset = 0;
    access->vaddr_static_byte_offset = (uint64_t)static_byte_offset;
  }
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_scratch_vaddr_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  const int64_t static_byte_offset = access->source.static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  if (static_byte_offset < offset_info->signed_min) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  if (static_byte_offset <= signed_max) {
    access->immediate_offset = static_byte_offset;
    access->vaddr_static_byte_offset = 0;
  } else {
    access->immediate_offset = 0;
    access->vaddr_static_byte_offset = (uint64_t)static_byte_offset;
  }
  if (access->vaddr_static_byte_offset > INT64_MAX ||
      !loom_amdgpu_memory_vaddr_offset_fits_u32(
          access, (int64_t)access->vaddr_static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_ds2_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.static_byte_offset < 0) {
    return false;
  }
  int64_t secondary_byte_offset = 0;
  if (!iree_checked_add_i64(access->source.static_byte_offset,
                            access->source.vector_lane_byte_stride,
                            &secondary_byte_offset) ||
      secondary_byte_offset < 0) {
    return false;
  }

  const uint64_t byte_offsets[] = {
      (uint64_t)access->source.static_byte_offset,
      (uint64_t)secondary_byte_offset,
  };
  uint32_t encoded_offsets[IREE_ARRAYSIZE(byte_offsets)] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(byte_offsets); ++i) {
    if ((byte_offsets[i] % offset_unit_byte_count) != 0) {
      return false;
    }
    const uint64_t encoded_offset = byte_offsets[i] / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      return false;
    }
    encoded_offsets[i] = (uint32_t)encoded_offset;
  }

  access->immediate_offset = encoded_offsets[0];
  access->secondary_immediate_offset = encoded_offsets[1];
  access->scalar_byte_offset = 0;
  if (!loom_amdgpu_source_memory_offset_fits_u32(&access->source,
                                                 secondary_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_select_ds2_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  loom_amdgpu_memory_descriptor_candidate_key_t candidate_key = 0;
  if (!loom_amdgpu_memory_descriptor_candidate_key(
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count,
          &candidate_key)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  bool found_kind_descriptor = false;
  const loom_amdgpu_memory_descriptor_candidate_range_t* range =
      loom_amdgpu_find_memory_descriptor_candidate_range(candidate_key);
  if (range == NULL) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  IREE_ASSERT((iree_host_size_t)range->first_candidate +
                  range->candidate_count <=
              IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates));
  const iree_host_size_t candidate_end =
      (iree_host_size_t)range->first_candidate + range->candidate_count;
  for (iree_host_size_t i = range->first_candidate; i < candidate_end; ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, candidate->descriptor_ref);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    found_kind_descriptor = true;
    loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
    if (!loom_amdgpu_descriptor_offset_immediate_info(
            descriptor_set, descriptor_ordinal, 2,
            LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
      return false;
    }
    if (!loom_amdgpu_memory_access_split_ds2_static_offset(
            access, offset_info.unit_byte_count, offset_info.unsigned_max,
            diagnostic)) {
      continue;
    }
    access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR;
    access->descriptor = loom_low_descriptor_set_descriptor_at(
        descriptor_set, descriptor_ordinal);
    IREE_ASSERT(access->descriptor != NULL);
    return true;
  }

  diagnostic->rejection_bits |=
      found_kind_descriptor
          ? LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE
          : LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
  return false;
}

static bool loom_amdgpu_try_select_ds_addtid_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  loom_target_workgroup_size_t workgroup_size = {0};
  const uint32_t wavefront_size = bundle != NULL && bundle->snapshot != NULL
                                      ? bundle->snapshot->subgroup_size
                                      : 0;
  if (wavefront_size == 0 ||
      !loom_amdgpu_required_workgroup_size(module, source_function, bundle,
                                           &workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.x > wavefront_size ||
      workgroup_size.y != 1 || workgroup_size.z != 1) {
    return false;
  }
  if (access->payload_register_count != 1 ||
      access->source.dynamic_term_count != 1 ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->source.dynamic_terms[0].source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      access->source.dynamic_terms[0].stride_value_count != 0 ||
      access->source.dynamic_terms[0].dimension != LOOM_KERNEL_DIMENSION_X ||
      access->source.dynamic_terms[0].byte_stride != 1 ||
      !loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor_candidate(
          descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count, &descriptor,
          &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return false;
  }

  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          &ignored_diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID;
  access->descriptor = descriptor;
  return true;
}

static iree_string_view_t loom_amdgpu_memory_ds_addtid_topology_reason_key(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle) {
  const uint32_t wavefront_size = bundle != NULL && bundle->snapshot != NULL
                                      ? bundle->snapshot->subgroup_size
                                      : 0;
  if (wavefront_size == 0) {
    return IREE_SV("wavefront_size_unknown");
  }
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_amdgpu_required_workgroup_size(module, source_function, bundle,
                                           &workgroup_size) ||
      workgroup_size.x == 0) {
    return IREE_SV("workgroup_size_unknown");
  }
  if (workgroup_size.x > wavefront_size) {
    return IREE_SV("cross_wave_workgroup");
  }
  if (workgroup_size.y != 1 || workgroup_size.z != 1) {
    return IREE_SV("multidimensional_workgroup");
  }
  return iree_string_view_empty();
}

iree_string_view_t loom_amdgpu_memory_ds_addtid_reason_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    return IREE_SV("selected");
  }
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return IREE_SV("not_applicable");
  }
  const iree_string_view_t topology_reason =
      loom_amdgpu_memory_ds_addtid_topology_reason_key(module, source_function,
                                                       bundle);
  if (!iree_string_view_is_empty(topology_reason)) {
    return topology_reason;
  }
  if (access->payload_register_count != 1) {
    return IREE_SV("payload_width");
  }
  if (access->source.dynamic_term_count != 1) {
    return IREE_SV("dynamic_term_count");
  }
  const loom_low_source_memory_dynamic_term_t* term =
      &access->source.dynamic_terms[0];
  if (access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
    return IREE_SV("dynamic_term_kind");
  }
  if (term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID) {
    return IREE_SV("dynamic_term_source");
  }
  if (term->stride_value_count != 0) {
    return IREE_SV("dynamic_stride_value");
  }
  if (term->dimension != LOOM_KERNEL_DIMENSION_X) {
    return IREE_SV("dynamic_dimension");
  }
  if (term->byte_stride != 1) {
    return IREE_SV("dynamic_stride_bytes");
  }
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    return IREE_SV("vector_lane_layout");
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor_candidate(
          descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count, &descriptor,
          &descriptor_ordinal)) {
    return IREE_SV("descriptor_missing");
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return IREE_SV("descriptor_offset_immediate");
  }
  loom_amdgpu_memory_access_t access_copy = *access;
  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_static_offset(
          &access_copy, offset_info.unit_byte_count, offset_info.unsigned_max,
          &ignored_diagnostic)) {
    return IREE_SV("static_offset_range");
  }
  return IREE_SV("not_selected");
}

static bool loom_amdgpu_memory_access_offset_is_aligned(
    const loom_amdgpu_memory_access_t* access, uint32_t byte_alignment,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (byte_alignment == 0) {
    return false;
  }
  if ((access->source.static_byte_offset % (int64_t)byte_alignment) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT;
    return false;
  }
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    const loom_value_facts_t byte_facts =
        access->source.dynamic_terms[i].byte_facts;
    if (!loom_value_facts_is_zero(byte_facts) &&
        !loom_value_facts_divisible_by(byte_facts, byte_alignment)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
      diagnostic->dynamic_term_index = i;
      return false;
    }
  }
  return true;
}

static void loom_amdgpu_memory_access_record_descriptor_missing(
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  diagnostic->rejection_bits |=
      access->payload_format ==
              LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_LOW_16BIT_FLOAT
          ? LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LOW16_DESCRIPTOR_MISSING
          : LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
}

static bool loom_amdgpu_memory_access_needs_signed_i16_repair(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->payload_format ==
             LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
}

static bool loom_amdgpu_memory_access_signed_i16_repair_is_available(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  if (!loom_amdgpu_memory_access_needs_signed_i16_repair(access)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE);
}

static bool loom_amdgpu_memory_access_try_select_buffer(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->payload_register_count == 4 &&
      !loom_amdgpu_memory_access_offset_is_aligned(
          access, /*byte_alignment=*/16, diagnostic)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access, LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind,
          &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  access->descriptor = descriptor;
  loom_amdgpu_memory_access_try_select_buffer_off_zero(descriptor_set, kind,
                                                       access);
  return true;
}

static bool loom_amdgpu_memory_access_uses_only_scalar_address_terms(
    const loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      return false;
    }
  }
  return true;
}

static bool
loom_amdgpu_memory_access_promote_scalar_materializable_terms_to_soffset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] ==
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      return false;
    }
    if (!loom_amdgpu_memory_dynamic_term_can_materialize_soffset(
            module, fact_table, view_regions,
            &access->source.dynamic_terms[i])) {
      return false;
    }
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  }
  return true;
}

static bool loom_amdgpu_memory_access_root_is_read_only(
    const loom_amdgpu_memory_access_t* access,
    const loom_view_region_table_t* view_regions) {
  if (view_regions == NULL ||
      access->source.root_value_id == LOOM_VALUE_ID_INVALID ||
      access->source.alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    return false;
  }
  const loom_view_access_flags_t access_flags =
      loom_view_region_table_root_access_flags(view_regions,
                                               access->source.root_value_id);
  return iree_all_bits_set(access_flags, LOOM_VIEW_ACCESS_READ) &&
         !iree_any_bit_set(access_flags, LOOM_VIEW_ACCESS_WRITE);
}

static bool loom_amdgpu_memory_access_try_select_global_smem(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    const loom_value_fact_table_t* fact_table,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access) {
  loom_amdgpu_memory_access_t candidate = *access;
  if (kind != LOOM_AMDGPU_MEMORY_OPERATION_LOAD ||
      (candidate.source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
       candidate.source.memory_space !=
           LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) ||
      loom_amdgpu_memory_cache_policy_is_present(
          &candidate.source.cache_policy) ||
      !loom_amdgpu_memory_access_promote_scalar_materializable_terms_to_soffset(
          module, fact_table, view_regions, &candidate) ||
      !loom_amdgpu_memory_access_uses_only_scalar_address_terms(&candidate) ||
      !loom_amdgpu_memory_access_root_is_read_only(&candidate, view_regions)) {
    return false;
  }

  candidate.payload_register_class =
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR;
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_smem_memory_descriptor(
          descriptor_set, &candidate, kind, &descriptor, &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return false;
  }
  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_global_smem_static_offset(
          &candidate, &offset_info, &ignored_diagnostic)) {
    return false;
  }
  candidate.address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM;
  candidate.descriptor = descriptor;
  *access = candidate;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_global_saddr(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_saddr_memory_descriptor(
          descriptor_set, access, kind, &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_global_saddr_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_dynamic_term_can_flat_address(
    const loom_module_t* module,
    const loom_low_source_memory_dynamic_term_t* term) {
  if (term->byte_stride == 1 && term->stride_value_count == 0 &&
      term->index < module->values.count &&
      (term->byte_shift == 0 ||
       term->byte_shift == LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE)) {
    const loom_type_t index_type = loom_module_value_type(module, term->index);
    if (loom_type_is_scalar(index_type) &&
        loom_type_element_type(index_type) == LOOM_SCALAR_TYPE_OFFSET) {
      return true;
    }
  }
  if (loom_value_facts_is_float(term->byte_facts) ||
      term->byte_facts.range_lo < 0 || term->byte_stride <= 0 ||
      term->byte_stride > UINT32_MAX) {
    return false;
  }
  if (term->byte_stride != 1 &&
      term->byte_shift == LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
    return false;
  }
  if (term->stride_value_count != 0) {
    return false;
  }
  if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
      term->byte_shift >= 32) {
    return false;
  }
  return term->byte_facts.range_hi / term->byte_stride <= UINT32_MAX;
}

void loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  diagnostic->rejection_bits |=
      LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS;
  diagnostic->dynamic_term_index = 0;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_flat_address(
            module, &source->dynamic_terms[i])) {
      diagnostic->dynamic_term_index = i;
      return;
    }
  }
}

static bool loom_amdgpu_memory_access_dynamic_terms_can_flat_address(
    const loom_module_t* module, const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_flat_address(
            module, &access->source.dynamic_terms[i])) {
      loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
          module, &access->source, diagnostic);
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_memory_dynamic_term_can_emit_flat_address(
    const loom_module_t* module,
    const loom_low_source_memory_dynamic_term_t* term) {
  if (!loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(module, term)) {
    return false;
  }
  if (term->byte_stride <= 0 || term->byte_stride > UINT32_MAX) {
    return false;
  }
  if (term->stride_value_count != 0) {
    return false;
  }
  return term->byte_shift == LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE ||
         term->byte_shift < 32;
}

static bool loom_amdgpu_memory_access_dynamic_terms_can_emit_flat_address(
    const loom_module_t* module, const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_emit_flat_address(
            module, &access->source.dynamic_terms[i])) {
      loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
          module, &access->source, diagnostic);
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_memory_access_dynamic_terms_can_vaddr(
    const loom_module_t* module, const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(
            module, &access->source.dynamic_terms[i])) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      diagnostic->dynamic_term_index = i;
      return false;
    }
  }
  return true;
}

void loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(
    loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
}

bool loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_access = (loom_amdgpu_memory_access_t){
      .source = *source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
      .descriptor = NULL,
  };
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          out_access->source.static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_include_alloca_root_byte_offset(
          fact_table, source_function, out_access, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(
          module, fact_table, view_regions, out_access, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_dynamic_terms_can_vaddr(module, out_access,
                                                         out_diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(out_access);
  if ((uint64_t)out_access->source.static_byte_offset > UINT32_MAX ||
      !loom_amdgpu_memory_vaddr_offset_fits_u32(
          out_access, out_access->source.static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }

  out_access->vaddr_static_byte_offset =
      (uint64_t)out_access->source.static_byte_offset;
  return true;
}

static bool loom_amdgpu_memory_access_select_global_flat_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_flat_memory_descriptor(
          descriptor_set, access, kind, &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_global_flat_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_global_flat(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_dynamic_terms_can_flat_address(module, access,
                                                                diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_global_flat_descriptor(
          descriptor_set, kind, access, diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(access);
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_scratch_vaddr(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_dynamic_terms_can_vaddr(module, access,
                                                         diagnostic)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_scratch_memory_descriptor(
          descriptor_set, access, kind, &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_scratch_vaddr_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(access);
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_lds_default(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor(descriptor_set, access, kind,
                                            &access->descriptor,
                                            &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  return loom_amdgpu_memory_access_split_lds_default_static_offset(
      access, offset_info.unit_byte_count, offset_info.unsigned_max,
      diagnostic);
}

static loom_amdgpu_memory_address_attempt_result_t
loom_amdgpu_memory_address_attempt_apply(
    const loom_amdgpu_memory_address_attempt_t* attempt,
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    loom_amdgpu_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (attempt->kind) {
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID:
      return loom_amdgpu_try_select_ds_addtid_memory_descriptor(
                 descriptor_set, module, source_function, bundle, access, kind)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR:
      if (loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (access->payload_register_count != 2 ||
          !loom_amdgpu_memory_access_has_32bit_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_select_ds2_memory_descriptor(descriptor_set, access,
                                                      kind, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT:
      return loom_amdgpu_memory_access_try_select_lds_default(
                 descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE:
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_buffer(descriptor_set, kind,
                                                         access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM:
      if (!allow_global_smem) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      return loom_amdgpu_memory_access_try_select_global_smem(
                 module, descriptor_set, view_regions, fact_table, kind, access)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR:
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_global_saddr(
                 descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_FLAT:
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_global_flat(
                 module, descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR:
      return loom_amdgpu_memory_access_try_select_scratch_vaddr(
                 module, descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
  }
  return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
}

static bool loom_amdgpu_memory_access_select_address_form(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    loom_amdgpu_memory_descriptor_domain_t descriptor_domain,
    loom_amdgpu_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_amdgpu_memory_address_attempt_t* attempts =
      kAmdgpuGlobalAddressAttempts;
  iree_host_size_t attempt_count = IREE_ARRAYSIZE(kAmdgpuGlobalAddressAttempts);
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    attempts = kAmdgpuLdsAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuLdsAddressAttempts);
  } else if (descriptor_domain ==
             LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    attempts = kAmdgpuScratchAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuScratchAddressAttempts);
  } else if (access->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    attempts = kAmdgpuDescriptorAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuDescriptorAddressAttempts);
  }
  for (iree_host_size_t i = 0; i < attempt_count; ++i) {
    const loom_amdgpu_memory_address_attempt_result_t result =
        loom_amdgpu_memory_address_attempt_apply(
            &attempts[i], module, descriptor_set, view_regions, fact_table,
            source_function, bundle, kind, allow_global_smem, access,
            diagnostic);
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED) {
      return true;
    }
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED) {
      return false;
    }
  }
  return false;
}

loom_amdgpu_memory_operation_kind_t
loom_amdgpu_memory_operation_kind_from_source(
    const loom_low_source_memory_access_plan_t* source) {
  switch (source->operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD:
      return LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE:
      return LOOM_AMDGPU_MEMORY_OPERATION_STORE;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH:
      IREE_ASSERT_UNREACHABLE("AMDGPU memory lowering selected non-load/store");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_UNREACHABLE("unknown source memory operation");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_type_t loom_amdgpu_memory_access_source_vector_type(
    const loom_module_t* module, const loom_op_t* source_op) {
  loom_type_t scalar_type = loom_type_none();
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      return loom_module_value_type(module, loom_vector_load_result(source_op));
    case LOOM_OP_VECTOR_STORE:
      return loom_module_value_type(module, loom_vector_store_value(source_op));
    case LOOM_OP_VIEW_LOAD:
      scalar_type =
          loom_module_value_type(module, loom_view_load_result(source_op));
      break;
    case LOOM_OP_VIEW_STORE:
      scalar_type =
          loom_module_value_type(module, loom_view_store_value(source_op));
      break;
    default:
      return loom_type_none();
  }
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                             loom_type_element_type(scalar_type),
                             loom_dim_pack_static(1), /*encoding_id=*/0);
}

static bool loom_amdgpu_memory_low16_float_use_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_use_t* use) {
  const loom_op_t* user_op = loom_use_user_op(*use);
  if (loom_scalar_extf_isa(user_op)) {
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    const loom_scalar_type_t element_type = loom_type_element_type(value_type);
    return loom_scalar_extf_input(user_op) == value_id &&
           (element_type == LOOM_SCALAR_TYPE_F16 ||
            element_type == LOOM_SCALAR_TYPE_BF16);
  }
  if (loom_view_store_isa(user_op) &&
      loom_view_store_value(user_op) == value_id) {
    return true;
  }
  if (loom_vector_from_elements_isa(user_op)) {
    uint32_t payload_bit_count = 0;
    uint32_t register_count = 0;
    if (!loom_amdgpu_type_packed_16bit_float_storage(
            loom_module_value_type(module,
                                   loom_vector_from_elements_result(user_op)),
            &payload_bit_count, &register_count) ||
        payload_bit_count == 0 || register_count == 0) {
      return false;
    }
    const loom_value_slice_t elements =
        loom_vector_from_elements_elements(user_op);
    for (iree_host_size_t i = 0; i < elements.count; ++i) {
      if (loom_value_slice_get(elements, i) == value_id) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_memory_access_selects_low16_float_descriptor(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t kind, loom_type_t vector_type) {
  if (kind != LOOM_AMDGPU_MEMORY_OPERATION_LOAD ||
      !loom_view_load_isa(source_op)) {
    return false;
  }

  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          vector_type, &payload_bit_count, &register_count) ||
      payload_bit_count != 16u || register_count != 1u) {
    return false;
  }

  const loom_value_id_t result_id = loom_view_load_result(source_op);
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

static bool loom_amdgpu_memory_access_selects_signed_i16_descriptor(
    const loom_op_t* source_op, loom_amdgpu_memory_operation_kind_t kind,
    loom_type_t vector_type) {
  return kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD &&
         loom_view_load_isa(source_op) &&
         loom_amdgpu_static_vector_lane_count(vector_type, LOOM_SCALAR_TYPE_I16,
                                              1) == 1;
}

static bool loom_amdgpu_memory_access_select_packet(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    loom_amdgpu_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  access->descriptor = NULL;
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(access->dynamic_term_kinds); ++i) {
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE;
  }
  if (access->payload_register_count == 0 ||
      access->payload_register_count > LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }

  if (access->source.vector_lane_byte_stride <= 0 ||
      access->source.vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          access->source.static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_include_alloca_root_byte_offset(
          fact_table, source_function, access, out_diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(
          module, fact_table, view_regions, access, out_diagnostic)) {
    return false;
  }

  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          access->source.memory_space, &descriptor_domain)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }

  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS ||
      descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    if (access->source.root_value_id >= module->values.count) {
      out_diagnostic->rejection_bits |=
          loom_amdgpu_memory_access_alloca_root_rejection_bit(
              access->source.memory_space);
      return false;
    }
    const loom_value_t* root_value =
        loom_module_value(module, access->source.root_value_id);
    const loom_op_t* root_op = loom_value_is_block_arg(root_value)
                                   ? NULL
                                   : loom_value_def_op(root_value);
    if (root_op == NULL || !loom_buffer_alloca_isa(root_op) ||
        loom_buffer_alloca_memory_space(root_op) !=
            access->source.memory_space) {
      out_diagnostic->rejection_bits |=
          loom_amdgpu_memory_access_alloca_root_rejection_bit(
              access->source.memory_space);
      return false;
    }
    if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
      loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(access);
    }
  }
  if (!loom_amdgpu_memory_access_signed_i16_repair_is_available(descriptor_set,
                                                                access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SIGNED_I16_REPAIR_DESCRIPTOR_MISSING;
    return false;
  }

  return loom_amdgpu_memory_access_select_address_form(
      module, descriptor_set, view_regions, fact_table, source_function, bundle,
      descriptor_domain, kind, allow_global_smem, access, out_diagnostic);
}

bool loom_amdgpu_memory_access_select_flat_global_address(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_access = (loom_amdgpu_memory_access_t){
      .source = *source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
      .descriptor = NULL,
  };
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  if (source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
      source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT &&
      source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }
  if (out_access->source.vector_lane_byte_stride <= 0 ||
      out_access->source.vector_lane_byte_stride > UINT32_MAX ||
      !loom_amdgpu_memory_access_has_contiguous_vector_lanes(out_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          out_access->source.static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_dynamic_terms_can_emit_flat_address(
          module, out_access, out_diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(out_access);
  if ((uint64_t)out_access->source.static_byte_offset > UINT32_MAX ||
      !loom_amdgpu_memory_vaddr_offset_fits_u32(
          out_access, out_access->source.static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  out_access->immediate_offset = 0;
  out_access->secondary_immediate_offset = 0;
  out_access->vaddr_static_byte_offset =
      (uint64_t)out_access->source.static_byte_offset;
  out_access->scalar_byte_offset = 0;
  out_access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
  return true;
}

static bool loom_amdgpu_memory_access_make_byte_chunk_source(
    const loom_low_source_memory_access_plan_t* source,
    uint32_t source_byte_offset, uint32_t packet_byte_count,
    loom_low_source_memory_access_plan_t* out_source,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_source = *source;
  if (source->element_byte_count == 0 ||
      packet_byte_count % source->element_byte_count != 0 ||
      packet_byte_count / source->element_byte_count > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  out_source->vector_lane_count =
      (uint32_t)(packet_byte_count / source->element_byte_count);
  const uint64_t static_delta_unsigned = source_byte_offset;
  if (static_delta_unsigned > INT64_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  const int64_t static_delta = (int64_t)static_delta_unsigned;
  if (source->static_byte_offset > INT64_MAX - static_delta) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  out_source->static_byte_offset = source->static_byte_offset + static_delta;
  return true;
}

static bool loom_amdgpu_memory_access_make_32bit_chunk_source(
    const loom_low_source_memory_access_plan_t* source,
    uint32_t source_register_offset, uint32_t source_register_count,
    loom_low_source_memory_access_plan_t* out_source,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  return loom_amdgpu_memory_access_make_byte_chunk_source(
      source, source_register_offset * 4u, source_register_count * 4u,
      out_source, out_diagnostic);
}

static void loom_amdgpu_memory_access_make_packet(
    const loom_low_source_memory_access_plan_t* source,
    uint32_t payload_register_count, uint32_t packet_byte_count,
    loom_amdgpu_memory_access_t* out_access) {
  *out_access = (loom_amdgpu_memory_access_t){
      .source = *source,
      .payload_register_class = LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR,
      .payload_register_count = payload_register_count,
      .packet_byte_count = packet_byte_count,
  };
}

static bool loom_amdgpu_memory_access_plan_push_packet(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    loom_amdgpu_memory_operation_kind_t kind, bool allow_global_smem,
    uint32_t source_register_offset, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_plan_t* out_plan,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  if (!loom_amdgpu_memory_access_select_packet(
          module, fact_table, descriptor_set, view_regions, source_function,
          bundle, kind, allow_global_smem, access, out_diagnostic)) {
    return false;
  }
  if (out_plan->packet_count >= IREE_ARRAYSIZE(out_plan->packets)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  out_plan->packets[out_plan->packet_count++] =
      (loom_amdgpu_memory_packet_plan_t){
          .access = *access,
          .opcode_id = LOOM_STRING_ID_INVALID,
          .source_register_offset = source_register_offset,
      };
  return true;
}

static bool loom_amdgpu_memory_access_plan_push_chunk_packet(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    loom_amdgpu_memory_operation_kind_t kind, bool allow_global_smem,
    const loom_low_source_memory_access_plan_t* source,
    uint32_t source_register_offset, uint32_t payload_register_count,
    uint32_t packet_byte_count, loom_amdgpu_memory_access_plan_t* out_plan,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  loom_low_source_memory_access_plan_t chunk_source = {0};
  if (!loom_amdgpu_memory_access_make_byte_chunk_source(
          source, source_register_offset * 4u, packet_byte_count, &chunk_source,
          out_diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_t packet_access = {0};
  loom_amdgpu_memory_access_make_packet(&chunk_source, payload_register_count,
                                        packet_byte_count, &packet_access);
  return loom_amdgpu_memory_access_plan_push_packet(
      module, fact_table, descriptor_set, view_regions, source_function, bundle,
      kind, allow_global_smem, source_register_offset, &packet_access, out_plan,
      out_diagnostic);
}

bool loom_amdgpu_memory_access_plan_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_view_region_table_t* view_regions,
    loom_func_like_t source_function, const loom_target_bundle_t* bundle,
    const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* out_source,
    loom_amdgpu_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_source = (loom_low_source_memory_access_plan_t){0};
  *out_plan = (loom_amdgpu_memory_access_plan_t){0};
  *out_source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  if (!loom_low_source_memory_access_plan_build(
          module, fact_table, source_op, out_source, out_source_diagnostic)) {
    return false;
  }

  const loom_amdgpu_memory_operation_kind_t kind =
      loom_amdgpu_memory_operation_kind_from_source(out_source);

  loom_amdgpu_memory_access_t access = {
      .source = *out_source,
  };
  const loom_type_t vector_type =
      loom_amdgpu_memory_access_source_vector_type(module, source_op);
  loom_amdgpu_memory_access_try_record_vector_width_diagnostic(
      out_source, vector_type, out_diagnostic);
  if (!loom_amdgpu_memory_access_register_footprint(vector_type, &access,
                                                    out_diagnostic)) {
    return false;
  }
  if (access.source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      loom_amdgpu_memory_access_selects_low16_float_descriptor(
          module, source_op, kind, vector_type)) {
    access.payload_format = LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_LOW_16BIT_FLOAT;
  }
  if (loom_amdgpu_memory_access_selects_signed_i16_descriptor(source_op, kind,
                                                              vector_type)) {
    access.payload_format =
        LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
  }
  const uint32_t whole_register_byte_count = access.payload_register_count * 4u;
  const bool whole_register_payload =
      access.packet_byte_count == whole_register_byte_count;
  if (access.payload_register_count <= LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES &&
      (whole_register_payload || access.payload_register_count == 1)) {
    const bool allow_global_smem =
        loom_amdgpu_type_is_32bit_memory_payload(vector_type);
    return loom_amdgpu_memory_access_plan_push_packet(
        module, fact_table, descriptor_set, view_regions, source_function,
        bundle, kind, allow_global_smem, 0, &access, out_plan, out_diagnostic);
  }
  if (access.payload_register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  const bool packed_16bit_float_tail =
      loom_amdgpu_memory_access_is_packed_16bit_float_tail(vector_type,
                                                           &access);
  if (!whole_register_payload && !packed_16bit_float_tail) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT;
    out_diagnostic->payload_type = vector_type;
    out_diagnostic->payload_bit_count = access.packet_byte_count * 8u;
    out_diagnostic->register_bit_count = access.payload_register_count * 32u;
    return false;
  }
  if (out_source->vector_lane_byte_stride <= 0 ||
      out_source->vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  if (packed_16bit_float_tail) {
    uint32_t source_register_offset = 0;
    uint32_t full_register_count = access.packet_byte_count / 4u;
    while (full_register_count != 0) {
      const uint32_t packet_register_count =
          iree_min(full_register_count, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES);
      if (!loom_amdgpu_memory_access_plan_push_chunk_packet(
              module, fact_table, descriptor_set, view_regions, source_function,
              bundle, kind, /*allow_global_smem=*/false, out_source,
              source_register_offset, packet_register_count,
              packet_register_count * 4u, out_plan, out_diagnostic)) {
        return false;
      }
      source_register_offset += packet_register_count;
      full_register_count -= packet_register_count;
    }
    const uint32_t tail_byte_count = access.packet_byte_count % 4u;
    if (tail_byte_count != 0 &&
        !loom_amdgpu_memory_access_plan_push_chunk_packet(
            module, fact_table, descriptor_set, view_regions, source_function,
            bundle, kind, /*allow_global_smem=*/false, out_source,
            source_register_offset, 1u, tail_byte_count, out_plan,
            out_diagnostic)) {
      return false;
    }
    return true;
  }

  uint32_t source_register_offset = 0;
  while (source_register_offset < access.payload_register_count) {
    const uint32_t remaining =
        access.payload_register_count - source_register_offset;
    const uint32_t packet_register_count =
        iree_min(remaining, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES);
    loom_low_source_memory_access_plan_t chunk_source = {0};
    if (!loom_amdgpu_memory_access_make_32bit_chunk_source(
            out_source, source_register_offset, packet_register_count,
            &chunk_source, out_diagnostic)) {
      return false;
    }
    loom_amdgpu_memory_access_t packet_access = {0};
    loom_amdgpu_memory_access_make_packet(&chunk_source, packet_register_count,
                                          packet_register_count * 4u,
                                          &packet_access);
    if (!loom_amdgpu_memory_access_plan_push_packet(
            module, fact_table, descriptor_set, view_regions, source_function,
            bundle, kind, /*allow_global_smem=*/true, source_register_offset,
            &packet_access, out_plan, out_diagnostic)) {
      return false;
    }
    source_register_offset += packet_register_count;
  }
  return true;
}

static iree_status_t loom_amdgpu_memory_access_plan_select_from_context(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  loom_low_source_memory_access_plan_t source = {0};
  if (!loom_amdgpu_memory_access_plan_select(
          module, loom_low_lower_context_fact_table(context),
          loom_low_lower_context_descriptor_set(context), view_regions,
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), source_op, &source, out_plan,
          &source_diagnostic, &diagnostic)) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_memory_access_plan_resolve(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_plan_t* selected_plan,
    loom_amdgpu_memory_access_plan_t* out_plan) {
  *out_plan = *selected_plan;
  for (uint32_t i = 0; i < out_plan->packet_count; ++i) {
    loom_amdgpu_memory_packet_plan_t* packet = &out_plan->packets[i];
    IREE_ASSERT(packet->access.descriptor != NULL);
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_row(
        context, packet->access.descriptor, &descriptor));
    packet->opcode_id = descriptor.opcode_id;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_memory_access_plan_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* plan) {
  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set,
                                                   &plan->packets[i].access)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_amdgpu_select_memory_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_memory_access_plan_t){0};
  *out_selected = false;
  loom_amdgpu_memory_access_plan_t selected_plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_plan_select_from_context(
      context, source_op, &selected_plan, &selected));
  if (!selected) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_memory_access_plan_cache_policy_can_lower(
          loom_low_lower_context_descriptor_set(context), &selected_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_plan_resolve(
      context, &selected_plan, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_memory_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_memory_access_plan_t){0};
  *out_selected = false;
  loom_amdgpu_memory_access_plan_t selected_plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_plan_select_from_context(
      context, source_op, &selected_plan, &selected));
  if (!selected) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_memory_access_plan_cache_policy_can_lower(
          loom_low_lower_context_descriptor_set(context), &selected_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_plan_resolve(
      context, &selected_plan, out_plan));
  *out_selected = true;
  return iree_ok_status();
}
