// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/source_value_analysis.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_table.h"

static loom_amdgpu_register_shape_t loom_amdgpu_register_shape(
    uint16_t class_id, uint32_t unit_count) {
  return (loom_amdgpu_register_shape_t){
      .class_id = class_id,
      .unit_count = unit_count,
  };
}

typedef enum loom_amdgpu_scalar_value_register_policy_e {
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE = 0,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED = 1,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1 = 2,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK = 3,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH = 4,
} loom_amdgpu_scalar_value_register_policy_t;

typedef uint32_t loom_amdgpu_scalar_value_register_flags_t;

enum loom_amdgpu_scalar_value_register_flag_bits_e {
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_NATURALLY_PREFERS_VGPR = 1u << 0,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_OP_RESULT_PREFERS_VGPR = 1u << 1,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_FALLBACK_RESULT_PREFERS_VGPR = 1u << 2,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_REQUIRES_ANALYSIS = 1u << 3,
};

typedef struct loom_amdgpu_scalar_value_register_mapping_t {
  // Type-only fallback shape used when no source value is available.
  loom_amdgpu_register_shape_t type_shape;
  // Default value shape before value-specific facts adjust placement.
  loom_amdgpu_register_shape_t default_shape;
  // Value-sensitive placement policy for this scalar type.
  loom_amdgpu_scalar_value_register_policy_t policy;
  // Placement flags that do not require inspecting the defining operation.
  loom_amdgpu_scalar_value_register_flags_t flags;
} loom_amdgpu_scalar_value_register_mapping_t;

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_SHAPES(                      \
    type_reg_class_id, type_register_unit_count, default_reg_class_id, \
    default_register_unit_count, selected_policy, selected_flags)      \
  {                                                                    \
      .type_shape = {.class_id = type_reg_class_id,                    \
                     .unit_count = type_register_unit_count},          \
      .default_shape = {.class_id = default_reg_class_id,              \
                        .unit_count = default_register_unit_count},    \
      .policy = selected_policy,                                       \
      .flags = selected_flags,                                         \
  }

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER(reg_class_id, register_unit_count,  \
                                          selected_policy, selected_flags)    \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_SHAPES(reg_class_id, register_unit_count, \
                                           reg_class_id, register_unit_count, \
                                           selected_policy, selected_flags)

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_TYPE_DEFAULT(                  \
    type_reg_class_id, type_register_unit_count, default_reg_class_id,   \
    default_register_unit_count, selected_policy, selected_flags)        \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_SHAPES(                              \
      type_reg_class_id, type_register_unit_count, default_reg_class_id, \
      default_register_unit_count, selected_policy, selected_flags)

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_NATURALLY_PREFERS_VGPR

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_OP_RESULT_PREFERS_VGPR

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FALLBACK_RESULT_VGPR \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_FALLBACK_RESULT_PREFERS_VGPR

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS \
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_REQUIRES_ANALYSIS

static const loom_amdgpu_scalar_value_register_mapping_t
    loom_amdgpu_scalar_value_register_mappings[LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_SCALAR_TYPE_INDEX] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_OFFSET] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_I1] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SCC, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_I8] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR),
        [LOOM_SCALAR_TYPE_I16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR),
        [LOOM_SCALAR_TYPE_I32] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_I64] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_F8E4M3] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR),
        [LOOM_SCALAR_TYPE_F8E5M2] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR),
        [LOOM_SCALAR_TYPE_F16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_TYPE_DEFAULT(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FALLBACK_RESULT_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_BF16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR),
        [LOOM_SCALAR_TYPE_F32] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_TYPE_DEFAULT(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FALLBACK_RESULT_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
        [LOOM_SCALAR_TYPE_F64] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR |
                LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS),
};
static_assert(IREE_ARRAYSIZE(loom_amdgpu_scalar_value_register_mappings) ==
                  LOOM_SCALAR_TYPE_COUNT_,
              "AMDGPU scalar value register mappings out of sync with scalar "
              "types");

#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FALLBACK_RESULT_VGPR
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_REQUIRES_ANALYSIS
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_OP_RESULT_VGPR
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_NATURAL_VGPR
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_TYPE_DEFAULT
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_SHAPES

static const loom_amdgpu_scalar_value_register_mapping_t*
loom_amdgpu_scalar_value_register_mapping_for_type(loom_type_t source_type) {
  if (!loom_type_is_scalar(source_type)) {
    return NULL;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  if (!loom_scalar_type_is_valid(element_type)) {
    return NULL;
  }
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      &loom_amdgpu_scalar_value_register_mappings[element_type];
  return mapping->policy == LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE
             ? NULL
             : mapping;
}

static bool loom_amdgpu_scalar_type_register_shape(
    loom_type_t source_type, loom_amdgpu_register_shape_t* out_shape) {
  *out_shape = (loom_amdgpu_register_shape_t){0};
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      loom_amdgpu_scalar_value_register_mapping_for_type(source_type);
  if (mapping == NULL || mapping->type_shape.unit_count == 0) {
    return false;
  }
  *out_shape = mapping->type_shape;
  return true;
}

bool loom_amdgpu_source_type_supported(void* user_data,
                                       const loom_module_t* module,
                                       loom_type_t source_type) {
  (void)user_data;
  (void)module;
  return loom_amdgpu_scalar_value_register_mapping_for_type(source_type) !=
         NULL;
}

static bool loom_amdgpu_scalar_type_has_register_flag(
    loom_type_t source_type, loom_amdgpu_scalar_value_register_flags_t flag) {
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      loom_amdgpu_scalar_value_register_mapping_for_type(source_type);
  return mapping != NULL && iree_any_bit_set(mapping->flags, flag);
}

bool loom_amdgpu_scalar_type_naturally_prefers_vgpr(loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_NATURALLY_PREFERS_VGPR);
}

bool loom_amdgpu_scalar_type_op_result_prefers_vgpr(loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_OP_RESULT_PREFERS_VGPR);
}

bool loom_amdgpu_scalar_type_fallback_result_prefers_vgpr(
    loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_FALLBACK_RESULT_PREFERS_VGPR);
}

bool loom_amdgpu_scalar_type_has_fixed_vgpr_storage(loom_type_t source_type) {
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      loom_amdgpu_scalar_value_register_mapping_for_type(source_type);
  return mapping != NULL &&
         mapping->policy == LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED &&
         mapping->default_shape.class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_source_scalar_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      loom_amdgpu_scalar_value_register_mapping_for_type(source_type);
  if (mapping == NULL) {
    return false;
  }
  *out_shape = mapping->default_shape;
  switch (mapping->policy) {
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED:
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1:
      if (loom_amdgpu_analyzed_source_value_is_native_i1_mask(
              module, fact_table, view_regions, analysis, source_value_id)) {
        *out_shape =
            loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
      } else if (loom_amdgpu_analyzed_source_value_is_durable_i1_bool(
                     module, fact_table, view_regions, analysis,
                     source_value_id)) {
        *out_shape =
            loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK:
      if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
              module, fact_table, view_regions, analysis, source_value_id)) {
        out_shape->class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH:
      if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
              module, fact_table, view_regions, analysis, source_value_id)) {
        out_shape->class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      }
      if (loom_amdgpu_source_address_value_needs_64bit(
              module, fact_table, source_value_id, source_type)) {
        out_shape->unit_count = 2;
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_source_vector_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    return false;
  }
  const loom_amdgpu_vector_storage_kind_flags_t storage_flags =
      loom_amdgpu_vector_storage_kind_flags(vector_storage.kind);
  if (iree_any_bit_set(storage_flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK)) {
    *out_shape = loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                            vector_storage.register_count);
    return true;
  }
  if (iree_any_bit_set(
          storage_flags,
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_ANALYZE_REGISTER_BANK)) {
    *out_shape = loom_amdgpu_register_shape(
        loom_amdgpu_analyzed_source_value_prefers_vgpr(
            module, fact_table, view_regions, analysis, source_value_id)
            ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
            : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
        vector_storage.register_count);
    return true;
  }
  if (iree_any_bit_set(storage_flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    *out_shape = loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                            vector_storage.register_count);
    return true;
  }
  return false;
}

static bool loom_amdgpu_source_buffer_value_register_shape(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value_id,
    loom_type_t source_type, loom_amdgpu_register_shape_t* out_shape) {
  if (!loom_type_is_buffer(source_type) || fact_table == NULL) return false;
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if (loom_value_facts_is_lane_varying(facts)) return false;
  loom_value_fact_buffer_reference_t reference = {0};
  if (!loom_value_facts_query_buffer_reference(&fact_table->context, facts,
                                               &reference)) {
    return false;
  }
  switch (reference.memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      *out_shape = loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return false;
  }
  return false;
}

static bool loom_amdgpu_source_value_register_shape_needs_analysis(
    loom_type_t source_type) {
  if (loom_type_is_scalar(source_type)) {
    return loom_amdgpu_scalar_type_has_register_flag(
        source_type, LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_REQUIRES_ANALYSIS);
  }

  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    return false;
  }
  return iree_any_bit_set(
      loom_amdgpu_vector_storage_kind_flags(vector_storage.kind),
      LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_ANALYZE_REGISTER_BANK);
}

static bool loom_amdgpu_source_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  return loom_amdgpu_source_buffer_value_register_shape(
             fact_table, source_value_id, source_type, out_shape) ||
         loom_amdgpu_source_scalar_value_register_shape(
             module, fact_table, view_regions, analysis, source_value_id,
             source_type, out_shape) ||
         loom_amdgpu_source_vector_value_register_shape(
             module, fact_table, view_regions, analysis, source_value_id,
             source_type, out_shape);
}

static bool loom_amdgpu_source_value_cached_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  bool has_shape = false;
  if (!loom_amdgpu_source_value_analysis_cached_register_shape(
          analysis, source_value_id, &has_shape, out_shape)) {
    has_shape = loom_amdgpu_source_value_register_shape(
        module, fact_table, view_regions, analysis, source_value_id,
        source_type, out_shape);
    loom_amdgpu_source_value_analysis_record_register_shape(
        analysis, source_value_id, out_shape, has_shape);
  }
  return has_shape;
}

iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type) {
  (void)user_data;
  loom_amdgpu_register_shape_t scalar_shape = {0};
  if (loom_amdgpu_scalar_type_register_shape(source_type, &scalar_shape)) {
    return loom_low_lower_make_register_type(
        context, scalar_shape.class_id, scalar_shape.unit_count, out_low_type);
  }
  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    const loom_amdgpu_vector_storage_kind_flags_t storage_flags =
        loom_amdgpu_vector_storage_kind_flags(vector_storage.kind);
    if (iree_any_bit_set(storage_flags,
                         LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK)) {
      return loom_amdgpu_make_sgpr_range_type(
          context, vector_storage.register_count, out_low_type);
    }
    if (iree_any_bit_set(
            storage_flags,
            LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_ANALYZE_REGISTER_BANK |
                LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
      return loom_low_lower_make_register_type(
          context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, vector_storage.register_count,
          out_low_type);
    }
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

iree_status_t loom_amdgpu_map_value(void* user_data,
                                    loom_low_lower_context_t* context,
                                    const loom_op_t* source_op,
                                    loom_value_id_t source_value_id,
                                    loom_type_t source_type,
                                    loom_type_t* out_low_type) {
  (void)user_data;
  const loom_view_region_table_t* view_regions = NULL;
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  if (loom_amdgpu_source_value_register_shape_needs_analysis(source_type)) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_context_view_regions(context, &view_regions));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_source_value_analysis_for_context(context, &analysis));
  }
  loom_amdgpu_register_shape_t shape = {0};
  if (loom_amdgpu_source_value_cached_register_shape(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), view_regions, analysis,
          source_value_id, source_type, &shape)) {
    return loom_low_lower_make_register_type(context, shape.class_id,
                                             shape.unit_count, out_low_type);
  }
  return loom_amdgpu_map_type(user_data, context, source_op, source_type,
                              out_low_type);
}

static void loom_amdgpu_map_contract_register(
    const loom_target_contract_query_environment_t* environment,
    uint16_t descriptor_register_class_id, uint32_t register_unit_count,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  IREE_ASSERT_LT(descriptor_register_class_id,
                 environment->descriptor_set->reg_class_count);
  *out_mapped_value = loom_low_lower_rule_mapped_value_register(
      descriptor_register_class_id, register_unit_count);
}

iree_status_t loom_amdgpu_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  if (loom_amdgpu_source_value_register_shape_needs_analysis(source_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_source_value_analysis_for_contract_query(
        environment, &analysis));
  }
  loom_amdgpu_register_shape_t shape = {0};
  if (loom_amdgpu_source_value_cached_register_shape(
          environment->module, environment->fact_table,
          environment->view_regions, analysis, source_value_id, source_type,
          &shape)) {
    loom_amdgpu_map_contract_register(environment, shape.class_id,
                                      shape.unit_count, out_mapped_value);
  }
  return iree_ok_status();
}
