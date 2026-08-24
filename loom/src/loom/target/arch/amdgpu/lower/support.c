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
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_table.h"

typedef struct loom_amdgpu_register_shape_t {
  // AMDGPU descriptor-set register class selected for the value.
  uint16_t class_id;
  // Number of 32-bit register units occupied by the value.
  uint32_t unit_count;
} loom_amdgpu_register_shape_t;

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
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
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

static bool loom_amdgpu_scalar_type_naturally_prefers_vgpr(
    loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_NATURALLY_PREFERS_VGPR);
}

static bool loom_amdgpu_scalar_type_op_result_prefers_vgpr(
    loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_OP_RESULT_PREFERS_VGPR);
}

static bool loom_amdgpu_scalar_type_fallback_result_prefers_vgpr(
    loom_type_t source_type) {
  return loom_amdgpu_scalar_type_has_register_flag(
      source_type,
      LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FLAG_FALLBACK_RESULT_PREFERS_VGPR);
}

static bool loom_amdgpu_scalar_type_has_fixed_vgpr_storage(
    loom_type_t source_type) {
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      loom_amdgpu_scalar_value_register_mapping_for_type(source_type);
  return mapping != NULL &&
         mapping->policy == LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED &&
         mapping->default_shape.class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

typedef uint8_t loom_amdgpu_source_value_analysis_bits_t;

enum loom_amdgpu_source_value_analysis_bit_e {
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_PREFERS_VGPR = 1u << 0,
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_NATIVE_I1_MASK = 1u << 1,
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_DURABLE_I1_BOOL = 1u << 2,
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE = 1u << 3,
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_SGPR_I1_BOOL = 1u << 4,
  LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_SCC_I1_BOOL = 1u << 5,
};

typedef struct loom_amdgpu_source_value_analysis_record_t {
  // Analysis bits whose values have been computed for this source value.
  loom_amdgpu_source_value_analysis_bits_t known_bits;
  // Analysis bits currently being computed for this source value.
  loom_amdgpu_source_value_analysis_bits_t active_bits;
  // Computed true/false values keyed by known_bits.
  loom_amdgpu_source_value_analysis_bits_t value_bits;
  // Cached register shape used when the REGISTER_SHAPE value bit is set.
  loom_amdgpu_register_shape_t register_shape;
} loom_amdgpu_source_value_analysis_record_t;

typedef struct loom_amdgpu_source_value_analysis_t {
  // Dense source value domain owned by the current source-to-low lowering run.
  const loom_local_value_domain_t* value_domain;
  // Fact table used to derive records.
  const loom_value_fact_table_t* fact_table;
  // Descriptor set governing target-specific source placement capabilities.
  const loom_low_descriptor_set_t* descriptor_set;
  // Cached target capabilities for descriptor_set.
  loom_amdgpu_descriptor_set_info_flags_t descriptor_set_info_flags;
  // Cached analysis records indexed by function-local source value ordinal.
  loom_amdgpu_source_value_analysis_record_t* records;
  // Number of initialized records.
  iree_host_size_t record_count;
  // Source function body covered by source_cfg_graph.
  const loom_region_t* source_body;
  // CFG graph for source_body, built once per source-to-low run.
  loom_cfg_graph_t source_cfg_graph;
  // True when source_cfg_graph has been initialized for a multi-block
  // source_body.
  bool source_cfg_graph_initialized;
} loom_amdgpu_source_value_analysis_t;

static int loom_amdgpu_source_value_analysis_state_key;

// Forward declarations for the mutually-recursive source placement analysis.
static bool loom_amdgpu_analyzed_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
static bool loom_amdgpu_analyzed_source_value_can_lower_as_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
static bool loom_amdgpu_analyzed_source_value_can_lower_as_sgpr_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
static bool loom_amdgpu_analyzed_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);

static iree_status_t loom_amdgpu_source_value_analysis_prepare(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_amdgpu_source_value_analysis_t* analysis) {
  if (analysis->descriptor_set != descriptor_set) {
    analysis->descriptor_set = descriptor_set;
    const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
        descriptor_set != NULL ? loom_amdgpu_target_info_descriptor_set_at(
                                     descriptor_set->descriptor_set_ordinal)
                               : NULL;
    analysis->descriptor_set_info_flags =
        descriptor_set_info != NULL ? descriptor_set_info->flags : 0;
    analysis->value_domain = NULL;
    analysis->fact_table = NULL;
  }
  if (analysis->value_domain != value_domain ||
      analysis->fact_table != fact_table ||
      (value_domain != NULL &&
       analysis->record_count < value_domain->value_count)) {
    analysis->value_domain = value_domain;
    analysis->fact_table = fact_table;
    analysis->records = NULL;
    analysis->record_count =
        value_domain != NULL ? value_domain->value_count : 0;
    if (analysis->record_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, analysis->record_count, sizeof(*analysis->records),
          (void**)&analysis->records));
      memset(analysis->records, 0,
             analysis->record_count * sizeof(*analysis->records));
    }
  }
  const loom_region_t* source_body = loom_func_like_body(source_function);
  if (analysis->source_body != source_body) {
    analysis->source_body = source_body;
    analysis->source_cfg_graph = (loom_cfg_graph_t){0};
    analysis->source_cfg_graph_initialized = false;
  }
  if (source_body != NULL && source_body->block_count > 1 &&
      !analysis->source_cfg_graph_initialized) {
    IREE_RETURN_IF_ERROR(loom_cfg_graph_build(module, source_body, arena,
                                              &analysis->source_cfg_graph));
    analysis->source_cfg_graph_initialized = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_value_analysis_for_context(
    loom_low_lower_context_t* context,
    loom_amdgpu_source_value_analysis_t** out_analysis) {
  *out_analysis = NULL;
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_source_value_analysis_state_key, sizeof(*analysis),
      (void**)&analysis));
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_value_analysis_prepare(
      loom_low_lower_context_module(context),
      loom_low_lower_context_source_function(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_value_domain(context),
      loom_low_lower_context_descriptor_set(context),
      loom_low_lower_context_function_arena(context), analysis));
  *out_analysis = analysis;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_value_analysis_for_target_low_legality(
    loom_target_low_legality_context_t* context,
    loom_amdgpu_source_value_analysis_t** out_analysis) {
  *out_analysis = NULL;
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_get_or_allocate_target_state(
      context, &loom_amdgpu_source_value_analysis_state_key, sizeof(*analysis),
      (void**)&analysis));
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_value_analysis_prepare(
      loom_target_low_legality_module(context),
      loom_target_low_legality_function(context),
      loom_target_low_legality_fact_table(context),
      loom_target_low_legality_value_domain(context),
      loom_target_low_legality_descriptor_set(context),
      loom_target_low_legality_scratch_arena(context), analysis));
  *out_analysis = analysis;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_value_analysis_for_contract_query(
    const loom_target_contract_query_environment_t* environment,
    loom_amdgpu_source_value_analysis_t** out_analysis) {
  *out_analysis = NULL;
  if (environment->value_domain == NULL ||
      environment->target_state_allocator.fn == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_get_or_allocate_target_state(
      environment, &loom_amdgpu_source_value_analysis_state_key,
      sizeof(*analysis), (void**)&analysis));
  if (analysis == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_value_analysis_prepare(
      environment->module, environment->function, environment->fact_table,
      environment->value_domain, environment->descriptor_set,
      environment->arena, analysis));
  *out_analysis = analysis;
  return iree_ok_status();
}

static const loom_cfg_graph_t* loom_amdgpu_source_value_analysis_cfg_graph(
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_region_t* region) {
  if (analysis == NULL || !analysis->source_cfg_graph_initialized ||
      analysis->source_body != region || analysis->source_cfg_graph.malformed) {
    return NULL;
  }
  return &analysis->source_cfg_graph;
}

static loom_amdgpu_source_value_analysis_record_t*
loom_amdgpu_source_value_analysis_lookup(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  if (analysis == NULL || analysis->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(analysis->value_domain)) {
    return NULL;
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(analysis->value_domain,
                                          source_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= analysis->record_count) {
    return NULL;
  }
  return &analysis->records[value_ordinal];
}

static bool loom_amdgpu_source_value_analysis_cached_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit, bool* out_value) {
  *out_value = false;
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record == NULL || !iree_any_bit_set(record->known_bits, bit)) {
    return false;
  }
  *out_value = iree_any_bit_set(record->value_bits, bit);
  return true;
}

static void loom_amdgpu_source_value_analysis_record_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit, bool value) {
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record == NULL) {
    return;
  }
  record->known_bits |= bit;
  if (value) {
    record->value_bits |= bit;
  } else {
    record->value_bits &= (loom_amdgpu_source_value_analysis_bits_t)~bit;
  }
}

static bool loom_amdgpu_source_value_analysis_begin_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit) {
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record == NULL) {
    return true;
  }
  if (iree_any_bit_set(record->active_bits, bit)) {
    return false;
  }
  record->active_bits |= bit;
  return true;
}

static void loom_amdgpu_source_value_analysis_end_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit, bool value) {
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record != NULL) {
    record->active_bits &= (loom_amdgpu_source_value_analysis_bits_t)~bit;
  }
  loom_amdgpu_source_value_analysis_record_bit(analysis, source_value_id, bit,
                                               value);
}

static bool loom_amdgpu_source_value_analysis_cached_register_shape(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, bool* out_has_shape,
    loom_amdgpu_register_shape_t* out_shape) {
  *out_has_shape = false;
  *out_shape = (loom_amdgpu_register_shape_t){0};
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record == NULL ||
      !iree_any_bit_set(record->known_bits,
                        LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE)) {
    return false;
  }
  *out_shape = record->register_shape;
  *out_has_shape = iree_any_bit_set(
      record->value_bits, LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE);
  return true;
}

static void loom_amdgpu_source_value_analysis_record_register_shape(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_amdgpu_register_shape_t* shape,
    bool valid) {
  loom_amdgpu_source_value_analysis_record_t* record =
      loom_amdgpu_source_value_analysis_lookup(analysis, source_value_id);
  if (record == NULL) {
    return;
  }
  record->known_bits |= LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE;
  if (valid) {
    record->value_bits |= LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE;
    record->register_shape = *shape;
  } else {
    record->value_bits &=
        (loom_amdgpu_source_value_analysis_bits_t)~LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_REGISTER_SHAPE;
    record->register_shape = (loom_amdgpu_register_shape_t){0};
  }
}

static bool loom_amdgpu_source_memory_root_is_read_only(
    const loom_low_source_memory_access_plan_t* plan,
    const loom_view_region_table_t* view_regions) {
  if (view_regions == NULL ||
      plan->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    return false;
  }
  const loom_view_access_flags_t access_flags =
      loom_view_region_table_root_access_flags(view_regions,
                                               plan->root_value_id);
  return iree_all_bits_set(access_flags, LOOM_VIEW_ACCESS_READ) &&
         !iree_any_bit_set(access_flags, LOOM_VIEW_ACCESS_WRITE);
}

static bool loom_amdgpu_source_memory_terms_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_low_source_memory_access_plan_t* plan) {
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &plan->dynamic_terms[i];
    switch (term->source) {
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE:
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
        return true;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
        if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
                module, fact_table, view_regions, analysis, term->index)) {
          return true;
        }
        break;
    }
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
              module, fact_table, view_regions, analysis,
              term->stride_values[stride_ordinal])) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_source_memory_access_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* source_op,
    loom_type_t source_type) {
  if (!loom_memory_access_isa(loom_memory_access_cast(module, source_op))) {
    return false;
  }
  if (fact_table == NULL || view_regions == NULL) {
    return true;
  }
  if (!loom_amdgpu_type_is_32bit_memory_payload(source_type)) {
    return true;
  }
  loom_low_source_memory_access_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(view_regions, source_op, &plan,
                                                &diagnostic)) {
    return true;
  }
  if (plan.operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      (plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
       plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) ||
      iree_any_bit_set(
          plan.cache_policy.build_flags,
          LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
              LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL) ||
      !loom_amdgpu_source_memory_root_is_read_only(&plan, view_regions) ||
      loom_amdgpu_source_memory_terms_prefer_vgpr(
          module, fact_table, view_regions, analysis, &plan)) {
    return true;
  }
  return false;
}

static bool loom_amdgpu_vector_extract_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* source_op) {
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] == INT64_MIN) {
    return true;
  }
  return loom_amdgpu_analyzed_source_value_prefers_vgpr(
      module, fact_table, view_regions, analysis,
      loom_vector_extract_source(source_op));
}

static bool loom_amdgpu_source_value_naturally_prefers_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_scalar_type_naturally_prefers_vgpr(source_type)) {
    return true;
  }
  loom_amdgpu_vector_storage_t vector_storage = {0};
  return loom_amdgpu_type_vector_storage(source_type, &vector_storage) &&
         !iree_any_bit_set(
             loom_amdgpu_vector_storage_kind_flags(vector_storage.kind),
             LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK);
}

static bool loom_amdgpu_source_value_facts_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_i1(source_type)) {
    return false;
  }
  return loom_value_facts_is_lane_varying(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_known_distribution_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_value_facts_t* out_facts) {
  if (out_facts != NULL) {
    *out_facts = loom_value_facts_unknown();
  }
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_value_fact_table_has_entry(fact_table, source_value_id)) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if ((facts.flags & LOOM_VALUE_FACT_DISTRIBUTION_MASK) == 0) {
    return false;
  }
  if (out_facts != NULL) {
    *out_facts = facts;
  }
  return true;
}

typedef uint32_t loom_amdgpu_source_producer_flags_t;

enum loom_amdgpu_source_producer_flag_bits_e {
  LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION = 1u << 0,
  LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR = 1u << 1,
  LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR = 1u << 2,
  LOOM_AMDGPU_SOURCE_PRODUCER_INDEX_CAST = 1u << 3,
  LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_EXTRACT = 1u << 4,
  LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_REDUCE = 1u << 5,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCF_SELECT = 1u << 6,
  LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE = 1u << 7,
  LOOM_AMDGPU_SOURCE_PRODUCER_MEMORY_ACCESS = 1u << 8,
  LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT = 1u << 9,
  LOOM_AMDGPU_SOURCE_PRODUCER_INT_CONVERSION_RESULT = 1u << 10,
  LOOM_AMDGPU_SOURCE_PRODUCER_FOLLOWS_OPERAND = 1u << 11,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_BITCAST = 1u << 12,
  LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE = 1u << 13,
  LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE_SUPPORTS_SGPR_BOOL = 1u << 14,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CMPI = 1u << 15,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_LOGICAL_BINARY = 1u << 16,
  LOOM_AMDGPU_SOURCE_PRODUCER_RESULT1_NATIVE_I1_MASK = 1u << 17,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CONSTANT = 1u << 18,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC = 1u << 19,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION = 1u << 20,
  LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_COMPARE = 1u << 21,
};

#define LOOM_AMDGPU_OP_INDEX(kind_) ((kind_) & 0xFF)

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuKernelSourceProducerFlags[LOOM_OP_KERNEL_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_ID)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ID)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_LANE_ID)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_REDUCE)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SCAN)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_REDUCE)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_SCAN)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SHUFFLE)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR |
            LOOM_AMDGPU_SOURCE_PRODUCER_RESULT1_NATIVE_I1_MASK,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuKernelSourceProducerFlags) ==
                  LOOM_OP_KERNEL_COUNT_,
              "AMDGPU kernel source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuIndexSourceProducerFlags[LOOM_OP_INDEX_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CAST)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_INDEX_CAST,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SUB)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_MUL)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SHLI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CMP)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE |
            LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE_SUPPORTS_SGPR_BOOL,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuIndexSourceProducerFlags) ==
                  LOOM_OP_INDEX_COUNT_,
              "AMDGPU index source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuScfSourceProducerFlags[LOOM_OP_SCF_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCF_SELECT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCF_SELECT,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuScfSourceProducerFlags) ==
                  LOOM_OP_SCF_COUNT_,
              "AMDGPU scf source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuBufferSourceProducerFlags[LOOM_OP_BUFFER_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_BUFFER_LOAD_I8_U)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_MEMORY_ACCESS,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuBufferSourceProducerFlags) ==
                  LOOM_OP_BUFFER_COUNT_,
              "AMDGPU buffer source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuVectorSourceProducerFlags[LOOM_OP_VECTOR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CONSTANT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FROM_ELEMENTS)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SELECT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SPLAT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_EXTRACT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_EXTRACT,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_LOAD)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_MEMORY_ACCESS,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_MMA)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_REDUCE)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_REDUCE,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuVectorSourceProducerFlags) ==
                  LOOM_OP_VECTOR_COUNT_,
              "AMDGPU vector source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuViewSourceProducerFlags[LOOM_OP_VIEW_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_LOAD)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_MEMORY_ACCESS,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_RMW)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_CMPXCHG)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuViewSourceProducerFlags) ==
                  LOOM_OP_VIEW_COUNT_,
              "AMDGPU view source producer table out of sync");

static const loom_amdgpu_source_producer_flags_t
    kAmdgpuScalarSourceProducerFlags[LOOM_OP_SCALAR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CONSTANT)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CONSTANT,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ADDF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SUBF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MULF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MINNUMF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MAXNUMF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CEILF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_FLOORF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ROUNDEVENF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_TRUNCF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SITOFP)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_UITOFP)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_FPTOSI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_INT_CONVERSION_RESULT |
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_FPTOUI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_INT_CONVERSION_RESULT |
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_EXTF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_FPTRUNC)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_EXTSI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_FOLLOWS_OPERAND,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_EXTUI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_FOLLOWS_OPERAND,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_BITCAST)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_BITCAST,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CMPI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE |
            LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE_SUPPORTS_SGPR_BOOL |
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CMPI,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CMPF)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE |
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_COMPARE,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ANDI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_LOGICAL_BINARY,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ORI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_LOGICAL_BINARY,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_XORI)] =
            LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_LOGICAL_BINARY,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuScalarSourceProducerFlags) ==
                  LOOM_OP_SCALAR_COUNT_,
              "AMDGPU scalar source producer table out of sync");

#undef LOOM_AMDGPU_OP_INDEX

typedef struct loom_amdgpu_source_producer_dialect_table_t {
  // Borrowed op-indexed flag rows for one source dialect.
  const loom_amdgpu_source_producer_flags_t* entries;
  // Number of op-indexed entries in the flag row table.
  iree_host_size_t count;
} loom_amdgpu_source_producer_dialect_table_t;

#define LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(table_) \
  {.entries = (table_), .count = IREE_ARRAYSIZE(table_)}

static const loom_amdgpu_source_producer_dialect_table_t
    kAmdgpuSourceProducerDialectTables[LOOM_DIALECT_BUILTIN_COUNT_] = {
        [LOOM_DIALECT_KERNEL] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuKernelSourceProducerFlags),
        [LOOM_DIALECT_INDEX] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuIndexSourceProducerFlags),
        [LOOM_DIALECT_SCF] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuScfSourceProducerFlags),
        [LOOM_DIALECT_BUFFER] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuBufferSourceProducerFlags),
        [LOOM_DIALECT_VECTOR] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuVectorSourceProducerFlags),
        [LOOM_DIALECT_VIEW] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuViewSourceProducerFlags),
        [LOOM_DIALECT_SCALAR] = LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE(
            kAmdgpuScalarSourceProducerFlags),
};
static_assert(IREE_ARRAYSIZE(kAmdgpuSourceProducerDialectTables) ==
                  LOOM_DIALECT_BUILTIN_COUNT_,
              "AMDGPU source producer dialect table out of sync");

#undef LOOM_AMDGPU_SOURCE_PRODUCER_DIALECT_TABLE

static loom_amdgpu_source_producer_flags_t loom_amdgpu_source_producer_flags(
    loom_op_kind_t kind) {
  const uint8_t dialect_id = loom_op_dialect_id(kind);
  if (dialect_id >= IREE_ARRAYSIZE(kAmdgpuSourceProducerDialectTables)) {
    return 0;
  }
  const loom_amdgpu_source_producer_dialect_table_t* table =
      &kAmdgpuSourceProducerDialectTables[dialect_id];
  const uint8_t dialect_index = loom_op_dialect_index(kind);
  return table->entries != NULL && dialect_index < table->count
             ? table->entries[dialect_index]
             : 0;
}

static bool loom_amdgpu_source_scalar_float_conversion_is_native(
    const loom_module_t* module, const loom_op_t* defining_op,
    loom_type_t result_type) {
  if (defining_op->operand_count != 1) {
    return false;
  }
  const loom_type_t input_type =
      loom_module_value_type(module, loom_op_const_operands(defining_op)[0]);
  switch (defining_op->kind) {
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
      return loom_amdgpu_type_is_i32(input_type) &&
             loom_amdgpu_type_is_f32(result_type);
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
      return loom_amdgpu_type_is_f32(input_type) &&
             loom_amdgpu_type_is_i32(result_type);
    case LOOM_OP_SCALAR_EXTF:
      return loom_type_is_scalar(input_type) &&
             loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_F16 &&
             loom_amdgpu_type_is_f32(result_type);
    case LOOM_OP_SCALAR_FPTRUNC:
      return loom_amdgpu_type_is_f32(input_type) &&
             loom_type_is_scalar(result_type) &&
             loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_F16;
    default:
      return false;
  }
}

static bool loom_amdgpu_source_scalar_float_result_follows_operands(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* defining_op,
    loom_amdgpu_source_producer_flags_t producer_flags) {
  if (analysis == NULL ||
      loom_value_def_index(loom_module_value(module, source_value_id)) != 0) {
    return false;
  }
  const loom_type_t result_type =
      loom_module_value_type(module, source_value_id);
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_ARITHMETIC)) {
    return iree_all_bits_set(
               analysis->descriptor_set_info_flags,
               LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC) &&
           (loom_amdgpu_type_is_f32(result_type) ||
            (loom_type_is_scalar(result_type) &&
             loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_F16));
  }
  return iree_any_bit_set(
             producer_flags,
             LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_CONVERSION) &&
         iree_all_bits_set(
             analysis->descriptor_set_info_flags,
             LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION) &&
         loom_amdgpu_source_scalar_float_conversion_is_native(
             module, defining_op, result_type);
}

static bool loom_amdgpu_workitem_dimension_is_valid(
    const loom_op_t* defining_op) {
  return iree_any_bit_set(loom_amdgpu_source_producer_flags(defining_op->kind),
                          LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION) &&
         defining_op->attribute_count != 0 &&
         loom_attr_as_enum(loom_op_attrs(defining_op)[0]) <
             LOOM_KERNEL_DIMENSION_COUNT_;
}

static bool loom_amdgpu_distribution_transfer_result_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  loom_value_facts_t distribution_facts = loom_value_facts_unknown();
  if (loom_amdgpu_source_value_known_distribution_facts(
          module, fact_table, source_value_id, &distribution_facts)) {
    if (loom_value_facts_is_lane_varying(distribution_facts)) {
      return true;
    }
  }

  const loom_value_id_t* operands = loom_op_const_operands(defining_op);
  for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
    const loom_value_id_t operand = operands[i];
    if (loom_amdgpu_analyzed_source_value_is_native_i1_mask(
            module, fact_table, view_regions, analysis, operand) ||
        loom_amdgpu_analyzed_source_value_prefers_vgpr(
            module, fact_table, view_regions, analysis, operand)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_producer_result_requires_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_amdgpu_source_producer_flags_t producer_flags) {
  if (!iree_any_bit_set(producer_flags,
                        LOOM_AMDGPU_SOURCE_PRODUCER_INT_CONVERSION_RESULT)) {
    return false;
  }
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0) {
    return false;
  }
  const loom_type_t value_type =
      loom_module_value_type(module, source_value_id);
  return loom_amdgpu_type_is_i8(value_type) ||
         loom_amdgpu_type_is_i16(value_type) ||
         loom_amdgpu_type_is_i32(value_type);
}

static bool loom_amdgpu_source_producer_result_follows_operand_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    const loom_op_t* defining_op,
    loom_amdgpu_source_producer_flags_t producer_flags,
    loom_value_id_t* out_operand) {
  *out_operand = LOOM_VALUE_ID_INVALID;
  if (!iree_any_bit_set(producer_flags,
                        LOOM_AMDGPU_SOURCE_PRODUCER_FOLLOWS_OPERAND)) {
    return false;
  }
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0 ||
      defining_op->operand_count != 1) {
    return false;
  }
  *out_operand = loom_op_const_operands(defining_op)[0];
  return *out_operand != source_value_id;
}

static bool
loom_amdgpu_distribution_transfer_binary_result_follows_operand_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    const loom_op_t* defining_op, loom_value_id_t* out_lhs,
    loom_value_id_t* out_rhs) {
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0 ||
      defining_op->operand_count != 2 || defining_op->result_count != 1 ||
      !loom_traits_have_distribution_transfer(
          loom_op_effective_traits(module, defining_op)) ||
      (!loom_amdgpu_type_is_i32(
           loom_module_value_type(module, source_value_id)) &&
       !loom_amdgpu_type_is_i64(
           loom_module_value_type(module, source_value_id)))) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(defining_op);
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  return true;
}

static bool loom_amdgpu_source_value_facts_are_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_value_facts_is_lane_predicate(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_facts_are_uniform_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_value_facts_is_subgroup_uniform(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_value_facts_t* out_facts) {
  if (out_facts != NULL) {
    *out_facts = loom_value_facts_unknown();
  }
  if (fact_table == NULL || source_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_amdgpu_type_is_i32(source_type) &&
      !loom_amdgpu_type_is_i64(source_type)) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if (!loom_value_facts_is_subgroup_lane_mask(facts)) {
    return false;
  }
  if (out_facts != NULL) {
    *out_facts = facts;
  }
  return true;
}

static bool loom_amdgpu_block_branches_to_with_arg(
    const loom_block_t* source_block, const loom_block_t* target_block,
    uint16_t target_arg_index, loom_value_id_t* out_arg) {
  *out_arg = LOOM_VALUE_ID_INVALID;
  if (source_block == NULL || source_block->op_count == 0) {
    return false;
  }
  const loom_op_t* terminator = loom_block_const_last_op(source_block);
  const loom_value_id_t* args = NULL;
  uint16_t arg_count = 0;
  if (!loom_cfg_terminator_payload_for_successor(terminator, target_block,
                                                 &args, &arg_count)) {
    return false;
  }
  if (target_arg_index >= arg_count) {
    return false;
  }
  *out_arg = args[target_arg_index];
  return true;
}

static bool loom_amdgpu_source_value_fact_identity_operand(
    const loom_value_t* value, const loom_op_t* defining_op,
    loom_trait_flags_t defining_op_traits, loom_value_id_t source_value_id,
    loom_value_id_t* out_operand) {
  *out_operand = LOOM_VALUE_ID_INVALID;
  if (!loom_traits_are_fact_identity(defining_op_traits)) {
    return false;
  }
  const uint16_t result_index = loom_value_def_index(value);
  if (result_index < defining_op->operand_count) {
    const loom_value_id_t operand =
        loom_op_const_operands(defining_op)[result_index];
    if (operand != source_value_id) {
      *out_operand = operand;
    }
  }
  return true;
}

static bool loom_amdgpu_fact_identity_use_result(
    const loom_module_t* module, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  if (user_op == NULL || operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != source_value_id ||
      operand_index >= user_op->result_count ||
      !loom_traits_are_fact_identity(
          loom_op_effective_traits(module, user_op))) {
    return false;
  }
  *out_result = loom_op_const_results(user_op)[operand_index];
  return *out_result != source_value_id;
}

static bool loom_amdgpu_source_value_directly_prefers_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t excluded_value_id) {
  while (true) {
    if (source_value_id == excluded_value_id ||
        source_value_id >= module->values.count) {
      return false;
    }
    if (loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                        source_value_id)) {
      return true;
    }
    const loom_value_t* value = loom_module_value(module, source_value_id);
    if (loom_value_is_block_arg(value)) {
      return false;
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) {
      return false;
    }

    const loom_trait_flags_t defining_op_traits =
        loom_op_effective_traits(module, defining_op);
    loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_source_value_fact_identity_operand(
            value, defining_op, defining_op_traits, source_value_id,
            &identity_operand)) {
      if (identity_operand == LOOM_VALUE_ID_INVALID) {
        return false;
      }
      source_value_id = identity_operand;
      continue;
    }

    const loom_amdgpu_source_producer_flags_t producer_flags =
        loom_amdgpu_source_producer_flags(defining_op->kind);
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION |
                             LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR)) {
      return true;
    }
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR)) {
      return loom_value_def_index(value) == 0;
    }
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_INDEX_CAST)) {
      const loom_value_id_t next_value_id = loom_index_cast_input(defining_op);
      if (next_value_id == source_value_id) {
        return false;
      }
      source_value_id = next_value_id;
      continue;
    }
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_EXTRACT)) {
      return loom_amdgpu_source_value_naturally_prefers_vgpr(
          module, loom_vector_extract_source(defining_op));
    }
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_REDUCE)) {
      return loom_amdgpu_vector_32bit_register_count(loom_module_value_type(
                 module, loom_vector_reduce_input(defining_op))) != 0;
    }
    if (iree_any_bit_set(producer_flags,
                         LOOM_AMDGPU_SOURCE_PRODUCER_SCF_SELECT)) {
      return loom_value_def_index(value) == 0 &&
             (loom_amdgpu_source_value_directly_prefers_vgpr(
                  module, loom_scf_select_true_value(defining_op),
                  excluded_value_id) ||
              loom_amdgpu_source_value_directly_prefers_vgpr(
                  module, loom_scf_select_false_value(defining_op),
                  excluded_value_id));
    }

    loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_distribution_transfer_binary_result_follows_operand_vgpr(
            module, source_value_id, defining_op, &lhs, &rhs)) {
      return loom_amdgpu_source_value_directly_prefers_vgpr(module, lhs,
                                                            source_value_id) ||
             loom_amdgpu_source_value_directly_prefers_vgpr(module, rhs,
                                                            source_value_id);
    }
    if (loom_amdgpu_source_producer_result_requires_vgpr(
            module, source_value_id, producer_flags)) {
      return true;
    }
    loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
    return loom_amdgpu_source_producer_result_follows_operand_vgpr(
               module, source_value_id, defining_op, producer_flags,
               &operand) &&
           loom_amdgpu_source_value_directly_prefers_vgpr(module, operand,
                                                          source_value_id);
  }
}

static bool loom_amdgpu_select_payload_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_value_id_t condition_value_id) {
  if (source_value_id == condition_value_id ||
      source_value_id >= module->values.count) {
    return false;
  }
  if (loom_amdgpu_scalar_type_has_fixed_vgpr_storage(
          loom_module_value_type(module, source_value_id))) {
    return true;
  }
  // Statusless module-local queries have no active-bit table. Keep their
  // select inspection bounded because condition-mask and result placement
  // query each other.
  if (analysis == NULL) {
    return loom_amdgpu_source_value_facts_prefer_vgpr(module, fact_table,
                                                      source_value_id) ||
           loom_amdgpu_source_value_directly_prefers_vgpr(
               module, source_value_id, condition_value_id);
  }
  return loom_amdgpu_analyzed_source_value_prefers_vgpr(
      module, fact_table, view_regions, analysis, source_value_id);
}

static bool loom_amdgpu_op_results_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* op,
    loom_value_id_t excluded_value_id) {
  if (op == NULL) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (loom_amdgpu_select_payload_prefers_vgpr(
            module, fact_table, view_regions, analysis, results[i],
            excluded_value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_op_operands_with_role_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* op,
    loom_operand_role_t role, loom_value_id_t excluded_value_id) {
  if (op == NULL) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (vtable == NULL || !iree_any_bit_set(vtable->operand_role_mask,
                                          loom_operand_role_mask_bit(role))) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_op_operand_role_at(vtable, op, i) != role) {
      continue;
    }
    if (loom_amdgpu_select_payload_prefers_vgpr(
            module, fact_table, view_regions, analysis, operands[i],
            excluded_value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_select_condition_use_needs_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id) {
  if (operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != source_value_id ||
      !loom_op_operand_has_role(module, user_op, operand_index,
                                LOOM_OPERAND_ROLE_SELECT_CONDITION)) {
    return false;
  }
  return loom_amdgpu_op_results_prefer_vgpr(module, fact_table, view_regions,
                                            analysis, user_op,
                                            source_value_id) ||
         loom_amdgpu_op_operands_with_role_prefer_vgpr(
             module, fact_table, view_regions, analysis, user_op,
             LOOM_OPERAND_ROLE_SELECT_PAYLOAD, source_value_id);
}

static bool loom_amdgpu_vector_i1_constructor_use_needs_native_mask(
    const loom_module_t* module, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id) {
  if (operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != source_value_id) {
    return false;
  }
  const loom_operand_role_t role =
      loom_op_operand_role(module, user_op, operand_index);
  if (role != LOOM_OPERAND_ROLE_BROADCAST_SOURCE &&
      role != LOOM_OPERAND_ROLE_COMPOSITE_ELEMENT) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(user_op);
  for (uint16_t i = 0; i < user_op->result_count; ++i) {
    if (loom_amdgpu_vector_i1_lane_count(
            loom_module_value_type(module, results[i])) != 0) {
      return true;
    }
  }
  return false;
}

typedef uint8_t loom_amdgpu_i1_compare_flags_t;

enum loom_amdgpu_i1_compare_flag_bits_e {
  LOOM_AMDGPU_I1_COMPARE_SUPPORTS_SGPR_BOOL = 1u << 0,
  LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK = 1u << 1,
};

typedef struct loom_amdgpu_i1_compare_values_t {
  // First source operand of the comparison.
  loom_value_id_t lhs;
  // Second source operand of the comparison.
  loom_value_id_t rhs;
  // Placement constraints implied by the comparison operation.
  loom_amdgpu_i1_compare_flags_t flags;
} loom_amdgpu_i1_compare_values_t;

static bool loom_amdgpu_scalar_cmpi_i64_requires_native_mask(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_amdgpu_i1_compare_values_t* values) {
  if (values->lhs >= module->values.count ||
      values->rhs >= module->values.count ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, values->lhs)) ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, values->rhs)) ||
      source_op->attribute_count == 0) {
    return false;
  }
  const uint8_t predicate = loom_attr_as_enum(loom_op_attrs(source_op)[0]);
  return predicate != LOOM_SCALAR_CMPI_PREDICATE_EQ &&
         predicate != LOOM_SCALAR_CMPI_PREDICATE_NE;
}

static bool loom_amdgpu_i1_compare_values(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_op_t* source_op, loom_amdgpu_i1_compare_values_t* out_values) {
  out_values->lhs = LOOM_VALUE_ID_INVALID;
  out_values->rhs = LOOM_VALUE_ID_INVALID;
  out_values->flags = 0;
  const loom_amdgpu_source_producer_flags_t producer_flags =
      loom_amdgpu_source_producer_flags(source_op->kind);
  if (!iree_any_bit_set(producer_flags,
                        LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE) ||
      source_op->operand_count != 2 || source_op->result_count != 1) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  out_values->lhs = operands[0];
  out_values->rhs = operands[1];
  if (iree_any_bit_set(
          producer_flags,
          LOOM_AMDGPU_SOURCE_PRODUCER_I1_COMPARE_SUPPORTS_SGPR_BOOL)) {
    out_values->flags |= LOOM_AMDGPU_I1_COMPARE_SUPPORTS_SGPR_BOOL;
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_FLOAT_COMPARE)) {
    if (analysis != NULL &&
        iree_all_bits_set(
            analysis->descriptor_set_info_flags,
            LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE)) {
      out_values->flags |= LOOM_AMDGPU_I1_COMPARE_SUPPORTS_SGPR_BOOL;
    } else {
      out_values->flags |= LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK;
    }
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CMPI) &&
      loom_amdgpu_scalar_cmpi_i64_requires_native_mask(module, source_op,
                                                       out_values)) {
    out_values->flags |= LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK;
  }
  return true;
}

static bool loom_amdgpu_i1_compare_has_direct_vgpr_operand(
    const loom_module_t* module, const loom_amdgpu_i1_compare_values_t* values,
    loom_value_id_t excluded_value_id) {
  return loom_amdgpu_source_value_directly_prefers_vgpr(module, values->lhs,
                                                        excluded_value_id) ||
         loom_amdgpu_source_value_directly_prefers_vgpr(module, values->rhs,
                                                        excluded_value_id);
}

static bool loom_amdgpu_i1_compare_has_vgpr_operand(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    const loom_amdgpu_i1_compare_values_t* values) {
  return loom_amdgpu_analyzed_source_value_prefers_vgpr(
             module, fact_table, view_regions, analysis, values->lhs) ||
         loom_amdgpu_analyzed_source_value_prefers_vgpr(
             module, fact_table, view_regions, analysis, values->rhs);
}

static bool loom_amdgpu_source_value_is_direct_native_i1_mask_except(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  loom_amdgpu_i1_compare_values_t compare;
  if (loom_value_def_index(value) == 0 &&
      loom_amdgpu_i1_compare_values(module, /*analysis=*/NULL, defining_op,
                                    &compare)) {
    return iree_any_bit_set(compare.flags,
                            LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK) ||
           loom_amdgpu_i1_compare_has_direct_vgpr_operand(module, &compare,
                                                          excluded_value_id);
  }

  return iree_any_bit_set(loom_amdgpu_source_producer_flags(defining_op->kind),
                          LOOM_AMDGPU_SOURCE_PRODUCER_RESULT1_NATIVE_I1_MASK) &&
         loom_value_def_index(value) == 1;
}

static bool loom_amdgpu_cond_br_targets_block(const loom_op_t* terminator,
                                              const loom_block_t* block) {
  return loom_cfg_cond_br_isa(terminator) &&
         (loom_cfg_cond_br_true_dest(terminator) == block ||
          loom_cfg_cond_br_false_dest(terminator) == block);
}

static bool loom_amdgpu_block_arg_merges_native_mask_diamond_from_graph(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_block_t* block, uint16_t arg_index,
    loom_value_id_t excluded_value_id) {
  const iree_host_size_t block_index = loom_cfg_graph_block_index(graph, block);
  if (block_index == IREE_HOST_SIZE_MAX) {
    return false;
  }
  const loom_cfg_edge_index_span_t incoming_edges =
      loom_cfg_graph_predecessor_edges(graph, (uint16_t)block_index);
  for (iree_host_size_t incoming_ordinal = 0;
       incoming_ordinal < incoming_edges.count; ++incoming_ordinal) {
    const loom_cfg_edge_info_t* incoming_edge =
        loom_cfg_graph_edge(graph, incoming_edges.values[incoming_ordinal]);
    if (incoming_edge == NULL) {
      continue;
    }
    const loom_op_t* incoming_terminator = incoming_edge->terminator;
    const loom_value_id_t* incoming_args = NULL;
    uint16_t incoming_arg_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(
            incoming_terminator, block, &incoming_args, &incoming_arg_count) ||
        arg_index >= incoming_arg_count) {
      continue;
    }

    const loom_cfg_edge_index_span_t arm_predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph,
                                         incoming_edge->source_block_index);
    for (iree_host_size_t arm_ordinal = 0;
         arm_ordinal < arm_predecessor_edges.count; ++arm_ordinal) {
      const loom_cfg_edge_info_t* arm_edge =
          loom_cfg_graph_edge(graph, arm_predecessor_edges.values[arm_ordinal]);
      if (arm_edge == NULL) {
        continue;
      }
      const loom_op_t* guard_terminator = arm_edge->terminator;
      if (!loom_amdgpu_cond_br_targets_block(
              guard_terminator, incoming_terminator->parent_block) ||
          !loom_amdgpu_source_value_is_direct_native_i1_mask_except(
              module, loom_cfg_cond_br_condition(guard_terminator),
              excluded_value_id)) {
        continue;
      }

      loom_value_id_t true_arg = LOOM_VALUE_ID_INVALID;
      loom_value_id_t false_arg = LOOM_VALUE_ID_INVALID;
      if (loom_amdgpu_block_branches_to_with_arg(
              loom_cfg_cond_br_true_dest(guard_terminator), block, arg_index,
              &true_arg) &&
          loom_amdgpu_block_branches_to_with_arg(
              loom_cfg_cond_br_false_dest(guard_terminator), block, arg_index,
              &false_arg)) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_block_arg_merges_native_mask_diamond(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_block_t* block, uint16_t arg_index,
    loom_value_id_t excluded_value_id) {
  const loom_region_t* region = block->parent_region;
  if (region == NULL || block == loom_region_const_entry_block(region)) {
    return false;
  }

  const loom_cfg_graph_t* graph =
      loom_amdgpu_source_value_analysis_cfg_graph(analysis, region);
  if (graph != NULL) {
    return loom_amdgpu_block_arg_merges_native_mask_diamond_from_graph(
        module, graph, block, arg_index, excluded_value_id);
  }

  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* guard = loom_region_const_block(region, block_index);
    if (guard == NULL || guard->op_count == 0) {
      continue;
    }
    const loom_op_t* terminator = loom_block_const_last_op(guard);
    if (!loom_cfg_cond_br_isa(terminator) ||
        !loom_amdgpu_source_value_is_direct_native_i1_mask_except(
            module, loom_cfg_cond_br_condition(terminator),
            excluded_value_id)) {
      continue;
    }

    loom_value_id_t true_arg = LOOM_VALUE_ID_INVALID;
    loom_value_id_t false_arg = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_block_branches_to_with_arg(
            loom_cfg_cond_br_true_dest(terminator), block, arg_index,
            &true_arg) &&
        loom_amdgpu_block_branches_to_with_arg(
            loom_cfg_cond_br_false_dest(terminator), block, arg_index,
            &false_arg)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_value_feeds_native_mask_merge_arg(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_value_t* value, loom_value_id_t value_id) {
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    const uint16_t operand_index = loom_use_operand_index(uses[i]);
    if (user_op == NULL || user_op->successor_count != 1) {
      continue;
    }
    loom_block_t* const* successors = loom_op_const_successors(user_op);
    const loom_block_t* dest = successors[0];
    const loom_value_id_t* args = NULL;
    uint16_t arg_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(user_op, dest, &args,
                                                   &arg_count) ||
        operand_index >= arg_count || args[operand_index] != value_id) {
      continue;
    }
    if (loom_amdgpu_block_arg_merges_native_mask_diamond(
            module, analysis, dest, operand_index, value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_block_arg_has_cfg_predecessor(
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_block_t* block, uint16_t arg_index) {
  const loom_region_t* region = block != NULL ? block->parent_region : NULL;
  if (region == NULL || arg_index >= block->arg_count) {
    return false;
  }
  const loom_cfg_graph_t* graph =
      loom_amdgpu_source_value_analysis_cfg_graph(analysis, region);
  if (graph != NULL) {
    const iree_host_size_t block_index =
        loom_cfg_graph_block_index(graph, block);
    if (block_index == IREE_HOST_SIZE_MAX) {
      return false;
    }
    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, predecessor_edges.values[i]);
      const loom_value_id_t* edge_args = NULL;
      uint16_t edge_arg_count = 0;
      if (edge != NULL &&
          loom_cfg_terminator_payload_for_successor(
              edge->terminator, block, &edge_args, &edge_arg_count) &&
          arg_index < edge_arg_count) {
        return true;
      }
    }
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* predecessor =
        loom_region_const_block(region, block_index);
    if (predecessor == NULL || predecessor->op_count == 0) {
      continue;
    }
    const loom_op_t* terminator = loom_block_const_last_op(predecessor);
    const loom_value_id_t* args = NULL;
    uint16_t arg_count = 0;
    if (loom_cfg_terminator_payload_for_successor(terminator, block, &args,
                                                  &arg_count) &&
        arg_index < arg_count) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_branch_arg_payload_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* terminator,
    const loom_block_t* dest_block, uint16_t arg_index,
    loom_value_id_t excluded_value_id) {
  const loom_value_id_t* args = NULL;
  uint16_t arg_count = 0;
  if (!loom_cfg_terminator_payload_for_successor(terminator, dest_block, &args,
                                                 &arg_count) ||
      arg_index >= arg_count) {
    return false;
  }
  const loom_value_id_t incoming_value_id = args[arg_index];
  if (incoming_value_id == excluded_value_id) {
    return false;
  }
  if (loom_amdgpu_source_value_facts_prefer_vgpr(module, fact_table,
                                                 incoming_value_id) ||
      loom_amdgpu_source_value_directly_prefers_vgpr(module, incoming_value_id,
                                                     excluded_value_id)) {
    return true;
  }
  const loom_value_t* incoming_value =
      loom_module_value(module, incoming_value_id);
  if (loom_value_is_block_arg(incoming_value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(incoming_value);
  if (defining_op == NULL || loom_value_def_index(incoming_value) != 0) {
    return false;
  }
  return loom_amdgpu_source_memory_access_prefers_vgpr(
      module, fact_table, view_regions, analysis, defining_op,
      loom_module_value_type(module, incoming_value_id));
}

static bool loom_amdgpu_block_arg_incoming_payload_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_block_t* block,
    uint16_t arg_index, loom_value_id_t excluded_value_id) {
  const loom_region_t* region = block != NULL ? block->parent_region : NULL;
  if (region == NULL || arg_index >= block->arg_count) {
    return false;
  }
  const loom_cfg_graph_t* graph =
      loom_amdgpu_source_value_analysis_cfg_graph(analysis, region);
  if (graph != NULL) {
    const iree_host_size_t block_index =
        loom_cfg_graph_block_index(graph, block);
    if (block_index == IREE_HOST_SIZE_MAX) {
      return false;
    }
    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, (uint16_t)block_index);
    for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
      const loom_cfg_edge_info_t* edge =
          loom_cfg_graph_edge(graph, predecessor_edges.values[i]);
      if (edge == NULL) {
        continue;
      }
      if (loom_amdgpu_branch_arg_payload_prefers_vgpr(
              module, fact_table, view_regions, analysis, edge->terminator,
              block, arg_index, excluded_value_id)) {
        return true;
      }
    }
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* predecessor =
        loom_region_const_block(region, block_index);
    if (predecessor == NULL || predecessor->op_count == 0) {
      continue;
    }
    const loom_op_t* terminator = loom_block_const_last_op(predecessor);
    if (loom_amdgpu_branch_arg_payload_prefers_vgpr(
            module, fact_table, view_regions, analysis, terminator, block,
            arg_index, excluded_value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_i1_use_is_control_condition(
    const loom_module_t* module, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id) {
  return operand_index < user_op->operand_count &&
         loom_op_const_operands(user_op)[operand_index] == source_value_id &&
         loom_op_operand_has_role(module, user_op, operand_index,
                                  LOOM_OPERAND_ROLE_CONTROL_CONDITION);
}

static bool loom_amdgpu_i1_use_is_condition_operand(
    const loom_module_t* module, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id) {
  if (operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != source_value_id) {
    return false;
  }
  const loom_operand_role_t role =
      loom_op_operand_role(module, user_op, operand_index);
  return role == LOOM_OPERAND_ROLE_CONTROL_CONDITION ||
         role == LOOM_OPERAND_ROLE_SELECT_CONDITION;
}

static bool loom_amdgpu_source_i1_value_has_cross_block_use(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || defining_op->parent_block == NULL) {
    return false;
  }

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op != NULL && user_op->parent_block != defining_op->parent_block) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_scalar_logical_binary_values(
    const loom_op_t* op, loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_result) {
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (!iree_any_bit_set(loom_amdgpu_source_producer_flags(op->kind),
                        LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_LOGICAL_BINARY) ||
      op->operand_count != 2 || op->result_count != 1) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_result = results[0];
  return true;
}

static bool loom_amdgpu_source_value_can_lower_as_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  const loom_value_id_t next_excluded_value_id =
      excluded_value_id == LOOM_VALUE_ID_INVALID ? source_value_id
                                                 : excluded_value_id;
  if (loom_amdgpu_source_value_facts_are_native_i1_mask(module, fact_table,
                                                        source_value_id)) {
    return false;
  }
  if (loom_amdgpu_source_value_facts_are_uniform_i1(module, fact_table,
                                                    source_value_id)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    return !loom_amdgpu_block_arg_has_cfg_predecessor(
        analysis, block, loom_value_def_index(value));
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return true;
  }

  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);
  loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_source_value_fact_identity_operand(
          value, defining_op, defining_op_traits, source_value_id,
          &identity_operand)) {
    return identity_operand != LOOM_VALUE_ID_INVALID &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions, analysis, identity_operand,
               next_excluded_value_id);
  }

  if (loom_value_def_index(value) != 0) {
    return false;
  }

  const loom_amdgpu_source_producer_flags_t producer_flags =
      loom_amdgpu_source_producer_flags(defining_op->kind);
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_CONSTANT)) {
    return true;
  }

  loom_amdgpu_i1_compare_values_t compare;
  if (loom_amdgpu_i1_compare_values(module, analysis, defining_op, &compare)) {
    return !iree_any_bit_set(compare.flags,
                             LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK) &&
           !loom_amdgpu_i1_compare_has_vgpr_operand(
               module, fact_table, view_regions, analysis, &compare);
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_scalar_logical_binary_values(defining_op, &lhs, &rhs,
                                               &result) &&
      result == source_value_id) {
    return loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions, analysis, lhs,
               next_excluded_value_id) &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions, analysis, rhs,
               next_excluded_value_id);
  }

  return false;
}

static bool loom_amdgpu_source_value_can_lower_as_sgpr_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  const loom_value_id_t next_excluded_value_id =
      excluded_value_id == LOOM_VALUE_ID_INVALID ? source_value_id
                                                 : excluded_value_id;
  if (loom_amdgpu_source_value_facts_are_native_i1_mask(module, fact_table,
                                                        source_value_id)) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return false;
  }

  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);
  loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_source_value_fact_identity_operand(
          value, defining_op, defining_op_traits, source_value_id,
          &identity_operand)) {
    return identity_operand != LOOM_VALUE_ID_INVALID &&
           loom_amdgpu_source_value_can_lower_as_sgpr_i1_bool(
               module, fact_table, view_regions, analysis, identity_operand,
               next_excluded_value_id);
  }

  if (loom_value_def_index(value) != 0) {
    return false;
  }

  loom_amdgpu_i1_compare_values_t compare;
  if (loom_amdgpu_i1_compare_values(module, analysis, defining_op, &compare)) {
    return iree_any_bit_set(compare.flags,
                            LOOM_AMDGPU_I1_COMPARE_SUPPORTS_SGPR_BOOL) &&
           !iree_any_bit_set(compare.flags,
                             LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK) &&
           !loom_amdgpu_i1_compare_has_vgpr_operand(
               module, fact_table, view_regions, analysis, &compare);
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_scalar_logical_binary_values(defining_op, &lhs, &rhs,
                                               &result) &&
      result == source_value_id) {
    const bool lhs_is_scalar_bool =
        loom_amdgpu_source_value_can_lower_as_scc_i1(
            module, fact_table, view_regions, analysis, lhs,
            next_excluded_value_id) ||
        loom_amdgpu_source_value_can_lower_as_sgpr_i1_bool(
            module, fact_table, view_regions, analysis, lhs,
            next_excluded_value_id);
    const bool rhs_is_scalar_bool =
        loom_amdgpu_source_value_can_lower_as_scc_i1(
            module, fact_table, view_regions, analysis, rhs,
            next_excluded_value_id) ||
        loom_amdgpu_source_value_can_lower_as_sgpr_i1_bool(
            module, fact_table, view_regions, analysis, rhs,
            next_excluded_value_id);
    return lhs_is_scalar_bool && rhs_is_scalar_bool;
  }

  return false;
}

static bool loom_amdgpu_source_value_is_native_i1_mask_excluding(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  const loom_value_id_t next_excluded_value_id =
      excluded_value_id == LOOM_VALUE_ID_INVALID ? source_value_id
                                                 : excluded_value_id;
  if (loom_amdgpu_source_value_facts_are_native_i1_mask(module, fact_table,
                                                        source_value_id)) {
    return true;
  }

  const bool source_can_lower_as_scc =
      loom_amdgpu_source_value_can_lower_as_scc_i1(
          module, fact_table, view_regions, analysis, source_value_id,
          next_excluded_value_id);
  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (loom_amdgpu_select_condition_use_needs_native_i1_mask(
            module, fact_table, view_regions, analysis, user_op, operand_index,
            source_value_id) ||
        loom_amdgpu_vector_i1_constructor_use_needs_native_mask(
            module, user_op, operand_index, source_value_id)) {
      return true;
    }
    loom_value_id_t identity_result = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_fact_identity_use_result(module, user_op, operand_index,
                                             source_value_id,
                                             &identity_result) &&
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, analysis, identity_result,
            next_excluded_value_id)) {
      return true;
    }
    if (source_can_lower_as_scc) {
      continue;
    }
    if (user_op == NULL || user_op->successor_count != 1) {
      continue;
    }
    loom_block_t* const* successors = loom_op_const_successors(user_op);
    const loom_block_t* dest = successors[0];
    const loom_value_id_t* args = NULL;
    uint16_t arg_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(user_op, dest, &args,
                                                   &arg_count) ||
        operand_index >= arg_count || args[operand_index] != source_value_id) {
      continue;
    }
    const loom_value_id_t dest_arg = loom_block_arg_id(dest, operand_index);
    if (dest_arg != source_value_id &&
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, analysis, dest_arg,
            next_excluded_value_id)) {
      return true;
    }
  }
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);
  loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_source_value_fact_identity_operand(
          value, defining_op, defining_op_traits, source_value_id,
          &identity_operand)) {
    return identity_operand != LOOM_VALUE_ID_INVALID &&
           loom_amdgpu_source_value_is_native_i1_mask_excluding(
               module, fact_table, view_regions, analysis, identity_operand,
               next_excluded_value_id);
  }

  loom_amdgpu_i1_compare_values_t compare;
  if (loom_value_def_index(value) == 0 &&
      loom_amdgpu_i1_compare_values(module, analysis, defining_op, &compare)) {
    return iree_any_bit_set(compare.flags,
                            LOOM_AMDGPU_I1_COMPARE_REQUIRES_NATIVE_MASK) ||
           loom_amdgpu_i1_compare_has_vgpr_operand(
               module, fact_table, view_regions, analysis, &compare);
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (loom_value_def_index(value) == 0 &&
      loom_amdgpu_scalar_logical_binary_values(defining_op, &lhs, &rhs,
                                               &result) &&
      result == source_value_id) {
    const bool lhs_is_mask =
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, analysis, lhs,
            next_excluded_value_id);
    const bool rhs_is_mask =
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, analysis, rhs,
            next_excluded_value_id);
    if (!lhs_is_mask && !rhs_is_mask) {
      return false;
    }
    return (lhs_is_mask || loom_amdgpu_source_value_can_lower_as_scc_i1(
                               module, fact_table, view_regions, analysis, lhs,
                               next_excluded_value_id)) &&
           (rhs_is_mask || loom_amdgpu_source_value_can_lower_as_scc_i1(
                               module, fact_table, view_regions, analysis, rhs,
                               next_excluded_value_id));
  }

  return iree_any_bit_set(loom_amdgpu_source_producer_flags(defining_op->kind),
                          LOOM_AMDGPU_SOURCE_PRODUCER_RESULT1_NATIVE_I1_MASK) &&
         loom_value_def_index(value) == 1;
}

static bool loom_amdgpu_analyzed_source_value_can_lower_as_sgpr_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_amdgpu_source_value_analysis_bits_t bit =
      LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_SGPR_I1_BOOL;
  bool value = false;
  if (loom_amdgpu_source_value_analysis_cached_bit(analysis, source_value_id,
                                                   bit, &value)) {
    return value;
  }
  if (!loom_amdgpu_source_value_analysis_begin_bit(analysis, source_value_id,
                                                   bit)) {
    return false;
  }
  value = loom_amdgpu_source_value_can_lower_as_sgpr_i1_bool(
      module, fact_table, view_regions, analysis, source_value_id,
      LOOM_VALUE_ID_INVALID);
  loom_amdgpu_source_value_analysis_end_bit(analysis, source_value_id, bit,
                                            value);
  return value;
}

static bool loom_amdgpu_i1_use_needs_native_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* user_op,
    uint16_t operand_index, loom_value_id_t source_value_id) {
  if (loom_amdgpu_select_condition_use_needs_native_i1_mask(
          module, fact_table, view_regions, analysis, user_op, operand_index,
          source_value_id) ||
      loom_amdgpu_vector_i1_constructor_use_needs_native_mask(
          module, user_op, operand_index, source_value_id)) {
    return true;
  }

  loom_value_id_t identity_result = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_fact_identity_use_result(module, user_op, operand_index,
                                           source_value_id, &identity_result)) {
    return loom_amdgpu_analyzed_source_value_is_native_i1_mask(
        module, fact_table, view_regions, analysis, identity_result);
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t logical_result = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_scalar_logical_binary_values(user_op, &lhs, &rhs,
                                           &logical_result);
  return logical_result != LOOM_VALUE_ID_INVALID &&
         (lhs == source_value_id || rhs == source_value_id) &&
         loom_amdgpu_analyzed_source_value_is_native_i1_mask(
             module, fact_table, view_regions, analysis, logical_result);
}

static bool loom_amdgpu_source_i1_value_has_same_block_branch_and_mask_use(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  const loom_block_t* defining_block =
      defining_op != NULL ? defining_op->parent_block : NULL;
  if (defining_block == NULL) {
    return false;
  }

  bool has_branch_use = false;
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (user_op == NULL || user_op->parent_block != defining_block) {
      continue;
    }
    if (loom_amdgpu_i1_use_is_control_condition(
            module, user_op, loom_use_operand_index(*use), source_value_id)) {
      has_branch_use = true;
      break;
    }
  }
  if (!has_branch_use) {
    return false;
  }

  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (user_op == NULL || user_op->parent_block != defining_block) {
      continue;
    }
    if (loom_amdgpu_i1_use_needs_native_mask(
            module, fact_table, view_regions, analysis, user_op,
            loom_use_operand_index(*use), source_value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_value_is_later_same_block_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* defining_op,
    loom_value_id_t candidate_value_id) {
  if (candidate_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, candidate_value_id))) {
    return false;
  }
  const loom_value_t* candidate_value =
      loom_module_value(module, candidate_value_id);
  if (loom_value_is_block_arg(candidate_value)) {
    return false;
  }
  const loom_op_t* candidate_defining_op = loom_value_def_op(candidate_value);
  return candidate_defining_op != NULL &&
         candidate_defining_op->parent_block == defining_op->parent_block &&
         candidate_defining_op->block_ordinal > defining_op->block_ordinal &&
         loom_amdgpu_analyzed_source_value_can_lower_as_scc_i1(
             module, fact_table, view_regions, analysis, candidate_value_id);
}

// Returns true when an operand is a later scalar condition or was produced by
// a control operation with a later scalar condition. The latter recognizes
// nested selects without depending on a particular source dialect.
static bool loom_amdgpu_source_operand_has_later_scc_i1_condition(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* defining_op,
    loom_value_id_t operand_value_id) {
  if (loom_amdgpu_source_value_is_later_same_block_scc_i1(
          module, fact_table, view_regions, analysis, defining_op,
          operand_value_id)) {
    return true;
  }
  const loom_value_t* operand_value =
      loom_module_value(module, operand_value_id);
  if (loom_value_is_block_arg(operand_value)) {
    return false;
  }
  const loom_op_t* operand_defining_op = loom_value_def_op(operand_value);
  if (operand_defining_op == NULL ||
      operand_defining_op->parent_block != defining_op->parent_block) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(operand_defining_op);
  for (uint16_t i = 0; i < operand_defining_op->operand_count; ++i) {
    if (loom_amdgpu_i1_use_is_condition_operand(module, operand_defining_op, i,
                                                operands[i]) &&
        loom_amdgpu_source_value_is_later_same_block_scc_i1(
            module, fact_table, view_regions, analysis, defining_op,
            operands[i])) {
      return true;
    }
  }
  return false;
}

// SCC is ephemeral architectural state. A control condition that precedes a
// later condition needed by its payload cannot remain in SCC: low scheduling
// must materialize the payload before consuming the outer condition. Retaining
// the outer condition in an SGPR prevents source order from imposing a state
// dependency cycle.
static bool loom_amdgpu_source_i1_value_has_nested_control_dependency(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(source_value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(source_value);
  if (defining_op == NULL || defining_op->parent_block == NULL) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(source_value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t condition_operand_index = loom_use_operand_index(*use);
    if (user_op == NULL || user_op->parent_block != defining_op->parent_block ||
        !loom_amdgpu_i1_use_is_condition_operand(
            module, user_op, condition_operand_index, source_value_id)) {
      continue;
    }
    const loom_value_id_t* operands = loom_op_const_operands(user_op);
    for (uint16_t i = 0; i < user_op->operand_count; ++i) {
      if (i != condition_operand_index &&
          loom_amdgpu_source_operand_has_later_scc_i1_condition(
              module, fact_table, view_regions, analysis, defining_op,
              operands[i])) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_source_value_is_durable_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  return (loom_amdgpu_source_i1_value_has_cross_block_use(module,
                                                          source_value_id) ||
          loom_amdgpu_source_i1_value_has_same_block_branch_and_mask_use(
              module, fact_table, view_regions, analysis, source_value_id) ||
          loom_amdgpu_source_i1_value_has_nested_control_dependency(
              module, fact_table, view_regions, analysis, source_value_id)) &&
         loom_amdgpu_analyzed_source_value_can_lower_as_sgpr_i1_bool(
             module, fact_table, view_regions, analysis, source_value_id);
}

bool loom_amdgpu_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, /*analysis=*/NULL, source_value_id,
      LOOM_VALUE_ID_INVALID);
}

bool loom_amdgpu_source_value_is_uniform_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  return loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
             module, fact_table, source_value_id, &facts) &&
         loom_value_facts_is_subgroup_uniform(facts);
}

bool loom_amdgpu_source_value_is_divergent_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  return loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
             module, fact_table, source_value_id, &facts) &&
         loom_value_facts_is_lane_varying(facts);
}

static bool loom_amdgpu_source_value_memory_payload_use_requires_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    const loom_op_t* user_op, uint16_t operand_index) {
  loom_memory_access_t access = loom_memory_access_cast(module, user_op);
  if (!loom_memory_access_operand_index_is_payload(access, operand_index)) {
    return false;
  }
  return operand_index < user_op->operand_count &&
         loom_op_const_operands(user_op)[operand_index] == source_value_id;
}

static bool loom_amdgpu_source_value_select_payload_use_requires_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* user_op,
    uint16_t operand_index) {
  if (user_op == NULL || operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != source_value_id ||
      !loom_op_operand_has_role(module, user_op, operand_index,
                                LOOM_OPERAND_ROLE_SELECT_PAYLOAD)) {
    return false;
  }

  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (!loom_op_first_operand_with_role(
          module, user_op, LOOM_OPERAND_ROLE_SELECT_CONDITION, &condition)) {
    return false;
  }
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, analysis, condition, source_value_id);
}

static bool loom_amdgpu_select_result_requires_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  if (!loom_op_defines_value(defining_op, source_value_id)) {
    return false;
  }
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (!loom_op_first_operand_with_role(module, defining_op,
                                       LOOM_OPERAND_ROLE_SELECT_CONDITION,
                                       &condition)) {
    return false;
  }
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, analysis, condition, source_value_id);
}

static bool loom_amdgpu_source_value_has_vgpr_payload_use(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (loom_amdgpu_source_value_memory_payload_use_requires_vgpr(
            module, source_value_id, user_op, operand_index) ||
        loom_amdgpu_source_value_select_payload_use_requires_vgpr(
            module, fact_table, view_regions, analysis, source_value_id,
            user_op, operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_scalar_i32_to_i64_conversion_consumers_require_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_source_producer_flags_t producer_flags =
      loom_amdgpu_source_producer_flags(defining_op->kind);
  if (!loom_amdgpu_source_producer_result_follows_operand_vgpr(
          module, source_value_id, defining_op, producer_flags, &operand) ||
      !loom_amdgpu_type_is_i32(loom_module_value_type(module, operand)) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_amdgpu_source_value_has_vgpr_payload_use(
      module, fact_table, view_regions, analysis, source_value_id);
}

static bool loom_amdgpu_source_value_prefers_vgpr_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count) {
    return false;
  }
  if (loom_amdgpu_source_value_facts_prefer_vgpr(module, fact_table,
                                                 source_value_id)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    const uint16_t arg_index = loom_value_def_index(value);
    if (loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                        source_value_id) ||
        (loom_amdgpu_block_arg_has_cfg_predecessor(analysis, block,
                                                   arg_index) &&
         loom_amdgpu_source_value_has_vgpr_payload_use(
             module, fact_table, view_regions, analysis, source_value_id))) {
      return true;
    }
    return loom_amdgpu_block_arg_merges_native_mask_diamond(
               module, analysis, block, arg_index, LOOM_VALUE_ID_INVALID) ||
           loom_amdgpu_block_arg_incoming_payload_prefers_vgpr(
               module, fact_table, view_regions, analysis, block, arg_index,
               source_value_id);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                           source_value_id);
  }
  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);

  if (loom_amdgpu_value_feeds_native_mask_merge_arg(module, analysis, value,
                                                    source_value_id)) {
    return true;
  }

  if (loom_amdgpu_scalar_i32_to_i64_conversion_consumers_require_vgpr(
          module, fact_table, view_regions, analysis, source_value_id,
          defining_op)) {
    return true;
  }

  if (loom_amdgpu_select_result_requires_vgpr(module, fact_table, view_regions,
                                              analysis, source_value_id,
                                              defining_op)) {
    return true;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  const loom_amdgpu_source_producer_flags_t producer_flags =
      loom_amdgpu_source_producer_flags(defining_op->kind);
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT) &&
      loom_amdgpu_source_address_value_needs_64bit(
          module, fact_table, source_value_id, source_type)) {
    return true;
  }

  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCALAR_BITCAST) &&
      defining_op->operand_count == 1) {
    const loom_value_id_t input_value_id =
        loom_op_const_operands(defining_op)[0];
    return loom_amdgpu_analyzed_source_value_prefers_vgpr(
        module, fact_table, view_regions, analysis, input_value_id);
  }

  if (loom_amdgpu_source_scalar_float_result_follows_operands(
          module, analysis, source_value_id, defining_op, producer_flags)) {
    const loom_value_id_t* operands = loom_op_const_operands(defining_op);
    for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
      if (loom_amdgpu_analyzed_source_value_prefers_vgpr(
              module, fact_table, view_regions, analysis, operands[i])) {
        return true;
      }
    }
    return false;
  }

  // Recursive select propagation requires the active-bit table because the
  // selected result can participate in condition-mask classification.
  if (analysis != NULL &&
      iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_SCF_SELECT) &&
      loom_value_def_index(value) == 0) {
    return loom_amdgpu_analyzed_source_value_prefers_vgpr(
               module, fact_table, view_regions, analysis,
               loom_scf_select_true_value(defining_op)) ||
           loom_amdgpu_analyzed_source_value_prefers_vgpr(
               module, fact_table, view_regions, analysis,
               loom_scf_select_false_value(defining_op));
  }

  if (loom_amdgpu_scalar_type_op_result_prefers_vgpr(source_type)) {
    return true;
  }

  if (loom_value_def_index(value) == 0 &&
      loom_traits_have_distribution_transfer(defining_op_traits)) {
    return loom_amdgpu_distribution_transfer_result_prefers_vgpr(
        module, fact_table, view_regions, analysis, source_value_id,
        defining_op);
  }

  loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_source_value_fact_identity_operand(
          value, defining_op, defining_op_traits, source_value_id,
          &identity_operand)) {
    return identity_operand != LOOM_VALUE_ID_INVALID &&
           loom_amdgpu_analyzed_source_value_prefers_vgpr(
               module, fact_table, view_regions, analysis, identity_operand);
  }

  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_WORKITEM_DIMENSION)) {
    return loom_amdgpu_workitem_dimension_is_valid(defining_op);
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_ALWAYS_VGPR)) {
    return true;
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_RESULT0_VGPR)) {
    return loom_value_def_index(value) == 0;
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_STORAGE)) {
    loom_amdgpu_vector_storage_t storage = {0};
    return loom_value_def_index(value) == 0 &&
           loom_amdgpu_type_vector_storage(source_type, &storage) &&
           !iree_any_bit_set(
               loom_amdgpu_vector_storage_kind_flags(storage.kind),
               LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_SGPR_MASK);
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_EXTRACT)) {
    return loom_amdgpu_vector_extract_prefers_vgpr(
        module, fact_table, view_regions, analysis, defining_op);
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_MEMORY_ACCESS)) {
    return loom_amdgpu_source_memory_access_prefers_vgpr(
        module, fact_table, view_regions, analysis, defining_op, source_type);
  }
  if (iree_any_bit_set(producer_flags,
                       LOOM_AMDGPU_SOURCE_PRODUCER_VECTOR_REDUCE)) {
    return loom_amdgpu_vector_32bit_register_count(loom_module_value_type(
               module, loom_vector_reduce_input(defining_op))) != 0;
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_distribution_transfer_binary_result_follows_operand_vgpr(
          module, source_value_id, defining_op, &lhs, &rhs)) {
    return loom_amdgpu_analyzed_source_value_prefers_vgpr(
               module, fact_table, view_regions, analysis, lhs) ||
           loom_amdgpu_analyzed_source_value_prefers_vgpr(
               module, fact_table, view_regions, analysis, rhs);
  }
  if (loom_amdgpu_source_producer_result_requires_vgpr(module, source_value_id,
                                                       producer_flags) ||
      loom_amdgpu_scalar_type_fallback_result_prefers_vgpr(source_type) ||
      loom_amdgpu_vector_32bit_register_count(source_type) != 0) {
    return true;
  }
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_source_producer_result_follows_operand_vgpr(
             module, source_value_id, defining_op, producer_flags, &operand) &&
         loom_amdgpu_analyzed_source_value_prefers_vgpr(
             module, fact_table, view_regions, analysis, operand);
}

bool loom_amdgpu_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  return loom_amdgpu_source_value_prefers_vgpr_impl(
      module, fact_table, view_regions, /*analysis=*/NULL, source_value_id);
}

static bool loom_amdgpu_analyzed_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_amdgpu_source_value_analysis_bits_t bit =
      LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_PREFERS_VGPR;
  bool value = false;
  if (loom_amdgpu_source_value_analysis_cached_bit(analysis, source_value_id,
                                                   bit, &value)) {
    return value;
  }
  if (!loom_amdgpu_source_value_analysis_begin_bit(analysis, source_value_id,
                                                   bit)) {
    return false;
  }
  value = loom_amdgpu_source_value_prefers_vgpr_impl(
      module, fact_table, view_regions, analysis, source_value_id);
  loom_amdgpu_source_value_analysis_end_bit(analysis, source_value_id, bit,
                                            value);
  return value;
}

static bool loom_amdgpu_analyzed_source_value_can_lower_as_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_amdgpu_source_value_analysis_bits_t bit =
      LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_SCC_I1_BOOL;
  bool value = false;
  if (loom_amdgpu_source_value_analysis_cached_bit(analysis, source_value_id,
                                                   bit, &value)) {
    return value;
  }
  if (!loom_amdgpu_source_value_analysis_begin_bit(analysis, source_value_id,
                                                   bit)) {
    return false;
  }
  value = loom_amdgpu_source_value_can_lower_as_scc_i1(
      module, fact_table, view_regions, analysis, source_value_id,
      LOOM_VALUE_ID_INVALID);
  loom_amdgpu_source_value_analysis_end_bit(analysis, source_value_id, bit,
                                            value);
  return value;
}

static bool loom_amdgpu_analyzed_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_amdgpu_source_value_analysis_bits_t bit =
      LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_NATIVE_I1_MASK;
  bool value = false;
  if (loom_amdgpu_source_value_analysis_cached_bit(analysis, source_value_id,
                                                   bit, &value)) {
    return value;
  }
  if (!loom_amdgpu_source_value_analysis_begin_bit(analysis, source_value_id,
                                                   bit)) {
    return value;
  }
  value = loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, analysis, source_value_id,
      LOOM_VALUE_ID_INVALID);
  loom_amdgpu_source_value_analysis_end_bit(analysis, source_value_id, bit,
                                            value);
  return value;
}

static bool loom_amdgpu_analyzed_source_value_is_durable_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id) {
  const loom_amdgpu_source_value_analysis_bits_t bit =
      LOOM_AMDGPU_SOURCE_VALUE_ANALYSIS_DURABLE_I1_BOOL;
  bool value = false;
  if (loom_amdgpu_source_value_analysis_cached_bit(analysis, source_value_id,
                                                   bit, &value)) {
    return value;
  }
  if (!loom_amdgpu_source_value_analysis_begin_bit(analysis, source_value_id,
                                                   bit)) {
    return value;
  }
  value = loom_amdgpu_source_value_is_durable_i1_bool(
      module, fact_table, view_regions, analysis, source_value_id);
  loom_amdgpu_source_value_analysis_end_bit(analysis, source_value_id, bit,
                                            value);
  return value;
}

iree_status_t loom_amdgpu_context_value_prefers_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_prefers_vgpr) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_context(context, &analysis));
  *out_prefers_vgpr = loom_amdgpu_analyzed_source_value_prefers_vgpr(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions, analysis,
      source_value_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_low_legality_value_prefers_vgpr(
    loom_target_low_legality_context_t* context,
    loom_value_id_t source_value_id, bool* out_prefers_vgpr) {
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_target_low_legality(context,
                                                                &analysis));
  *out_prefers_vgpr = loom_amdgpu_analyzed_source_value_prefers_vgpr(
      loom_target_low_legality_module(context),
      loom_target_low_legality_fact_table(context), view_regions, analysis,
      source_value_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_context_value_is_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_native_mask) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_context(context, &analysis));
  *out_is_native_mask = loom_amdgpu_analyzed_source_value_is_native_i1_mask(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions, analysis,
      source_value_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_target_low_legality_value_is_native_i1_mask(
    loom_target_low_legality_context_t* context,
    loom_value_id_t source_value_id, bool* out_is_native_mask) {
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  loom_amdgpu_source_value_analysis_t* analysis = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_value_analysis_for_target_low_legality(context,
                                                                &analysis));
  *out_is_native_mask = loom_amdgpu_analyzed_source_value_is_native_i1_mask(
      loom_target_low_legality_module(context),
      loom_target_low_legality_fact_table(context), view_regions, analysis,
      source_value_id);
  return iree_ok_status();
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
