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

bool loom_amdgpu_source_value_facts_prefer_vgpr(
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
            LOOM_AMDGPU_SOURCE_PRODUCER_FOLLOWS_OPERAND |
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

loom_amdgpu_source_producer_flags_t loom_amdgpu_source_producer_flags(
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

bool loom_amdgpu_source_value_directly_prefers_vgpr(
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

bool loom_amdgpu_op_results_prefer_vgpr(
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

bool loom_amdgpu_op_operands_with_role_prefer_vgpr(
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

// Returns true when |source_value_id| is an expression whose value depends on
// a lane-select condition. Branch joins may otherwise scalarize uniform VGPR
// payloads with readfirstlane, which is important for uniform address loops.
// Chase only distribution-preserving expression producers here: an arbitrary
// nested VGPR producer does not by itself make scalarization invalid.
static bool loom_amdgpu_branch_payload_has_vgpr_select_dependency(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || loom_value_def_index(value) != 0) {
    return false;
  }
  if (loom_scf_select_isa(defining_op)) {
    return loom_amdgpu_analyzed_source_value_prefers_vgpr(
        module, fact_table, view_regions, analysis, source_value_id);
  }

  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);
  loom_value_id_t identity_operand = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_source_value_fact_identity_operand(
          value, defining_op, defining_op_traits, source_value_id,
          &identity_operand)) {
    return identity_operand != LOOM_VALUE_ID_INVALID &&
           loom_amdgpu_branch_payload_has_vgpr_select_dependency(
               module, fact_table, view_regions, analysis, identity_operand,
               excluded_value_id);
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_distribution_transfer_binary_result_follows_operand_vgpr(
          module, source_value_id, defining_op, &lhs, &rhs)) {
    return loom_amdgpu_branch_payload_has_vgpr_select_dependency(
               module, fact_table, view_regions, analysis, lhs,
               excluded_value_id) ||
           loom_amdgpu_branch_payload_has_vgpr_select_dependency(
               module, fact_table, view_regions, analysis, rhs,
               excluded_value_id);
  }

  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_source_producer_flags_t producer_flags =
      loom_amdgpu_source_producer_flags(defining_op->kind);
  return loom_amdgpu_source_producer_result_follows_operand_vgpr(
             module, source_value_id, defining_op, producer_flags, &operand) &&
         loom_amdgpu_branch_payload_has_vgpr_select_dependency(
             module, fact_table, view_regions, analysis, operand,
             excluded_value_id);
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
  if (analysis != NULL && loom_amdgpu_branch_payload_has_vgpr_select_dependency(
                              module, fact_table, view_regions, analysis,
                              incoming_value_id, excluded_value_id)) {
    return true;
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

bool loom_amdgpu_analyzed_source_value_prefers_vgpr(
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
