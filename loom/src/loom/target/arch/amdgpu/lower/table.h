// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for register table source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_TABLE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_TABLE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/byte_permute.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_table_index_kind_e {
  LOOM_AMDGPU_TABLE_INDEX_KIND_NONE = 0,
  LOOM_AMDGPU_TABLE_INDEX_KIND_I32 = 1,
  LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8 = 2,
  LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I16 = 3,
  LOOM_AMDGPU_TABLE_INDEX_KIND_I64 = 4,
} loom_amdgpu_table_index_kind_t;

typedef enum loom_amdgpu_table_lookup_strategy_e {
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_NONE = 0,
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_F32_LADDER = 1,
  LOOM_AMDGPU_TABLE_LOOKUP_STRATEGY_PACKED_BYTE_PERMUTE_TREE = 2,
} loom_amdgpu_table_lookup_strategy_t;

typedef struct loom_amdgpu_table_lookup_plan_t {
  // Register table value selected by each index lane.
  loom_value_id_t table;
  // Index vector selecting dynamic table lanes, or invalid when all are static.
  loom_value_id_t indices;
  // Result vector receiving selected table lanes.
  loom_value_id_t result;
  // Selected lowering strategy.
  loom_amdgpu_table_lookup_strategy_t strategy;
  // Descriptor row selected for index-lane equality comparisons.
  loom_low_lower_resolved_descriptor_t compare_register_descriptor;
  // Optional descriptor row selected when the compare rhs ordinal is inline.
  loom_low_lower_resolved_descriptor_t compare_src1_inline_descriptor;
  // Descriptor row selected for register-register table lane selects.
  loom_low_lower_resolved_descriptor_t select_register_descriptor;
  // Optional descriptor row selected when the true table lane is a literal.
  loom_low_lower_resolved_descriptor_t select_src1_literal_descriptor;
  // Descriptor row selected for packed byte table permutation.
  loom_low_lower_resolved_descriptor_t permute_descriptor;
  // Fixed-selector permutation form used to pack non-byte index lanes.
  loom_amdgpu_byte_permute_plan_t index_pack_permute;
  // Selected index payload representation.
  loom_amdgpu_table_index_kind_t index_kind;
  // Static number of table lanes.
  uint32_t table_lane_count;
  // Number of 32-bit registers occupied by the table vector.
  uint32_t table_register_count;
  // Static number of result lanes.
  uint32_t result_lane_count;
  // Number of 32-bit registers occupied by the index vector.
  uint32_t index_register_count;
  // Number of 32-bit registers occupied by the result vector.
  uint32_t result_register_count;
  // Table lanes selecting the literal form, in ascending payload order.
  uint32_t literal_lane_mask;
  // Packed literal bits for set lanes; NULL when all selects use registers.
  const uint32_t* literal_bits;
  // Selected table lane for each F32 result, or UINT8_MAX for a dynamic index.
  uint8_t table_lane_indices[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_table_lookup_plan_t;

// Selects an AMDGPU register-table lookup plan.
iree_status_t loom_amdgpu_select_vector_table_lookup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_table_lookup_plan_t* out_plan, bool* out_selected);

// Publishes the selected lookup carrier, including singleton-table aliases.
iree_status_t loom_amdgpu_finalize_vector_table_lookup_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_table_lookup_plan_t* plan);

// Lowers a source vector.table.lookup op with its selected native recipe.
// Packed byte recipes emit independent permutes for each result register,
// including the semantic low bytes of a partial final register.
iree_status_t loom_amdgpu_lower_vector_table_lookup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan);

// Returns whether the current shapes, index facts and target descriptors admit
// a native recipe. Shared legalization uses this same selection predicate to
// preserve native lookup forms while expanding unsupported forms.
bool loom_amdgpu_vector_table_lookup_is_supported(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op);

// Verifies source vector table op legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_table(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_TABLE_H_
