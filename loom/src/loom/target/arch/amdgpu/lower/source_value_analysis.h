// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared state and recursion boundaries for AMDGPU source value analysis.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_VALUE_ANALYSIS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_VALUE_ANALYSIS_H_

#include <stdint.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/low_legality.h"
#include "loom/util/cfg_graph.h"

//===----------------------------------------------------------------------===//
// Register mapping
//===----------------------------------------------------------------------===//

typedef struct loom_amdgpu_register_shape_t {
  // AMDGPU descriptor-set register class selected for the value.
  uint16_t class_id;
  // Number of 32-bit register units occupied by the value.
  uint32_t unit_count;
} loom_amdgpu_register_shape_t;

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
  // True when source_cfg_graph has been initialized for a multi-block body.
  bool source_cfg_graph_initialized;
} loom_amdgpu_source_value_analysis_t;

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

bool loom_amdgpu_scalar_type_naturally_prefers_vgpr(loom_type_t source_type);
bool loom_amdgpu_scalar_type_op_result_prefers_vgpr(loom_type_t source_type);
bool loom_amdgpu_scalar_type_fallback_result_prefers_vgpr(
    loom_type_t source_type);
bool loom_amdgpu_scalar_type_has_fixed_vgpr_storage(loom_type_t source_type);

//===----------------------------------------------------------------------===//
// Cached analysis state
//===----------------------------------------------------------------------===//

iree_status_t loom_amdgpu_source_value_analysis_for_context(
    loom_low_lower_context_t* context,
    loom_amdgpu_source_value_analysis_t** out_analysis);
iree_status_t loom_amdgpu_source_value_analysis_for_target_low_legality(
    loom_target_low_legality_context_t* context,
    loom_amdgpu_source_value_analysis_t** out_analysis);
iree_status_t loom_amdgpu_source_value_analysis_for_contract_query(
    const loom_target_contract_query_environment_t* environment,
    loom_amdgpu_source_value_analysis_t** out_analysis);
const loom_cfg_graph_t* loom_amdgpu_source_value_analysis_cfg_graph(
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_region_t* region);
bool loom_amdgpu_source_value_analysis_cached_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit, bool* out_value);
bool loom_amdgpu_source_value_analysis_begin_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit);
void loom_amdgpu_source_value_analysis_end_bit(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id,
    loom_amdgpu_source_value_analysis_bits_t bit, bool value);
bool loom_amdgpu_source_value_analysis_cached_register_shape(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, bool* out_has_shape,
    loom_amdgpu_register_shape_t* out_shape);
void loom_amdgpu_source_value_analysis_record_register_shape(
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_amdgpu_register_shape_t* shape,
    bool valid);

//===----------------------------------------------------------------------===//
// VGPR placement
//===----------------------------------------------------------------------===//

loom_amdgpu_source_producer_flags_t loom_amdgpu_source_producer_flags(
    loom_op_kind_t kind);
bool loom_amdgpu_source_value_facts_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id);
bool loom_amdgpu_source_value_directly_prefers_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t excluded_value_id);
bool loom_amdgpu_op_results_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* op,
    loom_value_id_t excluded_value_id);
bool loom_amdgpu_op_operands_with_role_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis, const loom_op_t* op,
    loom_operand_role_t role, loom_value_id_t excluded_value_id);
bool loom_amdgpu_analyzed_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);

//===----------------------------------------------------------------------===//
// i1 representation
//===----------------------------------------------------------------------===//

bool loom_amdgpu_source_value_fact_identity_operand(
    const loom_value_t* value, const loom_op_t* defining_op,
    loom_trait_flags_t defining_op_traits, loom_value_id_t source_value_id,
    loom_value_id_t* out_operand);
bool loom_amdgpu_block_arg_merges_native_mask_diamond(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_block_t* block, uint16_t arg_index,
    loom_value_id_t excluded_value_id);
bool loom_amdgpu_value_feeds_native_mask_merge_arg(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_value_t* value, loom_value_id_t value_id);
bool loom_amdgpu_block_arg_has_cfg_predecessor(
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_block_t* block, uint16_t arg_index);
bool loom_amdgpu_select_result_requires_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id, const loom_op_t* defining_op);
bool loom_amdgpu_source_value_has_vgpr_payload_use(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
bool loom_amdgpu_analyzed_source_value_can_lower_as_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
bool loom_amdgpu_analyzed_source_value_can_lower_as_sgpr_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
bool loom_amdgpu_analyzed_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);
bool loom_amdgpu_analyzed_source_value_is_durable_i1_bool(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_amdgpu_source_value_analysis_t* analysis,
    loom_value_id_t source_value_id);

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_VALUE_ANALYSIS_H_
