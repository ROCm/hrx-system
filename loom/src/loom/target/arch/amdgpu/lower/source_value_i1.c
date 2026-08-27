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

bool loom_amdgpu_source_value_fact_identity_operand(
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

static bool loom_amdgpu_index_cmp_requires_native_mask(
    const loom_module_t* module,
    const loom_amdgpu_source_value_analysis_t* analysis,
    const loom_op_t* source_op) {
  // Full-width address comparisons are assembled from 32-bit vector compares
  // and therefore produce a native lane mask. Address comparisons proven to
  // fit in 32 bits retain the scalar SCC path.
  if (!loom_index_cmp_isa(source_op)) return false;
  const loom_value_id_t lhs = loom_index_cmp_lhs(source_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(source_op);
  if (lhs >= module->values.count || rhs >= module->values.count) return false;
  const loom_value_fact_table_t* fact_table =
      analysis != NULL ? analysis->fact_table : NULL;
  return loom_amdgpu_source_address_value_needs_64bit(
             module, fact_table, lhs, loom_module_value_type(module, lhs)) ||
         loom_amdgpu_source_address_value_needs_64bit(
             module, fact_table, rhs, loom_module_value_type(module, rhs));
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
  if (loom_amdgpu_index_cmp_requires_native_mask(module, analysis, source_op)) {
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

bool loom_amdgpu_block_arg_merges_native_mask_diamond(
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

bool loom_amdgpu_value_feeds_native_mask_merge_arg(
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

bool loom_amdgpu_block_arg_has_cfg_predecessor(
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

bool loom_amdgpu_analyzed_source_value_can_lower_as_sgpr_i1_bool(
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

bool loom_amdgpu_select_result_requires_vgpr(
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

bool loom_amdgpu_source_value_has_vgpr_payload_use(
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

bool loom_amdgpu_analyzed_source_value_can_lower_as_scc_i1(
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

bool loom_amdgpu_analyzed_source_value_is_native_i1_mask(
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

bool loom_amdgpu_analyzed_source_value_is_durable_i1_bool(
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
