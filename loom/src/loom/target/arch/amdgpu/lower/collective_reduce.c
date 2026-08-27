// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/cfg_graph.h"

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u

static const loom_amdgpu_collective_combine_dpp_form_t
    kLoomAmdgpuSubgroupReduceDppCombineForms[] = {
        LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_DPP16,
        LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_LEGACY,
};

static uint32_t loom_amdgpu_subgroup_u32_log2(uint32_t value) {
  uint32_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

static bool loom_amdgpu_subgroup_optional_attr_is_present(const loom_op_t* op,
                                                          uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static bool loom_amdgpu_subgroup_reduce_has_cluster_attrs(const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_reduce_cluster_size_ATTR_INDEX) ||
         loom_amdgpu_subgroup_optional_attr_is_present(
             op, loom_kernel_subgroup_reduce_cluster_stride_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_reduce_has_cluster_size(const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
      op, loom_kernel_subgroup_reduce_cluster_size_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_reduce_has_cluster_stride(
    const loom_op_t* op) {
  return loom_amdgpu_subgroup_optional_attr_is_present(
      op, loom_kernel_subgroup_reduce_cluster_stride_ATTR_INDEX);
}

static bool loom_amdgpu_subgroup_cluster_size_is_supported(
    int64_t cluster_size, uint32_t wavefront_size) {
  return cluster_size > 0 && cluster_size <= (int64_t)wavefront_size &&
         loom_amdgpu_u32_is_power_of_two((uint32_t)cluster_size);
}

typedef uint8_t loom_amdgpu_subgroup_reduce_shape_flags_t;

enum loom_amdgpu_subgroup_reduce_shape_flag_bits_e {
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED = 1u << 0,
};

typedef enum loom_amdgpu_subgroup_reduce_shape_failure_e {
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE = 1,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE = 2,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE = 3,
  LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE = 4,
} loom_amdgpu_subgroup_reduce_shape_failure_t;

typedef struct loom_amdgpu_subgroup_reduce_shape_t {
  // Semantic lanes participating in the selected subgroup reduce.
  uint32_t active_lane_count;
  // Shape properties derived from subgroup-reduce attributes and launch facts.
  loom_amdgpu_subgroup_reduce_shape_flags_t flags;
} loom_amdgpu_subgroup_reduce_shape_t;

static bool loom_amdgpu_subgroup_reduce_resolve_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, const loom_op_t* op,
    uint32_t wavefront_size, loom_amdgpu_subgroup_reduce_shape_t* out_shape,
    loom_amdgpu_subgroup_reduce_shape_failure_t* out_failure) {
  *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){0};
  *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;

  const bool has_cluster_attrs =
      loom_amdgpu_subgroup_reduce_has_cluster_attrs(op);
  int64_t cluster_size = 0;
  if (has_cluster_attrs) {
    if (loom_amdgpu_subgroup_reduce_has_cluster_stride(op)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE;
      return false;
    }
    if (!loom_amdgpu_subgroup_reduce_has_cluster_size(op)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE;
      return false;
    }

    cluster_size = loom_kernel_subgroup_reduce_cluster_size(op);
    if (!loom_amdgpu_subgroup_cluster_size_is_supported(cluster_size,
                                                        wavefront_size)) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE;
      return false;
    }
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    *out_failure =
        has_cluster_attrs
            ? LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE
            : LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
    return false;
  }

  if (!has_cluster_attrs) {
    if (flat_workgroup_size <= wavefront_size) {
      *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
          .active_lane_count = flat_workgroup_size,
      };
      return true;
    }
    if ((flat_workgroup_size % wavefront_size) != 0) {
      *out_failure =
          LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
      return false;
    }
    *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
        .active_lane_count = wavefront_size,
    };
    return true;
  }

  if (flat_workgroup_size <= wavefront_size) {
    if ((flat_workgroup_size % (uint32_t)cluster_size) != 0) {
      *out_failure =
          LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE;
      return false;
    }
  } else if ((flat_workgroup_size % wavefront_size) != 0) {
    *out_failure =
        LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE;
    return false;
  }

  *out_shape = (loom_amdgpu_subgroup_reduce_shape_t){
      .active_lane_count = (uint32_t)cluster_size,
      .flags = LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED,
  };
  return true;
}

static iree_string_view_t loom_amdgpu_subgroup_reduce_shape_failure_key(
    loom_amdgpu_subgroup_reduce_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_STRIDE:
      return IREE_SV("subgroup_reduce.cluster_stride");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_CLUSTER_SIZE:
      return IREE_SV("subgroup_reduce.power_of_two_cluster_size");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_CLUSTER_MULTIPLE:
      return IREE_SV("subgroup_reduce.fixed_workgroup_cluster_multiple");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_WORKGROUP_WAVE_MULTIPLE:
      return IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple");
    case LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU subgroup reduce shape failure requires reason");
  return IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple");
}

static bool loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(
    uint32_t wavefront_size, uint32_t active_lane_count) {
  if (active_lane_count == 0 || active_lane_count > wavefront_size) {
    return false;
  }
  if (active_lane_count <= LOOM_AMDGPU_DPP_ROW_LANE_COUNT) {
    return true;
  }
  return loom_amdgpu_u32_is_power_of_two(active_lane_count);
}

static bool loom_amdgpu_subgroup_reduce_dpp_row_bpermute_is_applicable(
    uint32_t wavefront_size, uint32_t active_lane_count) {
  if (active_lane_count == 0 || active_lane_count > wavefront_size) {
    return false;
  }
  if (active_lane_count <= LOOM_AMDGPU_DPP_ROW_LANE_COUNT) {
    return true;
  }
  return active_lane_count == wavefront_size && wavefront_size == 32;
}

static iree_status_t loom_amdgpu_resolve_subgroup_reduce_dpp_combine_descriptor(
    loom_low_lower_context_t* context, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* out_descriptor, bool* out_present) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_present = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuSubgroupReduceDppCombineForms); ++i) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_collective_combine_dpp_descriptor_ref(
            kind, payload_kind, kLoomAmdgpuSubgroupReduceDppCombineForms[i],
            &descriptor_ref)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, descriptor_ref, out_descriptor, out_present));
    if (*out_present) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_subgroup_reduce_crosslane_kind(
    loom_low_lower_context_t* context, uint32_t wavefront_size,
    uint32_t active_lane_count, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_low_lower_resolved_descriptor_t* dpp_combine,
    loom_low_lower_resolved_descriptor_t* permlanex16,
    loom_amdgpu_subgroup_reduce_crosslane_kind_t* out_crosslane_kind) {
  *dpp_move = (loom_low_lower_resolved_descriptor_t){0};
  *dpp_combine = (loom_low_lower_resolved_descriptor_t){0};
  *permlanex16 = (loom_low_lower_resolved_descriptor_t){0};
  *out_crosslane_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE;
  const bool dpp_row_bpermute_applicable =
      loom_amdgpu_subgroup_reduce_dpp_row_bpermute_is_applicable(
          wavefront_size, active_lane_count);
  const bool dpp_row_permlanex16_applicable =
      active_lane_count > LOOM_AMDGPU_DPP_ROW_LANE_COUNT &&
      loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(wavefront_size,
                                                        active_lane_count);
  if (!dpp_row_bpermute_applicable && !dpp_row_permlanex16_applicable) {
    return iree_ok_status();
  }

  bool dpp_combine_descriptor_present = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_subgroup_reduce_dpp_combine_descriptor(
          context, kind, payload_kind, dpp_combine,
          &dpp_combine_descriptor_present));

  bool dpp_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16, dpp_move,
      &dpp_descriptor_present));
  if (!dpp_descriptor_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP, dpp_move,
        &dpp_descriptor_present));
  }
  if (!dpp_combine_descriptor_present && !dpp_descriptor_present) {
    return iree_ok_status();
  }

  if (dpp_row_permlanex16_applicable) {
    bool permlanex16_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERMLANEX16_B32_SRC12_INLINE,
        permlanex16, &permlanex16_descriptor_present));
    if (permlanex16_descriptor_present) {
      *out_crosslane_kind =
          LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16;
      return iree_ok_status();
    }
  }
  if (dpp_row_bpermute_applicable) {
    *out_crosslane_kind =
        LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE;
  }
  return iree_ok_status();
}

typedef uint8_t loom_amdgpu_participant_zero_bits_t;

enum loom_amdgpu_participant_zero_bit_e {
  LOOM_AMDGPU_PARTICIPANT_ZERO_WORKITEM_X = 1u << 0,
  LOOM_AMDGPU_PARTICIPANT_ZERO_SUBGROUP_LANE = 1u << 1,
  LOOM_AMDGPU_PARTICIPANT_ZERO_ANY = LOOM_AMDGPU_PARTICIPANT_ZERO_WORKITEM_X |
                                     LOOM_AMDGPU_PARTICIPANT_ZERO_SUBGROUP_LANE,
};

static loom_amdgpu_participant_zero_bits_t
loom_amdgpu_value_participant_id_bits(const loom_value_fact_table_t* fact_table,
                                      loom_value_id_t value_id) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
  loom_amdgpu_participant_zero_bits_t bits = 0;
  if (iree_all_bits_set(facts.flags, LOOM_VALUE_FACT_TOPOLOGY_WORKITEM_X)) {
    bits |= LOOM_AMDGPU_PARTICIPANT_ZERO_WORKITEM_X;
  }
  if (iree_all_bits_set(facts.flags, LOOM_VALUE_FACT_TOPOLOGY_SUBGROUP_LANE)) {
    bits |= LOOM_AMDGPU_PARTICIPANT_ZERO_SUBGROUP_LANE;
  }
  return bits;
}

static loom_amdgpu_participant_zero_bits_t
loom_amdgpu_symbolic_expr_participant_id_bits(
    const loom_value_fact_table_t* fact_table,
    const loom_symbolic_expr_t* expression) {
  if (!loom_symbolic_expr_is_linear(expression) || expression->constant != 0 ||
      expression->term_count != 1 || expression->terms[0].coefficient != 1) {
    return 0;
  }
  return loom_amdgpu_value_participant_id_bits(fact_table,
                                               expression->terms[0].value_id) |
         loom_amdgpu_value_participant_id_bits(
             fact_table, expression->terms[0].relation_value_id);
}

static bool loom_amdgpu_symbolic_expr_is_zero(
    const loom_symbolic_expr_t* expression) {
  return loom_symbolic_expr_is_constant(expression) &&
         expression->constant == 0;
}

static iree_status_t loom_amdgpu_condition_participant_zero_bits(
    loom_low_lower_context_t* context, loom_value_id_t condition,
    bool assumed_truth,
    loom_amdgpu_participant_zero_bits_t* out_participant_zero_bits) {
  *out_participant_zero_bits = 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);

  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t condition_facts = {0};
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_condition_facts_query(
      loom_low_lower_context_condition_query(context), fact_table, condition,
      assumed_truth, &condition_facts, &complete));
  if (!complete) {
    return iree_ok_status();
  }

  loom_symbolic_expr_context_t* symbolic_context =
      loom_low_lower_context_symbolic_expr_context(context);

  for (iree_host_size_t i = 0; i < condition_facts.integer_relation_count;
       ++i) {
    const loom_condition_integer_relation_t* relation =
        &condition_facts.integer_relations[i];
    if (relation->relation != LOOM_SYMBOLIC_INTEGER_RELATION_EQ ||
        relation->left.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
        relation->right.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
      continue;
    }

    loom_symbolic_expr_t left_expression = {0};
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        symbolic_context, relation->left.value_id, &left_expression));
    loom_symbolic_expr_t right_expression = {0};
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        symbolic_context, relation->right.value_id, &right_expression));
    if (loom_amdgpu_symbolic_expr_is_zero(&right_expression)) {
      *out_participant_zero_bits |=
          loom_amdgpu_symbolic_expr_participant_id_bits(fact_table,
                                                        &left_expression);
    }
    if (loom_amdgpu_symbolic_expr_is_zero(&left_expression)) {
      *out_participant_zero_bits |=
          loom_amdgpu_symbolic_expr_participant_id_bits(fact_table,
                                                        &right_expression);
    }
    if (*out_participant_zero_bits == LOOM_AMDGPU_PARTICIPANT_ZERO_ANY) {
      break;
    }
  }

  return iree_ok_status();
}

typedef struct loom_amdgpu_participant_zero_analysis_t {
  // CFG region containing the collective result definition.
  const loom_region_t* region;
  // Dense CFG retained by the lowering fact table.
  const loom_cfg_graph_t* graph;
  // Per-block participant IDs that can reach the block without a zero guard.
  loom_amdgpu_participant_zero_bits_t* unguarded_reachable_bits;
} loom_amdgpu_participant_zero_analysis_t;

static iree_status_t loom_amdgpu_participant_zero_analysis_initialize(
    loom_low_lower_context_t* context, const loom_region_t* region,
    loom_amdgpu_participant_zero_analysis_t* out_analysis) {
  memset(out_analysis, 0, sizeof(*out_analysis));
  out_analysis->region = region;
  if (region == NULL || region->block_count == 0) {
    return iree_ok_status();
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_cfg_graph_t* graph =
      loom_value_fact_table_lookup_cfg_graph(fact_table, region);
  if (graph == NULL || graph->malformed) {
    return iree_ok_status();
  }
  out_analysis->graph = graph;

  loom_amdgpu_participant_zero_bits_t* edge_guard_bits = NULL;
  if (graph->edge_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
        context, graph->edge_count, sizeof(*edge_guard_bits),
        (void**)&edge_guard_bits));
    memset(edge_guard_bits, 0, graph->edge_count * sizeof(*edge_guard_bits));
    for (loom_cfg_edge_index_t edge_index = 0; edge_index < graph->edge_count;
         ++edge_index) {
      const loom_cfg_edge_info_t* edge = loom_cfg_graph_edge(graph, edge_index);
      if (!loom_cfg_cond_br_isa(edge->terminator)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_condition_participant_zero_bits(
          context, edge->selector_value_id,
          /*assumed_truth=*/edge->successor_index == 0,
          &edge_guard_bits[edge_index]));
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
      context, graph->block_count,
      sizeof(*out_analysis->unguarded_reachable_bits),
      (void**)&out_analysis->unguarded_reachable_bits));
  memset(out_analysis->unguarded_reachable_bits, 0,
         graph->block_count * sizeof(*out_analysis->unguarded_reachable_bits));

  // Propagate identities that still have an unguarded path from the entry.
  // Each of the two bits can enter each block only once, bounding the worklist
  // to two entries per block even when the CFG contains cycles.
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
      context, graph->block_count * 2, sizeof(*worklist), (void**)&worklist));
  iree_host_size_t worklist_count = 1;
  worklist[0] = 0;
  out_analysis->unguarded_reachable_bits[0] = LOOM_AMDGPU_PARTICIPANT_ZERO_ANY;
  while (worklist_count > 0) {
    const uint16_t source_block_index = worklist[--worklist_count];
    const loom_amdgpu_participant_zero_bits_t source_bits =
        out_analysis->unguarded_reachable_bits[source_block_index];
    const loom_cfg_edge_index_span_t successor_edges =
        loom_cfg_graph_successor_edges(graph, source_block_index);
    for (iree_host_size_t i = 0; i < successor_edges.count; ++i) {
      const loom_cfg_edge_index_t edge_index = successor_edges.values[i];
      const loom_cfg_edge_info_t* edge = loom_cfg_graph_edge(graph, edge_index);
      const loom_amdgpu_participant_zero_bits_t propagated_bits =
          source_bits & ~edge_guard_bits[edge_index];
      const loom_amdgpu_participant_zero_bits_t added_bits =
          propagated_bits &
          ~out_analysis->unguarded_reachable_bits[edge->target_block_index];
      if (added_bits == 0) {
        continue;
      }
      out_analysis->unguarded_reachable_bits[edge->target_block_index] |=
          added_bits;
      worklist[worklist_count++] = edge->target_block_index;
    }
  }
  return iree_ok_status();
}

static loom_amdgpu_participant_zero_bits_t
loom_amdgpu_participant_zero_analysis_block_bits(
    const loom_amdgpu_participant_zero_analysis_t* analysis,
    const loom_block_t* block) {
  if (analysis->graph == NULL || block == NULL) {
    return 0;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(analysis->region, block, &block_index) ||
      !loom_cfg_graph_block_is_reachable(analysis->graph, block_index)) {
    return 0;
  }
  return LOOM_AMDGPU_PARTICIPANT_ZERO_ANY &
         ~analysis->unguarded_reachable_bits[block_index];
}

static iree_status_t loom_amdgpu_use_participant_zero_bits(
    loom_low_lower_context_t* context,
    const loom_amdgpu_participant_zero_analysis_t* analysis,
    const loom_op_t* use_op,
    loom_amdgpu_participant_zero_bits_t* out_participant_zero_bits) {
  *out_participant_zero_bits = 0;
  for (const loom_op_t* current = use_op; current != NULL;
       current = current->parent_op) {
    if (current->parent_block != NULL &&
        current->parent_block->parent_region == analysis->region) {
      *out_participant_zero_bits |=
          loom_amdgpu_participant_zero_analysis_block_bits(
              analysis, current->parent_block);
    }

    const loom_op_t* parent_op = current->parent_op;
    if (parent_op == NULL || !loom_scf_if_isa(parent_op) ||
        current->parent_block == NULL) {
      continue;
    }
    const loom_region_t* current_region = current->parent_block->parent_region;
    bool assumed_truth = true;
    if (current_region == loom_scf_if_then_region(parent_op)) {
      assumed_truth = true;
    } else if (current_region == loom_scf_if_else_region(parent_op)) {
      assumed_truth = false;
    } else {
      continue;
    }
    loom_amdgpu_participant_zero_bits_t guard_bits = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_condition_participant_zero_bits(
        context, loom_scf_if_condition(parent_op), assumed_truth, &guard_bits));
    *out_participant_zero_bits |= guard_bits;
  }
  return iree_ok_status();
}

typedef enum loom_amdgpu_collective_result_demand_e {
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS = 0,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM = 1,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE = 2,
} loom_amdgpu_collective_result_demand_t;

typedef uint8_t loom_amdgpu_collective_result_demand_bits_t;

enum loom_amdgpu_collective_result_demand_bits_e {
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE =
      1u << LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE,
  LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ANY =
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS |
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM |
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE,
};

typedef uint8_t loom_amdgpu_workgroup_reduce_publication_rule_flags_t;

enum loom_amdgpu_workgroup_reduce_publication_rule_flags_e {
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE = 1u << 0,
  LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER = 1u << 1,
};

typedef struct loom_amdgpu_workgroup_reduce_publication_rule_t {
  // Publication strategy selected when this rule matches.
  loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind;
  // Result-demand kinds accepted by this rule.
  loom_amdgpu_collective_result_demand_bits_t demand_bits;
  // Workgroup/payload facts required by this rule.
  loom_amdgpu_workgroup_reduce_publication_rule_flags_t required_flags;
  // Compile-report key describing the selected publication strategy.
  iree_string_view_t report_key;
} loom_amdgpu_workgroup_reduce_publication_rule_t;

static const loom_amdgpu_workgroup_reduce_publication_rule_t
    kLoomAmdgpuWorkgroupReducePublicationRules[] = {
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_LEADER_WORKITEM,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "leader_workitem"),
        },
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_SUBGROUP_LEADER_LANE,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "redundant_subgroup_leader_lane"),
        },
        {
            .publication_kind =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP,
            .demand_bits =
                LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ALL_WORKITEMS,
            .required_flags =
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE |
                LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication."
                                   "redundant_subgroup"),
        },
        {
            .publication_kind = LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS,
            .demand_bits = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_BIT_ANY,
            .required_flags = 0,
            .report_key = IREE_SVL("amdgpu.workgroup_reduce.publication.lds"),
        },
};

static iree_status_t loom_amdgpu_collective_result_demand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_collective_result_demand_t* out_result_demand) {
  *out_result_demand = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_kernel_workgroup_reduce_result(source_op);

  loom_target_workgroup_size_t workgroup_size = {0};
  const bool has_one_dimensional_workgroup =
      loom_amdgpu_required_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &workgroup_size) &&
      workgroup_size.x != 0 && workgroup_size.y == 1 && workgroup_size.z == 1;

  if (result >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* result_value = loom_module_value(module, result);
  if (result_value->use_count == 0) {
    return iree_ok_status();
  }

  const loom_block_t* source_block = source_op->parent_block;
  loom_amdgpu_participant_zero_analysis_t participant_zero_analysis = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_participant_zero_analysis_initialize(
      context, source_block != NULL ? source_block->parent_region : NULL,
      &participant_zero_analysis));

  loom_amdgpu_collective_result_demand_t result_demand =
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_LEADER_WORKITEM;
  const loom_use_t* uses = loom_value_uses(result_value);
  for (uint32_t i = 0; i < result_value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    loom_amdgpu_participant_zero_bits_t participant_zero_bits = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_use_participant_zero_bits(
        context, &participant_zero_analysis, user_op, &participant_zero_bits));
    if (iree_any_bit_set(participant_zero_bits,
                         LOOM_AMDGPU_PARTICIPANT_ZERO_WORKITEM_X) &&
        has_one_dimensional_workgroup) {
      continue;
    }

    if (!iree_any_bit_set(participant_zero_bits,
                          LOOM_AMDGPU_PARTICIPANT_ZERO_SUBGROUP_LANE)) {
      return iree_ok_status();
    }
    result_demand = LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_SUBGROUP_LEADER_LANE;
  }

  *out_result_demand = result_demand;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_workgroup_reduce_publication_kind(
    uint32_t flat_workgroup_size, uint32_t wavefront_size,
    uint32_t register_count, loom_low_lower_context_t* context,
    const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_publication_kind_t* out_publication_kind) {
  *out_publication_kind = LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS;
  if (flat_workgroup_size <= wavefront_size) {
    return iree_ok_status();
  }

  loom_amdgpu_collective_result_demand_t result_demand =
      LOOM_AMDGPU_COLLECTIVE_RESULT_DEMAND_ALL_WORKITEMS;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_collective_result_demand(context, source_op, &result_demand));

  loom_amdgpu_workgroup_reduce_publication_rule_flags_t available_flags =
      LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_MULTI_WAVE;
  if (register_count == 1) {
    available_flags |=
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_RULE_SINGLE_REGISTER;
  }

  const loom_amdgpu_collective_result_demand_bits_t demand_bits =
      (uint8_t)(1u << (uint32_t)result_demand);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuWorkgroupReducePublicationRules); ++i) {
    const loom_amdgpu_workgroup_reduce_publication_rule_t* rule =
        &kLoomAmdgpuWorkgroupReducePublicationRules[i];
    if ((rule->demand_bits & demand_bits) == 0 ||
        (rule->required_flags & available_flags) != rule->required_flags) {
      continue;
    }
    *out_publication_kind = rule->publication_kind;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_string_view_t loom_amdgpu_workgroup_reduce_publication_report_key(
    loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuWorkgroupReducePublicationRules); ++i) {
    const loom_amdgpu_workgroup_reduce_publication_rule_t* rule =
        &kLoomAmdgpuWorkgroupReducePublicationRules[i];
    if (rule->publication_kind == publication_kind) {
      return rule->report_key;
    }
  }
  return iree_string_view_empty();
}

static bool loom_amdgpu_subgroup_reduce_dpp_row_descriptor_is_present(
    const loom_low_descriptor_set_t* descriptor_set, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuSubgroupReduceDppCombineForms); ++i) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (loom_amdgpu_collective_combine_dpp_descriptor_ref(
            kind, payload_kind, kLoomAmdgpuSubgroupReduceDppCombineForms[i],
            &descriptor_ref) &&
        loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
      return true;
    }
  }
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16) ||
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP);
}

static bool loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
    uint32_t register_count, uint32_t wavefront_size,
    uint32_t active_lane_count, bool has_cluster_attrs) {
  return register_count > 0 && wavefront_size == 64 &&
         active_lane_count == 64 && !has_cluster_attrs;
}

static iree_string_view_t loom_amdgpu_workgroup_reduce_shape_failure_key(
    loom_amdgpu_workgroup_collective_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE:
      return IREE_SV("workgroup_reduce.fixed_workgroup_size");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WAVE_COUNT:
      return IREE_SV("workgroup_reduce.wave_count");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_SCRATCH_BYTE_LENGTH:
      return IREE_SV("workgroup_reduce.scratch_byte_length");
    case LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU workgroup reduce shape failure requires reason");
  return IREE_SV("workgroup_reduce.fixed_workgroup_size");
}

iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_subgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_wavefront_size(context, &wavefront_size)) {
    return iree_ok_status();
  }
  loom_amdgpu_subgroup_reduce_shape_t shape = {0};
  loom_amdgpu_subgroup_reduce_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_reduce_resolve_shape(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), source_op, wavefront_size,
          &shape, &shape_failure)) {
    return iree_ok_status();
  }
  const uint32_t active_lane_count = shape.active_lane_count;
  const bool is_clustered = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED);
  const bool direct_width_selected = loom_amdgpu_select_direct_subgroup_width(
      context, wavefront_size, active_lane_count);
  loom_amdgpu_subgroup_reduce_publication_kind_t publication_kind =
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES;
  if (!direct_width_selected) {
    if (!loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
            register_count, wavefront_size, active_lane_count, is_clustered)) {
      return iree_ok_status();
    }
    publication_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST;
  }

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {{
      .descriptor_ref = combine_descriptor_ref,
      .out_descriptor = &out_plan->combine_descriptor,
  }};
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }
  if (publication_kind == LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        &out_plan->bpermute_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE,
        &out_plan->readlane_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
    loom_low_lower_resolved_descriptor_t unused_copy_descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        &unused_copy_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, wavefront_size, active_lane_count, kind, payload_kind,
      &out_plan->dpp_descriptor, &out_plan->dpp_combine_descriptor,
      &out_plan->permlanex16_descriptor, &out_plan->crosslane_kind));
  if (publication_kind ==
          LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST &&
      out_plan->crosslane_kind !=
          LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16) {
    return iree_ok_status();
  }

  uint32_t identity_bits = 0;
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
    const loom_amdgpu_descriptor_resolution_t guard_resolutions[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            .out_descriptor = &out_plan->guard_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            .out_descriptor = &out_plan->select_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, guard_resolutions, IREE_ARRAYSIZE(guard_resolutions),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->active_lane_count = active_lane_count;
  out_plan->identity_bits = identity_bits;
  out_plan->publication_kind = publication_kind;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_workgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_workgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_workgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_reduce_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  const loom_combining_kind_t kind =
      loom_kernel_workgroup_reduce_kind(source_op);
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_wavefront_size(context, &wavefront_size)) {
    return iree_ok_status();
  }
  const uint32_t partition_wavefront_size =
      loom_amdgpu_target_native_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_low_lower_context_target_facts(context)),
          wavefront_size);

  loom_amdgpu_workgroup_collective_shape_t shape = {0};
  loom_amdgpu_workgroup_collective_shape_failure_t shape_failure =
      LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_collective_resolve_workgroup_shape(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), partition_wavefront_size,
          register_count, &shape, &shape_failure)) {
    return iree_ok_status();
  }
  const bool is_multi_wave = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_tail = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);
  const bool needs_cross_wave_identity =
      is_multi_wave &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(shape.wave_count));
  const bool needs_subgroup_identity =
      shape.flat_workgroup_size < partition_wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(shape.flat_workgroup_size);
  const bool needs_identity_guard =
      needs_subgroup_identity || needs_cross_wave_identity;

  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
          .out_descriptor = &out_plan->bpermute_descriptor,
      },
      {
          .descriptor_ref = combine_descriptor_ref,
          .out_descriptor = &out_plan->combine_descriptor,
      },
  };
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  const uint32_t per_wave_active_lane_count =
      is_multi_wave ? partition_wavefront_size : shape.flat_workgroup_size;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, partition_wavefront_size, per_wave_active_lane_count, kind,
      payload_kind, &out_plan->dpp_descriptor,
      &out_plan->dpp_combine_descriptor, &out_plan->permlanex16_descriptor,
      &out_plan->crosslane_kind));

  uint32_t identity_bits = 0;
  if (needs_identity_guard) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }
  if (needs_identity_guard || is_multi_wave) {
    const loom_amdgpu_descriptor_resolution_t guard_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            .out_descriptor = &out_plan->guard_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, guard_resolution, IREE_ARRAYSIZE(guard_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }
  if (is_multi_wave) {
    loom_low_lower_resolved_descriptor_t unused_lane_mask_descriptor = {0};
    const loom_amdgpu_descriptor_resolution_t lane_mask_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
            .out_descriptor = &unused_lane_mask_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, lane_mask_resolution, IREE_ARRAYSIZE(lane_mask_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }
  if (has_partial_tail) {
    const loom_amdgpu_descriptor_resolution_t lane_ge_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
            .out_descriptor = &out_plan->lane_ge_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, lane_ge_resolution, IREE_ARRAYSIZE(lane_ge_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  if (needs_identity_guard) {
    const loom_amdgpu_descriptor_resolution_t select_resolution[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            .out_descriptor = &out_plan->select_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, select_resolution, IREE_ARRAYSIZE(select_resolution),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_resolve_cross_wave_descriptors(
        context, &out_plan->cross_wave, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->partition_wavefront_size = partition_wavefront_size;
  out_plan->flat_workgroup_size = shape.flat_workgroup_size;
  out_plan->identity_bits = identity_bits;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_workgroup_reduce_publication_kind(
      shape.flat_workgroup_size, partition_wavefront_size, register_count,
      context, source_op, &out_plan->publication_kind));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_permlanex16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[2];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("selector_low"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("selector_high"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &source_value, 1,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_dpp_combine_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t dpp_ctrl, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_partition_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t linear_id, uint32_t flat_workgroup_size,
    uint32_t partition_wavefront_size, loom_type_t lane_type,
    loom_value_id_t* out_lane_id) {
  *out_lane_id = LOOM_VALUE_ID_INVALID;
  if (flat_workgroup_size <= partition_wavefront_size) {
    *out_lane_id = linear_id;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, linear_id,
      partition_wavefront_size - 1, lane_type, out_lane_id);
}

static iree_status_t loom_amdgpu_emit_subgroup_xor_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  if (offset == 0) {
    *out_source_lane = lane_id;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_subgroup_reduce_first_offset(
    uint32_t active_lane_count) {
  if (active_lane_count <= 1) {
    return 0;
  }
  uint32_t offset = 1;
  while ((offset << 1) < active_lane_count) {
    offset <<= 1;
  }
  return offset;
}

static iree_status_t loom_amdgpu_emit_subgroup_select_peer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* select_descriptor,
    loom_value_id_t identity, loom_value_id_t peer, loom_value_id_t guard,
    loom_type_t lane_type, loom_value_id_t* out_selected_peer) {
  *out_selected_peer = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      identity,
      peer,
      guard,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, select_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_selected_peer = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_guard) {
  *out_guard = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_guard = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_xor_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static uint32_t loom_amdgpu_subgroup_reduce_dpp_ctrl(uint32_t lane_count) {
  switch (lane_count) {
    case 2:
      return LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1;
    case 4:
      return LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2;
    case 8:
      return LOOM_AMDGPU_DPP_CTRL_ROW_HALF_MIRROR;
    case 16:
      return LOOM_AMDGPU_DPP_CTRL_ROW_MIRROR;
    default:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU subgroup DPP reduce lowering requires a supported step");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(
    const loom_amdgpu_subgroup_reduce_plan_t* plan, uint32_t offset) {
  return plan->crosslane_kind ==
             LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16 &&
         offset == LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_cross_row_xor_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  if (precompute_step_values) {
    for (uint32_t offset = LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
         offset < plan->active_lane_count; offset <<= 1) {
      if (loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(plan,
                                                                 offset)) {
        continue;
      }
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
         offset < plan->active_lane_count; offset <<= 1) {
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      if (loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(plan,
                                                                 offset)) {
        IREE_ASSERT(plan->permlanex16_descriptor.descriptor != NULL);
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_permlanex16_register(
            context, source_op, &plan->permlanex16_descriptor, accumulator,
            lane_type, &peer));
      } else {
        loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
        if (precompute_step_values) {
          low_source_byte_offset = source_byte_offsets[step_index++];
        } else {
          loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
              context, source_op, lane_id, offset, lane_type, &source_lane));
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
              context, source_op, source_lane, lane_type,
              &low_source_byte_offset));
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor,
            low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
            lane_type, &peer));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_dpp_row_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    const uint32_t row_lane_count =
        iree_min(plan->active_lane_count, LOOM_AMDGPU_DPP_ROW_LANE_COUNT);
    for (uint32_t lane_count = 2; lane_count <= row_lane_count;
         lane_count <<= 1) {
      const uint32_t dpp_ctrl =
          loom_amdgpu_subgroup_reduce_dpp_ctrl(lane_count);
      if (plan->dpp_combine_descriptor.descriptor != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_dpp_combine_register(
            context, source_op, &plan->dpp_combine_descriptor, accumulator,
            accumulator, dpp_ctrl, lane_type, &accumulator));
        continue;
      }
      IREE_ASSERT(plan->dpp_descriptor.descriptor != NULL);
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
          context, source_op, &plan->dpp_descriptor, LOOM_AMDGPU_CROSSLANE_DPP,
          accumulator, dpp_ctrl, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return loom_amdgpu_emit_subgroup_reduce_cross_row_xor_tree(
      context, source_op, plan, lane_id, lane_type, inout_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_add_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_down_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers) {
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t active_lane_count = dynamic_active_lane_count;
  if (active_lane_count == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->active_lane_count, lane_type, &active_lane_count));
  }
  loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      plan->identity_bits, lane_type, &identity));
  loom_value_id_t first_lane_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &first_lane_offset));

  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, source_lane,
          active_lane_count, mask_type, &guards[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      loom_value_id_t guard = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
        guard = guards[step_index - 1];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
            context, source_op, &plan->guard_descriptor, source_lane,
            active_lane_count, mask_type, &guard));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
          context, source_op, &plan->select_descriptor, identity, peer, guard,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->bpermute_descriptor, first_lane_offset,
        /*static_byte_offset=*/0, accumulator, lane_type, &inout_registers[i]));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers) {
  if (dynamic_active_lane_count == LOOM_VALUE_ID_INVALID &&
      (plan->active_lane_count <= 1 ||
       loom_amdgpu_u32_is_power_of_two(plan->active_lane_count))) {
    if ((plan->crosslane_kind ==
             LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE ||
         plan->crosslane_kind ==
             LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16) &&
        loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(
            plan->wavefront_size, plan->active_lane_count)) {
      return loom_amdgpu_emit_subgroup_reduce_dpp_row_tree(
          context, source_op, plan, lane_id, lane_type, inout_registers);
    }
    return loom_amdgpu_emit_subgroup_reduce_xor_tree(
        context, source_op, plan, lane_id, lane_type, inout_registers);
  }
  return loom_amdgpu_emit_subgroup_reduce_down_tree(
      context, source_op, plan, lane_id, lane_type, dynamic_active_lane_count,
      inout_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_scalar_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  IREE_ASSERT_EQ(plan->wavefront_size, 64);
  IREE_ASSERT_EQ(plan->active_lane_count, 64);
  IREE_ASSERT(plan->readlane_descriptor.descriptor != NULL);

  loom_amdgpu_subgroup_reduce_plan_t half_wave_plan = *plan;
  half_wave_plan.active_lane_count = plan->wavefront_size / 2u;
  half_wave_plan.publication_kind =
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, &half_wave_plan, lane_id, lane_type,
      LOOM_VALUE_ID_INVALID, inout_registers));

  // Lane 0 holds each low-half result and lane 63 receives the full result
  // after combining it with the corresponding high-half result.
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_half = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_register(
        context, source_op, &plan->readlane_descriptor, inout_registers[i], 0,
        sgpr_type, &low_half));

    loom_value_id_t full_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
        context, source_op, &plan->combine_descriptor, low_half,
        inout_registers[i], lane_type, &full_result));

    loom_value_id_t full_result_scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_register(
        context, source_op, &plan->readlane_descriptor, full_result, 63,
        sgpr_type, &full_result_scalar));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, full_result_scalar, &inout_registers[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &result_registers[i]));
  }
  if (plan->publication_kind ==
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_SCALAR_BROADCAST) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_scalar_broadcast(
        context, source_op, plan, lane_id, lane_type, result_registers));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, plan, lane_id, lane_type, LOOM_VALUE_ID_INVALID,
        result_registers));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, loom_value_id_t dynamic_byte_offset,
    uint32_t static_byte_offset, loom_type_t lane_type,
    loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, scratch_base,
      dynamic_byte_offset, lane_type, out_address));
  if (static_byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      *out_address, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t scratch_base, uint32_t static_byte_offset,
    loom_type_t lane_type, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  if (static_byte_offset == 0) {
    *out_address = scratch_base;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      scratch_base, static_byte_offset, lane_type, out_address);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_write(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_value_id_t value) {
  const loom_value_id_t operands[] = {
      address,
      value,
  };
  loom_op_t* write_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.lds_write_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &write_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_scratch_read(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t address,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {address};
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.lds_read_descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &lane_type,
      1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &read_op));
  *out_value = loom_value_slice_get(loom_low_op_results(read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  return loom_amdgpu_collective_emit_cross_wave_barrier(context, source_op,
                                                        &plan->cross_wave);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t guard,
    loom_type_t mask_type, loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {mask_type, active_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.saveexec_descriptor, &guard, 1,
      loom_make_named_attr_slice(NULL, 0), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &saveexec_op));
  *out_saved_exec = loom_op_const_results(saveexec_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t saved_exec) {
  loom_op_t* restore_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->cross_wave.restore_exec_descriptor, &saved_exec, 1,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &restore_op);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan, loom_value_id_t linear_id,
    loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* out_leader_guard) {
  *out_leader_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t leader_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
      &leader_count));
  return loom_amdgpu_emit_subgroup_lane_compare(
      context, source_op, &plan->guard_descriptor, linear_id, leader_count,
      mask_type, out_leader_guard);
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_serial_scratch_fold(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t scratch_base, loom_value_id_t linear_id,
    uint32_t wave_count, loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* inout_registers) {
  loom_value_id_t leader_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
      context, source_op, plan, linear_id, lane_type, mask_type,
      &leader_guard));
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
      context, source_op, plan, leader_guard, mask_type, &saved_exec));

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
        context, source_op, plan, address, lane_type, &accumulator));
    for (uint32_t wave_ordinal = 1; wave_ordinal < wave_count; ++wave_ordinal) {
      const uint32_t scratch_byte_offset =
          register_byte_offset + wave_ordinal * 4u;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
              context, source_op, scratch_base, scratch_byte_offset, lane_type,
              &address));
      loom_value_id_t partial = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &partial));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, partial,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return loom_amdgpu_emit_workgroup_reduce_restore_exec(context, source_op,
                                                        plan, saved_exec);
}

static iree_status_t
loom_amdgpu_emit_workgroup_reduce_serial_scratch_publication(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan,
    loom_value_id_t scratch_base, loom_value_id_t linear_id,
    uint32_t wave_count, loom_type_t lane_type, loom_type_t mask_type,
    loom_value_id_t* inout_registers) {
  loom_value_id_t leader_guard = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_linear_leader_guard(
      context, source_op, plan, linear_id, lane_type, mask_type,
      &leader_guard));
  loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
      context, source_op, plan, leader_guard, mask_type, &saved_exec));
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
        context, source_op, plan, address, inout_registers[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
      context, source_op, plan, saved_exec));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint32_t register_byte_offset = i * wave_count * 4u;
    loom_value_id_t address = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
            context, source_op, scratch_base, register_byte_offset, lane_type,
            &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
        context, source_op, plan, address, lane_type, &inout_registers[i]));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  if (plan->flat_workgroup_size <= plan->partition_wavefront_size) {
    const loom_amdgpu_subgroup_reduce_plan_t subgroup_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .dpp_combine_descriptor = plan->dpp_combine_descriptor,
        .permlanex16_descriptor = plan->permlanex16_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = plan->register_count,
        .wavefront_size = plan->partition_wavefront_size,
        .active_lane_count = plan->flat_workgroup_size,
        .identity_bits = plan->identity_bits,
        .crosslane_kind = plan->crosslane_kind,
    };
    return loom_amdgpu_lower_kernel_subgroup_reduce(context, source_op,
                                                    &subgroup_plan);
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t linear_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_workitem_linear_id(
      context, source_op, lane_type, &linear_id));
  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_partition_lane_id(
      context, source_op, linear_id, plan->flat_workgroup_size,
      plan->partition_wavefront_size, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t flat_workgroup_size = plan->flat_workgroup_size;
  const uint32_t partition_wavefront_size = plan->partition_wavefront_size;
  const bool has_partial_tail =
      (flat_workgroup_size % partition_wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + partition_wavefront_size - 1) /
      partition_wavefront_size;
  const uint32_t tail_lane_count =
      flat_workgroup_size % partition_wavefront_size;
  const uint32_t scratch_slot_count = wave_count;
  const int64_t scratch_byte_length =
      (int64_t)((uint64_t)scratch_slot_count * register_count * 4u);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, scratch_byte_length, /*byte_alignment=*/4,
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), source_op->location,
      &storage_op));
  loom_op_t* storage_address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      lane_type, source_op->location, &storage_address_op));
  const loom_value_id_t scratch_base =
      loom_low_storage_address_result(storage_address_op);

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t dynamic_active_lane_count = LOOM_VALUE_ID_INVALID;
  if (has_partial_tail) {
    loom_value_id_t tail_wave_first_linear_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        (wave_count - 1) * partition_wavefront_size, lane_type,
        &tail_wave_first_linear_id));
    loom_value_id_t tail_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->lane_ge_descriptor, linear_id,
        tail_wave_first_linear_id, mask_type, &tail_wave_guard));

    loom_value_id_t full_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        partition_wavefront_size, lane_type, &full_lane_count_value));
    loom_value_id_t tail_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        tail_lane_count, lane_type, &tail_lane_count_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
        context, source_op, &plan->select_descriptor, full_lane_count_value,
        tail_lane_count_value, tail_wave_guard, lane_type,
        &dynamic_active_lane_count));
  }

  {
    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < register_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
          context, source_op, register_count, low_value, i, lane_type,
          &result_registers[i]));
    }

    const loom_amdgpu_subgroup_reduce_plan_t per_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .dpp_combine_descriptor = plan->dpp_combine_descriptor,
        .permlanex16_descriptor = plan->permlanex16_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = register_count,
        .wavefront_size = partition_wavefront_size,
        .active_lane_count = partition_wavefront_size,
        .identity_bits = plan->identity_bits,
        .crosslane_kind = plan->crosslane_kind,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &per_wave_plan, lane_id, lane_type,
        dynamic_active_lane_count, result_registers));

    loom_value_id_t subgroup_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_subgroup_u32_log2(partition_wavefront_size), linear_id,
        lane_type, &subgroup_id));
    loom_value_id_t subgroup_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_vgpr_scale_u32(context, source_op, subgroup_id, 4,
                                        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE,
                                        lane_type, &subgroup_byte_offset));

    loom_value_id_t lane_zero_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
        &lane_zero_count));
    loom_value_id_t lane_zero_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, lane_id, lane_zero_count,
        mask_type, &lane_zero_guard));
    loom_value_id_t saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type, &saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, subgroup_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));

    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    // RDNA wave64 workgroups execute as native 32-lane partitions. Fold the
    // partition partials through LDS under linear workitem 0 instead of forming
    // a synthetic cross-lane subgroup from partition leaders.
    if (plan->wavefront_size > partition_wavefront_size) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_serial_scratch_fold(
              context, source_op, plan, scratch_base, linear_id, wave_count,
              lane_type, mask_type, result_registers));
      if (plan->publication_kind ==
          LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM) {
        return loom_amdgpu_collective_bind_payload_result(
            context, source_op, plan->result, register_count, result_registers);
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_serial_scratch_publication(
              context, source_op, plan, scratch_base, linear_id, wave_count,
              lane_type, mask_type, result_registers));
      return loom_amdgpu_collective_bind_payload_result(
          context, source_op, plan->result, register_count, result_registers);
    }

    if (plan->publication_kind ==
            LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP ||
        plan->publication_kind ==
            LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE) {
      loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
          lane_type, &wave_count_value));
      loom_value_id_t wave_participant_guard = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, lane_id,
          wave_count_value, mask_type, &wave_participant_guard));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
          context, source_op, plan, wave_participant_guard, mask_type,
          &saved_exec));

      loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, lane_id, lane_type, &lane_byte_offset));
      for (uint32_t i = 0; i < register_count; ++i) {
        const uint32_t register_byte_offset = i * wave_count * 4u;
        loom_value_id_t address = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
            context, source_op, scratch_base, lane_byte_offset,
            register_byte_offset, lane_type, &address));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
            context, source_op, plan, address, lane_type,
            &result_registers[i]));
      }

      const loom_amdgpu_subgroup_reduce_plan_t cross_wave_plan = {
          .bpermute_descriptor = plan->bpermute_descriptor,
          .combine_descriptor = plan->combine_descriptor,
          .guard_descriptor = plan->guard_descriptor,
          .select_descriptor = plan->select_descriptor,
          .value = plan->value,
          .result = plan->result,
          .payload_kind = plan->payload_kind,
          .register_count = register_count,
          .wavefront_size = partition_wavefront_size,
          .active_lane_count = wave_count,
          .identity_bits = plan->identity_bits,
          .crosslane_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE,
      };
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
          context, source_op, &cross_wave_plan, lane_id, lane_type,
          LOOM_VALUE_ID_INVALID, result_registers));

      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
          context, source_op, plan, saved_exec));

      if (plan->publication_kind ==
          LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE) {
        return loom_amdgpu_collective_bind_payload_result(
            context, source_op, plan->result, register_count, result_registers);
      }

      loom_value_id_t first_lane_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &first_lane_offset));
      for (uint32_t i = 0; i < register_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor, first_lane_offset,
            /*static_byte_offset=*/0, result_registers[i], lane_type,
            &result_registers[i]));
      }

      return loom_amdgpu_collective_bind_payload_result(
          context, source_op, plan->result, register_count, result_registers);
    }

    loom_value_id_t wave_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, wave_count,
        lane_type, &wave_count_value));
    loom_value_id_t first_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->guard_descriptor, linear_id,
        wave_count_value, mask_type, &first_wave_guard));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, first_wave_guard, mask_type, &saved_exec));

    loom_value_id_t lane_byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
        context, source_op, lane_id, lane_type, &lane_byte_offset));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    const loom_amdgpu_subgroup_reduce_plan_t cross_wave_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = register_count,
        .wavefront_size = partition_wavefront_size,
        .active_lane_count = wave_count,
        .identity_bits = plan->identity_bits,
        .crosslane_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &cross_wave_plan, lane_id, lane_type,
        LOOM_VALUE_ID_INVALID, result_registers));

    if (plan->publication_kind ==
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
          context, source_op, plan, saved_exec));
      return loom_amdgpu_collective_bind_payload_result(
          context, source_op, plan->result, register_count, result_registers);
    }

    loom_value_id_t publish_saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_saveexec(
        context, source_op, plan, lane_zero_guard, mask_type,
        &publish_saved_exec));
    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_address(
          context, source_op, scratch_base, lane_byte_offset,
          register_byte_offset, lane_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_write(
          context, source_op, plan, address, result_registers[i]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, publish_saved_exec));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_restore_exec(
        context, source_op, plan, saved_exec));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_workgroup_reduce_barrier(context, source_op, plan));

    for (uint32_t i = 0; i < register_count; ++i) {
      const uint32_t register_byte_offset = i * wave_count * 4u;
      loom_value_id_t address = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_workgroup_reduce_static_scratch_address(
              context, source_op, scratch_base, register_byte_offset, lane_type,
              &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_reduce_scratch_read(
          context, source_op, plan, address, lane_type, &result_registers[i]));
    }

    return loom_amdgpu_collective_bind_payload_result(
        context, source_op, plan->result, register_count, result_registers);
  }
}
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);

  const loom_value_id_t value = loom_kernel_subgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.combining_kind"));
  }

  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.wavefront_size"));
  }
  loom_amdgpu_subgroup_reduce_shape_t shape = {0};
  loom_amdgpu_subgroup_reduce_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_reduce_resolve_shape(
          module, loom_target_low_legality_function(context), bundle, op,
          wavefront_size, &shape, &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_subgroup_reduce_shape_failure_key(shape_failure));
  }
  const uint32_t active_lane_count = shape.active_lane_count;
  const bool is_clustered = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_SUBGROUP_REDUCE_SHAPE_CLUSTERED);
  const bool direct_width_supported =
      loom_amdgpu_target_supports_direct_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          wavefront_size, active_lane_count);
  bool scalar_broadcast_supported = false;
  if (!direct_width_supported) {
    scalar_broadcast_supported =
        loom_amdgpu_subgroup_reduce_scalar_broadcast_is_supported(
            unused_register_count, wavefront_size, active_lane_count,
            is_clustered);
    if (!scalar_broadcast_supported) {
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("subgroup_reduce.native_width.scalar_broadcast"));
    }
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("subgroup_reduce.identity"));
    }
  }

  if (scalar_broadcast_supported) {
    const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.reduce_combine"),
            .descriptor_ref = combine_descriptor_ref,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_readlane_b32"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_permlanex16_b32"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_PERMLANEX16_B32_SRC12_INLINE,
        },
    };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, requirements, IREE_ARRAYSIZE(requirements)));
    if (!loom_amdgpu_subgroup_reduce_dpp_row_descriptor_is_present(
            loom_target_low_legality_descriptor_set(context), kind,
            payload_kind)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("descriptor.reduce_dpp_row"));
    }
  } else {
    const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.reduce_combine"),
            .descriptor_ref = combine_descriptor_ref,
        },
    };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, requirements, IREE_ARRAYSIZE(requirements)));
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    static const loom_amdgpu_low_legality_descriptor_requirement_t
        active_lane_requirements[] = {
            {
                .constraint_key = IREE_SVL("descriptor.v_cmp_ult_u32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.v_cndmask_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
            },
        };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, active_lane_requirements,
            IREE_ARRAYSIZE(active_lane_requirements)));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_workgroup_reduce_value(op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("workgroup_reduce.payload"));
  }

  loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_combining_kind_t kind = loom_kernel_workgroup_reduce_kind(op);
  if (!loom_amdgpu_collective_combine_descriptor_ref(kind, payload_kind,
                                                     &combine_descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.combining_kind"));
  }

  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wavefront_size"));
  }
  const uint32_t partition_wavefront_size =
      loom_amdgpu_target_native_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          wavefront_size);

  loom_amdgpu_workgroup_collective_shape_t shape = {0};
  loom_amdgpu_workgroup_collective_shape_failure_t shape_failure =
      LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_collective_resolve_workgroup_shape(
          module, loom_target_low_legality_function(context), bundle,
          partition_wavefront_size, register_count, &shape, &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_workgroup_reduce_shape_failure_key(shape_failure));
  }
  const bool is_multi_wave = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE);
  const bool has_partial_tail = iree_all_bits_set(
      shape.flags, LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL);
  const bool needs_cross_wave_identity =
      is_multi_wave &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(shape.wave_count));
  const bool needs_subgroup_identity =
      shape.flat_workgroup_size < partition_wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(shape.flat_workgroup_size);
  const bool needs_identity_guard =
      needs_subgroup_identity || needs_cross_wave_identity;
  if (needs_identity_guard) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("workgroup_reduce.identity"));
    }
  }

  const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.ds_bpermute_b32"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      },
      {
          .constraint_key = IREE_SVL("descriptor.combine"),
          .descriptor_ref = combine_descriptor_ref,
      },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
  if (needs_identity_guard || is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        IREE_SV("descriptor.v_cmp_ult_u32")));
  }
  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        IREE_SV("descriptor.v_and_b32_lit")));
  }
  if (has_partial_tail) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
        IREE_SV("descriptor.v_cmp_uge_u32")));
  }
  if (needs_identity_guard) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        IREE_SV("descriptor.v_cndmask_b32")));
  }
  if (is_multi_wave) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_collective_verify_cross_wave_descriptor_requirements(
            context, op));
  }

  return iree_ok_status();
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
