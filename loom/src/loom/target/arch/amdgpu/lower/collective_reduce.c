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

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u
#define LOOM_AMDGPU_DPP_ROW_LANE_COUNT 16u
#define LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1 UINT32_C(0x0B1)
#define LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2 UINT32_C(0x04E)
#define LOOM_AMDGPU_DPP_CTRL_ROW_HALF_MIRROR UINT32_C(0x141)
#define LOOM_AMDGPU_DPP_CTRL_ROW_MIRROR UINT32_C(0x140)

static bool loom_amdgpu_subgroup_wavefront_size_is_supported(
    uint32_t wavefront_size) {
  return wavefront_size == 32 || wavefront_size == 64;
}

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

static bool loom_amdgpu_subgroup_reduce_active_lane_count(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size,
    uint32_t* out_active_lane_count) {
  *out_active_lane_count = 0;
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return false;
  }
  if (flat_workgroup_size <= wavefront_size) {
    *out_active_lane_count = flat_workgroup_size;
    return true;
  }
  if ((flat_workgroup_size % wavefront_size) != 0) {
    return false;
  }
  *out_active_lane_count = wavefront_size;
  return true;
}

static bool loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(
    uint32_t wavefront_size, uint32_t active_lane_count) {
  // Keep the DPP row strategy on wave32. Wave64 targets use the lane-addressed
  // bpermute tree so full-wave reductions do not depend on family-specific DPP
  // row-control behavior.
  return active_lane_count == wavefront_size && wavefront_size == 32;
}

static iree_status_t loom_amdgpu_select_subgroup_reduce_crosslane_kind(
    loom_low_lower_context_t* context, uint32_t wavefront_size,
    uint32_t active_lane_count, loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_amdgpu_subgroup_reduce_crosslane_kind_t* out_crosslane_kind) {
  *dpp_move = (loom_low_lower_resolved_descriptor_t){0};
  *out_crosslane_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE;
  if (!loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(wavefront_size,
                                                         active_lane_count)) {
    return iree_ok_status();
  }

  bool dpp_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16, dpp_move,
      &dpp_descriptor_present));
  if (!dpp_descriptor_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP, dpp_move,
        &dpp_descriptor_present));
  }
  if (dpp_descriptor_present) {
    *out_crosslane_kind =
        LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_value_is_workitem_x(const loom_module_t* module,
                                            loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op != NULL && loom_kernel_workitem_id_isa(defining_op) &&
         loom_kernel_workitem_id_dimension(defining_op) ==
             LOOM_KERNEL_DIMENSION_X;
}

static bool loom_amdgpu_value_is_subgroup_lane_id(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op != NULL && loom_kernel_subgroup_lane_id_isa(defining_op);
}

typedef enum loom_amdgpu_participant_id_kind_e {
  LOOM_AMDGPU_PARTICIPANT_ID_WORKITEM_X = 0,
  LOOM_AMDGPU_PARTICIPANT_ID_SUBGROUP_LANE = 1,
} loom_amdgpu_participant_id_kind_t;

static bool loom_amdgpu_value_is_participant_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_participant_id_kind_t participant_id_kind) {
  switch (participant_id_kind) {
    case LOOM_AMDGPU_PARTICIPANT_ID_WORKITEM_X:
      return loom_amdgpu_value_is_workitem_x(module, value_id);
    case LOOM_AMDGPU_PARTICIPANT_ID_SUBGROUP_LANE:
      return loom_amdgpu_value_is_subgroup_lane_id(module, value_id);
    default:
      return false;
  }
}

static bool loom_amdgpu_symbolic_expr_is_participant_id(
    const loom_module_t* module, const loom_symbolic_expr_t* expression,
    loom_amdgpu_participant_id_kind_t participant_id_kind) {
  return loom_symbolic_expr_is_linear(expression) &&
         expression->constant == 0 && expression->term_count == 1 &&
         expression->terms[0].coefficient == 1 &&
         (loom_amdgpu_value_is_participant_id(
              module, expression->terms[0].value_id, participant_id_kind) ||
          loom_amdgpu_value_is_participant_id(
              module, expression->terms[0].relation_value_id,
              participant_id_kind));
}

static bool loom_amdgpu_symbolic_expr_is_zero(
    const loom_symbolic_expr_t* expression) {
  return loom_symbolic_expr_is_constant(expression) &&
         expression->constant == 0;
}

static iree_status_t loom_amdgpu_condition_implies_participant_zero(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition,
    loom_amdgpu_participant_id_kind_t participant_id_kind,
    bool* out_implies_participant_zero) {
  *out_implies_participant_zero = false;

  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t condition_facts = {0};
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  if (!loom_condition_facts_query(module, fact_table, condition,
                                  /*assumed_truth=*/true, &condition_facts)) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_symbolic_expr_context_t symbolic_context = {0};
  loom_symbolic_expr_context_initialize(module, fact_table, &arena,
                                        &symbolic_context);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < condition_facts.integer_relation_count && iree_status_is_ok(status);
       ++i) {
    const loom_condition_integer_relation_t* relation =
        &condition_facts.integer_relations[i];
    if (relation->relation != LOOM_SYMBOLIC_INTEGER_RELATION_EQ ||
        relation->left.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
        relation->right.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
      continue;
    }

    loom_symbolic_expr_t left_expression = {0};
    status = loom_symbolic_expr_from_value(
        &symbolic_context, relation->left.value_id, &left_expression);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_symbolic_expr_t right_expression = {0};
    status = loom_symbolic_expr_from_value(
        &symbolic_context, relation->right.value_id, &right_expression);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if ((loom_amdgpu_symbolic_expr_is_participant_id(module, &left_expression,
                                                     participant_id_kind) &&
         loom_amdgpu_symbolic_expr_is_zero(&right_expression)) ||
        (loom_amdgpu_symbolic_expr_is_participant_id(module, &right_expression,
                                                     participant_id_kind) &&
         loom_amdgpu_symbolic_expr_is_zero(&left_expression))) {
      *out_implies_participant_zero = true;
      break;
    }
  }

  iree_arena_deinitialize(&arena);
  return status;
}

static const loom_op_t* loom_amdgpu_use_enclosing_resultless_then_if(
    const loom_op_t* use_op) {
  for (const loom_op_t* current = use_op; current != NULL;
       current = current->parent_op) {
    const loom_op_t* parent_op = current->parent_op;
    if (parent_op == NULL || !loom_scf_if_isa(parent_op) ||
        parent_op->result_count != 0) {
      continue;
    }
    if (current->parent_block != NULL &&
        current->parent_block->parent_region ==
            loom_scf_if_then_region(parent_op)) {
      return parent_op;
    }
  }
  return NULL;
}

static loom_value_id_t loom_amdgpu_use_cfg_true_guard_condition(
    const loom_op_t* use_op) {
  const loom_block_t* block = use_op != NULL ? use_op->parent_block : NULL;
  const loom_region_t* region = block != NULL ? block->parent_region : NULL;
  if (region == NULL || block == loom_region_const_entry_block(region)) {
    return LOOM_VALUE_ID_INVALID;
  }

  loom_value_id_t guard_condition = LOOM_VALUE_ID_INVALID;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* predecessor =
        loom_region_const_block(region, block_index);
    if (predecessor == NULL || predecessor->op_count == 0) {
      continue;
    }

    const loom_op_t* terminator = loom_block_const_last_op(predecessor);
    if (loom_cfg_br_isa(terminator)) {
      if (loom_cfg_br_dest(terminator) == block) {
        return LOOM_VALUE_ID_INVALID;
      }
      continue;
    }
    if (!loom_cfg_cond_br_isa(terminator)) {
      continue;
    }

    const bool true_edge_reaches_block =
        loom_cfg_cond_br_true_dest(terminator) == block;
    const bool false_edge_reaches_block =
        loom_cfg_cond_br_false_dest(terminator) == block;
    if (!true_edge_reaches_block && !false_edge_reaches_block) {
      continue;
    }
    if (!true_edge_reaches_block || false_edge_reaches_block ||
        guard_condition != LOOM_VALUE_ID_INVALID) {
      return LOOM_VALUE_ID_INVALID;
    }
    guard_condition = loom_cfg_cond_br_condition(terminator);
  }
  return guard_condition;
}

typedef enum loom_amdgpu_workgroup_reduce_result_demand_e {
  LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_ALL_WORKITEMS = 0,
  LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_LEADER_WORKITEM = 1,
  LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_SUBGROUP_LEADER_LANE = 2,
} loom_amdgpu_workgroup_reduce_result_demand_t;

static iree_status_t loom_amdgpu_workgroup_reduce_result_demand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_result_demand_t* out_result_demand) {
  *out_result_demand = LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_ALL_WORKITEMS;

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_target_workgroup_size_t workgroup_size = {0};
  const bool has_one_dimensional_workgroup =
      loom_amdgpu_required_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &workgroup_size) &&
      workgroup_size.x != 0 && workgroup_size.y == 1 && workgroup_size.z == 1;

  if (module == NULL) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_kernel_workgroup_reduce_result(source_op);
  if (result >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* result_value = loom_module_value(module, result);
  if (result_value->use_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_workgroup_reduce_result_demand_t result_demand =
      LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_LEADER_WORKITEM;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_use_t* uses = loom_value_uses(result_value);
  for (uint32_t i = 0; i < result_value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    loom_value_id_t guard_condition = LOOM_VALUE_ID_INVALID;
    const loom_op_t* guard_op =
        loom_amdgpu_use_enclosing_resultless_then_if(user_op);
    if (guard_op != NULL) {
      guard_condition = loom_scf_if_condition(guard_op);
    } else {
      guard_condition = loom_amdgpu_use_cfg_true_guard_condition(user_op);
    }
    if (guard_condition == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }

    bool guard_is_leader_workitem = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_condition_implies_participant_zero(
        module, fact_table, guard_condition,
        LOOM_AMDGPU_PARTICIPANT_ID_WORKITEM_X, &guard_is_leader_workitem));
    if (guard_is_leader_workitem && has_one_dimensional_workgroup) {
      continue;
    }

    bool guard_is_subgroup_leader_lane = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_condition_implies_participant_zero(
        module, fact_table, guard_condition,
        LOOM_AMDGPU_PARTICIPANT_ID_SUBGROUP_LANE,
        &guard_is_subgroup_leader_lane));
    if (!guard_is_subgroup_leader_lane) {
      return iree_ok_status();
    }
    result_demand =
        LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_SUBGROUP_LEADER_LANE;
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

  loom_amdgpu_workgroup_reduce_result_demand_t result_demand =
      LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_ALL_WORKITEMS;
  IREE_RETURN_IF_ERROR(loom_amdgpu_workgroup_reduce_result_demand(
      context, source_op, &result_demand));
  if (result_demand ==
      LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_LEADER_WORKITEM) {
    *out_publication_kind =
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM;
    return iree_ok_status();
  }
  if (register_count == 1) {
    if (result_demand ==
        LOOM_AMDGPU_WORKGROUP_REDUCE_RESULT_DEMAND_SUBGROUP_LEADER_LANE) {
      *out_publication_kind =
          LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE;
      return iree_ok_status();
    }
    *out_publication_kind =
        LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_reduce_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_reduce_isa(source_op)) {
    return iree_ok_status();
  }
  if (loom_amdgpu_subgroup_reduce_has_cluster_attrs(source_op)) {
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }
  uint32_t active_lane_count = 0;
  if (!loom_amdgpu_subgroup_reduce_active_lane_count(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), wavefront_size,
          &active_lane_count)) {
    return iree_ok_status();
  }

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

  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, wavefront_size, active_lane_count, &out_plan->dpp_descriptor,
      &out_plan->crosslane_kind));

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
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context), &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return iree_ok_status();
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return iree_ok_status();
  }
  const bool needs_cross_wave_identity =
      flat_workgroup_size > wavefront_size &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(wave_count));
  const bool needs_subgroup_identity =
      flat_workgroup_size < wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(flat_workgroup_size);
  const bool needs_identity_guard =
      needs_subgroup_identity || needs_cross_wave_identity;
  const uint32_t scratch_slot_count = wave_count;
  const uint64_t scratch_byte_length =
      (uint64_t)scratch_slot_count * register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
    return iree_ok_status();
  }

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
      flat_workgroup_size > wavefront_size ? wavefront_size
                                           : flat_workgroup_size;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_reduce_crosslane_kind(
      context, wavefront_size, per_wave_active_lane_count,
      &out_plan->dpp_descriptor, &out_plan->crosslane_kind));

  uint32_t identity_bits = 0;
  if (needs_identity_guard) {
    if (!loom_amdgpu_collective_combine_identity_bits(kind, &identity_bits)) {
      return iree_ok_status();
    }
  }
  if (needs_identity_guard || flat_workgroup_size > wavefront_size) {
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

  if (flat_workgroup_size > wavefront_size) {
    const loom_amdgpu_descriptor_resolution_t scratch_resolutions[] = {
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
            .out_descriptor = &out_plan->lds_read_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
            .out_descriptor = &out_plan->lds_write_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
            .out_descriptor = &out_plan->barrier_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
            .out_descriptor = &out_plan->saveexec_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
            .out_descriptor = &out_plan->restore_exec_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, scratch_resolutions, IREE_ARRAYSIZE(scratch_resolutions),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_workgroup_reduce_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->wavefront_size = wavefront_size;
  out_plan->flat_workgroup_size = flat_workgroup_size;
  out_plan->identity_bits = identity_bits;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_workgroup_reduce_publication_kind(
      flat_workgroup_size, wavefront_size, register_count, context, source_op,
      &out_plan->publication_kind));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_bpermute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_byte_offset, loom_value_id_t low_source_value,
    loom_type_t lane_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_byte_offset,
      low_source_value,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_dpp_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_value, uint32_t dpp_ctrl, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &low_source_value, 1,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane, loom_type_t lane_type,
    loom_value_id_t* out_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2, lane,
      lane_type, out_byte_offset);
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
          low_source_byte_offset, accumulator, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_subgroup_reduce_dpp_ctrl(
    uint32_t lane_count, uint32_t* out_dpp_ctrl) {
  *out_dpp_ctrl = 0;
  switch (lane_count) {
    case 2:
      *out_dpp_ctrl = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1;
      return iree_ok_status();
    case 4:
      *out_dpp_ctrl = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2;
      return iree_ok_status();
    case 8:
      *out_dpp_ctrl = LOOM_AMDGPU_DPP_CTRL_ROW_HALF_MIRROR;
      return iree_ok_status();
    case 16:
      *out_dpp_ctrl = LOOM_AMDGPU_DPP_CTRL_ROW_MIRROR;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU subgroup DPP reduce step %u",
                              lane_count);
  }
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
          low_source_byte_offset, accumulator, lane_type, &peer));
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
    for (uint32_t lane_count = 2; lane_count <= LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
         lane_count <<= 1) {
      uint32_t dpp_ctrl = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_subgroup_reduce_dpp_ctrl(lane_count, &dpp_ctrl));
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_dpp_register(
          context, source_op, &plan->dpp_descriptor, accumulator, dpp_ctrl,
          lane_type, &peer));
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
          low_source_byte_offset, accumulator, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select_peer(
          context, source_op, &plan->select_descriptor, identity, peer, guard,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->bpermute_descriptor, first_lane_offset,
        accumulator, lane_type, &inout_registers[i]));
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
    if (plan->crosslane_kind ==
            LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE &&
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, plan, lane_id, lane_type, LOOM_VALUE_ID_INVALID,
      result_registers));

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
      context, &plan->lds_write_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
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
      context, &plan->lds_read_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &read_op));
  *out_value = loom_value_slice_get(loom_low_op_results(read_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_reduce_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  loom_op_t* barrier_op = NULL;
  return loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->barrier_descriptor, /*operands=*/NULL,
      /*operand_count=*/0, loom_make_named_attr_slice(NULL, 0),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &barrier_op);
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
      context, &plan->saveexec_descriptor, &guard, 1,
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
      context, &plan->restore_exec_descriptor, &saved_exec, 1,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &restore_op);
}

iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  if (plan->flat_workgroup_size <= plan->wavefront_size) {
    const loom_amdgpu_subgroup_reduce_plan_t subgroup_plan = {
        .bpermute_descriptor = plan->bpermute_descriptor,
        .dpp_descriptor = plan->dpp_descriptor,
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = plan->register_count,
        .wavefront_size = plan->wavefront_size,
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  const uint32_t register_count = plan->register_count;
  const uint32_t flat_workgroup_size = plan->flat_workgroup_size;
  const uint32_t wavefront_size = plan->wavefront_size;
  const bool has_partial_tail = (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  const uint32_t tail_lane_count = flat_workgroup_size % wavefront_size;
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
        (wave_count - 1) * wavefront_size, lane_type,
        &tail_wave_first_linear_id));
    loom_value_id_t tail_wave_guard = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
        context, source_op, &plan->lane_ge_descriptor, linear_id,
        tail_wave_first_linear_id, mask_type, &tail_wave_guard));

    loom_value_id_t full_lane_count_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        wavefront_size, lane_type, &full_lane_count_value));
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
        .combine_descriptor = plan->combine_descriptor,
        .guard_descriptor = plan->guard_descriptor,
        .select_descriptor = plan->select_descriptor,
        .value = plan->value,
        .result = plan->result,
        .payload_kind = plan->payload_kind,
        .register_count = register_count,
        .wavefront_size = wavefront_size,
        .active_lane_count = wavefront_size,
        .identity_bits = plan->identity_bits,
        .crosslane_kind = plan->crosslane_kind,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
        context, source_op, &per_wave_plan, lane_id, lane_type,
        dynamic_active_lane_count, result_registers));

    loom_value_id_t subgroup_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        loom_amdgpu_subgroup_u32_log2(wavefront_size), linear_id, lane_type,
        &subgroup_id));
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
          .wavefront_size = wavefront_size,
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
            result_registers[i], lane_type, &result_registers[i]));
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
        .wavefront_size = wavefront_size,
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
static iree_status_t loom_amdgpu_low_legality_verify_subgroup_wavefront(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, out_wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(*out_wavefront_size)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
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
  if (loom_amdgpu_subgroup_reduce_has_cluster_attrs(op)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.full_subgroup"));
  }

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

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.wavefront_size"));
  }
  uint32_t active_lane_count = 0;
  if (!loom_amdgpu_subgroup_reduce_active_lane_count(
          module, loom_target_low_legality_function(context), bundle,
          wavefront_size, &active_lane_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_reduce.fixed_workgroup_wave_multiple"));
  }
  if (!loom_amdgpu_u32_is_power_of_two(active_lane_count)) {
    uint32_t unused_identity_bits = 0;
    if (!loom_amdgpu_collective_combine_identity_bits(kind,
                                                      &unused_identity_bits)) {
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("subgroup_reduce.identity"));
    }
  }

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
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements)));
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

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wavefront_size"));
  }
  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(
          module, loom_target_low_legality_function(context), bundle,
          &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.fixed_workgroup_size"));
  }
  const bool has_partial_tail = flat_workgroup_size > wavefront_size &&
                                (flat_workgroup_size % wavefront_size) != 0;
  const uint32_t wave_count =
      (flat_workgroup_size + wavefront_size - 1) / wavefront_size;
  if (flat_workgroup_size > wavefront_size && wave_count > wavefront_size) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.wave_count"));
  }
  const bool needs_cross_wave_identity =
      flat_workgroup_size > wavefront_size &&
      (has_partial_tail || !loom_amdgpu_u32_is_power_of_two(wave_count));
  const bool needs_subgroup_identity =
      flat_workgroup_size < wavefront_size &&
      !loom_amdgpu_u32_is_power_of_two(flat_workgroup_size);
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
  const uint32_t scratch_slot_count = wave_count;
  const uint64_t scratch_byte_length =
      (uint64_t)scratch_slot_count * register_count * 4u;
  if (scratch_byte_length > UINT32_MAX) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("workgroup_reduce.scratch_byte_length"));
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
  if (needs_identity_guard || flat_workgroup_size > wavefront_size) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
        IREE_SV("descriptor.v_cmp_ult_u32")));
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
  if (flat_workgroup_size > wavefront_size) {
    static const loom_amdgpu_low_legality_descriptor_requirement_t
        scratch_requirements[] = {
            {
                .constraint_key = IREE_SVL("descriptor.ds_read_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.ds_write_b32"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_barrier"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_and_saveexec_b64"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
            },
            {
                .constraint_key = IREE_SVL("descriptor.s_mov_b64_exec"),
                .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
            },
        };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_low_legality_verify_descriptor_requirements(
            context, op, scratch_requirements,
            IREE_ARRAYSIZE(scratch_requirements)));
  }

  return iree_ok_status();
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
