// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION = 0,
  LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND = 1,
  LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP = 2,
};

typedef struct loom_amdgpu_masked_region_t {
  // Shared producer's exact graph edge identities for this region's exits.
  loom_cfg_edge_index_span_t exit_edges;
  // Guard false destination reached when no lane enters the masked region.
  loom_block_t* guard_false_dest;
  // Shared continuation reached by region exits.
  loom_block_t* continuation;
  // Optional transparent false path carrying fallback values to continuation.
  struct {
    // Unconditional forwarding terminator, or NULL for a nontransparent path.
    const loom_op_t* terminator;
    // True when region exits also reach this shared forwarding block.
    bool is_region_exit;
  } false_path;
} loom_amdgpu_masked_region_t;

typedef struct loom_amdgpu_branch_plan_t {
  // Low-only block that restores EXEC before leaving a narrowed region.
  loom_block_t* restore_block;
  // Effective destination after EXEC has been restored.
  loom_block_t* restore_dest;
  // False-arm terminator used when restore_block bypasses a pass-through block.
  const loom_op_t* restore_passthrough_terminator;
  // Low-only block that restores EXEC and merges pass-through false values.
  loom_block_t* merge_restore_block;
  // Shared continuation reached after merge_restore_block.
  loom_block_t* merge_restore_dest;
  // False-arm terminator providing pass-through merge fallback values.
  const loom_op_t* false_passthrough_terminator;
  // Low-only block that computes the inactive else mask for if/else diamonds.
  loom_block_t* else_dispatch_block;
  // Low-only block that enters the else body when no true lanes were active.
  loom_block_t* no_true_else_entry_block;
  // Original low destination for the else body.
  loom_block_t* else_body_block;
  // Source values yielded by the true arm of an if/else merge.
  loom_value_slice_t if_else_true_source_args;
  // Source values yielded by the false arm of an if/else merge.
  loom_value_slice_t if_else_false_source_args;
  // Low-only block that restores EXEC when no false lanes were active.
  loom_block_t* true_only_restore_block;
  // Number of values merged by an if/else diamond.
  uint16_t if_else_merge_arg_count;
} loom_amdgpu_branch_plan_t;

typedef struct loom_amdgpu_loop_plan_t loom_amdgpu_loop_plan_t;

typedef enum loom_amdgpu_loop_validation_e {
  LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED = 0,
  LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED,
  LOOM_AMDGPU_LOOP_VALIDATION_REJECTED,
} loom_amdgpu_loop_validation_t;

typedef struct loom_amdgpu_loop_exit_plan_t {
  // Shared saved-mask lifetime for the natural loop.
  loom_amdgpu_loop_plan_t* loop;
  // Source conditional terminator owning the exit successors.
  const loom_op_t* source_op;
  // Bit i is set when successor i exits the loop.
  uint8_t exit_successor_mask;
} loom_amdgpu_loop_exit_plan_t;

struct loom_amdgpu_loop_plan_t {
  // Low-only block that saves parent EXEC before the loop header.
  loom_block_t* entry_block;
  // Original low loop header destination reached after entry_block.
  loom_block_t* header_dest;
  // Low-only block that restores parent EXEC after loop retirement.
  loom_block_t* restore_block;
  // Shared source continuation reached after restore_block.
  loom_block_t* restore_dest;
  // Low value produced by entry_block, or INVALID before emission.
  loom_value_id_t saved_exec;
  // Number of direct exit edges collected during branch preparation.
  uint32_t exit_edge_count;
  // Number of populated records in exit_plans.
  uint32_t exit_plan_count;
  // Boundary validation state for shared retirement.
  loom_amdgpu_loop_validation_t validation;
  // Direct exit records, allocated inline to the loop's direct exit count.
  loom_amdgpu_loop_exit_plan_t exit_plans[];
};

static int loom_amdgpu_loop_plan_table_key;

typedef struct loom_amdgpu_zero_placeholder_t {
  // Low value type represented by value.
  loom_type_t type;
  // Reusable zero value for an inactive merge placeholder.
  loom_value_id_t value;
} loom_amdgpu_zero_placeholder_t;

static iree_status_t loom_amdgpu_emit_plain_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_op_t* low_cond_br_op = NULL;
  return loom_low_cond_br_build(loom_low_lower_context_builder(context),
                                low_condition, low_true_dest, low_false_dest,
                                source_op->location, &low_cond_br_op);
}

static bool loom_amdgpu_condition_is_reg_class(
    loom_low_lower_context_t* context, loom_type_t low_type,
    uint16_t reg_class_id, uint32_t unit_count) {
  const bool is_class =
      loom_amdgpu_low_type_is_register_class(context, low_type, reg_class_id);
  return is_class && loom_low_register_type_unit_count(low_type) == unit_count;
}

static bool loom_amdgpu_low_type_is_native_i1_mask(
    loom_low_lower_context_t* context, loom_type_t low_type) {
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return is_sgpr && loom_low_register_type_unit_count(low_type) == 2;
}

static bool loom_amdgpu_branch_condition_is_uniform(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_value_facts_is_subgroup_uniform(
      loom_value_fact_table_lookup(loom_low_lower_context_fact_table(context),
                                   loom_cfg_cond_br_condition(source_op)));
}

static bool loom_amdgpu_single_predecessor_cfg_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_condition, bool* out_assumed_truth) {
  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  loom_cfg_edge_index_span_t incoming = loom_cfg_graph_predecessor_edges(
      graph, source_op->parent_block->region_index);
  if (incoming.count != 1) {
    return false;
  }
  const loom_cfg_edge_info_t* edge = &graph->edges[incoming.values[0]];
  if (!loom_cfg_cond_br_isa(edge->terminator)) {
    return false;
  }
  *out_condition = edge->selector_value_id;
  *out_assumed_truth = edge->successor_index == 0;
  return true;
}

static iree_status_t loom_amdgpu_cfg_cond_br_edge_implied_bool(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition, bool* out_proven) {
  *out_condition = false;
  *out_proven = false;
  if (!loom_cfg_cond_br_isa(source_op)) {
    return iree_ok_status();
  }

  loom_value_id_t edge_condition = LOOM_VALUE_ID_INVALID;
  bool edge_assumed_truth = false;
  if (!loom_amdgpu_single_predecessor_cfg_condition(
          context, source_op, &edge_condition, &edge_assumed_truth)) {
    return iree_ok_status();
  }

  const loom_value_id_t condition = loom_cfg_cond_br_condition(source_op);
  if (condition == edge_condition) {
    *out_condition = edge_assumed_truth;
    *out_proven = true;
    return iree_ok_status();
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_condition_integer_relation_t relation_storage[16];
  loom_condition_fact_set_t edge_facts = {0};
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &edge_facts);
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_condition_facts_query(
      loom_low_lower_context_condition_query(context), fact_table,
      edge_condition, edge_assumed_truth, &edge_facts, &complete));
  if (!complete) {
    return iree_ok_status();
  }
  if (edge_facts.integer_relation_count == 0) {
    return iree_ok_status();
  }

  return loom_condition_fact_set_proves_condition(
      loom_low_lower_context_condition_query(context), fact_table, &edge_facts,
      condition, out_condition, out_proven);
}

// Materializes a subgroup-uniform scalar stored in one or two VGPRs as an
// SGPR branch payload without changing its canonical source mapping.
static iree_status_t loom_amdgpu_materialize_uniform_vgpr_scalar_as_sgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, loom_type_t required_low_type,
    loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t actual_type = loom_module_value_type(module, low_value_id);
  IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
      context, actual_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR));
  IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR));

  const uint32_t register_count =
      loom_low_register_type_unit_count(actual_type);
  IREE_ASSERT_GE(register_count, 1);
  IREE_ASSERT_LE(register_count, 2);
  const loom_type_t vgpr_type =
      loom_low_register_carrier_type_with_unit_count(actual_type, 1);
  const loom_type_t sgpr_type =
      loom_low_register_carrier_type_with_unit_count(required_low_type, 1);
  // Scalar carriers contain at most two 32-bit register units.
  loom_value_id_t sgpr_registers[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < register_count; ++i) {
    loom_value_id_t vgpr_register = low_value_id;
    if (register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_value_id, register_count, i, vgpr_type,
          &vgpr_register));
    }
    const loom_value_id_t operands[] = {vgpr_register};
    loom_op_t* readfirstlane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
        operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &sgpr_type, 1, &readfirstlane_op));
    sgpr_registers[i] =
        loom_value_slice_get(loom_low_op_results(readfirstlane_op), 0);
  }

  const loom_type_t result_type =
      loom_low_register_carrier_type_with_unit_count(required_low_type,
                                                     register_count);
  return loom_amdgpu_build_low_register_range(context, source_op,
                                              sgpr_registers, register_count,
                                              result_type, out_low_value_id);
}

static iree_status_t loom_amdgpu_materialize_branch_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = low_value_id;

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t actual_type = loom_module_value_type(module, *out_low_value_id);
  if (loom_type_equal(actual_type, required_low_type)) {
    return iree_ok_status();
  }

  const bool requires_sgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  const bool requires_vgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!requires_sgpr && !requires_vgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address branch payload selected non-register low type");
    IREE_BUILTIN_UNREACHABLE();
  }

  const bool actual_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, actual_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  const bool actual_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, actual_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (requires_vgpr && actual_is_sgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, *out_low_value_id, out_low_value_id));
    actual_type = loom_module_value_type(module, *out_low_value_id);
  } else if (requires_sgpr && actual_is_vgpr) {
    const loom_value_facts_t source_facts = loom_value_fact_table_lookup(
        loom_low_lower_context_fact_table(context), source_value_id);
    if (loom_value_facts_is_subgroup_uniform(source_facts)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_uniform_vgpr_scalar_as_sgpr(
          context, source_op, *out_low_value_id, required_low_type,
          out_low_value_id));
      actual_type = loom_module_value_type(module, *out_low_value_id);
    }
  }

  if (loom_type_equal(actual_type, required_low_type)) {
    return iree_ok_status();
  }

  const bool actual_matches_required_class =
      loom_amdgpu_low_type_is_register_class(
          context, actual_type,
          requires_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                        : LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!actual_matches_required_class) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address branch payload materialized wrong register class");
    IREE_BUILTIN_UNREACHABLE();
  }

  const uint32_t actual_unit_count =
      loom_low_register_type_unit_count(actual_type);
  const uint32_t required_unit_count =
      loom_low_register_type_unit_count(required_low_type);
  if (actual_unit_count == 2 && required_unit_count == 1) {
    // Entry ABI values remain 64-bit even when their retained range proves
    // that a destination block can use one address register.
    return loom_amdgpu_emit_low_slice(context, source_op, *out_low_value_id,
                                      /*offset=*/0, required_low_type,
                                      out_low_value_id);
  }
  if (actual_unit_count != 1 || required_unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address branch payload materialized wrong register shape");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(required_low_type, 1);
  loom_value_id_t high_zero = LOOM_VALUE_ID_INVALID;
  const uint16_t zero_descriptor = requires_vgpr
                                       ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                                       : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, zero_descriptor, 0, lane_type, &high_zero));

  const loom_value_id_t lanes[] = {*out_low_value_id, high_zero};
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), required_low_type,
      out_low_value_id);
}

static iree_status_t loom_amdgpu_materialize_control_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = low_value_id;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value_id);
  const bool requires_vgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  const bool requires_sgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if ((requires_sgpr || requires_vgpr) &&
      loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_materialize_branch_address(
        context, source_terminator, source_value_id, low_value_id,
        required_low_type, out_low_value_id);
  }
  if (requires_vgpr) {
    if (loom_amdgpu_type_is_i32(source_type) ||
        loom_amdgpu_vector_i32_register_count(source_type) != 0) {
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_terminator, source_value_id, out_low_value_id);
    }
    if (loom_amdgpu_type_is_f32(source_type) ||
        loom_amdgpu_vector_f32_register_count(source_type) != 0) {
      return loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_terminator, source_value_id, out_low_value_id);
    }
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_terminator, low_value_id, out_low_value_id);
  }

  if (requires_sgpr && loom_amdgpu_type_is_i1(source_type)) {
    if (loom_low_register_type_unit_count(required_low_type) == 2) {
      return loom_amdgpu_lookup_or_materialize_native_i1_mask(
          context, source_terminator, source_value_id, out_low_value_id);
    }
    // A scalar destination has a uniformity proof. Its source can still use a
    // native mask (for example, a Boolean select), or transient SCC state.
    // Capture the truth value without changing the source's canonical mapping.
    loom_value_id_t low_condition = low_value_id;
    const loom_type_t actual_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_value_id);
    if (loom_low_register_type_unit_count(actual_type) == 2) {
      const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(
          loom_low_lower_context_bundle(context));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
          context, source_terminator, low_value_id, wavefront_size,
          &low_condition));
    }
    loom_value_id_t low_false = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_terminator, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
        required_low_type, &low_false));
    loom_value_id_t low_true = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_terminator, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 1,
        required_low_type, &low_true));
    const loom_value_id_t operands[] = {low_true, low_false, low_condition};
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_terminator, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
        operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
        &required_low_type, 1, &select_op));
    *out_low_value_id = loom_value_slice_get(loom_low_op_results(select_op), 0);
    return iree_ok_status();
  }

  if (requires_sgpr && loom_type_is_scalar(source_type)) {
    // Collective results and their arithmetic can use VGPR instructions even
    // when the destination join selects scalar storage. Only the retained
    // uniformity proof makes selecting one active lane semantics-preserving.
    IREE_ASSERT(
        loom_value_facts_is_subgroup_uniform(loom_value_fact_table_lookup(
            loom_low_lower_context_fact_table(context), source_value_id)),
        "SGPR branch payload requires a uniform source value");
    return loom_amdgpu_materialize_uniform_vgpr_scalar_as_sgpr(
        context, source_terminator, low_value_id, required_low_type,
        out_low_value_id);
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU branch argument materializer selected unsupported type");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_materialize_branch_arg(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator, uint8_t successor_index,
    uint16_t arg_index, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id, loom_type_t required_low_type,
    loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_materialize_control_operand(
      context, source_terminator, source_value_id, low_value_id,
      required_low_type, out_low_value_id);
}

static iree_status_t loom_amdgpu_materialize_full_low_vgpr_b32_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && unit_count == 1) {
    return loom_amdgpu_materialize_full_low_vgpr_b32(context, source_op,
                                                     low_value, out_low_value);
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_materialize_structural_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id) {
  (void)user_data;
  (void)operand_index;
  *out_low_value_id = low_value_id;
  if (loom_low_lower_source_op_is_callable_exit(context, source_op) &&
      !loom_type_equal(
          loom_module_value_type(loom_low_lower_context_module(context),
                                 low_value_id),
          required_low_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_control_operand(
        context, source_op, source_value_id, low_value_id, required_low_type,
        out_low_value_id));
  }

  const bool requires_vgpr = loom_amdgpu_low_type_is_register_class(
      context, required_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!requires_vgpr) {
    return iree_ok_status();
  }
  return loom_amdgpu_materialize_full_low_vgpr_b32_registers(
      context, source_op, *out_low_value_id, out_low_value_id);
}

static iree_status_t loom_amdgpu_emit_sgpr_bool_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));

  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero));

  const loom_value_id_t compare_operands[] = {
      low_condition,
      low_zero,
  };
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32,
      compare_operands, IREE_ARRAYSIZE(compare_operands),
      loom_named_attr_slice_empty(), &scc_type, 1, &compare_op));
  return loom_amdgpu_emit_plain_cond_branch(
      context, source_op, loom_op_const_results(compare_op)[0], low_true_dest,
      low_false_dest);
}

static bool loom_amdgpu_try_false_passthrough_continuation(
    loom_block_t* false_dest, loom_block_t** out_continuation,
    const loom_op_t** out_false_terminator) {
  *out_continuation = NULL;
  *out_false_terminator = NULL;
  if (false_dest->arg_count != 0 || false_dest->op_count != 1) {
    return false;
  }
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* continuation = loom_cfg_br_dest(false_terminator);
  if (continuation == false_dest) {
    return false;
  }
  *out_continuation = continuation;
  *out_false_terminator = false_terminator;
  return true;
}

static iree_status_t loom_amdgpu_analyze_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_masked_region_t* out_region) {
  *out_region = (loom_amdgpu_masked_region_t){0};
  loom_block_t* source_entry = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_destination = loom_cfg_cond_br_false_dest(source_op);
  loom_block_t* continuation = false_destination;
  const loom_op_t* false_terminator = NULL;
  loom_block_t* passthrough = NULL;
  loom_amdgpu_try_false_passthrough_continuation(
      false_destination, &passthrough, &false_terminator);
  if (passthrough) {
    continuation = passthrough;
  }
  if (source_entry->arg_count || false_destination->arg_count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("destination_block_arguments_absent"));
  }
  if (source_entry == continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  uint16_t entry_index = source_entry->region_index;
  if (facts->dominance.entry_predecessors[entry_index] !=
      source_op->parent_block->region_index) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_entry"));
  }
  const loom_cfg_region_t* region = &facts->regions.blocks[entry_index];
  loom_cfg_edge_index_span_t direct_exits = loom_cfg_regions_edges_to(
      &facts->graph, &facts->dominance, &facts->regions, entry_index,
      continuation->region_index);
  loom_cfg_edge_index_span_t forwarded_exits = {0};
  if (false_terminator) {
    forwarded_exits = loom_cfg_regions_edges_to(
        &facts->graph, &facts->dominance, &facts->regions, entry_index,
        false_destination->region_index);
  }
  if (!region->exit_count ||
      region->exit_count != direct_exits.count + forwarded_exits.count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_exits_by_cfg"));
  }
  if (forwarded_exits.count) {
    loom_cfg_edge_index_span_t incoming = loom_cfg_regions_edges_to(
        &facts->graph, &facts->dominance, &facts->regions, 0,
        false_destination->region_index);
    // The guard's false edge is the sole external entry we bypass. All other
    // incoming paths must belong to the true region before sharing its exit.
    if (incoming.count != forwarded_exits.count + 1) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_single_entry"));
    }
  }
  *out_region = (loom_amdgpu_masked_region_t){
      .exit_edges = direct_exits,
      .guard_false_dest = false_destination,
      .continuation = continuation,
      .false_path =
          {
              .terminator = false_terminator,
              .is_region_exit = forwarded_exits.count != 0,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_insert_restore_block_before_dest(
    loom_low_lower_context_t* context, loom_block_t* low_dest,
    loom_block_t** out_restore_block) {
  *out_restore_block = NULL;
  loom_block_t* restore_block = NULL;
  loom_module_t* module = loom_low_lower_context_module(context);
  // The restored continuation stays beside the masked region instead of
  // turning every loop iteration into a trip through the function tail.
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      module, low_dest->parent_region, low_dest->region_index, &restore_block));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint16_t i = 0; i < low_dest->arg_count; ++i) {
    const loom_value_id_t dest_arg = loom_block_arg_id(low_dest, i);
    loom_value_id_t restore_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, restore_block, loom_module_value_type(module, dest_arg),
        &restore_arg));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, dest_arg, restore_arg));
  }
  *out_restore_block = restore_block;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_block_args_like_dest(
    loom_low_lower_context_t* context, loom_block_t* dest_block,
    loom_block_t* block, loom_value_id_t* out_arg_ids) {
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  for (uint16_t i = 0; i < dest_block->arg_count; ++i) {
    const loom_value_id_t dest_arg = loom_block_arg_id(dest_block, i);
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, block, loom_module_value_type(module, dest_arg), &block_arg));
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, dest_arg, block_arg));
    if (out_arg_ids != NULL) {
      out_arg_ids[i] = block_arg;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_then_masked_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_masked_region_t region = {0};
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_analyze_then_masked_region(context, source_op, &region));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  loom_block_t* guard_false_low_dest = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
      context, region.guard_false_dest, &guard_false_low_dest));
  IREE_RETURN_IF_ERROR(loom_amdgpu_insert_restore_block_before_dest(
      context, guard_false_low_dest, &plan->restore_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->restore_block, &plan->restore_dest));
  if (region.false_path.terminator != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
        context, region.false_path.terminator, 0, &plan->restore_dest));
    plan->restore_passthrough_terminator = region.false_path.terminator;
  }

  loom_block_t* exit_restore_block = plan->restore_block;
  loom_block_t* exit_restore_dest = plan->restore_dest;
  if (region.continuation != region.guard_false_dest) {
    loom_block_t* continuation_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_block(
        context, region.continuation, &continuation_low_dest));
    IREE_RETURN_IF_ERROR(loom_amdgpu_insert_restore_block_before_dest(
        context, continuation_low_dest, &exit_restore_block));
    plan->merge_restore_block = exit_restore_block;
    plan->merge_restore_dest = exit_restore_dest;
    plan->false_passthrough_terminator = region.false_path.terminator;
  }

  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  for (iree_host_size_t i = 0; i < region.exit_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[region.exit_edges.values[i]];
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, edge->terminator, edge->successor_index, exit_restore_block,
        &previous_low_dest));
    if (previous_low_dest != exit_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }
  if (region.false_path.is_region_exit) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, region.false_path.terminator, 0, exit_restore_block,
        &previous_low_dest));
    if (previous_low_dest != exit_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION,
                               plan));
}

static bool loom_amdgpu_try_if_else_diamond(const loom_op_t* source_op) {
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(source_op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(source_op);
  if (true_dest->arg_count != 0 || false_dest->arg_count != 0 ||
      true_dest == false_dest || true_dest->op_count == 0 ||
      false_dest->op_count == 0) {
    return false;
  }
  const loom_op_t* true_terminator = loom_block_const_last_op(true_dest);
  const loom_op_t* false_terminator = loom_block_const_last_op(false_dest);
  if (!loom_cfg_br_isa(true_terminator) || !loom_cfg_br_isa(false_terminator)) {
    return false;
  }
  loom_block_t* merge_block = loom_cfg_br_dest(true_terminator);
  if (merge_block != loom_cfg_br_dest(false_terminator) ||
      merge_block == true_dest || merge_block == false_dest) {
    return false;
  }
  return true;
}

static iree_status_t loom_amdgpu_verify_if_else_merge_args(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_merge_dest, loom_block_t* merge_low_dest,
    uint16_t merge_arg_count) {
  loom_module_t* module = loom_low_lower_context_module(context);
  IREE_ASSERT_EQ(source_merge_dest->arg_count, merge_arg_count);
  for (uint16_t i = 0; i < merge_arg_count; ++i) {
    const loom_value_id_t merge_arg = loom_block_arg_id(merge_low_dest, i);
    const loom_type_t merge_type = loom_module_value_type(module, merge_arg);
    const loom_value_id_t source_merge_arg =
        loom_block_arg_id(source_merge_dest, i);
    const loom_type_t source_merge_type =
        loom_module_value_type(module, source_merge_arg);
    if (loom_amdgpu_type_is_i1(source_merge_type) &&
        loom_amdgpu_low_type_is_native_i1_mask(context, merge_type)) {
      continue;
    }

    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, merge_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    const uint32_t lane_count = loom_low_register_type_unit_count(merge_type);
    if (!is_vgpr || lane_count == 0 ||
        lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
    }
  }
  return iree_ok_status();
}

static loom_value_slice_t loom_amdgpu_single_region_exit_args(
    const loom_cfg_graph_t* graph, const loom_cfg_region_t* region,
    uint16_t expected_count) {
  if (region->exit_edges.count != 1) {
    return (loom_value_slice_t){0};
  }
  const loom_cfg_edge_info_t* edge =
      &graph->edges[region->exit_edges.values[0]];
  if (!loom_cfg_br_isa(edge->terminator)) {
    return (loom_value_slice_t){0};
  }
  loom_value_slice_t args = loom_cfg_br_args(edge->terminator);
  return args.count == expected_count ? args : (loom_value_slice_t){0};
}

static bool loom_amdgpu_try_if_else_regions(
    const loom_value_fact_cfg_region_t* facts, const loom_op_t* source_op,
    const loom_cfg_region_t** out_true_region,
    const loom_cfg_region_t** out_false_region) {
  const loom_block_t* true_entry = loom_cfg_cond_br_true_dest(source_op);
  const loom_block_t* false_entry = loom_cfg_cond_br_false_dest(source_op);
  if (true_entry->arg_count || false_entry->arg_count ||
      true_entry == false_entry) {
    return false;
  }
  uint16_t guard = source_op->parent_block->region_index;
  if (facts->dominance.entry_predecessors[true_entry->region_index] != guard ||
      facts->dominance.entry_predecessors[false_entry->region_index] != guard) {
    return false;
  }
  const loom_cfg_region_t* true_region =
      &facts->regions.blocks[true_entry->region_index];
  const loom_cfg_region_t* false_region =
      &facts->regions.blocks[false_entry->region_index];
  if (true_region->continuation_index == LOOM_CFG_REGION_CONTINUATION_NONE ||
      true_region->continuation_index != false_region->continuation_index ||
      true_region->continuation_index == guard) {
    return false;
  }
  *out_true_region = true_region;
  *out_false_region = false_region;
  return true;
}

static iree_status_t loom_amdgpu_prepare_if_else_regions(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_cfg_region_t* true_region,
    const loom_cfg_region_t* false_region) {
  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  const loom_block_t* continuation =
      graph->blocks[true_region->continuation_index].block;
  loom_block_t* merge_low_dest = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_block(context, continuation, &merge_low_dest));

  loom_amdgpu_branch_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_branch_plan_t){0};
  plan->if_else_merge_arg_count = merge_low_dest->arg_count;
  plan->if_else_true_source_args = loom_amdgpu_single_region_exit_args(
      graph, true_region, plan->if_else_merge_arg_count);
  plan->if_else_false_source_args = loom_amdgpu_single_region_exit_args(
      graph, false_region, plan->if_else_merge_arg_count);

  if (plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_append_low_block(context, &plan->restore_block));

    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, source_op, 1, plan->else_dispatch_block,
        &plan->else_body_block));

    loom_block_t* common_restore_dest = NULL;
    for (iree_host_size_t i = 0; i < true_region->exit_edges.count; ++i) {
      loom_block_t* previous_low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
          context, graph->edges[true_region->exit_edges.values[i]].terminator,
          graph->edges[true_region->exit_edges.values[i]].successor_index,
          plan->else_dispatch_block, &previous_low_dest));
      if (common_restore_dest == NULL) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }

    for (iree_host_size_t i = 0; i < false_region->exit_edges.count; ++i) {
      loom_block_t* previous_low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
          context, graph->edges[false_region->exit_edges.values[i]].terminator,
          graph->edges[false_region->exit_edges.values[i]].successor_index,
          plan->restore_block, &previous_low_dest));
      if (common_restore_dest == NULL) {
        common_restore_dest = previous_low_dest;
      } else if (previous_low_dest != common_restore_dest) {
        return loom_low_lower_emit_branch_constraint(
            context, source_op,
            IREE_SV("masked_region_exit_edge_interposition"));
      }
    }
    if (common_restore_dest == NULL) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_single_continuation"));
    }
    plan->restore_dest = common_restore_dest;

    return loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND,
                                 plan));
  }

  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_if_else_merge_args(
      context, source_op, continuation, merge_low_dest,
      plan->if_else_merge_arg_count));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->else_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->else_dispatch_block, NULL));
  IREE_RETURN_IF_ERROR(loom_low_lower_append_low_block(
      context, &plan->no_true_else_entry_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->true_only_restore_block));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->merge_restore_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, merge_low_dest, plan->merge_restore_block, NULL));

  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, source_op, 1, plan->no_true_else_entry_block,
      &plan->else_body_block));

  loom_block_t* common_restore_dest = NULL;
  for (iree_host_size_t i = 0; i < true_region->exit_edges.count; ++i) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, graph->edges[true_region->exit_edges.values[i]].terminator,
        graph->edges[true_region->exit_edges.values[i]].successor_index,
        plan->else_dispatch_block, &previous_low_dest));
    if (common_restore_dest == NULL) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }
  for (iree_host_size_t i = 0; i < false_region->exit_edges.count; ++i) {
    loom_block_t* previous_low_dest = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
        context, graph->edges[false_region->exit_edges.values[i]].terminator,
        graph->edges[false_region->exit_edges.values[i]].successor_index,
        plan->merge_restore_block, &previous_low_dest));
    if (common_restore_dest == NULL) {
      common_restore_dest = previous_low_dest;
    } else if (previous_low_dest != common_restore_dest) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  if (common_restore_dest == NULL) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_single_continuation"));
  }
  plan->merge_restore_dest = common_restore_dest;

  return loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND, plan));
}

static bool loom_amdgpu_try_find_loop_exit(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           uint16_t* out_loop_index,
                                           uint8_t* out_exit_successor_mask) {
  *out_loop_index = LOOM_CFG_LOOP_NEST_NONE;
  *out_exit_successor_mask = 0;
  const loom_cfg_loop_nest_t* loops =
      &loom_low_lower_context_cfg(context)->loops;
  const loom_cfg_graph_t* graph = loops->graph;
  const uint16_t block_index =
      (uint16_t)loom_cfg_graph_block_index(graph, source_op->parent_block);
  const uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return false;
  }
  const loom_cfg_edge_index_span_t outgoing =
      loom_cfg_graph_successor_edges(graph, block_index);
  const loom_cfg_edge_info_t* true_edge = &graph->edges[outgoing.values[0]];
  const loom_cfg_edge_info_t* false_edge = &graph->edges[outgoing.values[1]];
  const bool true_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, true_edge->target_block_index);
  const bool false_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, false_edge->target_block_index);
  if (true_inside && false_inside) {
    return false;
  }
  *out_loop_index = loop_index;
  *out_exit_successor_mask = (true_inside ? 0 : 1) | (false_inside ? 0 : 2);
  return true;
}

static iree_status_t loom_amdgpu_allocate_loop_plan(
    loom_low_lower_context_t* context, uint32_t exit_capacity,
    loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const iree_host_size_t plan_length =
      sizeof(**out_plan) + exit_capacity * sizeof(loom_amdgpu_loop_exit_plan_t);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(context, plan_length,
                                                         (void**)out_plan));
  **out_plan = (loom_amdgpu_loop_plan_t){
      .saved_exec = LOOM_VALUE_ID_INVALID,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_or_allocate_loop_plan(
    loom_low_lower_context_t* context, const loom_cfg_loop_nest_t* loops,
    uint16_t loop_index, loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->exits.count == 1) {
    return loom_amdgpu_allocate_loop_plan(context, loop->direct_exit_count,
                                          out_plan);
  }

  loom_amdgpu_loop_plan_t** plans = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_loop_plan_table_key,
      loops->loop_count * sizeof(*plans), (void**)&plans));
  if (plans[loop_index] == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_allocate_loop_plan(
        context, loop->direct_exit_count, &plans[loop_index]));
  }
  *out_plan = plans[loop_index];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_validate_divergent_loop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_cfg_loop_nest_t* loops, uint16_t loop_index) {
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->entries.count != 1) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_entry"));
  }
  if (loop->continuation_index == LOOM_CFG_LOOP_CONTINUATION_NONE) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_continuation"));
  }
  if (loop->direct_exit_count != loop->exits.count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_direct_exits"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_finalize_loop_plan(
    loom_low_lower_context_t* context, const loom_cfg_loop_nest_t* loops,
    uint16_t loop_index, loom_amdgpu_loop_plan_t* plan) {
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  const loom_cfg_graph_t* graph = loops->graph;
  const loom_cfg_edge_info_t* entry_edge =
      &graph->edges[loop->entries.unique_index];
  const loom_block_t* header_block = graph->blocks[loop->header_index].block;

  loom_block_t* low_header = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_block(context, header_block, &low_header));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->entry_block));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_block_args_like_dest(
      context, low_header, plan->entry_block, NULL));
  IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
      context, entry_edge->terminator, (uint8_t)entry_edge->successor_index,
      plan->entry_block, &plan->header_dest));

  IREE_RETURN_IF_ERROR(
      loom_low_lower_append_low_block(context, &plan->restore_block));
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      loom_block_t* previous_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_interpose_successor_dest(
          context, exit->source_op, successor_index, plan->restore_block,
          &previous_dest));
      if (plan->restore_dest == NULL) {
        plan->restore_dest = previous_dest;
      } else {
        IREE_ASSERT_EQ(plan->restore_dest, previous_dest);
      }
    }
  }
  IREE_ASSERT(plan->restore_dest != NULL);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_prepare_loop_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, bool* out_handled) {
  *out_handled = false;
  const loom_cfg_loop_nest_t* loops =
      &loom_low_lower_context_cfg(context)->loops;
  uint16_t loop_index = LOOM_CFG_LOOP_NEST_NONE;
  uint8_t exit_successor_mask = 0;
  if (!loom_amdgpu_try_find_loop_exit(context, source_op, &loop_index,
                                      &exit_successor_mask)) {
    return iree_ok_status();
  }
  *out_handled = true;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->exits.count == 1 && !is_divergent) {
    return iree_ok_status();
  }

  loom_amdgpu_loop_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_or_allocate_loop_plan(context, loops, loop_index, &plan));
  if (is_divergent &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED) {
    const uint32_t previous_error_count =
        loom_low_lower_context_error_count(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_validate_divergent_loop(
        context, source_op, loops, loop_index));
    plan->validation =
        loom_low_lower_context_error_count(context) == previous_error_count
            ? LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED
            : LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
  }
  if (plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_REJECTED) {
    return iree_ok_status();
  }

  IREE_ASSERT_LT(plan->exit_plan_count, loop->direct_exit_count);
  loom_amdgpu_loop_exit_plan_t* exit_plan =
      &plan->exit_plans[plan->exit_plan_count++];
  *exit_plan = (loom_amdgpu_loop_exit_plan_t){
      .loop = plan,
      .source_op = source_op,
      .exit_successor_mask = exit_successor_mask,
  };
  plan->exit_edge_count += exit_successor_mask == 3 ? 2 : 1;
  IREE_ASSERT_LE(plan->exit_edge_count, loop->direct_exit_count);
  if (is_divergent) {
    IREE_RETURN_IF_ERROR(loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP,
                                 exit_plan)));
  }
  if (plan->exit_edge_count == loop->exits.count &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_finalize_loop_plan(context, loops, loop_index, plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_prepare_exec_mask_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  bool immediate_diamond = loom_amdgpu_try_if_else_diamond(source_op);
  loom_block_t* passthrough_continuation = NULL;
  const loom_op_t* passthrough_terminator = NULL;
  bool has_false_passthrough = loom_amdgpu_try_false_passthrough_continuation(
      loom_cfg_cond_br_false_dest(source_op), &passthrough_continuation,
      &passthrough_terminator);
  bool false_path_is_direct_passthrough = false;
  if (has_false_passthrough) {
    const uint16_t false_entry =
        loom_cfg_cond_br_false_dest(source_op)->region_index;
    false_path_is_direct_passthrough =
        facts->regions.blocks[false_entry].continuation_index ==
        passthrough_continuation->region_index;
  }
  const loom_cfg_region_t* true_region = NULL;
  const loom_cfg_region_t* false_region = NULL;
  // A one-branch false block is only a passthrough when it jumps directly to
  // its retained region continuation. A loop preheader has the same local
  // shape but enters the complete false region before converging.
  if ((immediate_diamond || !false_path_is_direct_passthrough) &&
      loom_amdgpu_try_if_else_regions(facts, source_op, &true_region,
                                      &false_region)) {
    return loom_amdgpu_prepare_if_else_regions(context, source_op, true_region,
                                               false_region);
  }
  return loom_amdgpu_prepare_then_masked_region(context, source_op);
}

iree_status_t loom_amdgpu_prepare_branch(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_terminator,
    iree_arena_allocator_t* analysis_arena) {
  (void)user_data;
  (void)analysis_arena;
  if (!loom_cfg_cond_br_isa(source_terminator)) {
    return iree_ok_status();
  }
  bool edge_implied_condition = false;
  bool edge_condition_proven = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_cfg_cond_br_edge_implied_bool(
      context, source_terminator, &edge_implied_condition,
      &edge_condition_proven));
  if (edge_condition_proven) {
    bool handled_loop = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_prepare_loop_exit(
        context, source_terminator, /*is_divergent=*/false, &handled_loop));
    return iree_ok_status();
  }

  loom_type_t condition_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
      context, source_terminator, loom_cfg_cond_br_condition(source_terminator),
      &condition_type));

  const bool is_divergent =
      loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2) &&
      !loom_amdgpu_branch_condition_is_uniform(context, source_terminator);
  bool handled_loop = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_prepare_loop_exit(
      context, source_terminator, is_divergent, &handled_loop));
  if (handled_loop || !is_divergent) {
    return iree_ok_status();
  }
  return loom_amdgpu_prepare_exec_mask_branch(context, source_terminator);
}

static iree_status_t loom_amdgpu_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest) {
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, restore_block->arg_ids,
                               restore_block->arg_count, source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_value_id_t* args,
    uint16_t arg_count) {
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, args, arg_count,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_restore_passthrough_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest, const loom_op_t* passthrough_terminator) {
  IREE_ASSERT(loom_cfg_br_isa(passthrough_terminator));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);

  // Edge materialization may read EXEC or emit vector copies. The skipped
  // arm's payload belongs to the restored parent lanes, not the narrowed arm.
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  loom_value_slice_t args = loom_cfg_br_args(passthrough_terminator);
  loom_value_id_t* low_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_remap_successor_args(
        context, passthrough_terminator, 0, restore_dest, args.values,
        args.count, &low_args);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, low_args, args.count,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_zero_vgpr_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  if (lane_count == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_value);
  }

  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &zero_lane));
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lanes[i] = zero_lane;
  }
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_value);
}

static iree_status_t loom_amdgpu_emit_zero_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, lane_type,
      &zero_lane));
  const loom_value_id_t lanes[] = {zero_lane, zero_lane};
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), value_type, out_value);
}

static iree_status_t loom_amdgpu_emit_zero_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  if (loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, value_type,
                                                out_value);
  }
  return loom_amdgpu_emit_zero_vgpr_value(context, source_op, value_type,
                                          out_value);
}

static iree_status_t loom_amdgpu_emit_masked_merge_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t lane_type,
    loom_value_id_t* out_merged_value) {
  const loom_value_id_t select_operands[] = {
      low_false_value,
      low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_named_attr_slice_empty(), &lane_type, 1, &select_op));
  *out_merged_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static bool loom_amdgpu_source_value_as_i1_constant(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    bool* out_value) {
  *out_value = false;
  return source_value != LOOM_VALUE_ID_INVALID &&
         loom_amdgpu_value_as_i1_constant(context, source_value, out_value);
}

static iree_status_t loom_amdgpu_emit_native_i1_mask_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t source_false_value,
    loom_value_id_t source_true_value, loom_value_id_t low_false_value,
    loom_value_id_t low_true_value, loom_value_id_t low_condition,
    loom_type_t mask_type, loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  bool true_constant = false;
  const bool true_is_constant = loom_amdgpu_source_value_as_i1_constant(
      context, source_true_value, &true_constant);
  bool false_constant = false;
  const bool false_is_constant = loom_amdgpu_source_value_as_i1_constant(
      context, source_false_value, &false_constant);
  if (true_is_constant && false_is_constant) {
    if (true_constant && false_constant) {
      *out_merged_value = saved_exec;
      return iree_ok_status();
    }
    if (!true_constant && !false_constant) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
  }

  if (low_true_value == low_false_value) {
    *out_merged_value = low_true_value;
    return iree_ok_status();
  }

  if (true_is_constant && true_constant && false_is_constant &&
      !false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
        low_condition, mask_type, out_merged_value);
  }

  loom_value_id_t true_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
      low_condition, mask_type, &true_condition));

  if (true_is_constant && !true_constant && false_is_constant &&
      false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        true_condition, saved_exec, mask_type, out_merged_value);
  }

  loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
  bool true_mask_is_zero = false;
  if (true_is_constant && true_constant) {
    true_mask = true_condition;
  } else if (true_is_constant && !true_constant) {
    true_mask_is_zero = true;
  } else if (low_true_value == low_condition) {
    true_mask = true_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        true_condition, low_true_value, mask_type, &true_mask));
  }

  if ((false_is_constant && !false_constant) ||
      low_false_value == low_condition) {
    if (true_mask_is_zero) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
    *out_merged_value = true_mask;
    return iree_ok_status();
  }

  loom_value_id_t false_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, true_condition,
      saved_exec, mask_type, &false_condition));
  loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
  if (false_is_constant && false_constant) {
    false_mask = false_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        false_condition, low_false_value, mask_type, &false_mask));
  }
  if (true_mask_is_zero) {
    *out_merged_value = false_mask;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, true_mask,
      false_mask, mask_type, out_merged_value);
}

static iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t source_false_value,
    loom_value_id_t source_true_value, loom_value_id_t low_false_value,
    loom_value_id_t low_true_value, loom_value_id_t low_condition,
    loom_type_t value_type, loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_amdgpu_emit_native_i1_mask_merge(
        context, source_op, saved_exec, source_false_value, source_true_value,
        low_false_value, low_true_value, low_condition, value_type,
        out_merged_value);
  }

  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  if (lane_count == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_false_value, &low_false_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_true_value, &low_true_value));
    return loom_amdgpu_emit_masked_merge_lane(
        context, source_op, low_false_value, low_true_value, low_condition,
        lane_type, out_merged_value);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_false_value, &low_false_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_true_value, &low_true_value));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t false_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_false_value, i, lane_type, &false_lane));
    loom_value_id_t true_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_true_value, i, lane_type, &true_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_lane(
        context, source_op, false_lane, true_lane, low_condition, lane_type,
        &lanes[i]));
  }

  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_merged_value);
}

static iree_status_t loom_amdgpu_emit_masked_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT(plan->merge_restore_block != NULL);
  IREE_ASSERT(plan->merge_restore_dest != NULL);
  IREE_ASSERT(plan->false_passthrough_terminator != NULL);
  IREE_ASSERT_EQ(plan->merge_restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  loom_value_slice_t false_args =
      loom_cfg_br_args(plan->false_passthrough_terminator);
  IREE_ASSERT_EQ(false_args.count, plan->merge_restore_block->arg_count);
  IREE_ASSERT_EQ(false_args.count, plan->merge_restore_dest->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, false_args.count, sizeof(*merged_args), (void**)&merged_args);
  }
  loom_value_id_t* low_false_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_remap_successor_args(
        context, plan->false_passthrough_terminator, 0,
        plan->merge_restore_dest, false_args.values, false_args.count,
        &low_false_args);
  }
  for (uint16_t i = 0; i < false_args.count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t source_false_value =
        loom_value_slice_get(false_args, i);
    const loom_value_id_t low_false_value = low_false_args[i];
    const loom_value_id_t low_true_value =
        loom_block_arg_id(plan->merge_restore_block, i);
    loom_module_t* module = loom_low_lower_context_module(context);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, saved_exec, source_false_value,
        LOOM_VALUE_ID_INVALID, low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status =
        loom_low_br_build(builder, plan->merge_restore_dest, merged_args,
                          false_args.count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_if_else_merge_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT(plan->merge_restore_block != NULL);
  IREE_ASSERT(plan->merge_restore_dest != NULL);
  IREE_ASSERT_EQ(plan->merge_restore_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->merge_restore_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t arg_count = plan->if_else_merge_arg_count;
  IREE_ASSERT_EQ(arg_count, plan->merge_restore_block->arg_count);
  IREE_ASSERT_EQ(arg_count, plan->merge_restore_dest->arg_count);
  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*merged_args), (void**)&merged_args);
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t source_false_value =
        i < plan->if_else_false_source_args.count
            ? loom_value_slice_get(plan->if_else_false_source_args, i)
            : LOOM_VALUE_ID_INVALID;
    const loom_value_id_t source_true_value =
        i < plan->if_else_true_source_args.count
            ? loom_value_slice_get(plan->if_else_true_source_args, i)
            : LOOM_VALUE_ID_INVALID;
    const loom_value_id_t low_false_value =
        loom_block_arg_id(plan->merge_restore_block, i);
    const loom_value_id_t low_true_value =
        loom_block_arg_id(plan->else_dispatch_block, i);
    status = loom_amdgpu_emit_masked_merge_value(
        context, source_op, saved_exec, source_false_value, source_true_value,
        low_false_value, low_true_value, low_condition,
        loom_module_value_type(module, low_true_value), &merged_args[i]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, plan->merge_restore_dest, merged_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_no_true_else_entry_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT_EQ(plan->no_true_else_entry_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->no_true_else_entry_block);

  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);

  const uint16_t arg_count = plan->if_else_merge_arg_count;
  loom_value_id_t* zero_args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*zero_args), (void**)&zero_args);
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, arg_count, sizeof(*placeholders), (void**)&placeholders);
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < arg_count && iree_status_is_ok(status); ++i) {
    zero_args[i] = LOOM_VALUE_ID_INVALID;
    // The no-true path reaches this block with EXEC empty. Restore the original
    // lane mask before defining placeholder VGPRs so the later hardware select
    // never reads an unwritten register operand, even though the placeholders
    // are semantically ignored when every active lane takes the else arm.
    const loom_value_id_t true_arg =
        loom_block_arg_id(plan->else_dispatch_block, i);
    const loom_type_t value_type = loom_module_value_type(module, true_arg);
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, value_type)) {
        zero_args[i] = placeholders[j].value;
        break;
      }
    }
    if (zero_args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    status = loom_amdgpu_emit_zero_merge_value(context, source_op, value_type,
                                               &zero_args[i]);
    if (iree_status_is_ok(status)) {
      placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
          .type = value_type,
          .value = zero_args[i],
      };
    }
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, plan->else_dispatch_block, zero_args,
                               arg_count, source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_else_dispatch_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_value_id_t low_condition,
    loom_type_t condition_type, loom_type_t active_type,
    const loom_amdgpu_branch_plan_t* plan) {
  IREE_ASSERT_EQ(plan->else_dispatch_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->else_dispatch_block);

  loom_op_t* else_active_op = NULL;
  iree_status_t status = iree_ok_status();
  if (plan->if_else_merge_arg_count == 0) {
    // Store-only branches do not introduce merge operands, so no later VALU
    // select can observe values from lanes that skipped a branch body.
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), &active_type, 1,
        &else_active_op);
  } else {
    // Value-producing branches may reach this block either from the true body
    // or from a no-true placeholder block. Recompute the else mask from the
    // original saved EXEC instead of assuming the predecessor left EXEC set to
    // exactly the true lanes.
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
    const loom_value_id_t true_mask_operands[] = {
        saved_exec,
        low_condition,
    };
    loom_op_t* true_mask_op = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
          true_mask_operands, IREE_ARRAYSIZE(true_mask_operands),
          loom_named_attr_slice_empty(), &condition_type, 1, &true_mask_op);
    }
    if (iree_status_is_ok(status)) {
      const loom_value_id_t true_mask = loom_op_const_results(true_mask_op)[0];
      status = loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64_EXEC,
          &true_mask, 1, loom_named_attr_slice_empty(), &active_type, 1,
          &else_active_op);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_block_t* const restore_dest = plan->true_only_restore_block != NULL
                                           ? plan->true_only_restore_block
                                           : plan->restore_block;
    status = loom_amdgpu_emit_plain_cond_branch(
        context, source_op, loom_op_const_results(else_active_op)[0],
        plan->else_body_block, restore_dest);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_divergent_loop_blocks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t exec_mask_type, loom_amdgpu_loop_plan_t* plan,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = plan->saved_exec;
  if (plan->saved_exec != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT(plan->entry_block != NULL);
  IREE_ASSERT(plan->header_dest != NULL);
  IREE_ASSERT(plan->restore_block != NULL);
  IREE_ASSERT(plan->restore_dest != NULL);
  IREE_ASSERT_EQ(plan->entry_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, plan->entry_block);

  loom_op_t* read_exec_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
      &read_exec_op);
  if (iree_status_is_ok(status)) {
    *out_saved_exec =
        loom_value_slice_get(loom_low_op_results(read_exec_op), 0);
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(
        builder, plan->header_dest, plan->entry_block->arg_ids,
        plan->entry_block->arg_count, source_op->location, &branch_op);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_exec_restore_block(
        context, source_op, *out_saved_exec, plan->restore_block,
        plan->restore_dest);
  }
  if (iree_status_is_ok(status)) {
    plan->saved_exec = *out_saved_exec;
  }
  return status;
}

static iree_status_t loom_amdgpu_emit_exec_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_block_t* low_true_dest,
    loom_block_t* low_false_dest, loom_type_t condition_type) {
  loom_low_lower_plan_t branch_plan = loom_low_lower_plan_empty();
  if (!loom_low_lower_lookup_branch_plan(context, source_op, &branch_plan)) {
    IREE_ASSERT_UNREACHABLE("AMDGPU divergent branch has no prepared plan");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      branch_plan.id != LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP) {
    IREE_ASSERT_UNREACHABLE("AMDGPU divergent branch plan id is invalid");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_amdgpu_loop_exit_plan_t* loop_exit_plan = NULL;
  const loom_amdgpu_branch_plan_t* plan = NULL;
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP) {
    loop_exit_plan =
        (const loom_amdgpu_loop_exit_plan_t*)branch_plan.target_data;
    IREE_ASSERT(loop_exit_plan != NULL);
  } else {
    plan = (const loom_amdgpu_branch_plan_t*)branch_plan.target_data;
    IREE_ASSERT(plan != NULL);
  }

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  loom_value_id_t loop_saved_exec = LOOM_VALUE_ID_INVALID;
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_divergent_loop_blocks(
        context, source_op, condition_type, loop_exit_plan->loop,
        &loop_saved_exec));
    if (loop_exit_plan->exit_successor_mask == 3) {
      IREE_ASSERT_EQ(low_true_dest, low_false_dest);
      loom_op_t* branch_op = NULL;
      return loom_low_br_build(loom_low_lower_context_builder(context),
                               low_true_dest, /*args=*/NULL, 0,
                               source_op->location, &branch_op);
    }
    if (loop_exit_plan->exit_successor_mask == 1) {
      // Within the saved parent mask, XOR complements the exit condition.
      // AND_SAVEEXEC still excludes lanes retired on earlier iterations.
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          loop_saved_exec, low_condition, condition_type, &low_condition));
      loom_block_t* repeat_dest = low_false_dest;
      low_false_dest = low_true_dest;
      low_true_dest = repeat_dest;
    }
  }

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
  const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
  if (plan != NULL && plan->restore_block != NULL) {
    const loom_value_id_t restore_exec = saved_exec;
    if (plan->restore_passthrough_terminator != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_passthrough_block(
          context, source_op, restore_exec, plan->restore_block,
          plan->restore_dest, plan->restore_passthrough_terminator));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_block(
          context, source_op, restore_exec, plan->restore_block,
          plan->restore_dest));
    }
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_THEN_MASKED_REGION &&
      plan->merge_restore_block != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
  }

  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, condition_type,
        active_type, plan));
  }
  if (branch_plan.id == LOOM_AMDGPU_BRANCH_PLAN_IF_ELSE_DIAMOND &&
      plan->if_else_merge_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_no_true_else_entry_block(
        context, source_op, saved_exec, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_exec_restore_branch(
        context, source_op, saved_exec, plan->true_only_restore_block,
        plan->merge_restore_dest, plan->else_dispatch_block->arg_ids,
        plan->if_else_merge_arg_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_if_else_merge_restore_block(
        context, source_op, saved_exec, low_condition, plan));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_else_dispatch_block(
        context, source_op, saved_exec, low_condition, condition_type,
        active_type, plan));
  }

  return loom_amdgpu_emit_plain_cond_branch(context, source_op, active,
                                            low_true_dest, low_false_dest);
}

// A uniform predicate may still need a native mask for vector selects. Its
// storage does not make the branch divergent. Ignore inactive lanes without
// changing EXEC or introducing masked merges for the branch payloads.
static iree_status_t loom_amdgpu_emit_uniform_mask_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_type_t condition_type,
    loom_block_t* low_true_dest, loom_block_t* low_false_dest) {
  loom_op_t* exec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ, NULL,
      0, loom_named_attr_slice_empty(), &condition_type, 1, &exec_op));
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_type_t result_types[] = {condition_type, scc_type};
  const loom_value_id_t operands[] = {
      low_condition,
      loom_op_const_results(exec_op)[0],
  };
  loom_op_t* test_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &test_op));
  return loom_amdgpu_emit_plain_cond_branch(context, source_op,
                                            loom_op_const_results(test_op)[1],
                                            low_true_dest, low_false_dest);
}

iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest) {
  (void)user_data;
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t condition_type = loom_module_value_type(module, low_condition);
  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SCC, 1)) {
    return loom_amdgpu_emit_plain_cond_branch(context, source_op, low_condition,
                                              low_true_dest, low_false_dest);
  }

  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1)) {
    return loom_amdgpu_emit_sgpr_bool_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest);
  }

  if (loom_amdgpu_condition_is_reg_class(context, condition_type,
                                         LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2)) {
    bool edge_implied_condition = false;
    bool edge_condition_proven = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_cfg_cond_br_edge_implied_bool(
        context, source_op, &edge_implied_condition, &edge_condition_proven));
    if (edge_condition_proven) {
      loom_block_t* low_dest =
          edge_implied_condition ? low_true_dest : low_false_dest;
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(loom_low_lower_context_builder(context),
                               low_dest, NULL, 0, source_op->location,
                               &low_br_op);
    }
    if (loom_amdgpu_branch_condition_is_uniform(context, source_op)) {
      return loom_amdgpu_emit_uniform_mask_cond_branch(
          context, source_op, low_condition, condition_type, low_true_dest,
          low_false_dest);
    }
    return loom_amdgpu_emit_exec_mask_cond_branch(
        context, source_op, low_condition, low_true_dest, low_false_dest,
        condition_type);
  }

  return loom_low_lower_emit_branch_condition_type_unsupported(
      context, source_op, condition_type, IREE_SV("amdgpu.branch_condition"));
}
