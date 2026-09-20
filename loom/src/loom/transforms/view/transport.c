// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/transport.h"

#include "loom/analysis/view_regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/walk.h"

#define LOOM_VIEW_TRANSPORT_STATISTICS(V, statistics_type)            \
  V(statistics_type, values_decomposed, "values-decomposed",          \
    "Number of view arguments and results replaced by byte offsets.") \
  V(statistics_type, operands_decomposed, "operands-decomposed",      \
    "Number of control-flow operands replaced by byte offsets.")

LOOM_PASS_STATISTICS_DEFINE(loom_view_transport_statistics,
                            loom_view_transport_statistics_t,
                            LOOM_VIEW_TRANSPORT_STATISTICS)

static const loom_pass_info_t kPassInfo = {
    .name = IREE_SVL("decompose-view-transports"),
    .description = IREE_SVL("Carry common-buffer views as byte offsets."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_view_transport_statistics_layout,
};

#define LOOM_VIEW_ROOT_SELECTION_STATISTICS(V, statistics_type)           \
  V(statistics_type, selections_decomposed, "selections-decomposed",      \
    "Number of view selections replaced by correlated buffer and offset " \
    "selections.")

LOOM_PASS_STATISTICS_DEFINE(loom_view_root_selection_statistics,
                            loom_view_root_selection_statistics_t,
                            LOOM_VIEW_ROOT_SELECTION_STATISTICS)

static const loom_pass_info_t kRootSelectionPassInfo = {
    .name = IREE_SVL("decompose-view-root-selections"),
    .description =
        IREE_SVL("Carry distinct-root view selections as buffer-offset pairs."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_view_root_selection_statistics_layout,
};

const loom_pass_info_t* loom_decompose_view_transports_pass_info(void) {
  return &kPassInfo;
}

const loom_pass_info_t* loom_decompose_view_root_selections_pass_info(void) {
  return &kRootSelectionPassInfo;
}

//===----------------------------------------------------------------------===//
// Function-local transport plan
//===----------------------------------------------------------------------===//

typedef enum loom_view_transport_placement_e {
  LOOM_VIEW_TRANSPORT_BEFORE_ANCHOR = 0,
  LOOM_VIEW_TRANSPORT_AFTER_ANCHOR = 1,
} loom_view_transport_placement_t;

typedef struct loom_view_transport_value_t {
  // Original view value, reused as the offset value after decomposition.
  loom_value_id_t value_id;
  // Original view type reconstructed over the transported offset.
  loom_type_t type;
  // Materializing buffer retained by the view fact producer.
  loom_value_id_t buffer_value_id;
  // Operation anchoring reconstruction at the value's definition.
  loom_op_t* anchor;
  // Reconstruction side of anchor: before for arguments, after for results.
  loom_view_transport_placement_t placement;
  // First dependent selection edge, or IREE_HOST_SIZE_MAX.
  iree_host_size_t first_dependent;
  // Whether all required components are available for this transport.
  bool selected;
} loom_view_transport_value_t;

typedef struct loom_view_transport_dependency_t {
  // Transport requiring the source component to be decomposed.
  iree_host_size_t target;
  // Next dependent of the same source, or IREE_HOST_SIZE_MAX.
  iree_host_size_t next;
} loom_view_transport_dependency_t;

typedef struct loom_view_transport_offset_t {
  // Existing or once-materialized complete byte offset, or INVALID.
  loom_value_id_t value_id;
  // Existing scalar or selected view-transport offset added to expression.
  loom_value_id_t base_value_id;
  // Retained affine expression added to base_value_id, or NULL when
  // unavailable.
  const loom_symbolic_expr_t* expression;
  // Transport required to materialize base_value_id, or IREE_HOST_SIZE_MAX.
  iree_host_size_t dependency;
} loom_view_transport_offset_t;

typedef struct loom_view_transport_operand_t {
  // Operation containing the control-flow payload operand.
  loom_op_t* op;
  // Flat operand ordinal to rewrite.
  uint16_t operand_index;
  // Transport whose selection authorizes this operand rewrite.
  iree_host_size_t target;
  // View-region ordinal identifying the source's retained offset recipe.
  loom_view_region_id_t source_region;
} loom_view_transport_operand_t;

typedef struct loom_view_transport_plan_t {
  // Module being normalized.
  loom_module_t* module;
  // Pass arena owning every plan array.
  iree_arena_allocator_t* arena;
  // Compact list of candidate argument and result values.
  loom_view_transport_value_t* values;
  // Number of candidate values.
  iree_host_size_t value_count;
  // Allocated candidate capacity.
  iree_host_size_t value_capacity;
  // Acquired function-local correspondence domain.
  loom_local_value_domain_t domain;
  // Candidate index by local ordinal, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* indices;
  // Borrowed, completely populated view-region analysis.
  const loom_view_region_table_t* regions;
  // Offset recipes indexed by view-region ordinal.
  loom_view_transport_offset_t* offsets;
  // Selection dependencies, also linking members of one SCF signature.
  loom_view_transport_dependency_t* dependencies;
  // Number of populated dependency edges.
  iree_host_size_t dependency_count;
  // Allocated dependency capacity.
  iree_host_size_t dependency_capacity;
  // Recorded control-flow operand rewrites.
  loom_view_transport_operand_t* operands;
  // Number of recorded operand rewrites.
  iree_host_size_t operand_count;
  // Allocated operand capacity.
  iree_host_size_t operand_capacity;
} loom_view_transport_plan_t;

static iree_status_t loom_view_transport_add_value(
    loom_view_transport_plan_t* plan, loom_value_id_t value_id,
    loom_op_t* anchor, loom_view_transport_placement_t placement) {
  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_view(type)) {
    return iree_ok_status();
  }
  if (plan->value_count == plan->value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->value_count, plan->value_count + 1,
        sizeof(*plan->values), &plan->value_capacity, (void**)&plan->values));
  }
  plan->values[plan->value_count++] = (loom_view_transport_value_t){
      .value_id = value_id,
      .type = type,
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .anchor = anchor,
      .placement = placement,
      .first_dependent = IREE_HOST_SIZE_MAX,
  };
  return iree_ok_status();
}

static iree_status_t loom_view_transport_add_arguments(
    loom_view_transport_plan_t* plan, loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_view_transport_add_value(
        plan, loom_block_arg_id(block, i), block->first_op,
        LOOM_VIEW_TRANSPORT_BEFORE_ANCHOR));
  }
  return iree_ok_status();
}

static bool loom_view_transport_is_selection(const loom_module_t* module,
                                             const loom_op_t* op) {
  loom_value_id_t payload;
  return loom_op_first_operand_with_role(
      module, op, LOOM_OPERAND_ROLE_SELECT_PAYLOAD, &payload);
}

static iree_status_t loom_view_transport_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_view_transport_plan_t* plan = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (op == context->block->first_op &&
      context->block != loom_region_entry_block(context->region) &&
      iree_any_bit_set(context->region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    IREE_RETURN_IF_ERROR(
        loom_view_transport_add_arguments(plan, context->block));
  }
  const loom_loop_like_t loop = loom_loop_like_cast(plan->module, op);
  if (loom_loop_like_isa(loop)) {
    IREE_RETURN_IF_ERROR(loom_view_transport_add_arguments(
        plan, loom_region_entry_block(loom_loop_like_body(loop))));
    loom_region_t* condition = loom_loop_like_condition_region(loop);
    if (condition) {
      IREE_RETURN_IF_ERROR(loom_view_transport_add_arguments(
          plan, loom_region_entry_block(condition)));
    }
  }
  if (loom_loop_like_isa(loop) ||
      loom_region_branch_isa(loom_region_branch_cast(plan->module, op)) ||
      loom_view_transport_is_selection(plan->module, op)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_view_transport_add_value(
          plan, loom_op_results(op)[i], op, LOOM_VIEW_TRANSPORT_AFTER_ANCHOR));
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_view_transport_index(
    const loom_view_transport_plan_t* plan, loom_value_id_t value_id) {
  if (!plan->indices) {
    return IREE_HOST_SIZE_MAX;
  }
  return plan
      ->indices[loom_local_value_domain_ordinal(&plan->domain, value_id)];
}

static iree_status_t loom_view_transport_add_dependency(
    loom_view_transport_plan_t* plan, iree_host_size_t source,
    iree_host_size_t target) {
  if (source == IREE_HOST_SIZE_MAX || target == IREE_HOST_SIZE_MAX ||
      source == target) {
    return iree_ok_status();
  }
  if (plan->dependency_count == plan->dependency_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->dependency_count, plan->dependency_count + 1,
        sizeof(*plan->dependencies), &plan->dependency_capacity,
        (void**)&plan->dependencies));
  }
  plan->dependencies[plan->dependency_count] =
      (loom_view_transport_dependency_t){
          .target = target,
          .next = plan->values[source].first_dependent,
      };
  plan->values[source].first_dependent = plan->dependency_count++;
  return iree_ok_status();
}

static iree_status_t loom_view_transport_link_signature(
    loom_view_transport_plan_t* plan, loom_value_id_t left,
    loom_value_id_t right) {
  const iree_host_size_t left_index = loom_view_transport_index(plan, left);
  const iree_host_size_t right_index = loom_view_transport_index(plan, right);
  IREE_RETURN_IF_ERROR(
      loom_view_transport_add_dependency(plan, left_index, right_index));
  return loom_view_transport_add_dependency(plan, right_index, left_index);
}

static void loom_view_transport_plan_offset(
    const loom_view_transport_plan_t* plan, const loom_view_region_t* region,
    loom_view_transport_offset_t* offset) {
  *offset = (loom_view_transport_offset_t){
      .value_id = region->begin_value_id,
      .base_value_id = LOOM_VALUE_ID_INVALID,
      .dependency = IREE_HOST_SIZE_MAX,
  };
  if (offset->value_id != LOOM_VALUE_ID_INVALID) {
    return;
  }
  // An authored base retains its address-domain conversions and assumptions.
  // Expanding it into canonical terms could discard those producer facts.
  if (loom_symbolic_expr_is_linear(&region->projection_byte_offset)) {
    const iree_host_size_t base =
        loom_view_transport_index(plan, region->base_view_value_id);
    if (base != IREE_HOST_SIZE_MAX) {
      offset->base_value_id = region->base_view_value_id;
      offset->dependency = base;
    } else {
      const loom_view_region_t* base_region = NULL;
      if (loom_view_region_table_try_lookup(
              plan->regions, region->base_view_value_id, &base_region)) {
        offset->base_value_id = base_region->begin_value_id;
      }
    }
    if (offset->base_value_id != LOOM_VALUE_ID_INVALID) {
      offset->expression = &region->projection_byte_offset;
      return;
    }
  }
  if (loom_symbolic_expr_is_linear(&region->begin_byte_offset)) {
    offset->expression = &region->begin_byte_offset;
  }
}

static iree_status_t loom_view_transport_add_operand(
    loom_view_transport_plan_t* plan, loom_op_t* op, uint16_t operand_index,
    loom_value_id_t target_value) {
  const iree_host_size_t target = loom_view_transport_index(plan, target_value);
  if (target == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  const loom_view_region_t* source_region = NULL;
  const loom_value_id_t source = loom_op_operands(op)[operand_index];
  (void)loom_view_region_table_try_lookup(plan->regions, source,
                                          &source_region);
  const loom_view_transport_offset_t* offset =
      &plan->offsets[source_region->region_id];
  if (offset->value_id == LOOM_VALUE_ID_INVALID && !offset->expression) {
    plan->values[target].selected = false;
  }
  IREE_RETURN_IF_ERROR(
      loom_view_transport_add_dependency(plan, offset->dependency, target));
  if (plan->operand_count == plan->operand_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->operand_count, plan->operand_count + 1,
        sizeof(*plan->operands), &plan->operand_capacity,
        (void**)&plan->operands));
  }
  plan->operands[plan->operand_count++] = (loom_view_transport_operand_t){
      .op = op,
      .operand_index = operand_index,
      .target = target,
      .source_region = source_region->region_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_view_transport_plan_loop(
    loom_view_transport_plan_t* plan, loom_loop_like_t loop) {
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop);
  if (initial.count == 0) {
    return iree_ok_status();
  }
  loom_block_t* body = loom_region_entry_block(loom_loop_like_body(loop));
  loom_region_t* condition = loom_loop_like_condition_region(loop);
  loom_block_t* entry = condition ? loom_region_entry_block(condition) : body;
  const uint16_t argument_offset =
      loom_loop_like_iv(loop) == LOOM_VALUE_ID_INVALID ? 0 : 1;
  const uint16_t initial_offset =
      (uint16_t)(initial.values - loom_op_const_operands(loop.op));
  for (uint16_t i = 0; i < initial.count; ++i) {
    const loom_value_id_t entry_value =
        loom_block_arg_id(entry, i + argument_offset);
    const loom_value_id_t body_value =
        loom_block_arg_id(body, i + argument_offset);
    IREE_RETURN_IF_ERROR(loom_view_transport_link_signature(
        plan, entry_value, loom_op_results(loop.op)[i]));
    IREE_RETURN_IF_ERROR(
        loom_view_transport_link_signature(plan, entry_value, body_value));
    IREE_RETURN_IF_ERROR(loom_view_transport_add_operand(
        plan, loop.op, initial_offset + i, entry_value));
    IREE_RETURN_IF_ERROR(
        loom_view_transport_add_operand(plan, body->last_op, i, entry_value));
    if (condition) {
      IREE_RETURN_IF_ERROR(loom_view_transport_add_operand(plan, entry->last_op,
                                                           i + 1, body_value));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_view_transport_plan_operands(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_view_transport_plan_t* plan = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  const loom_loop_like_t loop = loom_loop_like_cast(plan->module, op);
  if (loom_loop_like_isa(loop)) {
    return loom_view_transport_plan_loop(plan, loop);
  }
  const loom_region_branch_t branch = loom_region_branch_cast(plan->module, op);
  if (loom_region_branch_isa(branch)) {
    for (uint8_t r = 0; r < op->region_count; ++r) {
      loom_op_t* yield =
          loom_region_branch_region_terminator(plan->module, branch, r);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_view_transport_add_operand(
            plan, yield, i, loom_op_results(op)[i]));
      }
    }
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(plan->module, op);
  uint16_t payload_index = 0;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_op_operand_role_at(vtable, op, i) !=
        LOOM_OPERAND_ROLE_SELECT_PAYLOAD) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_view_transport_add_operand(
        plan, op, i, loom_op_results(op)[payload_index++ % op->result_count]));
  }
  for (uint16_t s = 0; s < op->successor_count; ++s) {
    loom_block_t* successor = loom_op_successors(op)[s];
    const loom_value_id_t* sources = NULL;
    uint16_t count = 0;
    if (loom_cfg_terminator_payload_for_successor(op, successor, &sources,
                                                  &count)) {
      const uint16_t start = (uint16_t)(sources - loom_op_const_operands(op));
      for (uint16_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(loom_view_transport_add_operand(
            plan, op, start + i, loom_block_arg_id(successor, i)));
      }
    } else {
      for (uint16_t i = 0; i < successor->arg_count; ++i) {
        const iree_host_size_t target =
            loom_view_transport_index(plan, loom_block_arg_id(successor, i));
        if (target != IREE_HOST_SIZE_MAX) {
          plan->values[target].selected = false;
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_view_transport_select(
    loom_view_transport_plan_t* plan) {
  iree_host_size_t* unavailable = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, plan->value_count,
                                                 sizeof(*unavailable),
                                                 (void**)&unavailable));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < plan->value_count; ++i) {
    if (!plan->values[i].selected) {
      unavailable[count++] = i;
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    for (iree_host_size_t edge = plan->values[unavailable[i]].first_dependent;
         edge != IREE_HOST_SIZE_MAX; edge = plan->dependencies[edge].next) {
      const iree_host_size_t target = plan->dependencies[edge].target;
      if (!plan->values[target].selected) {
        continue;
      }
      plan->values[target].selected = false;
      unavailable[count++] = target;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Offset materialization and reconstruction
//===----------------------------------------------------------------------===//

static iree_status_t loom_view_transport_constant(loom_builder_t* builder,
                                                  int64_t value,
                                                  loom_location_id_t location,
                                                  loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_view_transport_materialize_offset(
    loom_view_transport_plan_t* plan, loom_rewriter_t* rewriter,
    loom_view_region_id_t region_id, loom_value_id_t* out_value) {
  loom_view_transport_offset_t* offset = &plan->offsets[region_id];
  if (offset->value_id != LOOM_VALUE_ID_INVALID) {
    *out_value = offset->value_id;
    return iree_ok_status();
  }
  const loom_symbolic_expr_t* expression = offset->expression;
  if (offset->base_value_id != LOOM_VALUE_ID_INVALID &&
      expression->constant == 0 && expression->term_count == 0) {
    offset->value_id = offset->base_value_id;
    *out_value = offset->value_id;
    return iree_ok_status();
  }
  const loom_value_t* view = loom_module_value(
      plan->module, plan->regions->regions[region_id].view_value_id);
  loom_builder_t* builder = &rewriter->builder;
  const loom_op_t* anchor = loom_value_is_block_arg(view)
                                ? loom_value_def_block(view)->first_op
                                : loom_value_def_op(view);
  if (loom_value_is_block_arg(view)) {
    loom_builder_set_before(builder, anchor);
  } else {
    loom_builder_set_after(builder, anchor);
  }
  loom_value_id_t sum = offset->base_value_id;
  // Physical byte expressions use the full offset width. Logical index
  // carriers may be narrower on the selected target. Signed intermediates
  // also preserve negative affine coefficients before the complete offset.
  const loom_type_t arithmetic_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  if (sum != LOOM_VALUE_ID_INVALID) {
    loom_op_t* cast = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        builder, sum, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
        arithmetic_type, anchor->location, &cast));
    sum = loom_index_cast_result(cast);
  }
  if (expression->constant != 0 ||
      (sum == LOOM_VALUE_ID_INVALID && expression->term_count == 0)) {
    loom_value_id_t constant;
    IREE_RETURN_IF_ERROR(loom_view_transport_constant(
        builder, expression->constant, anchor->location, &constant));
    if (sum == LOOM_VALUE_ID_INVALID) {
      sum = constant;
    } else {
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          builder, 0, sum, constant, arithmetic_type, anchor->location, &add));
      sum = loom_scalar_addi_result(add);
    }
  }
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t* term = &expression->terms[i];
    // The identity representative retains predicates established at its
    // producer while denoting the same numeric term as the canonical value.
    loom_value_id_t value = term->relation_value_id;
    const loom_type_t type = loom_module_value_type(plan->module, value);
    if (!loom_type_equal(type, arithmetic_type)) {
      loom_op_t* cast = NULL;
      const loom_scalar_type_t scalar_type = loom_type_element_type(type);
      if (scalar_type == LOOM_SCALAR_TYPE_I1) {
        IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
            builder, value, type, arithmetic_type, anchor->location, &cast));
      } else if (loom_scalar_type_is_integer(scalar_type)) {
        IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
            builder, value, type, arithmetic_type, anchor->location, &cast));
      } else {
        IREE_RETURN_IF_ERROR(loom_index_cast_build(
            builder, value, type, arithmetic_type, anchor->location, &cast));
      }
      value = loom_op_results(cast)[0];
    }
    if (term->coefficient != 1) {
      loom_value_id_t coefficient;
      IREE_RETURN_IF_ERROR(loom_view_transport_constant(
          builder, term->coefficient, anchor->location, &coefficient));
      loom_op_t* multiply = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(builder, 0, value,
                                                  coefficient, arithmetic_type,
                                                  anchor->location, &multiply));
      value = loom_scalar_muli_result(multiply);
    }
    if (sum == LOOM_VALUE_ID_INVALID) {
      sum = value;
    } else {
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          builder, 0, sum, value, arithmetic_type, anchor->location, &add));
      sum = loom_scalar_addi_result(add);
    }
  }
  loom_op_t* cast = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, sum, arithmetic_type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      anchor->location, &cast));
  offset->value_id = loom_index_cast_result(cast);
  *out_value = offset->value_id;
  return iree_ok_status();
}

static iree_status_t loom_view_transport_reconstruct(
    loom_rewriter_t* rewriter, const loom_view_transport_value_t* value) {
  loom_builder_t* builder = &rewriter->builder;
  if (value->placement == LOOM_VIEW_TRANSPORT_BEFORE_ANCHOR) {
    loom_builder_set_before(builder, value->anchor);
  } else {
    loom_builder_set_after(builder, value->anchor);
  }
  // Reserve the replacement before moving uses so predicate and type references
  // move with operand uses. The original value then becomes the offset operand
  // of the reconstructed view, without a self-replacement exception.
  loom_value_id_t replacement;
  IREE_RETURN_IF_ERROR(loom_builder_reserve_results(builder, 1, &replacement));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(rewriter->module, replacement, value->type));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_move_value_name(rewriter, value->value_id, replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      rewriter, replacement, value->value_id, IREE_SV("offset")));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, value->value_id, replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
      rewriter, value->value_id, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)));
  loom_op_t* view = NULL;
  return loom_buffer_view_build(builder, value->buffer_value_id,
                                value->value_id, value->type,
                                value->anchor->location, &view);
}

static iree_status_t loom_view_transport_rewrite(
    loom_view_transport_plan_t* plan, loom_rewriter_t* rewriter,
    loom_view_transport_statistics_t* statistics) {
  for (iree_host_size_t i = 0; i < plan->value_count; ++i) {
    if (!plan->values[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_view_transport_reconstruct(rewriter, &plan->values[i]));
    ++statistics->values_decomposed;
  }
  for (iree_host_size_t i = 0; i < plan->operand_count; ++i) {
    const loom_view_transport_operand_t* operand = &plan->operands[i];
    if (!plan->values[operand->target].selected) {
      continue;
    }
    loom_value_id_t offset;
    IREE_RETURN_IF_ERROR(loom_view_transport_materialize_offset(
        plan, rewriter, operand->source_region, &offset));
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, operand->op, operand->operand_index, offset));
    ++statistics->operands_decomposed;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass lifecycle
//===----------------------------------------------------------------------===//

static iree_status_t loom_view_transport_prepare(
    loom_view_transport_plan_t* plan, loom_pass_t* pass,
    loom_func_like_t function, loom_rewriter_t* rewriter) {
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, plan->module, loom_pass_value_fact_scope_function(function),
      &facts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->indices),
      (void**)&plan->indices));
  for (iree_host_size_t i = 0; i < plan->domain.value_count; ++i) {
    plan->indices[i] = IREE_HOST_SIZE_MAX;
  }
  loom_dominance_info_t dominance;
  IREE_RETURN_IF_ERROR(loom_dominance_info_initialize_region(
      plan->module, loom_func_like_body(function), plan->arena, &dominance));
  for (iree_host_size_t i = 0; i < plan->value_count; ++i) {
    loom_view_transport_value_t* value = &plan->values[i];
    plan->indices[loom_local_value_domain_ordinal(&plan->domain,
                                                  value->value_id)] = i;
    loom_value_fact_view_reference_t reference;
    if (!loom_value_facts_query_view_reference(
            &facts->context,
            loom_value_fact_table_lookup(facts, value->value_id), &reference) ||
        reference.buffer_value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    value->buffer_value_id = reference.buffer_value_id;
    value->selected =
        value->placement == LOOM_VIEW_TRANSPORT_BEFORE_ANCHOR
            ? loom_value_is_available_before_op(
                  &dominance, value->buffer_value_id, value->anchor)
            : loom_dominates_value(&dominance, value->buffer_value_id,
                                   value->anchor);
  }
  loom_symbolic_expr_context_t expressions;
  loom_symbolic_expr_context_initialize(plan->module, &plan->domain, facts,
                                        plan->arena, &expressions);
  loom_view_region_table_t regions;
  IREE_RETURN_IF_ERROR(
      loom_view_region_table_initialize(&plan->domain, &expressions, &regions));
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze(&regions));
  plan->regions = &regions;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, regions.region_count, sizeof(*plan->offsets),
      (void**)&plan->offsets));
  for (iree_host_size_t i = 0; i < regions.region_count; ++i) {
    loom_view_transport_plan_offset(plan, &regions.regions[i],
                                    &plan->offsets[i]);
  }
  loom_walk_result_t result;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      plan->module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_view_transport_plan_operands,
                             .user_data = plan},
      plan->arena, &result));
  IREE_RETURN_IF_ERROR(loom_view_transport_select(plan));
  return loom_view_transport_rewrite(plan, rewriter,
                                     loom_view_transport_statistics(pass));
}

iree_status_t loom_decompose_view_transports_run(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  loom_view_transport_plan_t plan = {.module = module, .arena = pass->arena};
  loom_walk_result_t result;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_view_transport_collect,
                             .user_data = &plan},
      pass->arena, &result));
  if (plan.value_count == 0) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_status_t status = loom_local_value_domain_acquire_for_region_tree(
      module, body, pass->arena, &plan.domain);
  if (iree_status_is_ok(status)) {
    status = loom_view_transport_prepare(&plan, pass, function, &rewriter);
  }
  if (iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }
  loom_local_value_domain_release(&plan.domain);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

//===----------------------------------------------------------------------===//
// Distinct-root selections
//===----------------------------------------------------------------------===//

typedef struct loom_view_root_selection_source_t {
  // Materializing buffer for a leaf view, or INVALID for a nested selection.
  loom_value_id_t buffer_value_id;
  // View-region offset recipe for a leaf view, or INVALID when nested.
  loom_view_region_id_t region_id;
  // Earlier selection supplying the source pair, or IREE_HOST_SIZE_MAX.
  iree_host_size_t selection_index;
} loom_view_root_selection_source_t;

typedef struct loom_view_root_selection_t {
  // Original scf.select operation replaced by this plan entry.
  loom_op_t* op;
  // Original view type reconstructed after selecting the storage coordinate.
  loom_type_t view_type;
  // True-value storage coordinate.
  loom_view_root_selection_source_t true_source;
  // False-value storage coordinate.
  loom_view_root_selection_source_t false_source;
  // Rewritten selected materializing buffer, or INVALID before rewriting.
  loom_value_id_t buffer_value_id;
  // Rewritten selected root-relative offset, or INVALID before rewriting.
  loom_value_id_t offset_value_id;
  // Whether both source coordinates can be materialized exactly.
  bool selected;
} loom_view_root_selection_t;

typedef struct loom_view_root_selection_plan_t {
  // Module being normalized.
  loom_module_t* module;
  // Pass arena owning every plan allocation.
  iree_arena_allocator_t* arena;
  // Acquired function-local correspondence domain.
  loom_local_value_domain_t domain;
  // Borrowed, completely populated function value facts.
  const loom_value_fact_table_t* facts;
  // Function dominance used to prove materializers available at selections.
  loom_dominance_info_t dominance;
  // Compact selection entries in definition order.
  loom_view_root_selection_t* selections;
  // Number of populated selection entries.
  iree_host_size_t selection_count;
  // Allocated selection entry capacity.
  iree_host_size_t selection_capacity;
  // Selection index by local result ordinal, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* selection_indices;
  // Shared root-relative offset analysis and materialization state.
  loom_view_transport_plan_t offsets;
} loom_view_root_selection_plan_t;

static iree_status_t loom_view_root_selection_collect(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_view_root_selection_plan_t* plan = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_scf_select_isa(op) ||
      !loom_type_is_view(
          loom_module_value_type(plan->module, loom_scf_select_result(op)))) {
    return iree_ok_status();
  }
  if (plan->selection_count == plan->selection_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->selection_count, plan->selection_count + 1,
        sizeof(*plan->selections), &plan->selection_capacity,
        (void**)&plan->selections));
  }
  plan->selections[plan->selection_count++] = (loom_view_root_selection_t){
      .op = op,
      .view_type =
          loom_module_value_type(plan->module, loom_scf_select_result(op)),
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .offset_value_id = LOOM_VALUE_ID_INVALID,
  };
  return iree_ok_status();
}

static bool loom_view_root_selection_plan_source(
    loom_view_root_selection_plan_t* plan, iree_host_size_t selection_index,
    loom_value_id_t source_value_id,
    loom_view_root_selection_source_t* out_source) {
  *out_source = (loom_view_root_selection_source_t){
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .region_id = LOOM_VIEW_REGION_ID_INVALID,
      .selection_index = IREE_HOST_SIZE_MAX,
  };
  const loom_value_ordinal_t source_ordinal =
      loom_local_value_domain_ordinal(&plan->domain, source_value_id);
  const iree_host_size_t source_selection_index =
      plan->selection_indices[source_ordinal];
  if (source_selection_index < selection_index &&
      plan->selections[source_selection_index].selected) {
    out_source->selection_index = source_selection_index;
    return true;
  }

  loom_value_fact_view_reference_t reference;
  if (!loom_value_facts_query_view_reference(
          &plan->facts->context,
          loom_value_fact_table_lookup(plan->facts, source_value_id),
          &reference) ||
      reference.buffer_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_value_is_available_before_op(
          &plan->dominance, reference.buffer_value_id,
          plan->selections[selection_index].op)) {
    return false;
  }
  const loom_view_region_t* region = NULL;
  if (!loom_view_region_table_try_lookup(plan->offsets.regions, source_value_id,
                                         &region)) {
    return false;
  }
  const loom_view_transport_offset_t* offset =
      &plan->offsets.offsets[region->region_id];
  if (offset->value_id == LOOM_VALUE_ID_INVALID && !offset->expression) {
    return false;
  }
  out_source->buffer_value_id = reference.buffer_value_id;
  out_source->region_id = region->region_id;
  return true;
}

static void loom_view_root_selection_plan_entries(
    loom_view_root_selection_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->selection_count; ++i) {
    loom_view_root_selection_t* selection = &plan->selections[i];
    loom_value_fact_view_reference_t result_reference;
    const loom_value_id_t result = loom_scf_select_result(selection->op);
    if (!loom_value_facts_query_view_reference(
            &plan->facts->context,
            loom_value_fact_table_lookup(plan->facts, result),
            &result_reference) ||
        result_reference.buffer_value_id != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    selection->selected =
        loom_view_root_selection_plan_source(
            plan, i, loom_scf_select_true_value(selection->op),
            &selection->true_source) &&
        loom_view_root_selection_plan_source(
            plan, i, loom_scf_select_false_value(selection->op),
            &selection->false_source);
  }
}

static iree_status_t loom_view_root_selection_materialize_source(
    loom_view_root_selection_plan_t* plan, loom_rewriter_t* rewriter,
    const loom_view_root_selection_source_t* source,
    loom_value_id_t* out_buffer_value_id,
    loom_value_id_t* out_offset_value_id) {
  if (source->selection_index != IREE_HOST_SIZE_MAX) {
    const loom_view_root_selection_t* selection =
        &plan->selections[source->selection_index];
    *out_buffer_value_id = selection->buffer_value_id;
    *out_offset_value_id = selection->offset_value_id;
    return iree_ok_status();
  }
  *out_buffer_value_id = source->buffer_value_id;
  return loom_view_transport_materialize_offset(
      &plan->offsets, rewriter, source->region_id, out_offset_value_id);
}

static iree_status_t loom_view_root_selection_rewrite(
    loom_view_root_selection_plan_t* plan, loom_rewriter_t* rewriter,
    loom_view_root_selection_statistics_t* statistics) {
  for (iree_host_size_t i = 0; i < plan->selection_count; ++i) {
    loom_view_root_selection_t* selection = &plan->selections[i];
    if (!selection->selected) {
      continue;
    }
    loom_value_id_t true_values[2];
    IREE_RETURN_IF_ERROR(loom_view_root_selection_materialize_source(
        plan, rewriter, &selection->true_source, &true_values[0],
        &true_values[1]));
    loom_value_id_t false_values[2];
    IREE_RETURN_IF_ERROR(loom_view_root_selection_materialize_source(
        plan, rewriter, &selection->false_source, &false_values[0],
        &false_values[1]));

    loom_builder_t* builder = &rewriter->builder;
    loom_builder_set_before(builder, selection->op);
    const loom_value_id_t value_checkpoint =
        loom_rewriter_value_checkpoint(rewriter);
    const loom_type_t result_types[] = {
        loom_type_buffer(), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET)};
    loom_op_t* if_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_if_build(
        builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
        loom_scf_select_condition(selection->op), result_types,
        IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
        /*tied_result_count=*/0, selection->op->location, &if_op));

    loom_builder_ip_t saved_ip = loom_builder_enter_region(
        builder, if_op, loom_scf_if_then_region(if_op));
    loom_op_t* yield_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(builder, true_values, IREE_ARRAYSIZE(true_values),
                             selection->op->location, &yield_op));
    loom_builder_restore(builder, saved_ip);
    saved_ip = loom_builder_enter_region(builder, if_op,
                                         loom_scf_if_else_region(if_op));
    IREE_RETURN_IF_ERROR(loom_scf_yield_build(
        builder, false_values, IREE_ARRAYSIZE(false_values),
        selection->op->location, &yield_op));
    loom_builder_restore(builder, saved_ip);

    const loom_value_slice_t if_results = loom_scf_if_results(if_op);
    selection->buffer_value_id = if_results.values[0];
    selection->offset_value_id = if_results.values[1];
    const loom_value_id_t old_result = loom_scf_select_result(selection->op);
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        rewriter, old_result, selection->buffer_value_id, IREE_SV("root")));
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        rewriter, old_result, selection->offset_value_id, IREE_SV("offset")));
    loom_op_t* view_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        builder, selection->buffer_value_id, selection->offset_value_id,
        selection->view_type, selection->op->location, &view_op));
    const loom_value_id_t replacement = loom_buffer_view_result(view_op);
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, selection->op, &replacement, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, selection->op, &replacement, 1));
    ++statistics->selections_decomposed;
  }
  return iree_ok_status();
}

static iree_status_t loom_view_root_selection_prepare(
    loom_view_root_selection_plan_t* plan, loom_pass_t* pass,
    loom_func_like_t function, loom_rewriter_t* rewriter) {
  loom_value_fact_table_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, plan->module, loom_pass_value_fact_scope_function(function),
      &facts));
  plan->facts = facts;
  IREE_RETURN_IF_ERROR(loom_dominance_info_initialize_region(
      plan->module, loom_func_like_body(function), plan->arena,
      &plan->dominance));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->selection_indices),
      (void**)&plan->selection_indices));
  for (iree_host_size_t i = 0; i < plan->domain.value_count; ++i) {
    plan->selection_indices[i] = IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t i = 0; i < plan->selection_count; ++i) {
    const loom_value_id_t result =
        loom_scf_select_result(plan->selections[i].op);
    plan->selection_indices[loom_local_value_domain_ordinal(&plan->domain,
                                                            result)] = i;
  }

  loom_symbolic_expr_context_t expressions;
  loom_symbolic_expr_context_initialize(plan->module, &plan->domain, facts,
                                        plan->arena, &expressions);
  loom_view_region_table_t regions;
  IREE_RETURN_IF_ERROR(
      loom_view_region_table_initialize(&plan->domain, &expressions, &regions));
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze(&regions));
  plan->offsets = (loom_view_transport_plan_t){
      .module = plan->module,
      .arena = plan->arena,
      .domain = plan->domain,
      .regions = &regions,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, regions.region_count, sizeof(*plan->offsets.offsets),
      (void**)&plan->offsets.offsets));
  for (iree_host_size_t i = 0; i < regions.region_count; ++i) {
    loom_view_transport_plan_offset(&plan->offsets, &regions.regions[i],
                                    &plan->offsets.offsets[i]);
  }
  loom_view_root_selection_plan_entries(plan);
  return loom_view_root_selection_rewrite(
      plan, rewriter, loom_view_root_selection_statistics(pass));
}

iree_status_t loom_decompose_view_root_selections_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  loom_view_root_selection_plan_t plan = {
      .module = module,
      .arena = pass->arena,
  };
  loom_walk_result_t result;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_view_root_selection_collect,
                             .user_data = &plan},
      pass->arena, &result));
  if (plan.selection_count == 0) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_status_t status = loom_local_value_domain_acquire_for_region_tree(
      module, body, pass->arena, &plan.domain);
  if (iree_status_is_ok(status)) {
    status = loom_view_root_selection_prepare(&plan, pass, function, &rewriter);
  }
  if (iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }
  loom_local_value_domain_release(&plan.domain);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
