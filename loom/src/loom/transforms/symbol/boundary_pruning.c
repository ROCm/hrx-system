// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/boundary_pruning.h"

#include <stdint.h>
#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/util/walk.h"

typedef struct loom_refine_boundaries_prune_plan_t {
  // Arguments to remove, indexed by the original callee argument ordinal.
  bool* prune_arguments;

  // Results to remove, indexed by the original callee result ordinal.
  bool* prune_results;

  // Original callee argument count for this plan.
  uint16_t argument_count;

  // Original callee result count for this plan.
  uint16_t result_count;

  // True when at least one argument is marked for pruning.
  bool has_prunable_arguments;

  // True when at least one result is marked for pruning.
  bool has_prunable_results;
} loom_refine_boundaries_prune_plan_t;

static bool loom_refine_boundaries_argument_is_prunable(
    const loom_module_t* module, loom_value_id_t argument) {
  if (argument == LOOM_VALUE_ID_INVALID || argument >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, argument);
  return value->use_count == 0 && !loom_value_has_attribute_uses(value) &&
         !loom_module_value_has_type_uses(module, argument);
}

static bool loom_refine_boundaries_logical_argument_is_prunable(
    const loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    uint16_t argument_index) {
  for (uint8_t projection_index = 0;
       projection_index < function_info->argument_projection_count;
       ++projection_index) {
    const loom_refine_boundaries_argument_projection_t* projection =
        &function_info->argument_projections[projection_index];
    if (!loom_refine_boundaries_argument_is_prunable(
            module,
            loom_block_arg_id(projection->entry_block, argument_index))) {
      return false;
    }
  }
  return true;
}

static bool loom_refine_boundaries_result_is_tied(const loom_op_t* op,
                                                  uint16_t result_index) {
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == result_index) {
      return true;
    }
  }
  return false;
}

static bool loom_refine_boundaries_result_is_prunable(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index) {
  if (result_index >= op->result_count) {
    return false;
  }
  if (loom_refine_boundaries_result_is_tied(op, result_index)) {
    return false;
  }
  loom_value_id_t result = loom_op_const_results(op)[result_index];
  if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, result);
  return value->use_count == 0 && !loom_value_has_attribute_uses(value) &&
         !loom_module_value_has_type_uses(module, result);
}

static iree_status_t loom_refine_boundaries_build_prune_plans(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena,
    loom_refine_boundaries_prune_plan_t** out_plans) {
  *out_plans = NULL;
  if (graph->function_count == 0) {
    return iree_ok_status();
  }

  loom_refine_boundaries_prune_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->function_count, sizeof(*plans), (void**)&plans));
  memset(plans, 0, graph->function_count * sizeof(*plans));

  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_function_t* function_info =
        &graph->functions[node];
    if (!function_info->can_refine_boundary) {
      continue;
    }

    if (function_info->argument_projection_count == 0) {
      continue;
    }

    loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (function_info->argument_count > 0) {
      plan->argument_count = function_info->argument_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, function_info->argument_count, sizeof(*plan->prune_arguments),
          (void**)&plan->prune_arguments));
      memset(plan->prune_arguments, 0,
             function_info->argument_count * sizeof(*plan->prune_arguments));

      for (uint16_t i = 0; i < function_info->argument_count; ++i) {
        if (!loom_refine_boundaries_logical_argument_is_prunable(
                module, function_info, i)) {
          continue;
        }
        plan->prune_arguments[i] = true;
        plan->has_prunable_arguments = true;
      }
    }

    if (function_info->result_count > 0) {
      plan->result_count = function_info->result_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, function_info->result_count, sizeof(*plan->prune_results),
          (void**)&plan->prune_results));
      memset(plan->prune_results, 0,
             function_info->result_count * sizeof(*plan->prune_results));

      for (uint16_t i = 0; i < function_info->result_count; ++i) {
        if (!loom_refine_boundaries_result_is_prunable(
                module, function_info->function.op, i)) {
          continue;
        }
        plan->prune_results[i] = true;
        plan->has_prunable_results = true;
      }
    }
  }

  *out_plans = plans;
  return iree_ok_status();
}

static void loom_refine_boundaries_recompute_prune_plan(
    loom_refine_boundaries_prune_plan_t* plan) {
  plan->has_prunable_arguments = false;
  for (uint16_t i = 0; i < plan->argument_count; ++i) {
    if (plan->prune_arguments[i]) {
      plan->has_prunable_arguments = true;
      break;
    }
  }
  plan->has_prunable_results = false;
  for (uint16_t i = 0; i < plan->result_count; ++i) {
    if (plan->prune_results[i]) {
      plan->has_prunable_results = true;
      break;
    }
  }
}

typedef struct loom_refine_boundaries_prune_call_walk_t {
  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Dense prune plans indexed by function node.
  loom_refine_boundaries_prune_plan_t* plans;
} loom_refine_boundaries_prune_call_walk_t;

static iree_status_t loom_refine_boundaries_preflight_pruned_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_prune_call_walk_t* walk =
      (loom_refine_boundaries_prune_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_call_like_t call = {0};
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->graph->module, op, &call, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_prune_plan_t* plan = &walk->plans[callee_node];
  if (!plan->has_prunable_arguments && !plan->has_prunable_results) {
    return iree_ok_status();
  }

  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  uint16_t operand_offset = loom_call_like_operand_offset(call);
  uint16_t result_offset = loom_call_like_result_offset(call);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    uint16_t operand_index = tied_results[i].operand_index;
    if (plan->has_prunable_arguments && operand_index >= operand_offset) {
      uint16_t argument_index = (uint16_t)(operand_index - operand_offset);
      plan->prune_arguments[argument_index] = false;
    }

    uint16_t result_index = tied_results[i].result_index;
    if (plan->has_prunable_results && result_index >= result_offset) {
      uint16_t call_result_index = (uint16_t)(result_index - result_offset);
      plan->prune_results[call_result_index] = false;
    }
  }

  if (plan->has_prunable_results) {
    for (uint16_t i = 0; i < plan->result_count; ++i) {
      if (!plan->prune_results[i]) {
        continue;
      }
      if (!loom_refine_boundaries_result_is_prunable(walk->graph->module, op,
                                                     result_offset + i)) {
        plan->prune_results[i] = false;
      }
    }
  }
  loom_refine_boundaries_recompute_prune_plan(plan);
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_copy_result_names(
    loom_module_t* module, const loom_op_t* old_op, loom_op_t* new_op,
    const uint16_t* old_to_new_result_indices) {
  const loom_value_id_t* old_results = loom_op_const_results(old_op);
  loom_value_id_t* new_results = loom_op_results(new_op);
  for (uint16_t i = 0; i < old_op->result_count; ++i) {
    uint16_t new_index = old_to_new_result_indices[i];
    if (new_index == UINT16_MAX) {
      continue;
    }
    loom_value_id_t old_result = old_results[i];
    loom_value_id_t new_result = new_results[new_index];
    if (old_result == LOOM_VALUE_ID_INVALID ||
        new_result == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_module_copy_value_name(module, old_result, new_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_build_pruned_call(
    loom_module_t* module, loom_op_t* op, loom_call_like_t call,
    loom_value_slice_t operands, loom_value_slice_t results,
    const loom_refine_boundaries_prune_plan_t* plan,
    iree_arena_allocator_t* arena, uint16_t** out_old_to_new_result_indices,
    loom_op_t** out_new_op) {
  *out_old_to_new_result_indices = NULL;
  *out_new_op = NULL;
  uint16_t operand_offset = loom_call_like_operand_offset(call);
  uint16_t result_offset = loom_call_like_result_offset(call);

  uint16_t* old_to_new_operand_indices = NULL;
  loom_value_id_t* new_operands = NULL;
  if (op->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op->operand_count, sizeof(*old_to_new_operand_indices),
        (void**)&old_to_new_operand_indices));
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      old_to_new_operand_indices[i] = UINT16_MAX;
    }
  }
  uint16_t kept_call_operand_count = 0;
  for (uint16_t i = 0; i < operands.count; ++i) {
    if (i < plan->argument_count && plan->prune_arguments[i]) {
      continue;
    }
    old_to_new_operand_indices[operand_offset + i] =
        (uint16_t)(operand_offset + kept_call_operand_count);
    ++kept_call_operand_count;
  }
  uint16_t new_operand_count =
      (uint16_t)(operand_offset + kept_call_operand_count);
  if (new_operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, new_operand_count,
                                                   sizeof(*new_operands),
                                                   (void**)&new_operands));
    const loom_value_id_t* old_operands = loom_op_const_operands(op);
    for (uint16_t i = 0; i < operand_offset; ++i) {
      old_to_new_operand_indices[i] = i;
      new_operands[i] = old_operands[i];
    }
    uint16_t kept_index = operand_offset;
    for (uint16_t i = 0; i < operands.count; ++i) {
      uint16_t old_index = (uint16_t)(operand_offset + i);
      if (old_to_new_operand_indices[old_index] == UINT16_MAX) {
        continue;
      }
      new_operands[kept_index++] = operands.values[i];
    }
  }

  uint16_t* old_to_new_result_indices = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, op->result_count, sizeof(*old_to_new_result_indices),
        (void**)&old_to_new_result_indices));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      old_to_new_result_indices[i] = UINT16_MAX;
    }
  }
  uint16_t kept_call_result_count = 0;
  for (uint16_t i = 0; i < results.count; ++i) {
    if (i < plan->result_count && plan->prune_results[i]) {
      continue;
    }
    old_to_new_result_indices[result_offset + i] =
        (uint16_t)(result_offset + kept_call_result_count);
    ++kept_call_result_count;
  }
  uint16_t new_result_count =
      (uint16_t)(result_offset + kept_call_result_count);

  loom_tied_result_t* tied_results = NULL;
  if (op->tied_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, op->tied_result_count,
                                                   sizeof(*tied_results),
                                                   (void**)&tied_results));
    const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
    for (uint16_t i = 0; i < op->tied_result_count; ++i) {
      tied_results[i] = old_tied_results[i];
      uint16_t old_operand_index = old_tied_results[i].operand_index;
      uint16_t new_operand_index =
          old_to_new_operand_indices[old_operand_index];
      tied_results[i].operand_index = new_operand_index;
      uint16_t old_result_index = old_tied_results[i].result_index;
      uint16_t new_result_index = old_to_new_result_indices[old_result_index];
      tied_results[i].result_index = new_result_index;
    }
  }

  loom_type_t* result_types = NULL;
  if (new_result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, new_result_count, sizeof(*result_types), (void**)&result_types));
    const loom_value_id_t* old_results = loom_op_const_results(op);
    for (uint16_t i = 0; i < result_offset; ++i) {
      old_to_new_result_indices[i] = i;
      result_types[i] = loom_module_value_type(module, old_results[i]);
    }
    for (uint16_t i = 0; i < results.count; ++i) {
      uint16_t old_index = (uint16_t)(result_offset + i);
      uint16_t new_index = old_to_new_result_indices[old_index];
      if (new_index == UINT16_MAX) {
        continue;
      }
      result_types[new_index] =
          loom_module_value_type(module, results.values[i]);
    }
  }
  *out_old_to_new_result_indices = old_to_new_result_indices;

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, op->parent_block, &builder);
  loom_builder_set_before(&builder, op);
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &builder, op->kind, new_operand_count, new_result_count,
      /*region_count=*/0, op->tied_result_count, op->attribute_count,
      op->location, out_new_op));
  if (new_operand_count > 0) {
    memcpy(loom_op_operands(*out_new_op), new_operands,
           (iree_host_size_t)new_operand_count * sizeof(*new_operands));
  }
  for (uint16_t i = 0; i < new_result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_value(
        &builder, result_types[i], &loom_op_results(*out_new_op)[i]));
  }
  if (op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(*out_new_op), tied_results,
           (iree_host_size_t)op->tied_result_count * sizeof(*tied_results));
  }
  if (op->attribute_count > 0) {
    memcpy(loom_op_attrs(*out_new_op), loom_op_const_attrs(op),
           (iree_host_size_t)op->attribute_count * sizeof(loom_attribute_t));
  }
  return loom_builder_finalize_op(&builder, *out_new_op);
}

typedef struct loom_refine_boundaries_rewrite_call_walk_t {
  // Module being rewritten.
  loom_module_t* module;

  // Function graph for resolving direct callees.
  const loom_refine_boundaries_graph_t* graph;

  // Dense prune plans indexed by function node.
  loom_refine_boundaries_prune_plan_t* plans;

  // Scratch arena for filtered operand and result type arrays.
  iree_arena_allocator_t* arena;
} loom_refine_boundaries_rewrite_call_walk_t;

static iree_status_t loom_refine_boundaries_rewrite_pruned_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;

  loom_refine_boundaries_rewrite_call_walk_t* walk =
      (loom_refine_boundaries_rewrite_call_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_call_like_t call = {0};
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->module, op, &call, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }

  loom_refine_boundaries_prune_plan_t* plan = &walk->plans[callee_node];
  if (!plan->has_prunable_arguments && !plan->has_prunable_results) {
    return iree_ok_status();
  }

  uint16_t* old_to_new_result_indices = NULL;
  loom_op_t* new_op = NULL;
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_build_pruned_call(
      walk->module, op, call, operands, results, plan, walk->arena,
      &old_to_new_result_indices, &new_op));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_copy_result_names(
      walk->module, op, new_op, old_to_new_result_indices));

  const loom_value_id_t* old_results = loom_op_const_results(op);
  const loom_value_id_t* new_results = loom_op_const_results(new_op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    uint16_t new_index = old_to_new_result_indices[i];
    if (new_index == UINT16_MAX) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
        walk->module, old_results[i], new_results[new_index]));
  }
  return loom_op_erase(walk->module, op);
}

static iree_status_t loom_refine_boundaries_preflight_pruned_calls(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans,
    iree_arena_allocator_t* walk_arena) {
  loom_refine_boundaries_prune_call_walk_t walk = {
      .graph = graph,
      .plans = plans,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(module, module->body, LOOM_WALK_PRE_ORDER,
                          (loom_walk_callback_t){
                              loom_refine_boundaries_preflight_pruned_call,
                              &walk,
                          },
                          walk_arena, &walk_result);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_calls(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans,
    iree_arena_allocator_t* walk_arena) {
  loom_refine_boundaries_rewrite_call_walk_t walk = {
      .module = module,
      .graph = graph,
      .plans = plans,
      .arena = walk_arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_region(module, module->body, LOOM_WALK_PRE_ORDER,
                          (loom_walk_callback_t){
                              loom_refine_boundaries_rewrite_pruned_call,
                              &walk,
                          },
                          walk_arena, &walk_result);
}

typedef struct loom_refine_boundaries_return_list_t {
  // Function body containing the exits being collected.
  loom_region_t* body;

  // Declared exit operation kind for direct blocks in |body|.
  loom_op_kind_t body_exit_kind;

  // Function return ops discovered before mutation.
  loom_op_t** ops;

  // Number of return ops in |ops|.
  iree_host_size_t count;

  // Allocated op pointer capacity.
  iree_host_size_t capacity;

  // Arena owning |ops|.
  iree_arena_allocator_t* arena;
} loom_refine_boundaries_return_list_t;

static iree_status_t loom_refine_boundaries_append_return_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_refine_boundaries_return_list_t* list =
      (loom_refine_boundaries_return_list_t*)user_data;
  if (op->kind != list->body_exit_kind ||
      op->parent_block->parent_region != list->body) {
    return iree_ok_status();
  }
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->ops),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_collect_return_ops(
    loom_module_t* module,
    const loom_refine_boundaries_function_t* function_info,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_return_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->body = function_info->body;
  out_list->body_exit_kind = function_info->body_exit_kind;
  out_list->arena = arena;
  out_list->capacity = 4;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, out_list->capacity,
                                                 sizeof(*out_list->ops),
                                                 (void**)&out_list->ops));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_arena_reset(walk_arena);
  return loom_walk_function(
      module, function_info->function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_append_return_op, out_list},
      walk_arena, &walk_result);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_return(
    loom_module_t* module, loom_op_t* return_op,
    const loom_refine_boundaries_prune_plan_t* plan,
    iree_arena_allocator_t* arena) {
  loom_value_slice_t operands = {
      .values = loom_op_operands(return_op),
      .count = return_op->operand_count,
  };

  loom_value_id_t* kept_operands = NULL;
  uint16_t kept_count = 0;
  if (operands.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, operands.count, sizeof(*kept_operands), (void**)&kept_operands));
    for (uint16_t i = 0; i < operands.count; ++i) {
      if (plan->prune_results[i]) {
        continue;
      }
      kept_operands[kept_count++] = operands.values[i];
    }
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, return_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, return_op);
  loom_op_t* new_return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &builder, return_op->kind, kept_count, /*result_count=*/0,
      /*region_count=*/0, /*tied_result_count=*/0, return_op->attribute_count,
      return_op->location, &new_return_op));
  if (kept_count > 0) {
    memcpy(loom_op_operands(new_return_op), kept_operands,
           (iree_host_size_t)kept_count * sizeof(*kept_operands));
  }
  if (return_op->attribute_count > 0) {
    memcpy(loom_op_attrs(new_return_op), loom_op_const_attrs(return_op),
           (iree_host_size_t)return_op->attribute_count *
               sizeof(loom_attribute_t));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&builder, new_return_op));
  return loom_op_erase(module, return_op);
}

static iree_status_t loom_refine_boundaries_rewrite_pruned_returns(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* walk_arena) {
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_results) {
      continue;
    }

    loom_refine_boundaries_return_list_t returns = {0};
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_return_ops(
        module, &graph->functions[node], arena, walk_arena, &returns));
    for (iree_host_size_t i = 0; i < returns.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_return(
          module, returns.ops[i], plan, arena));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_remove_pruned_results(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, iree_arena_allocator_t* arena,
    int64_t* out_pruned_count) {
  *out_pruned_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_results) {
      continue;
    }
    loom_op_t* function_op = graph->functions[node].function.op;
    uint16_t removed_count = 0;
    IREE_RETURN_IF_ERROR(loom_op_remove_results(
        module, function_op, plan->prune_results, arena, &removed_count));
    *out_pruned_count += removed_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_refine_boundaries_remove_pruned_arguments(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    loom_refine_boundaries_prune_plan_t* plans, int64_t* out_pruned_count) {
  *out_pruned_count = 0;
  for (iree_host_size_t node = 0; node < graph->function_count; ++node) {
    const loom_refine_boundaries_prune_plan_t* plan = &plans[node];
    if (!plan->has_prunable_arguments) {
      continue;
    }
    const loom_refine_boundaries_function_t* function_info =
        &graph->functions[node];
    for (uint8_t projection_index = 0;
         projection_index < function_info->argument_projection_count;
         ++projection_index) {
      loom_block_t* entry_block =
          function_info->argument_projections[projection_index].entry_block;
      for (uint16_t i = plan->argument_count; i > 0; --i) {
        uint16_t argument_index = (uint16_t)(i - 1);
        if (!plan->prune_arguments[argument_index]) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_block_remove_arg(module, entry_block, argument_index));
        *out_pruned_count += 1;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_refine_boundaries_prune_internal_boundaries(
    loom_module_t* module, const loom_refine_boundaries_graph_t* graph,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* walk_arena,
    int64_t* out_pruned_argument_count, int64_t* out_pruned_result_count) {
  *out_pruned_argument_count = 0;
  *out_pruned_result_count = 0;
  loom_refine_boundaries_prune_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(
      loom_refine_boundaries_build_prune_plans(module, graph, arena, &plans));
  if (!plans) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_refine_boundaries_preflight_pruned_calls(
      module, graph, plans, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_calls(
      module, graph, plans, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_rewrite_pruned_returns(
      module, graph, plans, arena, walk_arena));
  IREE_RETURN_IF_ERROR(loom_refine_boundaries_remove_pruned_results(
      module, graph, plans, arena, out_pruned_result_count));
  return loom_refine_boundaries_remove_pruned_arguments(
      module, graph, plans, out_pruned_argument_count);
}
