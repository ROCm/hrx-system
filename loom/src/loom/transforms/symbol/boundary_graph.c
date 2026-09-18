// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/boundary_graph.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/type_registry.h"
#include "loom/util/walk.h"

static bool loom_refine_boundaries_call_kind_participates(
    loom_call_like_kind_t kind) {
  switch (kind) {
    case LOOM_CALL_LIKE_KIND_SEMANTIC:
      return true;
    case LOOM_CALL_LIKE_KIND_LOW_INTERNAL:
    case LOOM_CALL_LIKE_KIND_LOW_INVOKE:
    case LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM:
    case LOOM_CALL_LIKE_KIND_NONE:
    default:
      return false;
  }
}

bool loom_refine_boundaries_read_call(const loom_module_t* module,
                                      loom_op_t* op, loom_call_like_t* out_call,
                                      loom_symbol_ref_t* out_callee,
                                      loom_value_slice_t* out_operands,
                                      loom_value_slice_t* out_results) {
  loom_call_like_t call = loom_call_like_cast(module, op);
  if (!loom_call_like_isa(call) ||
      !loom_refine_boundaries_call_kind_participates(
          loom_call_like_kind(call))) {
    if (out_call) {
      *out_call = (loom_call_like_t){0};
    }
    *out_callee = loom_symbol_ref_null();
    *out_operands = (loom_value_slice_t){0};
    *out_results = (loom_value_slice_t){0};
    return false;
  }
  if (out_call) {
    *out_call = call;
  }
  *out_callee = loom_call_like_callee(call);
  *out_operands = loom_call_like_operands(call);
  *out_results = loom_call_like_results(call);
  return true;
}

bool loom_refine_boundaries_callee_node(
    const loom_refine_boundaries_graph_t* graph, loom_symbol_ref_t callee,
    iree_host_size_t* out_node) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= graph->symbol_to_node_count) {
    return false;
  }
  iree_host_size_t node = graph->symbol_to_node[callee.symbol_id];
  if (node == IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_node = node;
  return true;
}

static bool loom_refine_boundaries_region_projects_arguments(
    const loom_module_t* module, loom_region_t* region,
    const loom_value_id_t* argument_ids, uint16_t argument_count) {
  if (!region || region->block_count == 0) {
    return false;
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  if (entry_block->arg_count != argument_count) {
    return false;
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    loom_value_id_t projected_argument = loom_block_arg_id(entry_block, i);
    if (!loom_type_equal(loom_module_value_type(module, argument_ids[i]),
                         loom_module_value_type(module, projected_argument))) {
      return false;
    }
  }
  return true;
}

static bool loom_refine_boundaries_region_declares_argument_projection(
    const loom_op_vtable_t* vtable, uint8_t region_index,
    uint8_t body_region_index) {
  if (region_index == body_region_index) {
    return true;
  }
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  return descriptor &&
         iree_any_bit_set(descriptor->flags, LOOM_REGION_PROJECT_FUNC_ARGS);
}

static iree_status_t loom_refine_boundaries_collect_argument_projections(
    const loom_module_t* module, loom_refine_boundaries_function_t* function,
    iree_arena_allocator_t* arena) {
  function->argument_projections = NULL;
  function->argument_projection_count = 0;

  loom_op_t* op = function->function.op;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  uint8_t body_region_index = function->function.vtable->body_region_index;
  uint8_t projection_count = 0;
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (!loom_refine_boundaries_region_declares_argument_projection(
            vtable, i, body_region_index)) {
      continue;
    }
    if (loom_refine_boundaries_region_projects_arguments(
            module, regions[i], function->argument_ids,
            function->argument_count)) {
      ++projection_count;
    }
  }
  if (projection_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, projection_count, sizeof(*function->argument_projections),
      (void**)&function->argument_projections));
  function->argument_projection_count = projection_count;

  uint8_t projection_index = 0;
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!loom_refine_boundaries_region_declares_argument_projection(
            vtable, i, body_region_index)) {
      continue;
    }
    if (!loom_refine_boundaries_region_projects_arguments(
            module, region, function->argument_ids, function->argument_count)) {
      continue;
    }
    function->argument_projections[projection_index++] =
        (loom_refine_boundaries_argument_projection_t){
            .region = region,
            .entry_block = loom_region_entry_block(region),
        };
  }
  return iree_ok_status();
}

typedef struct loom_refine_boundaries_successor_walk_t {
  // Function graph adapter.
  const loom_refine_boundaries_graph_t* graph;

  // SCC successor visitor.
  loom_scc_successor_callback_t visitor;
} loom_refine_boundaries_successor_walk_t;

static iree_status_t loom_refine_boundaries_visit_successor_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_refine_boundaries_successor_walk_t* walk =
      (loom_refine_boundaries_successor_walk_t*)user_data;
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  loom_value_slice_t operands = {0};
  loom_value_slice_t results = {0};
  if (!loom_refine_boundaries_read_call(walk->graph->module, op, NULL, &callee,
                                        &operands, &results)) {
    return iree_ok_status();
  }

  iree_host_size_t callee_node = IREE_HOST_SIZE_MAX;
  if (!loom_refine_boundaries_callee_node(walk->graph, callee, &callee_node)) {
    return iree_ok_status();
  }
  return walk->visitor.fn(walk->visitor.user_data, callee_node);
}

static iree_status_t loom_refine_boundaries_visit_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t visitor) {
  const loom_refine_boundaries_graph_t* graph =
      (const loom_refine_boundaries_graph_t*)user_data;

  loom_refine_boundaries_successor_walk_t walk = {
      .graph = graph,
      .visitor = visitor,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  // SCC successor callbacks may recursively walk callees while this function's
  // traversal is still live. Reclaim only the frames owned by this invocation.
  const iree_arena_checkpoint_t walk_checkpoint =
      iree_arena_checkpoint_save(graph->walk_arena);
  iree_status_t status = loom_walk_function(
      graph->module, graph->functions[node].function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_refine_boundaries_visit_successor_call,
                             &walk},
      graph->walk_arena, &walk_result);
  iree_arena_checkpoint_restore(&walk_checkpoint);
  return status;
}

iree_status_t loom_refine_boundaries_build_graph(
    loom_module_t* module, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* walk_arena,
    loom_refine_boundaries_graph_t* out_graph, loom_scc_list_t* out_sccs) {
  memset(out_graph, 0, sizeof(*out_graph));
  out_graph->module = module;
  out_graph->walk_arena = walk_arena;
  out_graph->symbol_to_node_count = module->symbols.count;
  if (module->symbols.count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*out_graph->symbol_to_node),
      (void**)&out_graph->symbol_to_node));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    out_graph->symbol_to_node[i] = IREE_HOST_SIZE_MAX;
  }

  iree_host_size_t function_count = 0;
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        !symbol->defining_op) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (loom_func_like_body(function)) {
      ++function_count;
    }
  }

  if (function_count == 0) {
    *out_sccs = (loom_scc_list_t){0};
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function_count, sizeof(*out_graph->functions),
      (void**)&out_graph->functions));
  memset(out_graph->functions, 0,
         function_count * sizeof(*out_graph->functions));
  out_graph->function_count = function_count;

  iree_host_size_t node = 0;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        !symbol->defining_op) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }

    loom_symbol_id_t symbol_id =
        (loom_symbol_id_t)(symbol - module->symbols.entries);
    loom_refine_boundaries_function_t* info = &out_graph->functions[node];
    info->function = function;
    info->argument_ids =
        loom_func_like_arg_ids(function, &info->argument_count);
    info->result_count = function.op->result_count;
    info->can_refine_boundary =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE) &&
        loom_func_like_repr_contract(function) == LOOM_STRING_ID_INVALID &&
        loom_func_like_is_module_internal(function);
    IREE_RETURN_IF_ERROR(loom_refine_boundaries_collect_argument_projections(
        module, info, arena));
    if (info->result_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count, sizeof(*info->return_facts),
          (void**)&info->return_facts));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count, sizeof(*info->return_fact_defined),
          (void**)&info->return_fact_defined));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count,
          sizeof(*info->return_forward_argument_indices),
          (void**)&info->return_forward_argument_indices));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, info->result_count,
          sizeof(*info->return_forward_result_indices),
          (void**)&info->return_forward_result_indices));
      memset(info->return_fact_defined, 0,
             info->result_count * sizeof(*info->return_fact_defined));
      for (uint16_t i = 0; i < info->result_count; ++i) {
        info->return_forward_argument_indices[i] =
            LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN;
        info->return_forward_result_indices[i] =
            LOOM_REFINE_BOUNDARIES_FORWARD_UNSEEN;
      }
    }
    out_graph->symbol_to_node[symbol_id] = node;
    ++node;
  }

  loom_scc_graph_t scc_graph = {
      .node_count = function_count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_refine_boundaries_visit_successors, out_graph),
  };
  return loom_scc_compute(&scc_graph, NULL, arena, out_sccs);
}
