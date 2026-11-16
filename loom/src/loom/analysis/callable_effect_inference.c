// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/callable_effect_inference.h"

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ops/callable_effects.h"
#include "loom/ops/op_defs.h"
#include "loom/util/call_graph.h"

typedef struct loom_callable_effect_inference_t {
  // Module being finalized.
  loom_module_t* module;
  // Transient analysis storage.
  iree_arena_allocator_t* arena;
  // Direct call graph and SCC assignment.
  loom_call_graph_t graph;
  // Direct contracts collected with discovery, indexed by graph node ordinal.
  loom_callable_effect_contract_t* local_contracts;
  // Allocated entries in local_contracts.
  iree_host_size_t local_capacity;
  // Completed contracts indexed by the graph's callee-first components.
  loom_callable_effect_contract_t* contracts;
  // Unannotated tracked calls awaiting a completed callee summary.
  loom_op_t** calls;
  // Number of entries in |calls|.
  iree_host_size_t call_count;
  // Allocated entry capacity in |calls|.
  iree_host_size_t call_capacity;
} loom_callable_effect_inference_t;

static loom_callable_effect_contract_t loom_callable_effect_join(
    loom_callable_effect_contract_t lhs, loom_callable_effect_contract_t rhs) {
  if (lhs == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED ||
      rhs == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  if (lhs == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY ||
      rhs == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
  }
  return LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
}

static loom_callable_effect_contract_t
loom_callable_effect_explicit_function_contract(const loom_module_t* module,
                                                loom_symbol_id_t symbol_id) {
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  const loom_func_like_t function =
      loom_func_like_cast(module, symbol->defining_op);
  const loom_callable_effect_contract_t contract =
      loom_func_like_effect_contract(function);
  return contract < LOOM_CALLABLE_EFFECT_CONTRACT_COUNT_
             ? contract
             : LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
}

static iree_status_t loom_callable_effect_record_call(
    loom_callable_effect_inference_t* inference, loom_op_t* op) {
  if (inference->call_count == inference->call_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        inference->arena, inference->call_count, inference->call_count + 1,
        sizeof(*inference->calls), &inference->call_capacity,
        (void**)&inference->calls));
  }
  inference->calls[inference->call_count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_callable_effect_begin_function(
    void* user_data, uint16_t node_index, loom_func_like_t function) {
  loom_callable_effect_inference_t* inference = user_data;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      inference->arena, node_index, (iree_host_size_t)node_index + 1,
      sizeof(*inference->local_contracts), &inference->local_capacity,
      (void**)&inference->local_contracts));
  const loom_callable_effect_contract_t authored_contract =
      loom_func_like_effect_contract(function);
  // Authored contracts are upper bounds even when the current body is pure.
  inference->local_contracts[node_index] =
      authored_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED
          ? LOOM_CALLABLE_EFFECT_CONTRACT_PURE
          : authored_contract;
  return iree_ok_status();
}

static iree_status_t loom_callable_effect_classify_op(
    void* user_data, uint16_t node_index, loom_op_t* op, loom_call_like_t call,
    const loom_walk_context_t* context) {
  (void)context;
  loom_callable_effect_inference_t* inference = user_data;
  loom_callable_effect_contract_t* contract =
      &inference->local_contracts[node_index];
  if (loom_call_like_isa(call)) {
    const loom_callable_effect_contract_t authored_contract =
        loom_call_like_effect_contract(call);
    if (authored_contract != LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
      *contract = loom_callable_effect_join(*contract, authored_contract);
    } else if (call.vtable->effect_contract_attr_index !=
               LOOM_ATTR_INDEX_NONE) {
      IREE_RETURN_IF_ERROR(loom_callable_effect_record_call(inference, op));
    }
    return iree_ok_status();
  }

  const loom_callable_effect_contract_t op_contract =
      loom_callable_effects_classify_traits(
          loom_op_effective_traits(inference->module, op));
  *contract = loom_callable_effect_join(*contract, op_contract);
  return iree_ok_status();
}

static loom_callable_effect_contract_t loom_callable_effect_symbol_contract(
    const loom_callable_effect_inference_t* inference,
    loom_symbol_id_t symbol_id) {
  const loom_call_graph_node_t* node =
      loom_call_graph_node(&inference->graph, symbol_id);
  if (node) {
    return inference->contracts[node->scc_id];
  }
  return loom_callable_effect_explicit_function_contract(inference->module,
                                                         symbol_id);
}

static iree_status_t loom_callable_effect_inference_build(
    loom_callable_effect_inference_t* inference) {
  const loom_call_graph_options_t options = {
      .visitor =
          {
              .user_data = inference,
              .function = loom_callable_effect_begin_function,
              .operation = loom_callable_effect_classify_op,
          },
  };
  IREE_RETURN_IF_ERROR(loom_call_graph_build(
      inference->module, &options, inference->arena, &inference->graph));
  if (inference->graph.node_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      inference->arena, inference->graph.components.count,
      sizeof(*inference->contracts), (void**)&inference->contracts));

  // Tarjan assigns cross-SCC callees lower IDs than their callers. Fold each
  // completed callee contract into its caller exactly once per deduplicated
  // call-graph edge.
  for (iree_host_size_t scc_id = 0; scc_id < inference->graph.components.count;
       ++scc_id) {
    const loom_scc_t* component = &inference->graph.components.values[scc_id];
    loom_callable_effect_contract_t contract =
        LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
    for (iree_host_size_t member = 0; member < component->node_count;
         ++member) {
      const iree_host_size_t node_index = component->nodes[member];
      const loom_call_graph_node_t* node = &inference->graph.nodes[node_index];
      contract = loom_callable_effect_join(
          contract, inference->local_contracts[node_index]);
      for (uint16_t i = 0; i < node->callee_count; ++i) {
        const loom_symbol_id_t callee_symbol_id = node->callees[i];
        const loom_call_graph_node_t* callee_node =
            loom_call_graph_node(&inference->graph, callee_symbol_id);
        if (callee_node && callee_node->scc_id == scc_id) {
          continue;
        }
        if (callee_node) {
          IREE_ASSERT_LT(callee_node->scc_id, scc_id);
        }
        contract = loom_callable_effect_join(
            contract,
            loom_callable_effect_symbol_contract(inference, callee_symbol_id));
      }
    }
    inference->contracts[scc_id] = contract;
  }

  // Calls retained during classification now receive a durable contract. The
  // generic setter refreshes effective traits and all ancestor summaries, so
  // verification and later passes never need to repeat this graph analysis.
  for (iree_host_size_t i = 0; i < inference->call_count; ++i) {
    const loom_call_like_t call =
        loom_call_like_cast(inference->module, inference->calls[i]);
    const loom_symbol_ref_t callee = loom_call_like_callee(call);
    const loom_callable_effect_contract_t contract =
        loom_callable_effect_symbol_contract(inference, callee.symbol_id);
    if (contract != LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
      loom_call_like_set_effect_contract(inference->module, call, contract);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_callable_effect_inference_run(loom_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  // Effect contracts decorate semantic and native calls. Command launches
  // contribute dependency edges but have no effect contract to finalize.
  if (!loom_module_has_source_calls(module) &&
      !loom_module_has_low_internal_calls(module)) {
    return iree_ok_status();
  }
  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_callable_effect_inference_t inference = {
      .module = module,
      .arena = &arena,
  };
  const iree_status_t status = loom_callable_effect_inference_build(&inference);
  iree_arena_deinitialize(&arena);
  return status;
}
