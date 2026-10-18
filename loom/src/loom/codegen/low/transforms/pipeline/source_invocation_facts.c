// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_invocation_facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/util/call_graph.h"
#include "loom/util/reference_transfer.h"
#include "loom/util/walk.h"

typedef struct loom_low_source_invocation_collect_t {
  // Module containing the source calls being summarized.
  const loom_module_t* module;

  // Complete call graph used to reject recursive callee boundaries.
  const loom_call_graph_t* call_graph;

  // Caller-local facts consumed while visiting call operands.
  const loom_value_fact_table_t* caller_facts;

  // Per-caller outgoing facts, merged into the durable summary after the
  // caller fact scope is released.
  loom_value_fact_table_t* outgoing_facts;
} loom_low_source_invocation_collect_t;

static iree_status_t loom_low_source_invocation_merge_fact(
    const loom_module_t* module, loom_value_fact_table_t* target,
    loom_value_id_t value_id, const loom_value_fact_table_t* source,
    loom_value_facts_t facts) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_value_fact_table_has_entry(target, value_id)) {
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        target, source, module, type, facts, &cloned_facts));
    return loom_value_fact_table_define(target, value_id, cloned_facts);
  }

  const loom_value_facts_t existing =
      loom_value_fact_table_lookup(target, value_id);
  loom_value_facts_t joined_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
      target, module, type, target, existing, source, facts, &joined_facts));
  return loom_value_fact_table_define(target, value_id, joined_facts);
}

static iree_status_t loom_low_source_invocation_merge_argument_fact(
    const loom_module_t* module, loom_value_fact_table_t* target,
    loom_func_like_t callee_function, loom_value_id_t argument_id,
    const loom_value_fact_table_t* source, loom_value_facts_t facts) {
  const loom_type_t type = loom_module_value_type(module, argument_id);
  loom_value_facts_t projected_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
      target, source, module, type, facts, &projected_facts));
  const loom_value_fact_reference_origin_t argument_origin = {
      .function_symbol_id = loom_func_like_callee(callee_function).symbol_id,
      .region_index = loom_func_like_body_region_index(callee_function),
      .kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
      .entry_value_id = argument_id,
  };
  IREE_RETURN_IF_ERROR(loom_value_facts_rebase_call_argument(
      &target->context, argument_id, argument_origin, &projected_facts));
  return loom_low_source_invocation_merge_fact(module, target, argument_id,
                                               target, projected_facts);
}

static bool loom_low_source_invocation_can_refine_callee(
    const loom_low_source_invocation_collect_t* collect,
    loom_symbol_ref_t callee, loom_func_like_t* out_function) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= collect->module->symbols.count ||
      loom_call_graph_is_recursive(collect->call_graph, callee.symbol_id)) {
    return false;
  }

  const loom_symbol_t* symbol =
      &collect->module->symbols.entries[callee.symbol_id];
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE) ||
      symbol->defining_op == NULL) {
    return false;
  }
  const loom_func_like_t function =
      loom_func_like_cast(collect->module, symbol->defining_op);
  if (!loom_func_like_isa(function) || loom_func_like_body(function) == NULL ||
      !loom_func_like_is_module_internal(function)) {
    return false;
  }

  *out_function = function;
  return true;
}

static iree_status_t loom_low_source_invocation_collect_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_low_source_invocation_collect_t* collect =
      (loom_low_source_invocation_collect_t*)user_data;
  const loom_call_like_t call = loom_call_like_cast(collect->module, op);
  if (!loom_call_like_isa(call) ||
      loom_call_like_kind(call) != LOOM_CALL_LIKE_KIND_SEMANTIC) {
    return iree_ok_status();
  }

  loom_func_like_t callee_function = {0};
  if (!loom_low_source_invocation_can_refine_callee(
          collect, loom_call_like_callee(call), &callee_function)) {
    return iree_ok_status();
  }

  const loom_value_slice_t operands = loom_call_like_operands(call);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(callee_function, &argument_count);
  IREE_ASSERT_EQ(operands.count, argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_facts_t operand_facts =
        loom_value_fact_table_lookup(collect->caller_facts, operands.values[i]);
    IREE_RETURN_IF_ERROR(loom_low_source_invocation_merge_argument_fact(
        collect->module, collect->outgoing_facts, callee_function, arguments[i],
        collect->caller_facts, operand_facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_merge_outgoing(
    const loom_module_t* module, const loom_value_fact_table_t* outgoing,
    loom_value_fact_table_t* arguments) {
  for (iree_host_size_t i = 0; i < outgoing->touched_count; ++i) {
    const loom_value_id_t value_id = outgoing->touched_values[i];
    IREE_RETURN_IF_ERROR(loom_low_source_invocation_merge_fact(
        module, arguments, value_id, outgoing,
        loom_value_fact_table_lookup(outgoing, value_id)));
  }
  return iree_ok_status();
}

static const loom_low_source_selection_t*
loom_low_source_invocation_find_selection(
    const loom_low_source_selection_t* const* selections_by_symbol,
    iree_host_size_t selection_count, loom_symbol_id_t symbol_id) {
  return symbol_id < selection_count ? selections_by_symbol[symbol_id] : NULL;
}

iree_status_t loom_low_source_invocation_facts_compute(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_source_selection_list_t* selections,
    iree_arena_allocator_t* arena,
    loom_low_source_invocation_facts_t* out_facts) {
  memset(out_facts, 0, sizeof(*out_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
      &out_facts->arguments, arena, module->values.count));
  loom_type_registry_configure_fact_context(&out_facts->arguments.context);
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(module->arena.block_pool, &analysis_arena);
  iree_arena_allocator_t outgoing_arena;
  iree_arena_initialize(module->arena.block_pool, &outgoing_arena);

  const loom_low_source_selection_t** selections_by_symbol = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &analysis_arena, module->symbols.count, sizeof(*selections_by_symbol),
      (void**)&selections_by_symbol);
  if (iree_status_is_ok(status)) {
    memset(selections_by_symbol, 0,
           module->symbols.count * sizeof(*selections_by_symbol));
    for (iree_host_size_t i = 0; i < selections->count; ++i) {
      const loom_low_source_selection_t* selection = &selections->values[i];
      if (selection->kind != LOOM_LOW_SOURCE_SELECTION_FUNCTION) {
        continue;
      }
      const loom_symbol_ref_t function_ref =
          loom_func_like_callee(selection->func);
      selections_by_symbol[function_ref.symbol_id] = selection;
    }
  }

  loom_call_graph_t call_graph = {0};
  if (iree_status_is_ok(status)) {
    status = loom_call_graph_build(module, &analysis_arena, &call_graph);
  }
  for (uint16_t scc = call_graph.scc_count;
       iree_status_is_ok(status) && scc > 0; --scc) {
    const uint16_t scc_id = (uint16_t)(scc - 1);
    for (uint16_t node_index = 0;
         iree_status_is_ok(status) && node_index < call_graph.node_count;
         ++node_index) {
      const loom_call_graph_node_t* node = &call_graph.nodes[node_index];
      if (node->scc_id != scc_id) {
        continue;
      }

      const loom_symbol_t* symbol = &module->symbols.entries[node->symbol_id];
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      const loom_low_source_selection_t* selection =
          loom_low_source_invocation_find_selection(
              selections_by_symbol, module->symbols.count, node->symbol_id);
      const loom_target_facts_t* target_facts =
          selection ? selection->target_facts : NULL;
      loom_pass_value_fact_scope_t scope =
          loom_pass_value_fact_scope_conditioned_function_for_target(
              function, target_facts);
      scope.seed_facts =
          loom_low_source_invocation_facts_for_function(out_facts, function);

      loom_value_fact_table_t* caller_facts = NULL;
      status =
          loom_pass_value_facts_acquire(pass, module, scope, &caller_facts);
      if (!iree_status_is_ok(status)) {
        break;
      }

      iree_arena_reset(&outgoing_arena);
      loom_value_fact_table_t outgoing_facts = {0};
      status =
          loom_value_fact_table_initialize(&outgoing_facts, &outgoing_arena, 0);
      loom_type_registry_configure_fact_context(&outgoing_facts.context);
      if (iree_status_is_ok(status)) {
        loom_low_source_invocation_collect_t collect = {
            .module = module,
            .call_graph = &call_graph,
            .caller_facts = caller_facts,
            .outgoing_facts = &outgoing_facts,
        };
        loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
        status = loom_walk_function(
            module, function, LOOM_WALK_PRE_ORDER,
            (loom_walk_callback_t){loom_low_source_invocation_collect_call,
                                   &collect},
            &walk_result);
      }
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
      if (iree_status_is_ok(status)) {
        status = loom_low_source_invocation_merge_outgoing(
            module, &outgoing_facts, &out_facts->arguments);
      }
    }
  }

  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  iree_arena_deinitialize(&outgoing_arena);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

loom_value_fact_table_view_t loom_low_source_invocation_facts_for_function(
    const loom_low_source_invocation_facts_t* facts,
    loom_func_like_t function) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  return (loom_value_fact_table_view_t){
      .table = &facts->arguments,
      .value_ids = arguments,
      .value_count = argument_count,
  };
}
