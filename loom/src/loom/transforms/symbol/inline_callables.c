// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_callables.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/selection.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_INLINE_CALLABLES_STATISTICS(V, statistics_type)     \
  V(statistics_type, required_edges, "required-edges",           \
    "Number of call edges required to inline by policy.")        \
  V(statistics_type, kept_edges, "kept-edges",                   \
    "Number of call edges left unchanged by policy.")            \
  V(statistics_type, calls_cloned, "calls-cloned",               \
    "Number of call sites inlined by cloning the callee body.")  \
  V(statistics_type, calls_transferred, "calls-transferred",     \
    "Number of call sites inlined by moving the callee body.")   \
  V(statistics_type, symbols_transferred, "symbols-transferred", \
    "Number of private callable symbols erased by transfer inline.")

LOOM_PASS_STATISTICS_DEFINE(loom_inline_callables_statistics,
                            loom_inline_callables_statistics_t,
                            LOOM_INLINE_CALLABLES_STATISTICS)

static const loom_pass_info_t loom_inline_callables_pass_info_storage = {
    .name = IREE_SVL("inline-callables"),
    .description =
        IREE_SVL("Inline call-like edges required by authored inline policy."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_inline_callables_statistics_layout,
};

const loom_pass_info_t* loom_inline_callables_pass_info(void) {
  return &loom_inline_callables_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Plan model
//===----------------------------------------------------------------------===//

#define LOOM_INLINE_PLAN_ENTRY_INVALID ((uint32_t)UINT32_MAX)

typedef enum loom_inline_plan_action_e {
  LOOM_INLINE_PLAN_ACTION_KEEP = 0,
  LOOM_INLINE_PLAN_ACTION_REQUIRED = 1,
  LOOM_INLINE_PLAN_ACTION_CLONE = 2,
  LOOM_INLINE_PLAN_ACTION_TRANSFER = 3,
  LOOM_INLINE_PLAN_ACTION_ERROR = 4,
} loom_inline_plan_action_t;

typedef enum loom_inline_blocker_e {
  LOOM_INLINE_BLOCKER_NONE = 0,
  LOOM_INLINE_BLOCKER_POLICY_CONFLICT = 1,
  LOOM_INLINE_BLOCKER_REQUIRED_CYCLE = 2,
  LOOM_INLINE_BLOCKER_CALL_NOT_CALL_LIKE = 3,
  LOOM_INLINE_BLOCKER_CALL_NOT_OWNED_BY_SYMBOL = 4,
  LOOM_INLINE_BLOCKER_INVALID_CALLEE_SYMBOL = 5,
  LOOM_INLINE_BLOCKER_CALLEE_NOT_FUNCTION_LIKE = 6,
  LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE = 7,
  LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND = 8,
  LOOM_INLINE_BLOCKER_NON_CALL_SHAPE = 9,
  LOOM_INLINE_BLOCKER_CALLEE_MISSING_BODY = 10,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_NOT_SINGLE_BLOCK = 11,
  LOOM_INLINE_BLOCKER_RECURSIVE_BODY = 12,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR = 13,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_RETURN = 14,
  LOOM_INLINE_BLOCKER_OPERAND_COUNT_MISMATCH = 15,
  LOOM_INLINE_BLOCKER_INVALID_OPERAND_OR_ARGUMENT = 16,
  LOOM_INLINE_BLOCKER_OPERAND_TYPE_MISMATCH = 17,
  LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH = 18,
  LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT = 19,
  LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH = 20,
} loom_inline_blocker_t;

typedef struct loom_inline_symbol_info_t {
  // Borrowed symbol table entry for this symbol id.
  const loom_symbol_t* symbol;
  // Function-like view of symbol->defining_op, or empty when not function-like.
  loom_func_like_t function;
  // Number of incoming direct call edges in the dependency snapshot.
  uint32_t incoming_call_count;
  // Number of incoming non-call symbol references in the dependency snapshot.
  uint32_t incoming_non_call_ref_count;
  // Number of incoming call edges selected for required inlining.
  uint32_t planned_call_removals;
} loom_inline_symbol_info_t;

typedef struct loom_inline_plan_entry_t {
  // Stable ordinal assigned in dependency edge order.
  uint32_t ordinal;
  // Dependency edge that produced this call plan entry.
  loom_symbol_dependency_edge_id_t dependency_edge_id;
  // Next required-inline entry with the same source symbol.
  uint32_t next_required_from_source;
  // Symbol whose definition owns call_op.
  loom_symbol_id_t source_symbol_id;
  // Symbol referenced by the call-like callee attr.
  loom_symbol_id_t target_symbol_id;
  // Direct call-like operation to rewrite.
  loom_op_t* call_op;
  // Call-like interface for call_op.
  loom_call_like_t call;
  // Function-like interface for the target definition.
  loom_func_like_t callee;
  // Inline policy read from the callee symbol definition.
  uint8_t callee_policy;
  // Inline policy read from the call site.
  uint8_t call_policy;
  // Effective edge inline policy after conflict resolution.
  uint8_t effective_policy;
  // Temperature hint read from the callee symbol definition.
  uint8_t callee_temperature;
  // Temperature hint read from the call site.
  uint8_t call_temperature;
  // Effective edge temperature hint after call-site override.
  uint8_t effective_temperature;
  // Planned action for this edge.
  loom_inline_plan_action_t action;
  // Stable blocker code for ACTION_ERROR.
  loom_inline_blocker_t blocker;
  // Execution order ordinal assigned after SCC ordering.
  uint32_t execution_ordinal;
} loom_inline_plan_entry_t;

typedef struct loom_inline_state_t {
  // Active pass invocation.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_inline_callables_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Dependency table built from the immutable module snapshot.
  loom_symbol_dependency_table_t dependencies;
  // Dense symbol summaries indexed by module symbol id.
  loom_inline_symbol_info_t* symbols;
  // Dense plan entries collected from direct call edges.
  loom_inline_plan_entry_t* entries;
  // Number of valid entries in entries.
  uint32_t entry_count;
  // First required-inline plan entry for each source symbol.
  uint32_t* first_required_by_source;
  // Required-inline SCCs in callee-before-caller order.
  loom_scc_list_t sccs;
  // SCC ordinal for each symbol id, or IREE_HOST_SIZE_MAX when absent.
  iree_host_size_t* component_by_symbol;
} loom_inline_state_t;

static bool loom_inline_policy_is_inline(uint8_t policy) {
  return policy == LOOM_FUNC_INLINE_POLICY_INLINE;
}

static bool loom_inline_policy_is_noinline(uint8_t policy) {
  return policy == LOOM_FUNC_INLINE_POLICY_NOINLINE;
}

static iree_string_view_t loom_inline_symbol_name(const loom_module_t* module,
                                                  loom_symbol_id_t symbol_id) {
  if (symbol_id < module->symbols.count) {
    loom_string_id_t name_id = module->symbols.entries[symbol_id].name_id;
    if (name_id < module->strings.count) {
      return module->strings.entries[name_id];
    }
  }
  return IREE_SV("<invalid>");
}

static bool loom_inline_symbol_is_transferable(const loom_module_t* module,
                                               const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return false;
  }
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    return false;
  }

  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return false;
  }
  if (loom_func_like_visibility(function) != 0) {
    return false;
  }
  if (loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID ||
      loom_func_like_export_attrs(function).count > 0) {
    return false;
  }
  return true;
}

static iree_string_view_t loom_inline_blocker_code(
    loom_inline_blocker_t blocker) {
  switch (blocker) {
    case LOOM_INLINE_BLOCKER_POLICY_CONFLICT:
      return IREE_SV("policy_conflict");
    case LOOM_INLINE_BLOCKER_REQUIRED_CYCLE:
      return IREE_SV("required_cycle");
    case LOOM_INLINE_BLOCKER_CALL_NOT_CALL_LIKE:
      return IREE_SV("call_not_call_like");
    case LOOM_INLINE_BLOCKER_CALL_NOT_OWNED_BY_SYMBOL:
      return IREE_SV("call_not_owned_by_symbol");
    case LOOM_INLINE_BLOCKER_INVALID_CALLEE_SYMBOL:
      return IREE_SV("invalid_callee_symbol");
    case LOOM_INLINE_BLOCKER_CALLEE_NOT_FUNCTION_LIKE:
      return IREE_SV("callee_not_function_like");
    case LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE:
      return IREE_SV("func_call_target_not_function_like");
    case LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND:
      return IREE_SV("unsupported_call_kind");
    case LOOM_INLINE_BLOCKER_NON_CALL_SHAPE:
      return IREE_SV("non_call_shape");
    case LOOM_INLINE_BLOCKER_CALLEE_MISSING_BODY:
      return IREE_SV("callee_missing_body");
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_NOT_SINGLE_BLOCK:
      return IREE_SV("callee_body_not_single_block");
    case LOOM_INLINE_BLOCKER_RECURSIVE_BODY:
      return IREE_SV("recursive_body");
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR:
      return IREE_SV("callee_body_missing_terminator");
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_RETURN:
      return IREE_SV("callee_body_missing_return");
    case LOOM_INLINE_BLOCKER_OPERAND_COUNT_MISMATCH:
      return IREE_SV("operand_count_mismatch");
    case LOOM_INLINE_BLOCKER_INVALID_OPERAND_OR_ARGUMENT:
      return IREE_SV("invalid_operand_or_argument");
    case LOOM_INLINE_BLOCKER_OPERAND_TYPE_MISMATCH:
      return IREE_SV("operand_type_mismatch");
    case LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH:
      return IREE_SV("return_count_mismatch");
    case LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT:
      return IREE_SV("invalid_return_or_result");
    case LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH:
      return IREE_SV("result_type_mismatch");
    case LOOM_INLINE_BLOCKER_NONE:
    default:
      return IREE_SV("unknown");
  }
}

static void loom_inline_mark_blocker(loom_inline_plan_entry_t* entry,
                                     loom_inline_blocker_t blocker) {
  if (entry->action == LOOM_INLINE_PLAN_ACTION_ERROR) {
    return;
  }
  entry->action = LOOM_INLINE_PLAN_ACTION_ERROR;
  entry->blocker = blocker;
}

static bool loom_inline_op_is_inside_region(const loom_op_t* op,
                                            const loom_region_t* region) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    const loom_region_t* parent_region =
        current->parent_block ? current->parent_block->parent_region : NULL;
    if (parent_region == region) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Plan collection
//===----------------------------------------------------------------------===//

static iree_status_t loom_inline_allocate_state(loom_inline_state_t* state) {
  loom_pass_t* pass = state->pass;
  loom_module_t* module = state->module;
  if (state->dependencies.edge_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "inline-callables dependency edge count exceeds "
                            "uint32_t range");
  }
  if (module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, module->symbols.count, sizeof(*state->symbols),
        (void**)&state->symbols));
    memset(state->symbols, 0, module->symbols.count * sizeof(*state->symbols));

    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(pass->arena, module->symbols.count,
                                  sizeof(*state->first_required_by_source),
                                  (void**)&state->first_required_by_source));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, module->symbols.count, sizeof(*state->component_by_symbol),
        (void**)&state->component_by_symbol));
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      state->first_required_by_source[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
      state->component_by_symbol[i] = IREE_HOST_SIZE_MAX;
    }
  }

  if (state->dependencies.edge_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, state->dependencies.edge_count, sizeof(*state->entries),
        (void**)&state->entries));
    memset(state->entries, 0,
           state->dependencies.edge_count * sizeof(*state->entries));
  }
  return iree_ok_status();
}

static void loom_inline_initialize_symbol_infos(loom_inline_state_t* state) {
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_inline_symbol_info_t* info = &state->symbols[i];
    info->symbol = &state->module->symbols.entries[i];
    info->function =
        loom_func_like_cast(state->module, info->symbol->defining_op);
  }
}

static uint8_t loom_inline_effective_temperature(uint8_t callee_temperature,
                                                 uint8_t call_temperature) {
  return call_temperature != 0 ? call_temperature : callee_temperature;
}

static void loom_inline_resolve_entry_policy(loom_inline_state_t* state,
                                             loom_inline_plan_entry_t* entry) {
  const bool callee_inline = loom_inline_policy_is_inline(entry->callee_policy);
  const bool callee_noinline =
      loom_inline_policy_is_noinline(entry->callee_policy);
  const bool call_inline = loom_inline_policy_is_inline(entry->call_policy);
  const bool call_noinline = loom_inline_policy_is_noinline(entry->call_policy);

  if ((callee_inline && call_noinline) || (callee_noinline && call_inline)) {
    entry->effective_policy = LOOM_FUNC_INLINE_POLICY_INLINE;
    loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_POLICY_CONFLICT);
    return;
  }

  if (callee_noinline || call_noinline) {
    entry->effective_policy = LOOM_FUNC_INLINE_POLICY_NOINLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics->kept_edges;
    return;
  }

  if (callee_inline || call_inline) {
    entry->effective_policy = LOOM_FUNC_INLINE_POLICY_INLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_REQUIRED;
    ++state->statistics->required_edges;
    return;
  }

  entry->effective_policy = 0;
  entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
  ++state->statistics->kept_edges;
}

static void loom_inline_add_required_graph_edge(loom_inline_state_t* state,
                                                uint32_t entry_index) {
  loom_inline_plan_entry_t* entry = &state->entries[entry_index];
  if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
    return;
  }
  if (entry->source_symbol_id >= state->module->symbols.count ||
      entry->target_symbol_id >= state->module->symbols.count) {
    return;
  }
  entry->next_required_from_source =
      state->first_required_by_source[entry->source_symbol_id];
  state->first_required_by_source[entry->source_symbol_id] = entry_index;
  state->symbols[entry->target_symbol_id].planned_call_removals++;
}

static void loom_inline_collect_dependency_counts(loom_inline_state_t* state) {
  for (iree_host_size_t i = 0; i < state->dependencies.edge_count; ++i) {
    const loom_symbol_dependency_edge_t* edge = &state->dependencies.edges[i];
    if (edge->target_symbol_id >= state->module->symbols.count) {
      continue;
    }
    if (edge->kind == LOOM_SYMBOL_DEPENDENCY_EDGE_CALL) {
      state->symbols[edge->target_symbol_id].incoming_call_count++;
    } else {
      state->symbols[edge->target_symbol_id].incoming_non_call_ref_count++;
    }
  }
}

static iree_status_t loom_inline_build_plan(loom_inline_state_t* state) {
  loom_inline_collect_dependency_counts(state);
  for (iree_host_size_t i = 0; i < state->dependencies.edge_count; ++i) {
    const loom_symbol_dependency_edge_t* edge = &state->dependencies.edges[i];
    if (edge->kind != LOOM_SYMBOL_DEPENDENCY_EDGE_CALL) {
      continue;
    }

    uint32_t entry_index = state->entry_count++;
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    entry->ordinal = entry_index;
    entry->dependency_edge_id = (loom_symbol_dependency_edge_id_t)i;
    entry->next_required_from_source = LOOM_INLINE_PLAN_ENTRY_INVALID;
    entry->source_symbol_id = edge->source_symbol_id;
    entry->target_symbol_id = edge->target_symbol_id;
    entry->call_op = (loom_op_t*)edge->user_op;
    entry->call = loom_call_like_cast(state->module, entry->call_op);
    if (entry->target_symbol_id < state->module->symbols.count) {
      entry->callee = state->symbols[entry->target_symbol_id].function;
    }
    entry->call_policy = loom_call_like_inline_policy(entry->call);
    entry->callee_policy = loom_func_like_inline_policy(entry->callee);
    entry->call_temperature = loom_call_like_temperature(entry->call);
    entry->callee_temperature = loom_func_like_temperature(entry->callee);
    entry->effective_temperature = loom_inline_effective_temperature(
        entry->callee_temperature, entry->call_temperature);
    entry->execution_ordinal = UINT32_MAX;

    loom_inline_resolve_entry_policy(state, entry);
    loom_inline_add_required_graph_edge(state, entry_index);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Required-inline graph
//===----------------------------------------------------------------------===//

static iree_status_t loom_inline_visit_required_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  loom_inline_state_t* state = (loom_inline_state_t*)user_data;
  if (node >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inline graph source node out of range");
  }
  for (uint32_t entry_index = state->first_required_by_source[node];
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    IREE_RETURN_IF_ERROR(
        successor.fn(successor.user_data, entry->target_symbol_id));
    entry_index = entry->next_required_from_source;
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_compute_required_sccs(
    loom_inline_state_t* state) {
  loom_scc_graph_t graph = {
      .node_count = state->module->symbols.count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_inline_visit_required_successors, state),
  };
  IREE_RETURN_IF_ERROR(loom_scc_compute(&graph, /*options=*/NULL,
                                        state->pass->arena, &state->sccs));

  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    for (iree_host_size_t i = 0; i < component->node_count; ++i) {
      state->component_by_symbol[component->nodes[i]] = component_index;
    }
  }
  return iree_ok_status();
}

static void loom_inline_mark_cycle_blockers(loom_inline_state_t* state) {
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    if (!component->is_cycle) {
      continue;
    }
    for (uint32_t entry_index = 0; entry_index < state->entry_count;
         ++entry_index) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
        continue;
      }
      if (entry->source_symbol_id >= state->module->symbols.count ||
          entry->target_symbol_id >= state->module->symbols.count) {
        continue;
      }
      if (state->component_by_symbol[entry->source_symbol_id] ==
              component_index &&
          state->component_by_symbol[entry->target_symbol_id] ==
              component_index) {
        loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_REQUIRED_CYCLE);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Preflight
//===----------------------------------------------------------------------===//

static loom_inline_blocker_t loom_inline_validate_call_kind(
    const loom_inline_plan_entry_t* entry) {
  switch (loom_call_like_kind(entry->call)) {
    case LOOM_CALL_LIKE_KIND_SEMANTIC:
      if (loom_func_like_isa(entry->callee)) {
        return LOOM_INLINE_BLOCKER_NONE;
      }
      return LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE;
    default:
      return LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND;
  }
}

static loom_inline_blocker_t loom_inline_validate_inline_body(
    const loom_module_t* module, const loom_inline_plan_entry_t* entry) {
  if (!loom_call_like_isa(entry->call)) {
    return LOOM_INLINE_BLOCKER_CALL_NOT_CALL_LIKE;
  }
  if (entry->source_symbol_id >= module->symbols.count) {
    return LOOM_INLINE_BLOCKER_CALL_NOT_OWNED_BY_SYMBOL;
  }
  if (entry->target_symbol_id >= module->symbols.count) {
    return LOOM_INLINE_BLOCKER_INVALID_CALLEE_SYMBOL;
  }
  if (!loom_func_like_isa(entry->callee)) {
    return LOOM_INLINE_BLOCKER_CALLEE_NOT_FUNCTION_LIKE;
  }
  loom_inline_blocker_t call_kind_blocker =
      loom_inline_validate_call_kind(entry);
  if (call_kind_blocker != LOOM_INLINE_BLOCKER_NONE) {
    return call_kind_blocker;
  }

  if (loom_call_like_operand_offset(entry->call) != 0 ||
      loom_call_like_result_offset(entry->call) != 0 ||
      entry->call_op->region_count != 0 ||
      entry->call_op->successor_count != 0) {
    return LOOM_INLINE_BLOCKER_NON_CALL_SHAPE;
  }

  loom_region_t* body = loom_func_like_body(entry->callee);
  if (!body) {
    return LOOM_INLINE_BLOCKER_CALLEE_MISSING_BODY;
  }
  if (body->block_count != 1) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_NOT_SINGLE_BLOCK;
  }
  if (loom_inline_op_is_inside_region(entry->call_op, body)) {
    return LOOM_INLINE_BLOCKER_RECURSIVE_BODY;
  }

  loom_block_t* entry_block = loom_region_entry_block(body);
  if (entry_block->op_count == 0) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR;
  }
  loom_op_t* return_op = loom_block_op(entry_block, entry_block->op_count - 1);
  if (!loom_func_return_isa(return_op)) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_RETURN;
  }

  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(entry->callee, &arg_count);
  loom_value_slice_t call_operands = loom_call_like_operands(entry->call);
  if (arg_count != call_operands.count) {
    return LOOM_INLINE_BLOCKER_OPERAND_COUNT_MISMATCH;
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    if (arg_ids[i] >= module->values.count ||
        call_operands.values[i] >= module->values.count) {
      return LOOM_INLINE_BLOCKER_INVALID_OPERAND_OR_ARGUMENT;
    }
    loom_type_t arg_type = loom_module_value_type(module, arg_ids[i]);
    loom_type_t operand_type =
        loom_module_value_type(module, call_operands.values[i]);
    if (!loom_type_equal(arg_type, operand_type)) {
      return LOOM_INLINE_BLOCKER_OPERAND_TYPE_MISMATCH;
    }
  }

  loom_value_slice_t return_operands = loom_func_return_operands(return_op);
  loom_value_slice_t call_results = loom_call_like_results(entry->call);
  if (return_operands.count != call_results.count) {
    return LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH;
  }
  for (uint16_t i = 0; i < call_results.count; ++i) {
    if (return_operands.values[i] >= module->values.count ||
        call_results.values[i] >= module->values.count) {
      return LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT;
    }
    loom_type_t return_type =
        loom_module_value_type(module, return_operands.values[i]);
    loom_type_t result_type =
        loom_module_value_type(module, call_results.values[i]);
    if (!loom_type_equal(return_type, result_type)) {
      return LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH;
    }
  }

  return LOOM_INLINE_BLOCKER_NONE;
}

static void loom_inline_preflight_required_entries(loom_inline_state_t* state) {
  for (uint32_t i = 0; i < state->entry_count; ++i) {
    loom_inline_plan_entry_t* entry = &state->entries[i];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
      continue;
    }
    loom_inline_blocker_t blocker =
        loom_inline_validate_inline_body(state->module, entry);
    if (blocker != LOOM_INLINE_BLOCKER_NONE) {
      loom_inline_mark_blocker(entry, blocker);
    }
  }
}

static iree_status_t loom_inline_emit_blockers(loom_inline_state_t* state) {
  for (uint32_t i = 0; i < state->entry_count; ++i) {
    const loom_inline_plan_entry_t* entry = &state->entries[i];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_ERROR) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, entry->call_op)),
        loom_param_string(state->pass->info->name),
        loom_param_string(
            loom_inline_symbol_name(state->module, entry->target_symbol_id)),
        loom_param_string(loom_inline_blocker_code(entry->blocker)),
    };
    loom_diagnostic_related_op_t related_op = {
        .label = IREE_SV("callee definition"),
        .op = entry->callee.op,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .op = entry->call_op,
        .error = LOOM_ERR_LOWERING_044,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = entry->callee.op ? &related_op : NULL,
        .related_op_count = entry->callee.op ? 1 : 0,
    };
    IREE_RETURN_IF_ERROR(
        iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution planning
//===----------------------------------------------------------------------===//

static void loom_inline_assign_execution_order(loom_inline_state_t* state) {
  uint32_t execution_ordinal = 0;
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    for (uint32_t entry_index = 0; entry_index < state->entry_count;
         ++entry_index) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
        continue;
      }
      if (entry->source_symbol_id >= state->module->symbols.count) {
        continue;
      }
      if (state->component_by_symbol[entry->source_symbol_id] !=
          component_index) {
        continue;
      }
      entry->execution_ordinal = execution_ordinal++;
    }
  }
}

static bool loom_inline_symbol_can_transfer(const loom_inline_state_t* state,
                                            loom_symbol_id_t symbol_id) {
  if (symbol_id >= state->module->symbols.count) {
    return false;
  }
  const loom_inline_symbol_info_t* info = &state->symbols[symbol_id];
  if (info->incoming_call_count != info->planned_call_removals) {
    return false;
  }
  if (info->incoming_non_call_ref_count != 0) {
    return false;
  }
  if (!loom_inline_symbol_is_transferable(state->module, info->symbol)) {
    return false;
  }
  return loom_func_like_body(info->function) != NULL;
}

static iree_status_t loom_inline_select_transfer_actions(
    loom_inline_state_t* state) {
  uint32_t* final_entry_by_symbol = NULL;
  if (state->module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, state->module->symbols.count,
        sizeof(*final_entry_by_symbol), (void**)&final_entry_by_symbol));
    for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
      final_entry_by_symbol[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
    }
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
      continue;
    }
    if (!loom_inline_symbol_can_transfer(state, entry->target_symbol_id)) {
      continue;
    }
    uint32_t previous_index = final_entry_by_symbol[entry->target_symbol_id];
    if (previous_index == LOOM_INLINE_PLAN_ENTRY_INVALID ||
        state->entries[previous_index].execution_ordinal <
            entry->execution_ordinal) {
      final_entry_by_symbol[entry->target_symbol_id] = entry_index;
    }
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
      continue;
    }
    if (final_entry_by_symbol[entry->target_symbol_id] == entry_index) {
      entry->action = LOOM_INLINE_PLAN_ACTION_TRANSFER;
    } else {
      entry->action = LOOM_INLINE_PLAN_ACTION_CLONE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_execute_entry(
    loom_inline_state_t* state, loom_rewriter_t* rewriter,
    loom_inline_plan_entry_t* entry) {
  switch (entry->action) {
    case LOOM_INLINE_PLAN_ACTION_CLONE: {
      IREE_RETURN_IF_ERROR(
          loom_callable_inline_call(rewriter, entry->call_op, entry->callee));
      loom_pass_mark_changed(state->pass);
      ++state->statistics->calls_cloned;
      return iree_ok_status();
    }
    case LOOM_INLINE_PLAN_ACTION_TRANSFER: {
      IREE_RETURN_IF_ERROR(loom_callable_inline_consuming_call(
          rewriter, entry->call_op, entry->callee));
      loom_pass_mark_changed(state->pass);
      ++state->statistics->calls_transferred;
      ++state->statistics->symbols_transferred;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_inline_execute_plan(loom_inline_state_t* state) {
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, state->module, state->pass->arena));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t component_index = 0;
       iree_status_is_ok(status) && component_index < state->sccs.count;
       ++component_index) {
    for (uint32_t entry_index = 0;
         iree_status_is_ok(status) && entry_index < state->entry_count;
         ++entry_index) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->source_symbol_id >= state->module->symbols.count) {
        continue;
      }
      if (state->component_by_symbol[entry->source_symbol_id] !=
          component_index) {
        continue;
      }
      status = loom_inline_execute_entry(state, &rewriter, entry);
    }
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

iree_status_t loom_inline_callables_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  loom_inline_state_t state = {
      .pass = pass,
      .statistics = loom_inline_callables_statistics(pass),
      .module = module,
  };
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(module, pass->arena,
                                                          &state.dependencies));
  IREE_RETURN_IF_ERROR(loom_inline_allocate_state(&state));
  loom_inline_initialize_symbol_infos(&state);
  IREE_RETURN_IF_ERROR(loom_inline_build_plan(&state));
  IREE_RETURN_IF_ERROR(loom_inline_compute_required_sccs(&state));
  loom_inline_mark_cycle_blockers(&state);
  loom_inline_preflight_required_entries(&state);
  IREE_RETURN_IF_ERROR(loom_inline_emit_blockers(&state));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }

  loom_inline_assign_execution_order(&state);
  IREE_RETURN_IF_ERROR(loom_inline_select_transfer_actions(&state));
  IREE_RETURN_IF_ERROR(loom_inline_execute_plan(&state));
  if (!pass->changed) {
    return iree_ok_status();
  }
  return loom_target_pass_compact_symbols_preserving_target_ref(
      pass, module, pass->arena, NULL);
}
