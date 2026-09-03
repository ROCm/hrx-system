// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_callables.h"

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/analysis/scc.h"
#include "loom/analysis/symbol_references.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/pipeline.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"

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

static const loom_pass_option_def_t kInlineCallablesOptions[] = {
    {IREE_SVL("policy"),
     IREE_SVL("Policy source: authored or target emission requirements.")},
};

static const loom_pass_info_t loom_inline_callables_pass_info_storage = {
    .name = IREE_SVL("inline-callables"),
    .description = IREE_SVL(
        "Inline call-like edges required by authored or target policy."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kInlineCallablesOptions,
    .option_count = IREE_ARRAYSIZE(kInlineCallablesOptions),
    .statistic_layout = &loom_inline_callables_statistics_layout,
};

const loom_pass_info_t* loom_inline_callables_pass_info(void) {
  return &loom_inline_callables_pass_info_storage;
}

typedef enum loom_inline_policy_source_e {
  LOOM_INLINE_POLICY_SOURCE_AUTHORED = 0,
  LOOM_INLINE_POLICY_SOURCE_TARGET = 1,
} loom_inline_policy_source_t;

typedef struct loom_inline_callables_pass_state_t {
  // Additional policy applied after authored inline/noinline resolution.
  loom_inline_policy_source_t policy_source;
} loom_inline_callables_pass_state_t;

static iree_status_t loom_inline_callables_parse_policy(
    iree_string_view_t value, loom_inline_callables_pass_state_t* state) {
  if (iree_string_view_equal(value, IREE_SV("authored"))) {
    state->policy_source = LOOM_INLINE_POLICY_SOURCE_AUTHORED;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("target"))) {
    state->policy_source = LOOM_INLINE_POLICY_SOURCE_TARGET;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "inline-callables option 'policy' must be 'authored' or 'target'");
}

static iree_status_t loom_inline_callables_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  if (iree_string_view_equal(name, IREE_SV("policy"))) {
    return loom_inline_callables_parse_policy(
        value, (loom_inline_callables_pass_state_t*)user_data);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'inline-callables'",
                          (int)name.size, name.data);
}

iree_status_t loom_inline_callables_create(loom_pass_t* pass,
                                           iree_string_view_t options) {
  loom_inline_callables_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  *state = (loom_inline_callables_pass_state_t){
      .policy_source = LOOM_INLINE_POLICY_SOURCE_AUTHORED,
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) continue;
      if (iree_string_view_equal(option->schema->name, IREE_SV("policy"))) {
        IREE_RETURN_IF_ERROR(loom_inline_callables_parse_policy(
            option->schema->enum_values[option->enum_value_index].value,
            state));
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown decoded option '%.*s' for pass 'inline-callables'",
          (int)option->schema->name.size, option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_inline_callables_parse_option,
                                    .user_data = state,
                                }));
  }
  pass->state = state;
  return iree_ok_status();
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
  LOOM_INLINE_BLOCKER_CALLEE_BODY_EMPTY = 11,
  LOOM_INLINE_BLOCKER_RECURSIVE_BODY = 12,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR = 13,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR = 14,
  LOOM_INLINE_BLOCKER_OPERAND_COUNT_MISMATCH = 15,
  LOOM_INLINE_BLOCKER_INVALID_OPERAND_OR_ARGUMENT = 16,
  LOOM_INLINE_BLOCKER_OPERAND_TYPE_MISMATCH = 17,
  LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH = 18,
  LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT = 19,
  LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH = 20,
  LOOM_INLINE_BLOCKER_TARGET_REQUIRES_INLINE = 21,
  LOOM_INLINE_BLOCKER_LOW_CALLEE_KIND = 22,
  LOOM_INLINE_BLOCKER_LOW_ALLOCATION = 23,
  LOOM_INLINE_BLOCKER_LOW_ENTRY_RESOURCE = 24,
  LOOM_INLINE_BLOCKER_LOW_LOCKED_NESTED_REGION = 25,
  LOOM_INLINE_BLOCKER_CALLEE_SUCCESSOR_OUTSIDE_BODY = 26,
  LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE = 27,
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
  // Whether the function body remains linear after its required call closure
  // is flattened in callee-before-caller order.
  bool body_will_be_linear;
} loom_inline_symbol_info_t;

typedef struct loom_inline_plan_entry_t {
  // Stable ordinal assigned in reference occurrence order.
  uint32_t ordinal;
  // Reference occurrence that produced this call plan entry.
  loom_symbol_reference_occurrence_id_t reference_occurrence_id;
  // Next required-inline entry with the same source symbol.
  uint32_t next_required_from_source;
  // Next required-inline entry executed in the same SCC component.
  uint32_t next_required_in_component;
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
  loom_inline_policy_t callee_policy;
  // Inline policy read from the call site.
  loom_inline_policy_t call_policy;
  // Effective edge inline policy after conflict resolution.
  loom_inline_policy_t effective_policy;
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
  // Erase the private callee after clone inlining the final live reference.
  bool erase_callee_after_clone;
  // Execution order ordinal assigned after SCC ordering.
  uint32_t execution_ordinal;
} loom_inline_plan_entry_t;

typedef struct loom_inline_component_entries_t {
  // First required-inline entry in stable occurrence order.
  uint32_t first_entry;
  // Last required-inline entry used to append in stable occurrence order.
  uint32_t last_entry;
} loom_inline_component_entries_t;

typedef struct loom_inline_state_t {
  // Active pass invocation.
  loom_pass_t* pass;
  // Immutable pass options.
  const loom_inline_callables_pass_state_t* options;
  // Typed statistics storage for the current pass invocation.
  loom_inline_callables_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Mutable compiler-version owner, or NULL outside target compilation.
  loom_function_version_owner_t* version_owner;
  // Reference table built from the immutable module snapshot.
  loom_symbol_reference_table_t references;
  // Target versions observed against the immutable module snapshot.
  loom_target_function_version_snapshot_t target_versions;
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
  // Required-inline entries indexed by their source SCC component.
  loom_inline_component_entries_t* component_entries;
  // Function versions whose definitions were erased during execution.
  iree_host_size_t erased_version_count;
} loom_inline_state_t;

static bool loom_inline_target_requires_low_call_inline(
    const loom_inline_state_t* state, loom_symbol_id_t caller_symbol_id) {
  if (!state->options ||
      state->options->policy_source != LOOM_INLINE_POLICY_SOURCE_TARGET) {
    return false;
  }
  const loom_target_function_version_t* caller_version =
      loom_target_function_version_snapshot_at(&state->target_versions,
                                               caller_symbol_id);
  if (!caller_version || !caller_version->resolved_target.provider) {
    return false;
  }
  const loom_target_provider_t* provider =
      caller_version->resolved_target.provider;
  return provider->select_low_call_policy != NULL &&
         provider->select_low_call_policy(&caller_version->resolved_target) ==
             LOOM_TARGET_LOW_CALL_POLICY_REQUIRE_INLINE;
}

static bool loom_inline_policy_is_inline(loom_inline_policy_t policy) {
  return policy == LOOM_INLINE_POLICY_INLINE;
}

static bool loom_inline_policy_is_noinline(loom_inline_policy_t policy) {
  return policy == LOOM_INLINE_POLICY_NOINLINE;
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
  return loom_func_like_is_module_internal(function);
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
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_EMPTY:
      return IREE_SV("callee_body_empty");
    case LOOM_INLINE_BLOCKER_RECURSIVE_BODY:
      return IREE_SV("recursive_body");
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR:
      return IREE_SV("callee_body_missing_terminator");
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR:
      return IREE_SV("callee_body_invalid_terminator");
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
    case LOOM_INLINE_BLOCKER_TARGET_REQUIRES_INLINE:
      return IREE_SV("target_requires_inline");
    case LOOM_INLINE_BLOCKER_LOW_CALLEE_KIND:
      return IREE_SV("low_callee_kind");
    case LOOM_INLINE_BLOCKER_LOW_ALLOCATION:
      return IREE_SV("low_allocation_not_virtual");
    case LOOM_INLINE_BLOCKER_LOW_ENTRY_RESOURCE:
      return IREE_SV("low_entry_resource");
    case LOOM_INLINE_BLOCKER_LOW_LOCKED_NESTED_REGION:
      return IREE_SV("low_locked_nested_region");
    case LOOM_INLINE_BLOCKER_CALLEE_SUCCESSOR_OUTSIDE_BODY:
      return IREE_SV("callee_successor_outside_body");
    case LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE:
      return IREE_SV("caller_region_not_cfg_capable");
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
  if (state->references.occurrence_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "inline-callables reference occurrence count exceeds "
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

  if (state->references.occurrence_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, state->references.occurrence_count,
        sizeof(*state->entries), (void**)&state->entries));
    memset(state->entries, 0,
           state->references.occurrence_count * sizeof(*state->entries));
  }
  return iree_ok_status();
}

static void loom_inline_initialize_symbol_infos(loom_inline_state_t* state) {
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_inline_symbol_info_t* info = &state->symbols[i];
    info->symbol = &state->module->symbols.entries[i];
    info->function =
        loom_func_like_cast(state->module, info->symbol->defining_op);
    info->body_will_be_linear =
        loom_callable_body_is_linear(state->module, info->function);
  }
}

static uint8_t loom_inline_effective_temperature(uint8_t callee_temperature,
                                                 uint8_t call_temperature) {
  return call_temperature != 0 ? call_temperature : callee_temperature;
}

static void loom_inline_resolve_entry_policy(loom_inline_state_t* state,
                                             loom_inline_plan_entry_t* entry) {
  const loom_call_like_kind_t call_kind = loom_call_like_kind(entry->call);
  if (call_kind == LOOM_CALL_LIKE_KIND_TEMPLATE) {
    entry->effective_policy = LOOM_INLINE_POLICY_INLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_REQUIRED;
    ++state->statistics->required_edges;
    return;
  }

  // low.invoke still carries source values. Its policy is applied after
  // source-to-Low normalizes the edge to low.func.call with register values.
  if (call_kind == LOOM_CALL_LIKE_KIND_LOW_INVOKE) {
    entry->effective_policy = 0;
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics->kept_edges;
    return;
  }

  const bool callee_inline = loom_inline_policy_is_inline(entry->callee_policy);
  const bool callee_noinline =
      loom_inline_policy_is_noinline(entry->callee_policy);
  const bool call_inline = loom_inline_policy_is_inline(entry->call_policy);
  const bool call_noinline = loom_inline_policy_is_noinline(entry->call_policy);

  if ((callee_inline && call_noinline) || (callee_noinline && call_inline)) {
    entry->effective_policy = LOOM_INLINE_POLICY_INLINE;
    loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_POLICY_CONFLICT);
    return;
  }

  const bool target_requires_inline =
      call_kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL &&
      loom_inline_target_requires_low_call_inline(state,
                                                  entry->source_symbol_id);
  if (target_requires_inline && (callee_noinline || call_noinline)) {
    entry->effective_policy = LOOM_INLINE_POLICY_NOINLINE;
    loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_TARGET_REQUIRES_INLINE);
    return;
  }

  if (callee_noinline || call_noinline) {
    entry->effective_policy = LOOM_INLINE_POLICY_NOINLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics->kept_edges;
    return;
  }

  if (callee_inline || call_inline || target_requires_inline) {
    entry->effective_policy = LOOM_INLINE_POLICY_INLINE;
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

static void loom_inline_collect_reference_counts(loom_inline_state_t* state) {
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* edge =
        &state->references.occurrences[i];
    if (!loom_symbol_reference_occurrence_is_dependency(edge)) {
      continue;
    }
    if (edge->target_symbol_id >= state->module->symbols.count) {
      continue;
    }
    if (edge->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL) {
      state->symbols[edge->target_symbol_id].incoming_call_count++;
    } else {
      state->symbols[edge->target_symbol_id].incoming_non_call_ref_count++;
    }
  }
}

static iree_status_t loom_inline_build_plan(loom_inline_state_t* state) {
  loom_inline_collect_reference_counts(state);
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* edge =
        &state->references.occurrences[i];
    if (!loom_symbol_reference_occurrence_is_dependency(edge) ||
        edge->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL) {
      continue;
    }

    uint32_t entry_index = state->entry_count++;
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    entry->ordinal = entry_index;
    entry->reference_occurrence_id = (loom_symbol_reference_occurrence_id_t)i;
    entry->next_required_from_source = LOOM_INLINE_PLAN_ENTRY_INVALID;
    entry->next_required_in_component = LOOM_INLINE_PLAN_ENTRY_INVALID;
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

static iree_status_t loom_inline_index_required_entries_by_component(
    loom_inline_state_t* state) {
  if (state->sccs.count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->sccs.count, sizeof(*state->component_entries),
      (void**)&state->component_entries));
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    state->component_entries[component_index] =
        (loom_inline_component_entries_t){
            .first_entry = LOOM_INLINE_PLAN_ENTRY_INVALID,
            .last_entry = LOOM_INLINE_PLAN_ENTRY_INVALID,
        };
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED ||
        entry->source_symbol_id >= state->module->symbols.count) {
      continue;
    }
    const iree_host_size_t component_index =
        state->component_by_symbol[entry->source_symbol_id];
    if (component_index >= state->sccs.count) continue;
    loom_inline_component_entries_t* component_entries =
        &state->component_entries[component_index];
    uint32_t previous_entry = component_entries->last_entry;
    if (previous_entry == LOOM_INLINE_PLAN_ENTRY_INVALID) {
      component_entries->first_entry = entry_index;
    } else {
      state->entries[previous_entry].next_required_in_component = entry_index;
    }
    component_entries->last_entry = entry_index;
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
    for (uint32_t entry_index =
             state->component_entries[component_index].first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_required_in_component) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
        continue;
      }
      if (entry->target_symbol_id >= state->module->symbols.count) {
        continue;
      }
      if (state->component_by_symbol[entry->target_symbol_id] ==
          component_index) {
        loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_REQUIRED_CYCLE);
      }
    }
  }
}

static bool loom_inline_call_is_in_function_body(
    const loom_inline_state_t* state, const loom_inline_plan_entry_t* entry) {
  if (entry->source_symbol_id >= state->module->symbols.count ||
      entry->call_op == NULL || entry->call_op->parent_block == NULL) {
    return false;
  }
  const loom_region_t* source_body =
      loom_func_like_body(state->symbols[entry->source_symbol_id].function);
  return source_body != NULL &&
         entry->call_op->parent_block->parent_region == source_body;
}

// Predicts the body shape seen when each required edge executes. Flattening a
// non-linear callee at a function-body call site makes that caller non-linear,
// and the effect propagates transitively through the acyclic required-inline
// graph. Calls in nested regions do not change the containing function body's
// block topology. SCC order is successor-before-predecessor, so every callee's
// final shape is known before its callers are visited.
static void loom_inline_propagate_required_cfg_shapes(
    loom_inline_state_t* state) {
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    if (component->is_cycle) continue;
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const loom_symbol_id_t source_symbol_id =
          (loom_symbol_id_t)component->nodes[node_index];
      loom_inline_symbol_info_t* source_info =
          &state->symbols[source_symbol_id];
      if (!source_info->body_will_be_linear) continue;
      for (uint32_t entry_index =
               state->first_required_by_source[source_symbol_id];
           entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
           entry_index =
               state->entries[entry_index].next_required_from_source) {
        const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
        if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED ||
            entry->target_symbol_id >= state->module->symbols.count ||
            !loom_inline_call_is_in_function_body(state, entry)) {
          continue;
        }
        if (!state->symbols[entry->target_symbol_id].body_will_be_linear) {
          source_info->body_will_be_linear = false;
          break;
        }
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
    case LOOM_CALL_LIKE_KIND_TEMPLATE:
      if (loom_func_like_isa(entry->callee)) {
        return LOOM_INLINE_BLOCKER_NONE;
      }
      return LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE;
    case LOOM_CALL_LIKE_KIND_LOW_INTERNAL:
      if (loom_low_func_def_isa(entry->callee.op)) {
        return LOOM_INLINE_BLOCKER_NONE;
      }
      return LOOM_INLINE_BLOCKER_LOW_CALLEE_KIND;
    default:
      return LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND;
  }
}

static loom_inline_blocker_t loom_inline_validate_low_body_op(
    const loom_op_t* op, bool schedule_locked) {
  if (loom_low_resource_isa(op) || loom_low_live_in_isa(op)) {
    return LOOM_INLINE_BLOCKER_LOW_ENTRY_RESOURCE;
  }
  if (schedule_locked && op->region_count != 0) {
    return LOOM_INLINE_BLOCKER_LOW_LOCKED_NESTED_REGION;
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = regions[region_index];
    if (!region) continue;
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_inline_blocker_t blocker =
            loom_inline_validate_low_body_op(child_op, schedule_locked);
        if (blocker != LOOM_INLINE_BLOCKER_NONE) return blocker;
      }
    }
  }
  return LOOM_INLINE_BLOCKER_NONE;
}

static loom_inline_blocker_t loom_inline_validate_inline_body(
    const loom_inline_state_t* state, const loom_inline_plan_entry_t* entry) {
  const loom_module_t* module = state->module;
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
  if (body->block_count == 0) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_EMPTY;
  }
  if (loom_inline_op_is_inside_region(entry->call_op, body)) {
    return LOOM_INLINE_BLOCKER_RECURSIVE_BODY;
  }

  const uint8_t body_region_index =
      loom_func_like_body_region_index(entry->callee);
  const loom_op_vtable_t* callee_vtable =
      loom_op_vtable(module, entry->callee.op);
  const loom_region_descriptor_t* body_descriptor =
      loom_op_vtable_region_descriptor(callee_vtable, body_region_index);
  if (!body_descriptor || body_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR;
  }

  const bool is_low_call =
      loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_LOW_INTERNAL;
  if (is_low_call) {
    const uint8_t allocation = loom_low_func_def_allocation(entry->callee.op);
    if (allocation != 0 && allocation != LOOM_LOW_ALLOCATION_VIRTUAL) {
      return LOOM_INLINE_BLOCKER_LOW_ALLOCATION;
    }
    const bool schedule_locked = loom_low_func_def_schedule(entry->callee.op) ==
                                 LOOM_LOW_SCHEDULE_LOCKED;
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* body_op = NULL;
      loom_block_for_each_op(block, body_op) {
        loom_inline_blocker_t low_blocker =
            loom_inline_validate_low_body_op(body_op, schedule_locked);
        if (low_blocker != LOOM_INLINE_BLOCKER_NONE) return low_blocker;
      }
    }
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

  loom_value_slice_t call_results = loom_call_like_results(entry->call);
  if (entry->callee.op->result_count != call_results.count) {
    return LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH;
  }
  const loom_value_id_t* callee_results =
      loom_op_const_results(entry->callee.op);
  for (uint16_t i = 0; i < call_results.count; ++i) {
    if (callee_results[i] >= module->values.count ||
        call_results.values[i] >= module->values.count) {
      return LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT;
    }
    loom_type_t callee_result_type =
        loom_module_value_type(module, callee_results[i]);
    loom_type_t result_type =
        loom_module_value_type(module, call_results.values[i]);
    if (!loom_type_equal(callee_result_type, result_type)) {
      return LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH;
    }
  }

  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    if (block->op_count == 0) {
      return LOOM_INLINE_BLOCKER_CALLEE_BODY_MISSING_TERMINATOR;
    }
    loom_op_t* terminator = loom_block_op(block, block->op_count - 1);
    if (!loom_op_has_trait(module, terminator, LOOM_TRAIT_TERMINATOR)) {
      return LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR;
    }
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      uint16_t ignored_index = 0;
      if (!loom_region_try_block_index(
              body, loom_op_successors(terminator)[successor_index],
              &ignored_index)) {
        return LOOM_INLINE_BLOCKER_CALLEE_SUCCESSOR_OUTSIDE_BODY;
      }
    }
    if (terminator->kind != body_descriptor->terminator) continue;
    if (terminator->operand_count != call_results.count) {
      return LOOM_INLINE_BLOCKER_RETURN_COUNT_MISMATCH;
    }
    const loom_value_id_t* return_operands = loom_op_const_operands(terminator);
    for (uint16_t i = 0; i < call_results.count; ++i) {
      if (return_operands[i] >= module->values.count) {
        return LOOM_INLINE_BLOCKER_INVALID_RETURN_OR_RESULT;
      }
      const loom_type_t return_type =
          loom_module_value_type(module, return_operands[i]);
      const loom_type_t result_type =
          loom_module_value_type(module, call_results.values[i]);
      if (!loom_type_equal(return_type, result_type)) {
        return LOOM_INLINE_BLOCKER_RESULT_TYPE_MISMATCH;
      }
    }
  }

  if (!state->symbols[entry->target_symbol_id].body_will_be_linear &&
      !loom_callable_call_site_allows_cfg_splice(module, entry->call_op)) {
    return LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE;
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
        loom_inline_validate_inline_body(state, entry);
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
    for (uint32_t entry_index =
             state->component_entries[component_index].first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_required_in_component) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED) {
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
  if (state->version_owner == NULL &&
      loom_target_function_version_snapshot_handle_at(&state->target_versions,
                                                      symbol_id) != NULL) {
    return false;
  }
  const loom_symbol_ref_t family =
      loom_func_like_template_family(info->function);
  if (loom_symbol_ref_is_valid(family) &&
      loom_symbol_reference_template_family_is_demanded(&state->references,
                                                        family.symbol_id)) {
    return false;
  }
  return loom_func_like_body(info->function) != NULL;
}

static iree_status_t loom_inline_select_transfer_actions(
    loom_inline_state_t* state, uint32_t* out_transfer_count) {
  *out_transfer_count = 0;
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
      if (state->symbols[entry->target_symbol_id].body_will_be_linear) {
        entry->action = LOOM_INLINE_PLAN_ACTION_TRANSFER;
        ++*out_transfer_count;
      } else {
        entry->action = LOOM_INLINE_PLAN_ACTION_CLONE;
        entry->erase_callee_after_clone = true;
      }
    } else {
      entry->action = LOOM_INLINE_PLAN_ACTION_CLONE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_execute_entry(
    loom_inline_state_t* state, loom_rewriter_t* rewriter,
    const loom_availability_analysis_t* transfer_availability,
    loom_inline_plan_entry_t* entry) {
  switch (entry->action) {
    case LOOM_INLINE_PLAN_ACTION_CLONE: {
      const loom_callable_build_branch_fn_t build_branch =
          loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_LOW_INTERNAL
              ? loom_low_br_build
              : loom_cfg_br_build;
      IREE_RETURN_IF_ERROR(loom_callable_inline_call_with_branch(
          rewriter, entry->call_op, entry->callee, build_branch));
      if (entry->erase_callee_after_clone) {
        loom_function_version_t* version =
            loom_target_function_version_snapshot_handle_at(
                &state->target_versions, entry->target_symbol_id);
        IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, entry->callee.op));
        if (version != NULL) {
          ++state->erased_version_count;
        }
        ++state->statistics->symbols_transferred;
      }
      loom_pass_mark_changed(state->pass);
      ++state->statistics->calls_cloned;
      return iree_ok_status();
    }
    case LOOM_INLINE_PLAN_ACTION_TRANSFER: {
      IREE_ASSERT(transfer_availability);
      loom_function_version_t* version =
          loom_target_function_version_snapshot_handle_at(
              &state->target_versions, entry->target_symbol_id);
      IREE_RETURN_IF_ERROR(loom_callable_inline_consuming_call(
          rewriter, transfer_availability, entry->call_op, entry->callee));
      if (version != NULL) {
        ++state->erased_version_count;
      }
      loom_pass_mark_changed(state->pass);
      ++state->statistics->calls_transferred;
      ++state->statistics->symbols_transferred;
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

// Reconciles the mutable owner once after the rewrite batch. Stable compaction
// keeps version observation order while avoiding a linear search and tail move
// for every erased helper.
static iree_host_size_t loom_inline_prune_erased_function_versions(
    loom_function_version_owner_t* owner) {
  if (owner == NULL) return 0;
  iree_host_size_t write_index = 0;
  iree_host_size_t removed_count = 0;
  for (iree_host_size_t read_index = 0; read_index < owner->list.count;
       ++read_index) {
    loom_function_version_t* version = owner->storage[read_index];
    if (version->function.op != NULL &&
        iree_any_bit_set(version->function.op->flags, LOOM_OP_FLAG_DEAD)) {
      ++removed_count;
      continue;
    }
    owner->storage[write_index++] = version;
  }
  for (iree_host_size_t i = write_index; i < owner->list.count; ++i) {
    owner->storage[i] = NULL;
  }
  owner->list.count = write_index;
  return removed_count;
}

// Makes a locked Low function's per-block source order explicit before its
// body crosses a callable boundary. Each nonempty block receives a leading
// fence and a fence after every non-terminator operation. The now-redundant
// function-level lock is cleared, making this normalization idempotent when a
// retained helper participates in a later inline pass. The generic CFG splice
// treats the fences as ordinary cloned IR and remains unaware of Low scheduling
// semantics.
static iree_status_t loom_inline_materialize_locked_low_schedules(
    loom_inline_state_t* state, loom_rewriter_t* rewriter) {
  uint8_t* normalized_symbols = NULL;
  if (state->module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, state->module->symbols.count,
        sizeof(*normalized_symbols), (void**)&normalized_symbols));
    memset(normalized_symbols, 0,
           state->module->symbols.count * sizeof(*normalized_symbols));
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if ((entry->action != LOOM_INLINE_PLAN_ACTION_CLONE &&
         entry->action != LOOM_INLINE_PLAN_ACTION_TRANSFER) ||
        loom_call_like_kind(entry->call) != LOOM_CALL_LIKE_KIND_LOW_INTERNAL ||
        normalized_symbols[entry->target_symbol_id] != 0 ||
        loom_low_func_def_schedule(entry->callee.op) !=
            LOOM_LOW_SCHEDULE_LOCKED) {
      continue;
    }
    normalized_symbols[entry->target_symbol_id] = 1;

    loom_region_t* body = loom_func_like_body(entry->callee);
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      loom_block_t* block = loom_region_block(body, block_index);
      loom_op_t* terminator = block->last_op;
      loom_op_t* first_op = block->first_op;
      if (first_op == terminator) continue;

      loom_builder_set_before(&rewriter->builder, first_op);
      loom_op_t* fence_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_schedule_fence_build(
          &rewriter->builder, first_op->location, &fence_op));
      for (loom_op_t* op = first_op; op != terminator;) {
        loom_op_t* next_op = op->next_op;
        loom_builder_set_after(&rewriter->builder, op);
        IREE_RETURN_IF_ERROR(loom_low_schedule_fence_build(
            &rewriter->builder, op->location, &fence_op));
        op = next_op;
      }
    }
    loom_op_attrs(entry->callee.op)[loom_low_func_def_schedule_ATTR_INDEX] =
        loom_attr_absent();
    loom_pass_mark_changed(state->pass);
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_execute_plan(loom_inline_state_t* state,
                                              uint32_t transfer_count) {
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, state->module, state->pass->arena));

  loom_availability_analysis_t transfer_availability = {0};
  iree_status_t status =
      loom_inline_materialize_locked_low_schedules(state, &rewriter);
  if (iree_status_is_ok(status) && transfer_count != 0) {
    status = loom_availability_analysis_initialize(
        state->module, state->pass->arena, &transfer_availability);
  }
  const loom_availability_analysis_t* transfer_availability_ptr =
      transfer_count > 0 ? &transfer_availability : NULL;
  for (iree_host_size_t component_index = 0;
       iree_status_is_ok(status) && component_index < state->sccs.count;
       ++component_index) {
    for (uint32_t entry_index =
             state->component_entries[component_index].first_entry;
         iree_status_is_ok(status) &&
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_required_in_component) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      status = loom_inline_execute_entry(state, &rewriter,
                                         transfer_availability_ptr, entry);
    }
  }

  const iree_host_size_t removed_version_count =
      state->erased_version_count > 0
          ? loom_inline_prune_erased_function_versions(state->version_owner)
          : 0;
  IREE_ASSERT_EQ(removed_version_count, state->erased_version_count);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

iree_status_t loom_inline_callables_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  const loom_inline_callables_pass_state_t default_options = {
      .policy_source = LOOM_INLINE_POLICY_SOURCE_AUTHORED,
  };
  const loom_inline_callables_pass_state_t* options =
      pass->state ? (const loom_inline_callables_pass_state_t*)pass->state
                  : &default_options;
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  loom_inline_state_t state = {
      .pass = pass,
      .options = options,
      .statistics = loom_inline_callables_statistics(pass),
      .module = module,
      .version_owner =
          loom_target_pass_capability_function_version_owner(target_capability),
  };
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(module, pass->arena,
                                                         &state.references));
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(target_capability),
      pass->arena, &state.target_versions));
  IREE_RETURN_IF_ERROR(loom_inline_allocate_state(&state));
  loom_inline_initialize_symbol_infos(&state);
  IREE_RETURN_IF_ERROR(loom_inline_build_plan(&state));
  IREE_RETURN_IF_ERROR(loom_inline_compute_required_sccs(&state));
  IREE_RETURN_IF_ERROR(loom_inline_index_required_entries_by_component(&state));
  loom_inline_mark_cycle_blockers(&state);
  loom_inline_propagate_required_cfg_shapes(&state);
  loom_inline_preflight_required_entries(&state);
  IREE_RETURN_IF_ERROR(loom_inline_emit_blockers(&state));
  if (loom_pass_has_error_diagnostics(pass)) {
    return iree_ok_status();
  }

  loom_inline_assign_execution_order(&state);
  uint32_t transfer_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_inline_select_transfer_actions(&state, &transfer_count));
  IREE_RETURN_IF_ERROR(loom_inline_execute_plan(&state, transfer_count));
  if (!pass->changed) {
    return iree_ok_status();
  }
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
