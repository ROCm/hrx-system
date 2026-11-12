// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_callables.h"

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/loop_execution.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/capture.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/options.h"
#include "loom/pass/pipeline.h"
#include "loom/rewrite/callable.h"
#include "loom/target/execution_scope.h"
#include "loom/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"
#include "loom/transforms/symbol/inline_execution.h"
#include "loom/transforms/symbol/inline_plan.h"
#include "loom/transforms/symbol/inline_profitability.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_statistic_field_t
    loom_inline_callables_statistics_fields[] = {
        LOOM_INLINE_CALLABLES_STATISTICS(LOOM_PASS_STATISTIC_LAYOUT_FIELD_,
                                         loom_inline_callables_statistics_t)};
static const loom_pass_statistic_layout_t
    loom_inline_callables_statistics_layout =
        LOOM_PASS_STATISTIC_LAYOUT(loom_inline_callables_statistics_t,
                                   loom_inline_callables_statistics_fields);

static const loom_pass_option_def_t kInlineCallablesOptions[] = {
    {IREE_SVL("budget"),
     IREE_SVL("Combined ordinary and hot materialization budget in expanded "
              "body units; zero disables default selection.")},
    {IREE_SVL("hot-budget"),
     IREE_SVL("Hot-only materialization budget in expanded body units; "
              "zero disables hot fallback selection.")},
    {IREE_SVL("policy"),
     IREE_SVL("Policy source: authored or target emission requirements.")},
    {IREE_SVL("stage"),
     IREE_SVL("Inlining stage: prepare, resume, or complete callable "
              "boundaries.")},
};

static const loom_pass_info_t loom_inline_callables_pass_info_storage = {
    .name = IREE_SVL("inline-callables"),
    .description = IREE_SVL(
        "Inline required call edges and profitable optional call cohorts."),
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

typedef enum loom_inline_stage_e {
  LOOM_INLINE_STAGE_COMPLETE = 0,
  LOOM_INLINE_STAGE_PREPARE = 1,
  LOOM_INLINE_STAGE_RESUME = 2,
} loom_inline_stage_t;

typedef struct loom_inline_callables_pass_state_t {
  // Additional policy applied after authored inline/noinline resolution.
  loom_inline_policy_source_t policy_source;
  // Whether callable exits may retain required boundaries for completion.
  loom_inline_stage_t stage;
  // Materialization budgets for otherwise optional direct calls.
  loom_inline_profitability_options_t profitability;
} loom_inline_callables_pass_state_t;

struct loom_inline_execution_targets_t {
  // Worker environments indexed by the plan's immutable reference snapshot.
  loom_target_execution_scope_table_t scopes;
  // Cached authored targets used by explicitly targeted regions.
  loom_symbol_fact_table_t symbol_facts;
  // Borrowed family-provider registry for the current compiler invocation.
  const loom_target_environment_t* environment;
};

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

static iree_status_t loom_inline_callables_parse_stage(
    iree_string_view_t value, loom_inline_callables_pass_state_t* state) {
  if (iree_string_view_equal(value, IREE_SV("complete"))) {
    state->stage = LOOM_INLINE_STAGE_COMPLETE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("prepare"))) {
    state->stage = LOOM_INLINE_STAGE_PREPARE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("resume"))) {
    state->stage = LOOM_INLINE_STAGE_RESUME;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "inline-callables option 'stage' must be 'prepare', 'resume', or "
      "'complete'");
}

static iree_status_t loom_inline_callables_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_inline_callables_pass_state_t* state = user_data;
  if (iree_string_view_equal(name, IREE_SV("policy"))) {
    return loom_inline_callables_parse_policy(value, state);
  }
  if (iree_string_view_equal(name, IREE_SV("stage"))) {
    return loom_inline_callables_parse_stage(value, state);
  }
  if (iree_string_view_equal(name, IREE_SV("budget"))) {
    return loom_pass_option_parse_uint32(IREE_SV("inline-callables"), name,
                                         value, &state->profitability.budget);
  }
  if (iree_string_view_equal(name, IREE_SV("hot-budget"))) {
    return loom_pass_option_parse_uint32(IREE_SV("inline-callables"), name,
                                         value,
                                         &state->profitability.hot_budget);
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
      .stage = LOOM_INLINE_STAGE_COMPLETE,
      .profitability =
          {
              .budget = LOOM_INLINE_PROFITABILITY_DEFAULT_BUDGET,
              .hot_budget = LOOM_INLINE_PROFITABILITY_DEFAULT_HOT_BUDGET,
          },
  };
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) {
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("policy"))) {
        IREE_RETURN_IF_ERROR(loom_inline_callables_parse_policy(
            option->schema->enum_values[option->enum_value_index].value,
            state));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("stage"))) {
        IREE_RETURN_IF_ERROR(loom_inline_callables_parse_stage(
            option->schema->enum_values[option->enum_value_index].value,
            state));
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("budget"))) {
        state->profitability.budget = option->uint32_value;
        continue;
      }
      if (iree_string_view_equal(option->schema->name, IREE_SV("hot-budget"))) {
        state->profitability.hot_budget = option->uint32_value;
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

static bool loom_inline_target_requires_call_inline(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry, loom_call_like_kind_t call_kind,
    loom_call_like_t call, loom_func_like_t callee) {
  if (!state->options.target_policy) {
    return false;
  }
  loom_resolved_target_t target = {0};
  if (entry->execution_scope) {
    const loom_target_execution_scope_t* scope =
        state->execution_targets->scopes.values[entry->execution_scope - 1];
    target = (loom_resolved_target_t){scope->provider, scope->facts};
  } else {
    const loom_target_function_version_t* caller_version =
        loom_target_function_version_snapshot_at(state->options.target_versions,
                                                 entry->source_symbol_id);
    if (caller_version) {
      target = caller_version->resolved_target;
    }
  }
  if (!target.provider) {
    return false;
  }
  return loom_target_provider_select_call_policy(&target, state->module,
                                                 call_kind, call, callee) ==
         LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
}

static iree_status_t loom_inline_prepare_execution_target(
    loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  if (state->execution_targets == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        state->pass->arena, sizeof(*state->execution_targets),
        (void**)&state->execution_targets));
    *state->execution_targets = (loom_inline_execution_targets_t){
        .environment = loom_target_pass_capability_target_environment(
            loom_target_pass_capability_from_pass(state->pass)),
    };
    loom_symbol_fact_table_initialize(&state->execution_targets->symbol_facts,
                                      state->pass->arena);
  }
  const loom_target_function_version_t* caller_version =
      loom_target_function_version_snapshot_at(state->options.target_versions,
                                               entry->source_symbol_id);
  const loom_func_like_t caller =
      state->symbols[entry->source_symbol_id].function;
  const loom_target_execution_scope_t inherited = {
      .witness = loom_func_like_isa(caller) ? loom_func_like_target(caller)
                                            : loom_symbol_ref_null(),
      .facts = caller_version ? caller_version->resolved_target.facts : NULL,
      .provider =
          caller_version ? caller_version->resolved_target.provider : NULL,
  };
  const loom_target_execution_scope_t* scope = NULL;
  return loom_target_execution_scope_resolve(
      &state->execution_targets->scopes, state->options.references,
      entry->execution_scope, state->module,
      &state->execution_targets->symbol_facts,
      state->execution_targets->environment, inherited, state->pass->arena,
      &scope);
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
      return loom_string_table_get(&module->strings, name_id);
    }
  }
  return IREE_SV("<invalid>");
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
    case LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR:
      return IREE_SV("callee_body_invalid_terminator");
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

static iree_status_t loom_inline_allocate_state(
    loom_inline_callables_plan_t* state) {
  loom_pass_t* pass = state->pass;
  loom_module_t* module = state->module;
  iree_host_size_t entry_capacity = 0;
  if (!iree_host_size_checked_add(state->options.references->calls.count,
                                  state->options.additional_call_capacity,
                                  &entry_capacity) ||
      entry_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "inline-callables plan entry count exceeds "
                            "uint32_t range");
  }
  if (module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, module->symbols.count, sizeof(*state->symbols),
        (void**)&state->symbols));
    memset(state->symbols, 0, module->symbols.count * sizeof(*state->symbols));

    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(pass->arena, module->symbols.count,
                                  sizeof(*state->first_graph_by_source),
                                  (void**)&state->first_graph_by_source));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, module->symbols.count, sizeof(*state->component_by_symbol),
        (void**)&state->component_by_symbol));
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      state->first_graph_by_source[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
      state->component_by_symbol[i] = IREE_HOST_SIZE_MAX;
    }
  }

  if (entry_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(pass->arena, entry_capacity,
                                                   sizeof(*state->entries),
                                                   (void**)&state->entries));
  }
  return iree_ok_status();
}

static void loom_inline_initialize_symbol_infos(
    loom_inline_callables_plan_t* state) {
  for (iree_host_size_t i = 0; i < state->module->symbols.count; ++i) {
    loom_inline_symbol_info_t* info = &state->symbols[i];
    info->symbol = &state->module->symbols.entries[i];
    info->function =
        loom_func_like_cast(state->module, info->symbol->defining_op);
    info->body_will_be_linear =
        loom_callable_body_is_linear(state->module, info->function);
    info->body_has_callable_exit =
        loom_symbol_reference_symbol_has_callable_exit(
            state->options.references, (loom_symbol_id_t)i);
    info->body_is_contextual = loom_symbol_reference_symbol_is_contextual(
        state->options.references, (loom_symbol_id_t)i);
  }
}

static uint8_t loom_inline_effective_temperature(uint8_t callee_temperature,
                                                 uint8_t call_temperature) {
  return call_temperature != 0 ? call_temperature : callee_temperature;
}

static bool loom_inline_temperature_is_hot(uint8_t temperature) {
  return temperature == LOOM_FUNC_TEMPERATURE_HOT;
}

static bool loom_inline_temperature_is_cold(uint8_t temperature) {
  return temperature == LOOM_FUNC_TEMPERATURE_COLD;
}

static bool loom_inline_scope_inherits_target(
    const loom_inline_callables_plan_t* state,
    loom_symbol_reference_execution_scope_id_t scope) {
  return scope == 0 ||
         !loom_symbol_ref_is_valid(
             state->options.references->execution_targets.values[scope - 1]);
}

// Selected templates expand in the application context established by their
// applicability owner. Ordinary contextual callees must retain their own
// explicit target until folding removes that dependency or both contexts agree.
static bool loom_inline_shares_target_context(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  const loom_call_like_kind_t kind = loom_call_like_kind(entry->call);
  if (kind == LOOM_CALL_LIKE_KIND_TEMPLATE) {
    return true;
  }
  const loom_target_function_version_t* callee_version =
      loom_target_function_version_snapshot_at(state->options.target_versions,
                                               entry->target_symbol_id);
  if (callee_version != NULL && entry->execution_scope != 0 &&
      (kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
       kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL)) {
    // A worker's projected context can have no authored symbol witness. Use
    // the same retained environment that selected its call policy instead of
    // comparing the callee with the enclosing construction's target.
    const loom_target_execution_scope_t* scope =
        state->execution_targets->scopes.values[entry->execution_scope - 1];
    return scope->provider == callee_version->resolved_target.provider &&
           scope->facts != NULL &&
           loom_target_facts_are_equivalent(
               scope->facts, callee_version->resolved_target.facts);
  }
  if (callee_version != NULL && entry->execution_scope == 0) {
    const loom_target_function_version_t* caller_version =
        loom_target_function_version_snapshot_at(state->options.target_versions,
                                                 entry->source_symbol_id);
    if (caller_version != NULL && callee_version->target_context_ordinal !=
                                      LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
      // Independent versions can retain equivalent resolved environments.
      // Ordinal equality proves identity; inequality alone proves nothing
      // about the target semantics carried by their indexed fact records.
      return callee_version->target_context_ordinal ==
                 caller_version->target_context_ordinal ||
             (callee_version->resolved_target.provider ==
                  caller_version->resolved_target.provider &&
              loom_target_facts_are_equivalent(
                  callee_version->resolved_target.facts,
                  caller_version->resolved_target.facts));
    }
  }
  if (callee_version != NULL && !callee_version->authored_target_is_exact) {
    return false;
  }
  const loom_symbol_ref_t callee_target = loom_func_like_target(entry->callee);
  if (!loom_symbol_ref_is_valid(callee_target)) {
    return true;
  }
  const loom_symbol_ref_t caller_target =
      loom_inline_scope_inherits_target(state, entry->execution_scope)
          ? loom_func_like_target(
                state->symbols[entry->source_symbol_id].function)
          : state->options.references->execution_targets
                .values[entry->execution_scope - 1];
  return callee_target.module_id == caller_target.module_id &&
         callee_target.symbol_id == caller_target.symbol_id;
}

// Structural composition shares an invocation, but a child may still require
// its own target environment or executable boundary. Keep that edge explicit
// until its materialization owner can preserve the requirement. Unconstrained
// children inherit the caller's environment and can be spliced directly.
static bool loom_inline_composition_shares_context(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  if (!loom_func_like_isa(entry->callee) ||
      entry->source_symbol_id >= state->module->symbols.count) {
    // Required-edge preflight owns missing callable definitions and owners.
    return true;
  }
  const loom_inline_symbol_info_t* caller =
      &state->symbols[entry->source_symbol_id];
  // A construction helper acquires its ordering context only when it expands
  // into a pipeline. Flattening its compositions earlier would erase child
  // completion boundaries when the helper is instantiated inside a serial
  // schedule. The next reference snapshot owns that newly exposed context.
  if (!loom_symbol_implements(caller->symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    return false;
  }
  const loom_symbol_ref_t callee_target = loom_func_like_target(entry->callee);
  const loom_symbol_ref_t caller_target =
      loom_func_like_target(caller->function);
  if (loom_symbol_ref_is_valid(callee_target) &&
      (callee_target.module_id != caller_target.module_id ||
       callee_target.symbol_id != caller_target.symbol_id)) {
    return false;
  }
  const loom_symbol_t* callee = state->symbols[entry->target_symbol_id].symbol;
  const uint8_t carrier_attr_index =
      loom_symbol_definition_product_carrier_attr_index(callee->definition);
  if (carrier_attr_index == LOOM_ATTR_INDEX_NONE ||
      loom_attr_is_absent(
          loom_op_const_attrs(callee->defining_op)[carrier_attr_index])) {
    return true;
  }
  return loom_symbol_definition_product_carrier(callee->definition,
                                                callee->defining_op) ==
         loom_symbol_definition_product_carrier(caller->symbol->definition,
                                                caller->symbol->defining_op);
}

static void loom_inline_resolve_entry_policy(
    loom_inline_callables_plan_t* state, loom_inline_plan_entry_t* entry,
    bool joins_composition) {
  const loom_call_like_kind_t call_kind = loom_call_like_kind(entry->call);
  if (call_kind == LOOM_CALL_LIKE_KIND_COMPOSITION &&
      (joins_composition ||
       !loom_inline_composition_shares_context(state, entry))) {
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics.kept_edges;
    return;
  }
  if (call_kind == LOOM_CALL_LIKE_KIND_TEMPLATE ||
      call_kind == LOOM_CALL_LIKE_KIND_COMPOSITION ||
      call_kind == LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM) {
    entry->effective_policy = LOOM_INLINE_POLICY_INLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_REQUIRED;
    ++state->statistics.required_edges;
    return;
  }

  // low.invoke still carries source values. Its policy is applied after
  // source-to-Low normalizes the edge to low.func.call with register values.
  if (call_kind == LOOM_CALL_LIKE_KIND_LOW_INVOKE) {
    entry->effective_policy = 0;
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics.kept_edges;
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
      (call_kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
       call_kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL) &&
      loom_inline_target_requires_call_inline(state, entry, call_kind,
                                              entry->call, entry->callee);
  if (target_requires_inline && (callee_noinline || call_noinline)) {
    entry->effective_policy = LOOM_INLINE_POLICY_NOINLINE;
    loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_TARGET_REQUIRES_INLINE);
    return;
  }

  if (callee_noinline || call_noinline) {
    entry->effective_policy = LOOM_INLINE_POLICY_NOINLINE;
    entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
    ++state->statistics.kept_edges;
    return;
  }

  if (callee_inline || call_inline || target_requires_inline) {
    entry->effective_policy = LOOM_INLINE_POLICY_INLINE;
    const bool preserves_callable_exit =
        (state->options.defer_callable_exits &&
         call_kind == LOOM_CALL_LIKE_KIND_SEMANTIC) ||
        call_kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL;
    if (preserves_callable_exit &&
        entry->target_symbol_id < state->module->symbols.count &&
        state->symbols[entry->target_symbol_id].body_has_callable_exit) {
      entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
      ++state->statistics.deferred_edges;
      return;
    }
    entry->action = LOOM_INLINE_PLAN_ACTION_REQUIRED;
    ++state->statistics.required_edges;
    return;
  }

  entry->effective_policy = 0;
  const bool profitability_enabled =
      state->options.profitability.budget != 0 ||
      (loom_inline_temperature_is_hot(entry->effective_temperature) &&
       state->options.profitability.hot_budget != 0);
  const bool preserves_callable_exit =
      (state->options.defer_callable_exits &&
       call_kind == LOOM_CALL_LIKE_KIND_SEMANTIC) ||
      call_kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL;
  // Optional body composition is local to an ordinary callable. Independent
  // execution scopes own separate scheduling and target environments; their
  // eventual outlined functions get another profitability decision. Non-
  // callable symbol owners such as test scenarios describe invocations rather
  // than bodies that may absorb the referenced function.
  if (profitability_enabled &&
      !loom_inline_temperature_is_cold(entry->effective_temperature) &&
      entry->execution_scope == 0 &&
      entry->source_symbol_id < state->module->symbols.count &&
      loom_func_like_isa(state->symbols[entry->source_symbol_id].function) &&
      (call_kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
       call_kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL) &&
      !(preserves_callable_exit &&
        entry->target_symbol_id < state->module->symbols.count &&
        state->symbols[entry->target_symbol_id].body_has_callable_exit)) {
    entry->action = LOOM_INLINE_PLAN_ACTION_CANDIDATE;
    ++state->candidate_count;
    return;
  }
  entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
  ++state->statistics.kept_edges;
}

static void loom_inline_add_graph_edge(loom_inline_callables_plan_t* state,
                                       uint32_t entry_index) {
  loom_inline_plan_entry_t* entry = &state->entries[entry_index];
  if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED &&
      entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
    return;
  }
  if (entry->source_symbol_id >= state->module->symbols.count ||
      entry->target_symbol_id >= state->module->symbols.count) {
    return;
  }
  entry->next_graph_from_source =
      state->first_graph_by_source[entry->source_symbol_id];
  state->first_graph_by_source[entry->source_symbol_id] = entry_index;
  if (entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED) {
    state->symbols[entry->target_symbol_id].planned_call_removals++;
  }
}

static loom_inline_plan_entry_t* loom_inline_append_entry(
    loom_inline_callables_plan_t* state, loom_symbol_id_t source_symbol_id,
    loom_symbol_reference_execution_scope_id_t execution_scope,
    loom_symbol_id_t target_symbol_id, loom_op_t* call_op) {
  uint32_t entry_index = state->entry_count++;
  loom_inline_plan_entry_t* entry = &state->entries[entry_index];
  *entry = (loom_inline_plan_entry_t){
      .ordinal = entry_index,
      .next_graph_from_source = LOOM_INLINE_PLAN_ENTRY_INVALID,
      .next_required_in_component = LOOM_INLINE_PLAN_ENTRY_INVALID,
      .source_symbol_id = source_symbol_id,
      .target_symbol_id = target_symbol_id,
      .execution_scope = execution_scope,
      .call_op = call_op,
      .call = loom_call_like_cast(state->module, call_op),
  };
  if (target_symbol_id < state->module->symbols.count) {
    entry->callee = state->symbols[target_symbol_id].function;
    ++state->symbols[target_symbol_id].incoming_call_count;
  }
  entry->call_policy = loom_call_like_inline_policy(entry->call);
  entry->callee_policy = loom_func_like_inline_policy(entry->callee);
  entry->call_temperature = loom_call_like_temperature(entry->call);
  entry->callee_temperature = loom_func_like_temperature(entry->callee);
  entry->effective_temperature = loom_inline_effective_temperature(
      entry->callee_temperature, entry->call_temperature);
  return entry;
}

void loom_inline_callables_plan_append(
    loom_inline_callables_plan_t* state, loom_symbol_id_t source_symbol_id,
    loom_symbol_reference_execution_scope_id_t execution_scope,
    loom_symbol_id_t target_symbol_id, loom_op_t* call_op) {
  loom_inline_plan_entry_t* entry = loom_inline_append_entry(
      state, source_symbol_id, execution_scope, target_symbol_id, call_op);
  loom_inline_resolve_entry_policy(state, entry, false);
  loom_inline_add_graph_edge(state, entry->ordinal);
}

static iree_status_t loom_inline_build_plan(
    loom_inline_callables_plan_t* state) {
  for (iree_host_size_t i = 0; i < state->options.references->occurrence_count;
       ++i) {
    const loom_symbol_reference_occurrence_t* edge =
        loom_symbol_reference_table_occurrence(state->options.references, i);
    if (!loom_symbol_reference_occurrence_is_dependency(edge) ||
        (state->options.live_symbols &&
         edge->source_symbol_id < state->module->symbols.count &&
         !state->options.live_symbols[edge->source_symbol_id])) {
      continue;
    }
    if (edge->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL) {
      if (edge->target_symbol_id < state->module->symbols.count) {
        ++state->symbols[edge->target_symbol_id].incoming_non_call_ref_count;
      }
      continue;
    }

    loom_inline_plan_entry_t* entry = loom_inline_append_entry(
        state, edge->source_symbol_id, edge->execution_scope,
        edge->target_symbol_id, (loom_op_t*)edge->user_op);
    const loom_call_like_kind_t kind = loom_call_like_kind(entry->call);
    if (entry->execution_scope &&
        (kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL) &&
        (state->options.target_policy ||
         loom_target_function_version_snapshot_at(
             state->options.target_versions, entry->target_symbol_id))) {
      IREE_RETURN_IF_ERROR(loom_inline_prepare_execution_target(state, entry));
    }
    if (kind == LOOM_CALL_LIKE_KIND_TEMPLATE) {
      bool eligible = false;
      const loom_inline_callables_eligibility_callback_t callback =
          state->options.template_eligibility;
      if (callback.fn) {
        IREE_RETURN_IF_ERROR(callback.fn(callback.user_data, edge, &eligible));
      }
      if (!eligible) {
        ++state->statistics.kept_edges;
        continue;
      }
    }
    loom_inline_resolve_entry_policy(state, entry, edge->joins_composition);
    if ((entry->action == LOOM_INLINE_PLAN_ACTION_KEEP ||
         entry->action == LOOM_INLINE_PLAN_ACTION_CANDIDATE) &&
        kind == LOOM_CALL_LIKE_KIND_SEMANTIC &&
        entry->source_symbol_id < state->module->symbols.count &&
        loom_inline_scope_inherits_target(state, entry->execution_scope)) {
      // A retained semantic invocation inherits its caller's environment even
      // though its body is not expanded by this inline plan.
      state->symbols[entry->source_symbol_id].body_is_contextual = true;
    }
    loom_inline_add_graph_edge(state, entry->ordinal);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Required-inline graph
//===----------------------------------------------------------------------===//

static iree_status_t loom_inline_visit_required_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  loom_inline_callables_plan_t* state =
      (loom_inline_callables_plan_t*)user_data;
  if (node >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inline graph source node out of range");
  }
  for (uint32_t entry_index = state->first_graph_by_source[node];
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED) {
      IREE_RETURN_IF_ERROR(
          successor.fn(successor.user_data, entry->target_symbol_id));
    }
    entry_index = entry->next_graph_from_source;
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_visit_active_successors(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  loom_inline_callables_plan_t* state = user_data;
  if (node >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inline graph source node out of range");
  }
  for (uint32_t entry_index = state->first_graph_by_source[node];
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED ||
        entry->action == LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
      IREE_RETURN_IF_ERROR(
          successor.fn(successor.user_data, entry->target_symbol_id));
    }
    entry_index = entry->next_graph_from_source;
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_compute_sccs(
    loom_inline_callables_plan_t* state,
    loom_scc_visit_successors_fn_t visit_successors, loom_scc_list_t* out_sccs,
    iree_host_size_t* out_component_by_symbol) {
  loom_scc_graph_t graph = {
      .node_count = state->module->symbols.count,
      .visit_successors =
          loom_scc_visit_successors_callback_make(visit_successors, state),
  };
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, /*options=*/NULL, state->pass->arena, out_sccs));
  for (iree_host_size_t component_index = 0; component_index < out_sccs->count;
       ++component_index) {
    const loom_scc_t* component = &out_sccs->values[component_index];
    for (iree_host_size_t i = 0; i < component->node_count; ++i) {
      out_component_by_symbol[component->nodes[i]] = component_index;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_inline_compute_required_sccs(
    loom_inline_callables_plan_t* state) {
  return loom_inline_compute_sccs(state, loom_inline_visit_required_successors,
                                  &state->sccs, state->component_by_symbol);
}

static iree_status_t loom_inline_index_required_entries_by_component(
    loom_inline_callables_plan_t* state) {
  if (state->sccs.count == 0) {
    return iree_ok_status();
  }
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
    entry->next_required_in_component = LOOM_INLINE_PLAN_ENTRY_INVALID;
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED ||
        entry->source_symbol_id >= state->module->symbols.count) {
      continue;
    }
    const iree_host_size_t component_index =
        state->component_by_symbol[entry->source_symbol_id];
    if (component_index >= state->sccs.count) {
      continue;
    }
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

static void loom_inline_mark_cycle_blockers(
    loom_inline_callables_plan_t* state) {
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    if (!component->is_cycle) {
      continue;
    }
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const loom_symbol_id_t source_symbol_id =
          (loom_symbol_id_t)component->nodes[node_index];
      for (uint32_t entry_index =
               state->first_graph_by_source[source_symbol_id];
           entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
           entry_index = state->entries[entry_index].next_graph_from_source) {
        loom_inline_plan_entry_t* entry = &state->entries[entry_index];
        if (entry->action == LOOM_INLINE_PLAN_ACTION_REQUIRED &&
            entry->target_symbol_id < state->module->symbols.count &&
            state->component_by_symbol[entry->target_symbol_id] ==
                component_index) {
          loom_inline_mark_blocker(entry, LOOM_INLINE_BLOCKER_REQUIRED_CYCLE);
        }
      }
    }
  }
}

static bool loom_inline_call_is_in_function_body(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  if (entry->source_symbol_id >= state->module->symbols.count ||
      entry->call_op == NULL || entry->call_op->parent_block == NULL) {
    return false;
  }
  const loom_region_t* source_body =
      loom_func_like_body(state->symbols[entry->source_symbol_id].function);
  return source_body != NULL &&
         entry->call_op->parent_block->parent_region == source_body;
}

// Propagates the expanded body's context dependency and CFG shape through the
// acyclic required-inline graph. A differing explicit target delays expansion
// while contextual operations remain. SCC order is callee-before-caller, so
// each decision includes the effects of the callee's own required edges.
static void loom_inline_propagate_required_bodies(
    loom_inline_callables_plan_t* state) {
  for (iree_host_size_t component_index = 0;
       component_index < state->sccs.count; ++component_index) {
    const loom_scc_t* component = &state->sccs.values[component_index];
    if (component->is_cycle) {
      continue;
    }
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const loom_symbol_id_t source_symbol_id =
          (loom_symbol_id_t)component->nodes[node_index];
      loom_inline_symbol_info_t* source_info =
          &state->symbols[source_symbol_id];
      for (uint32_t entry_index =
               state->first_graph_by_source[source_symbol_id];
           entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
           entry_index = state->entries[entry_index].next_graph_from_source) {
        loom_inline_plan_entry_t* entry = &state->entries[entry_index];
        if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED ||
            entry->target_symbol_id >= state->module->symbols.count) {
          continue;
        }
        const loom_inline_symbol_info_t* target_info =
            &state->symbols[entry->target_symbol_id];
        if (target_info->body_is_contextual &&
            !loom_inline_shares_target_context(state, entry)) {
          entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
          --state->symbols[entry->target_symbol_id].planned_call_removals;
          --state->statistics.required_edges;
          ++state->statistics.kept_edges;
          continue;
        }
        if (loom_inline_scope_inherits_target(state, entry->execution_scope)) {
          source_info->body_is_contextual |= target_info->body_is_contextual;
        }
        if (!target_info->body_will_be_linear &&
            loom_inline_call_is_in_function_body(state, entry)) {
          source_info->body_will_be_linear = false;
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
    case LOOM_CALL_LIKE_KIND_COMPOSITION:
    case LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM:
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
    if (!region) {
      continue;
    }
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_inline_blocker_t blocker =
            loom_inline_validate_low_body_op(child_op, schedule_locked);
        if (blocker != LOOM_INLINE_BLOCKER_NONE) {
          return blocker;
        }
      }
    }
  }
  return LOOM_INLINE_BLOCKER_NONE;
}

static loom_inline_blocker_t loom_inline_validate_inline_site(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
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

  const loom_region_descriptor_t* body_descriptor =
      loom_func_like_body_region_descriptor(module, entry->callee);
  if (!body_descriptor || body_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR;
  }

  // Input verification establishes the callee's signature and CFG. The call
  // site must still be able to represent the cloned control flow.
  if (!state->symbols[entry->target_symbol_id].body_will_be_linear &&
      !loom_callable_call_site_allows_cfg_splice(module, entry->call_op)) {
    return LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE;
  }

  return LOOM_INLINE_BLOCKER_NONE;
}

static loom_inline_blocker_t loom_inline_validate_callee_body(
    const loom_inline_plan_entry_t* entry) {
  if (loom_call_like_kind(entry->call) != LOOM_CALL_LIKE_KIND_LOW_INTERNAL) {
    return LOOM_INLINE_BLOCKER_NONE;
  }
  if (!loom_low_func_def_isa(entry->callee.op)) {
    return LOOM_INLINE_BLOCKER_NONE;
  }
  const loom_region_t* body = loom_func_like_body(entry->callee);
  if (!body || body->block_count == 0) {
    return LOOM_INLINE_BLOCKER_NONE;
  }
  const uint8_t allocation = loom_low_func_def_allocation(entry->callee.op);
  if (allocation != 0 && allocation != LOOM_LOW_ALLOCATION_VIRTUAL) {
    return LOOM_INLINE_BLOCKER_LOW_ALLOCATION;
  }
  const bool schedule_locked =
      loom_low_func_def_schedule(entry->callee.op) == LOOM_LOW_SCHEDULE_LOCKED;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* body_op = NULL;
    loom_block_for_each_op(block, body_op) {
      loom_inline_blocker_t low_blocker =
          loom_inline_validate_low_body_op(body_op, schedule_locked);
      if (low_blocker != LOOM_INLINE_BLOCKER_NONE) {
        return low_blocker;
      }
    }
  }
  return LOOM_INLINE_BLOCKER_NONE;
}

static loom_inline_blocker_t loom_inline_validate_inline_body(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  const loom_inline_blocker_t site_blocker =
      loom_inline_validate_inline_site(state, entry);
  return site_blocker != LOOM_INLINE_BLOCKER_NONE
             ? site_blocker
             : loom_inline_validate_callee_body(entry);
}

static void loom_inline_preflight_required_entries(
    loom_inline_callables_plan_t* state) {
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

static iree_status_t loom_inline_emit_blockers(
    loom_inline_callables_plan_t* state) {
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
        .module = state->module,
        .op = entry->callee.op,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .module = state->module,
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
// Profitability selection
//===----------------------------------------------------------------------===//

static void loom_inline_keep_candidate(loom_inline_callables_plan_t* state,
                                       loom_inline_plan_entry_t* entry) {
  if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
    return;
  }
  entry->action = LOOM_INLINE_PLAN_ACTION_KEEP;
  --state->candidate_count;
  ++state->statistics.kept_edges;
}

static void loom_inline_select_candidate(loom_inline_callables_plan_t* state,
                                         loom_inline_plan_entry_t* entry) {
  if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
    return;
  }
  entry->action = LOOM_INLINE_PLAN_ACTION_REQUIRED;
  --state->candidate_count;
  ++state->symbols[entry->target_symbol_id].planned_call_removals;
  ++state->statistics.profitable_edges;
}

static iree_status_t loom_inline_initialize_profitability(
    loom_inline_callables_plan_t* state) {
  if (state->candidate_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t symbol_count = state->module->symbols.count;
  if (symbol_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, symbol_count, sizeof(*state->profitability_symbols),
        (void**)&state->profitability_symbols));
    for (iree_host_size_t i = 0; i < symbol_count; ++i) {
      state->profitability_symbols[i] = (loom_inline_profitability_symbol_t){
          .expanded_body_cost = UINT32_MAX,
          .first_candidate_by_target = LOOM_INLINE_PLAN_ENTRY_INVALID,
          .direct_body_is_linear = loom_callable_body_is_linear(
              state->module, state->symbols[i].function),
          .direct_body_is_contextual =
              loom_symbol_reference_symbol_is_contextual(
                  state->options.references, (loom_symbol_id_t)i),
      };
    }
  }
  if (state->entry_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->pass->arena, state->entry_count,
                                  sizeof(*state->next_candidate_by_target),
                                  (void**)&state->next_candidate_by_target));
    for (uint32_t i = 0; i < state->entry_count; ++i) {
      state->next_candidate_by_target[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
    }
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
      continue;
    }
    if (entry->source_symbol_id >= symbol_count ||
        entry->target_symbol_id >= symbol_count) {
      loom_inline_keep_candidate(state, entry);
      continue;
    }
    loom_inline_profitability_symbol_t* target =
        &state->profitability_symbols[entry->target_symbol_id];
    state->next_candidate_by_target[entry_index] =
        target->first_candidate_by_target;
    target->first_candidate_by_target = entry_index;
  }
  return iree_ok_status();
}

static void loom_inline_preflight_candidates(
    loom_inline_callables_plan_t* state) {
  for (iree_host_size_t target_symbol_id = 0;
       target_symbol_id < state->module->symbols.count; ++target_symbol_id) {
    const uint32_t first_entry = state->profitability_symbols[target_symbol_id]
                                     .first_candidate_by_target;
    if (first_entry == LOOM_INLINE_PLAN_ENTRY_INVALID) {
      continue;
    }
    loom_inline_blocker_t body_blocker = LOOM_INLINE_BLOCKER_NONE;
    for (uint32_t entry_index = first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->next_candidate_by_target[entry_index]) {
      const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action == LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
        body_blocker = loom_inline_validate_callee_body(entry);
        break;
      }
    }
    for (uint32_t entry_index = first_entry;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->next_candidate_by_target[entry_index]) {
      loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if (entry->action == LOOM_INLINE_PLAN_ACTION_CANDIDATE &&
          (body_blocker != LOOM_INLINE_BLOCKER_NONE ||
           loom_inline_validate_inline_site(state, entry) !=
               LOOM_INLINE_BLOCKER_NONE)) {
        loom_inline_keep_candidate(state, entry);
      }
    }
  }
}

static void loom_inline_reject_recursive_candidates(
    loom_inline_callables_plan_t* state, const loom_scc_list_t* sccs,
    const iree_host_size_t* component_by_symbol) {
  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
      continue;
    }
    const iree_host_size_t source_component =
        component_by_symbol[entry->source_symbol_id];
    const iree_host_size_t target_component =
        component_by_symbol[entry->target_symbol_id];
    if (source_component == target_component &&
        source_component < sccs->count &&
        sccs->values[source_component].is_cycle) {
      loom_inline_keep_candidate(state, entry);
    }
  }
}

static uint32_t loom_inline_type_materialization_cost(loom_type_t type) {
  uint64_t element_count = 1;
  if (loom_type_is_vector(type) || loom_type_is_tile(type) ||
      loom_type_is_tensor(type)) {
    if (!loom_type_static_element_count(type, &element_count)) {
      return UINT32_MAX;
    }
  } else if (!loom_type_is_scalar(type)) {
    return 1;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (element_bit_count <= 0) {
    return UINT32_MAX;
  }
  const uint64_t words_per_element = ((uint64_t)element_bit_count + 31u) / 32u;
  return element_count > UINT32_MAX / words_per_element
             ? UINT32_MAX
             : (uint32_t)(element_count * words_per_element);
}

static uint32_t loom_inline_op_materialization_cost(const loom_module_t* module,
                                                    loom_op_t* op) {
  // Calls are replaced as one unit when their body expands, and terminator
  // payloads disappear or become control-flow forwarding during composition.
  // Charging either payload here would count it again through the producers.
  if (loom_call_like_isa(loom_call_like_cast(module, op)) ||
      loom_op_has_trait(module, op, LOOM_TRAIT_TERMINATOR)) {
    return 1;
  }
  uint32_t cost = 1;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const uint32_t value_cost = loom_inline_type_materialization_cost(
        loom_module_value_type(module, loom_op_operands(op)[i]));
    cost = iree_max(cost, value_cost);
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const uint32_t value_cost = loom_inline_type_materialization_cost(
        loom_module_value_type(module, loom_op_results(op)[i]));
    cost = iree_max(cost, value_cost);
  }
  return cost;
}

typedef struct loom_inline_body_cost_context_t {
  const loom_module_t* module;
  uint32_t cost;
} loom_inline_body_cost_context_t;

static iree_status_t loom_inline_accumulate_body_cost(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_inline_body_cost_context_t* cost_context = user_data;
  const uint32_t op_cost =
      loom_inline_op_materialization_cost(cost_context->module, op);
  if (op_cost > UINT32_MAX - cost_context->cost) {
    cost_context->cost = UINT32_MAX;
  } else {
    cost_context->cost += op_cost;
  }
  *out_result = LOOM_WALK_CONTINUE;
  return iree_ok_status();
}

static iree_status_t loom_inline_compute_direct_body_cost(
    loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id,
    uint32_t* out_cost) {
  loom_inline_body_cost_context_t cost_context = {
      .module = state->module,
  };
  loom_region_t* body = loom_func_like_body(state->symbols[symbol_id].function);
  if (!body) {
    *out_cost = 0;
    return iree_ok_status();
  }
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(
      loom_walk_region(state->module, body, LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           .fn = loom_inline_accumulate_body_cost,
                           .user_data = &cost_context,
                       },
                       &result));
  *out_cost = cost_context.cost;
  return iree_ok_status();
}

static iree_status_t loom_inline_recompute_profitability_symbol(
    loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id) {
  loom_inline_symbol_info_t* source_info = &state->symbols[symbol_id];
  loom_inline_profitability_symbol_t* profitability =
      &state->profitability_symbols[symbol_id];
  source_info->body_will_be_linear = profitability->direct_body_is_linear;
  source_info->body_is_contextual = profitability->direct_body_is_contextual;
  IREE_RETURN_IF_ERROR(loom_inline_compute_direct_body_cost(
      state, symbol_id, &profitability->expanded_body_cost));

  for (uint32_t entry_index = state->first_graph_by_source[symbol_id];
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
       entry_index = state->entries[entry_index].next_graph_from_source) {
    const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action == LOOM_INLINE_PLAN_ACTION_KEEP) {
      if (loom_call_like_kind(entry->call) == LOOM_CALL_LIKE_KIND_SEMANTIC &&
          loom_inline_scope_inherits_target(state, entry->execution_scope)) {
        source_info->body_is_contextual = true;
      }
      continue;
    }
    if (entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED ||
        entry->target_symbol_id >= state->module->symbols.count) {
      continue;
    }
    const loom_inline_symbol_info_t* target_info =
        &state->symbols[entry->target_symbol_id];
    const loom_inline_profitability_symbol_t* target_profitability =
        &state->profitability_symbols[entry->target_symbol_id];
    profitability->expanded_body_cost = loom_inline_profitability_add_expansion(
        profitability->expanded_body_cost,
        target_profitability->expanded_body_cost);
    if (loom_inline_scope_inherits_target(state, entry->execution_scope)) {
      source_info->body_is_contextual |= target_info->body_is_contextual;
    }
    if (!target_info->body_will_be_linear &&
        loom_inline_call_is_in_function_body(state, entry)) {
      source_info->body_will_be_linear = false;
    }
  }
  return iree_ok_status();
}

static void loom_inline_select_candidates_for_target(
    loom_inline_callables_plan_t* state, loom_symbol_id_t target_symbol_id) {
  loom_inline_profitability_symbol_t* profitability =
      &state->profitability_symbols[target_symbol_id];
  uint32_t ordinary_call_count = 0;
  uint32_t hot_call_count = 0;
  for (uint32_t entry_index = profitability->first_candidate_by_target;
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
       entry_index = state->next_candidate_by_target[entry_index]) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
      continue;
    }
    const loom_inline_symbol_info_t* target_info =
        &state->symbols[target_symbol_id];
    if ((target_info->body_is_contextual &&
         !loom_inline_shares_target_context(state, entry)) ||
        loom_inline_validate_inline_site(state, entry) !=
            LOOM_INLINE_BLOCKER_NONE) {
      loom_inline_keep_candidate(state, entry);
      continue;
    }
    if (loom_inline_temperature_is_hot(entry->effective_temperature)) {
      ++hot_call_count;
    } else {
      ++ordinary_call_count;
    }
  }

  const loom_inline_profitability_decision_t decision =
      loom_inline_profitability_select(state->options.profitability,
                                       profitability->expanded_body_cost,
                                       ordinary_call_count, hot_call_count);
  for (uint32_t entry_index = profitability->first_candidate_by_target;
       entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
       entry_index = state->next_candidate_by_target[entry_index]) {
    loom_inline_plan_entry_t* entry = &state->entries[entry_index];
    if (entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) {
      continue;
    }
    const bool is_hot =
        loom_inline_temperature_is_hot(entry->effective_temperature);
    if ((is_hot && decision.hot) || (!is_hot && decision.ordinary)) {
      loom_inline_select_candidate(state, entry);
    } else {
      loom_inline_keep_candidate(state, entry);
    }
  }
}

static iree_status_t loom_inline_select_profitable_entries(
    loom_inline_callables_plan_t* state, const loom_scc_list_t* required_sccs) {
  IREE_RETURN_IF_ERROR(loom_inline_initialize_profitability(state));
  if (state->candidate_count == 0) {
    return iree_ok_status();
  }
  loom_inline_preflight_candidates(state);
  if (state->candidate_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t symbol_count = state->module->symbols.count;
  iree_host_size_t* combined_component_by_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, symbol_count, sizeof(*combined_component_by_symbol),
      (void**)&combined_component_by_symbol));
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    combined_component_by_symbol[i] = IREE_HOST_SIZE_MAX;
  }
  loom_scc_list_t combined_sccs = {0};
  IREE_RETURN_IF_ERROR(
      loom_inline_compute_sccs(state, loom_inline_visit_active_successors,
                               &combined_sccs, combined_component_by_symbol));
  loom_inline_reject_recursive_candidates(state, &combined_sccs,
                                          combined_component_by_symbol);
  if (state->candidate_count == 0) {
    return iree_ok_status();
  }

  uint32_t* next_symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, symbol_count,
                                sizeof(*next_symbol), (void**)&next_symbol));
  uint32_t worklist_count = 0;
  for (iree_host_size_t symbol_id = 0; symbol_id < symbol_count; ++symbol_id) {
    loom_inline_profitability_symbol_t* profitability =
        &state->profitability_symbols[symbol_id];
    bool has_candidate = false;
    for (uint32_t entry_index = profitability->first_candidate_by_target;
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->next_candidate_by_target[entry_index]) {
      has_candidate |= state->entries[entry_index].action ==
                       LOOM_INLINE_PLAN_ACTION_CANDIDATE;
    }
    if (has_candidate) {
      profitability->expanded_body_cost = 0;
      next_symbol[worklist_count++] = (uint32_t)symbol_id;
    }
  }
  while (worklist_count > 0) {
    const loom_symbol_id_t source_symbol_id =
        (loom_symbol_id_t)next_symbol[--worklist_count];
    for (uint32_t entry_index = state->first_graph_by_source[source_symbol_id];
         entry_index != LOOM_INLINE_PLAN_ENTRY_INVALID;
         entry_index = state->entries[entry_index].next_graph_from_source) {
      const loom_inline_plan_entry_t* entry = &state->entries[entry_index];
      if ((entry->action != LOOM_INLINE_PLAN_ACTION_REQUIRED &&
           entry->action != LOOM_INLINE_PLAN_ACTION_CANDIDATE) ||
          entry->target_symbol_id >= symbol_count) {
        continue;
      }
      loom_inline_profitability_symbol_t* target =
          &state->profitability_symbols[entry->target_symbol_id];
      if (target->expanded_body_cost == UINT32_MAX) {
        target->expanded_body_cost = 0;
        next_symbol[worklist_count++] = entry->target_symbol_id;
      }
    }
  }

  uint32_t* first_symbol_by_component = NULL;
  uint32_t* last_symbol_by_component = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, combined_sccs.count,
      sizeof(*first_symbol_by_component), (void**)&first_symbol_by_component));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, combined_sccs.count,
      sizeof(*last_symbol_by_component), (void**)&last_symbol_by_component));
  for (iree_host_size_t i = 0; i < combined_sccs.count; ++i) {
    first_symbol_by_component[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
    last_symbol_by_component[i] = LOOM_INLINE_PLAN_ENTRY_INVALID;
  }
  for (iree_host_size_t component_index = 0;
       component_index < required_sccs->count; ++component_index) {
    const loom_scc_t* component = &required_sccs->values[component_index];
    for (iree_host_size_t node_index = 0; node_index < component->node_count;
         ++node_index) {
      const uint32_t symbol_id = (uint32_t)component->nodes[node_index];
      const iree_host_size_t combined_component =
          combined_component_by_symbol[symbol_id];
      next_symbol[symbol_id] = LOOM_INLINE_PLAN_ENTRY_INVALID;
      if (last_symbol_by_component[combined_component] ==
          LOOM_INLINE_PLAN_ENTRY_INVALID) {
        first_symbol_by_component[combined_component] = symbol_id;
      } else {
        next_symbol[last_symbol_by_component[combined_component]] = symbol_id;
      }
      last_symbol_by_component[combined_component] = symbol_id;
    }
  }

  state->sccs = combined_sccs;
  state->component_by_symbol = combined_component_by_symbol;
  for (iree_host_size_t component_index = 0;
       component_index < combined_sccs.count; ++component_index) {
    for (uint32_t symbol_id = first_symbol_by_component[component_index];
         symbol_id != LOOM_INLINE_PLAN_ENTRY_INVALID;
         symbol_id = next_symbol[symbol_id]) {
      if (state->profitability_symbols[symbol_id].expanded_body_cost ==
          UINT32_MAX) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_inline_recompute_profitability_symbol(state, symbol_id));
      loom_inline_select_candidates_for_target(state, symbol_id);
    }
  }

  for (uint32_t entry_index = 0; entry_index < state->entry_count;
       ++entry_index) {
    loom_inline_keep_candidate(state, &state->entries[entry_index]);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Plan and pass entry points
//===----------------------------------------------------------------------===//

static iree_status_t loom_inline_callables_plan_initialize(
    loom_pass_t* pass, loom_module_t* module,
    const loom_inline_callables_plan_options_t* options,
    loom_inline_callables_plan_t* state) {
  *state = (loom_inline_callables_plan_t){
      .pass = pass,
      .options = *options,
      .module = module,
      .version_owner = loom_target_pass_capability_function_version_owner(
          loom_target_pass_capability_from_pass(pass)),
  };
  // Non-call references only affect transfer policy when calls are present.
  if (options->references->calls.count == 0 &&
      options->additional_call_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_inline_allocate_state(state));
  loom_inline_initialize_symbol_infos(state);
  return loom_inline_build_plan(state);
}

iree_status_t loom_inline_callables_plan_allocate(
    loom_pass_t* pass, loom_module_t* module,
    const loom_inline_callables_plan_options_t* options,
    loom_inline_callables_plan_t** out_plan) {
  loom_inline_callables_plan_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(pass->arena, sizeof(*state), (void**)&state));
  IREE_RETURN_IF_ERROR(
      loom_inline_callables_plan_initialize(pass, module, options, state));
  *out_plan = state;
  return iree_ok_status();
}

iree_status_t loom_inline_callables_plan_execute(
    loom_inline_callables_plan_t* state) {
  // Kept calls have no cycle or rewrite obligations. Policy conflicts can
  // still require diagnostics even when no edge is required to inline.
  if (state->statistics.required_edges == 0 && state->candidate_count == 0) {
    return loom_inline_emit_blockers(state);
  }
  IREE_RETURN_IF_ERROR(loom_inline_compute_required_sccs(state));
  loom_inline_mark_cycle_blockers(state);
  loom_inline_propagate_required_bodies(state);
  loom_inline_preflight_required_entries(state);
  IREE_RETURN_IF_ERROR(loom_inline_emit_blockers(state));
  if (loom_pass_has_error_diagnostics(state->pass)) {
    return iree_ok_status();
  }
  if (state->candidate_count != 0) {
    const loom_scc_list_t required_sccs = state->sccs;
    IREE_RETURN_IF_ERROR(
        loom_inline_select_profitable_entries(state, &required_sccs));
    loom_inline_preflight_required_entries(state);
    IREE_RETURN_IF_ERROR(loom_inline_emit_blockers(state));
    if (loom_pass_has_error_diagnostics(state->pass)) {
      return iree_ok_status();
    }
  }
  if (state->statistics.required_edges == 0 &&
      state->statistics.profitable_edges == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_inline_index_required_entries_by_component(state));
  return loom_inline_execute_plan(state);
}

iree_status_t loom_inline_callables_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  const loom_inline_callables_pass_state_t default_options = {
      .policy_source = LOOM_INLINE_POLICY_SOURCE_AUTHORED,
      .stage = LOOM_INLINE_STAGE_COMPLETE,
      .profitability =
          {
              .budget = LOOM_INLINE_PROFITABILITY_DEFAULT_BUDGET,
              .hot_budget = LOOM_INLINE_PROFITABILITY_DEFAULT_HOT_BUDGET,
          },
  };
  const loom_inline_callables_pass_state_t* options =
      pass->state ? (const loom_inline_callables_pass_state_t*)pass->state
                  : &default_options;
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  loom_target_pass_state_t* target_state =
      loom_target_pass_capability_state(target_capability);
  if (options->stage == LOOM_INLINE_STAGE_RESUME) {
    if (target_state == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "inline-callables stage 'resume' requires target pipeline state");
    }
    if (!target_state->source_inline_completion_pending) {
      *loom_inline_callables_statistics(pass) =
          (loom_inline_callables_statistics_t){0};
      return iree_ok_status();
    }
  }
  loom_symbol_reference_table_t references = {0};
  loom_target_function_version_snapshot_t target_versions = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, pass->arena, &references));
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(target_capability),
      pass->arena, &target_versions));
  const loom_inline_callables_plan_options_t plan_options = {
      .references = &references,
      .target_versions = &target_versions,
      .demanded_families = references.template_demands.family_bits,
      .target_policy =
          options->policy_source == LOOM_INLINE_POLICY_SOURCE_TARGET,
      .defer_callable_exits = options->stage == LOOM_INLINE_STAGE_PREPARE,
      .profitability = options->profitability,
  };
  loom_inline_callables_plan_t state;
  IREE_RETURN_IF_ERROR(loom_inline_callables_plan_initialize(
      pass, module, &plan_options, &state));

  iree_status_t status = loom_inline_callables_plan_execute(&state);
  *loom_inline_callables_statistics(pass) = state.statistics;
  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
      options->stage == LOOM_INLINE_STAGE_PREPARE &&
      references.callable_exit_symbols != NULL && target_state != NULL) {
    target_state->source_inline_completion_pending = true;
  }
  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
      pass->changed) {
    status = loom_module_compact_symbols(module, pass->arena, NULL);
  }
  if (iree_status_is_ok(status) && !loom_pass_has_error_diagnostics(pass) &&
      options->stage == LOOM_INLINE_STAGE_RESUME) {
    target_state->source_inline_completion_pending = false;
  }
  return status;
}
