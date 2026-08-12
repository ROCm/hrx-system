// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_selection.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/func_provider_catalog.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/condition.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/symbol/symbol_equivalence.h"
#include "loom/transforms/symbol/symbol_pruning.h"
#include "loom/util/bstring.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

typedef enum loom_template_selection_mode_e {
  LOOM_TEMPLATE_SELECTION_MODE_EARLY = 0,
  LOOM_TEMPLATE_SELECTION_MODE_FINAL = 1,
} loom_template_selection_mode_t;

typedef struct loom_template_selection_pass_state_t {
  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // True when mode was explicitly provided.
  bool has_mode_option;
} loom_template_selection_pass_state_t;

static const loom_pass_option_def_t kTemplateSelectionOptions[] = {
    {IREE_SVL("mode"),
     IREE_SVL("Selection mode: early preserves unresolved applies, final "
              "emits diagnostics for every unresolved live apply.")},
};

#define LOOM_TEMPLATE_SELECTION_STATISTICS(V, statistics_type)            \
  V(statistics_type, apply_sites, "apply-sites",                          \
    "Number of live func.apply sites analyzed.")                          \
  V(statistics_type, selected_sites, "selected-sites",                    \
    "Number of func.apply sites resolved to inline calls.")               \
  V(statistics_type, fallback_selected_sites, "fallback-selected-sites",  \
    "Number of selected sites that used a lower-priority "                \
    "provider while a higher-priority candidate existed.")                \
  V(statistics_type, unresolved_sites, "unresolved-sites",                \
    "Number of live func.apply sites left unresolved.")                   \
  V(statistics_type, no_provider_sites, "no-provider-sites",              \
    "Number of unresolved sites with no provider for the "                \
    "requested contract.")                                                \
  V(statistics_type, target_mismatch_sites, "target-mismatch-sites",      \
    "Number of unresolved sites whose provider target identities "        \
    "are all disproven.")                                                 \
  V(statistics_type, rejected_sites, "rejected-sites",                    \
    "Number of unresolved sites whose providers were rejected by "        \
    "signature, caller context, target conditions, or value predicates.") \
  V(statistics_type, missing_fact_sites, "missing-fact-sites",            \
    "Number of unresolved sites with no proven provider because target, " \
    "context, or value facts remain unknown.")                            \
  V(statistics_type, ambiguous_sites, "ambiguous-sites",                  \
    "Number of unresolved sites with multiple best providers.")           \
  V(statistics_type, materialization_blocked_sites,                       \
    "materialization-blocked-sites",                                      \
    "Number of unresolved sites with a selected provider that "           \
    "could not be materialized.")                                         \
  V(statistics_type, provider_edges, "provider-edges",                    \
    "Number of apply-generated provider liveness edges.")                 \
  V(statistics_type, symbols_pruned, "symbols-pruned",                    \
    "Number of unreachable private symbols pruned after selection.")

LOOM_PASS_STATISTICS_DEFINE(loom_template_selection_statistics,
                            loom_template_selection_statistics_t,
                            LOOM_TEMPLATE_SELECTION_STATISTICS)

static const loom_pass_info_t loom_template_selection_pass_info_storage = {
    .name = IREE_SVL("select-templates"),
    .description = IREE_SVL(
        "Select func.template providers for live func.apply contract demands."),
    .kind = LOOM_PASS_MODULE,
    .option_defs = kTemplateSelectionOptions,
    .option_count = IREE_ARRAYSIZE(kTemplateSelectionOptions),
    .statistic_layout = &loom_template_selection_statistics_layout,
};

const loom_pass_info_t* loom_template_selection_pass_info(void) {
  return &loom_template_selection_pass_info_storage;
}

static iree_status_t loom_template_selection_parse_mode(
    iree_string_view_t value, loom_template_selection_pass_state_t* state) {
  if (state->has_mode_option) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate option 'mode' for pass 'select-templates'");
  }
  if (iree_string_view_equal(value, IREE_SV("early"))) {
    state->mode = LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  } else if (iree_string_view_equal(value, IREE_SV("final"))) {
    state->mode = LOOM_TEMPLATE_SELECTION_MODE_FINAL;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "select-templates option 'mode' expected 'early' or 'final', got "
        "'%.*s'",
        (int)value.size, value.data);
  }
  state->has_mode_option = true;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_template_selection_pass_state_t* state =
      (loom_template_selection_pass_state_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("mode"))) {
    return loom_template_selection_parse_mode(value, state);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'select-templates'",
                          (int)name.size, name.data);
}

iree_status_t loom_template_selection_create(loom_pass_t* pass,
                                             iree_string_view_t options) {
  loom_template_selection_pass_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->mode = LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) continue;
      if (iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
        IREE_RETURN_IF_ERROR(loom_template_selection_parse_mode(
            option->schema->enum_values[option->enum_value_index].value,
            state));
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'select-templates'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_pass_options_parse(pass->info->name, options,
                                (loom_pass_option_parse_callback_t){
                                    .fn = loom_template_selection_parse_option,
                                    .user_data = state,
                                }));
  }
  pass->state = state;
  return iree_ok_status();
}

static loom_template_selection_mode_t loom_template_selection_mode(
    const loom_pass_t* pass) {
  if (pass->state) {
    const loom_template_selection_pass_state_t* state =
        (const loom_template_selection_pass_state_t*)pass->state;
    return state->mode;
  }
  if (!pass->decoded_options) return LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
    const loom_pass_decoded_option_t* option =
        &pass->decoded_options->options[i];
    if (!option->present) continue;
    if (!iree_string_view_equal(option->schema->name, IREE_SV("mode"))) {
      continue;
    }
    return option->enum_value_index == 1 ? LOOM_TEMPLATE_SELECTION_MODE_FINAL
                                         : LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  }
  return LOOM_TEMPLATE_SELECTION_MODE_EARLY;
}

//===----------------------------------------------------------------------===//
// Plan model
//===----------------------------------------------------------------------===//

typedef enum loom_template_provider_feasibility_e {
  LOOM_TEMPLATE_PROVIDER_REJECT = 0,
  LOOM_TEMPLATE_PROVIDER_MAYBE = 1,
  LOOM_TEMPLATE_PROVIDER_MATCH = 2,
} loom_template_provider_feasibility_t;

typedef enum loom_template_provider_unresolved_reason_e {
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_NONE = 0,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY = 1,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_CALLER_CONTEXT = 2,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION = 3,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE = 4,
} loom_template_provider_unresolved_reason_t;

typedef struct loom_template_provider_classification_t {
  // Combined signature, target, context, condition, and predicate outcome.
  loom_template_provider_feasibility_t feasibility;

  // Target-identity outcome before other provider requirements are applied.
  loom_template_provider_feasibility_t target_feasibility;

  // First unresolved requirement category in deterministic evaluation order.
  loom_template_provider_unresolved_reason_t unresolved_reason;

  // First unresolved typed target condition, or NULL for other categories.
  const loom_target_condition_t* unresolved_target_condition;
} loom_template_provider_classification_t;

typedef enum loom_template_predicate_arg_kind_e {
  LOOM_TEMPLATE_PREDICATE_ARG_INVALID = 0,
  LOOM_TEMPLATE_PREDICATE_ARG_CONST = 1,
  LOOM_TEMPLATE_PREDICATE_ARG_VALUE = 2,
} loom_template_predicate_arg_kind_t;

typedef struct loom_template_predicate_arg_t {
  // Resolved argument category.
  loom_template_predicate_arg_kind_t kind;

  // Apply-site SSA value when kind is VALUE.
  loom_value_id_t value_id;

  // Integer literal when kind is CONST.
  int64_t constant;

  // Scalar facts for the literal or apply-site SSA value.
  loom_value_facts_t facts;
} loom_template_predicate_arg_t;

typedef enum loom_template_selection_action_e {
  LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED = 0,
  LOOM_TEMPLATE_SELECTION_ACTION_SELECT = 1,
} loom_template_selection_action_t;

typedef enum loom_template_selection_blocker_e {
  LOOM_TEMPLATE_SELECTION_BLOCKER_NONE = 0,
  LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER = 1,
  LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH = 2,
  LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED = 3,
  LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS = 4,
  LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS = 5,
  LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION = 6,
} loom_template_selection_blocker_t;

typedef struct loom_template_selection_entry_t {
  // Live func.apply operation to rewrite or diagnose.
  loom_op_t* apply_op;

  // Interned contract key demanded by apply_op.
  loom_string_id_t contract_id;

  // Borrowed contract key text.
  iree_string_view_t contract;

  // Selected provider when action is SELECT or materialization is blocked.
  const loom_func_provider_summary_t* selected_provider;

  // Highest-priority provider whose applicability remains unproven.
  const loom_func_provider_summary_t* unresolved_provider;

  // First unresolved target condition on unresolved_provider, or NULL.
  const loom_target_condition_t* unresolved_target_condition;

  // Selection action for this apply.
  loom_template_selection_action_t action;

  // Reason an unresolved apply could not be selected.
  loom_template_selection_blocker_t blocker;

  // First unresolved requirement category on unresolved_provider.
  loom_template_provider_unresolved_reason_t unresolved_reason;
} loom_template_selection_entry_t;

typedef struct loom_template_provider_equivalence_t {
  // First compared local provider symbol.
  loom_symbol_ref_t lhs;

  // Second compared local provider symbol.
  loom_symbol_ref_t rhs;

  // Structural equivalence result for the pair.
  bool equivalent;
} loom_template_provider_equivalence_t;

typedef struct loom_template_selection_state_t {
  // Active pass invocation.
  loom_pass_t* pass;

  // Typed statistics storage for the current pass invocation.
  loom_template_selection_statistics_t* statistics;

  // Module being transformed.
  loom_module_t* module;

  // Reusable condition traversal state for application path facts.
  loom_condition_query_t condition_query;

  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // Target-refined versions observed against the current module symbols.
  loom_target_function_version_snapshot_t target_versions;

  // Symbol facts backing the provider catalog.
  loom_symbol_fact_table_t fact_table;

  // Local provider catalog keyed by func.apply contract.
  loom_func_provider_catalog_t catalog;

  // Concrete symbol dependencies for this module snapshot.
  loom_symbol_dependency_table_t dependencies;

  // Symbol-pruning policy shared with the liveness root classifier.
  loom_symbol_pruning_options_t pruning_options;

  // Liveness result after apply-generated provider edges.
  loom_symbol_liveness_t liveness;

  // True when the caller requested pass report detail rows.
  bool reports_enabled;

  // Reachable apply-site selection entries.
  loom_template_selection_entry_t* entries;

  // Number of valid selection entries.
  iree_host_size_t entry_count;

  // Capacity of entries.
  iree_host_size_t entry_capacity;

  // Structural provider comparisons cached for this immutable analysis phase.
  struct {
    // Compared provider pairs.
    loom_template_provider_equivalence_t* entries;

    // Number of cached comparisons.
    iree_host_size_t count;

    // Allocated entries.
    iree_host_size_t capacity;
  } equivalence_cache;

  // Reusable branch-relation storage for the apply site being classified.
  struct {
    // Relations implied by structured control-flow ancestors.
    loom_condition_integer_relation_t* relations;

    // Allocated relation count.
    iree_host_size_t capacity;
  } application_path_scratch;
} loom_template_selection_state_t;

static iree_string_view_t loom_template_selection_contract_name(
    const loom_module_t* module, loom_string_id_t contract_id) {
  if (contract_id < module->strings.count) {
    return module->strings.entries[contract_id];
  }
  return IREE_SV("<invalid>");
}

static iree_string_view_t loom_template_selection_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t fallback) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return fallback;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return fallback;
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_template_selection_context_symbol_name(
    const loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context) {
  if (!context || !context->source_symbol) {
    return IREE_SV("<none>");
  }
  if (context->source_symbol->name_id >= state->module->strings.count) {
    return IREE_SV("<none>");
  }
  return state->module->strings.entries[context->source_symbol->name_id];
}

static iree_string_view_t loom_template_selection_unresolved_reason_code(
    loom_template_provider_unresolved_reason_t reason) {
  switch (reason) {
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY:
      return IREE_SV("target_identity");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_CALLER_CONTEXT:
      return IREE_SV("caller_context");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION:
      return IREE_SV("target_condition");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE:
      return IREE_SV("value_predicate");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_template_selection_target_condition_name(
    const loom_template_selection_state_t* state,
    const loom_target_condition_t* condition) {
  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(condition->value);
  const loom_parameterized_attr_descriptor_t* family =
      loom_context_resolve_parameterized_attr(state->module->context,
                                              family_kind);
  if (family == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "resolved target condition family disappeared from its context");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_bstring_view(family->name);
}

static iree_string_view_t loom_template_selection_blocker_code(
    loom_template_selection_blocker_t blocker) {
  switch (blocker) {
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER:
      return IREE_SV("no_provider");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH:
      return IREE_SV("target_mismatch");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED:
      return IREE_SV("all_rejected");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS:
      return IREE_SV("missing_facts");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS:
      return IREE_SV("ambiguous");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION:
      return IREE_SV("materialization_blocked");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NONE:
    default:
      return IREE_SV("unresolved");
  }
}

static iree_string_view_t loom_template_selection_outcome(
    const loom_template_selection_entry_t* entry,
    int64_t highest_provider_priority) {
  if (entry->action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
    if (entry->selected_provider &&
        entry->selected_provider->priority < highest_provider_priority) {
      return IREE_SV("fallback_selected");
    }
    return IREE_SV("selected");
  }
  switch (entry->blocker) {
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER:
      return IREE_SV("no_provider");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH:
      return IREE_SV("target_mismatch");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED:
      return IREE_SV("rejected");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS:
      return IREE_SV("missing_facts");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS:
      return IREE_SV("ambiguous");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION:
      return IREE_SV("materialization_blocked");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NONE:
    default:
      return IREE_SV("unresolved");
  }
}

static bool loom_template_provider_is_materializable(
    const loom_func_provider_summary_t* provider) {
  return provider->origin == LOOM_FUNC_PROVIDER_ORIGIN_LOCAL &&
         provider->kind == LOOM_FUNC_PROVIDER_KIND_TEMPLATE &&
         provider->has_body && loom_symbol_ref_is_valid(provider->symbol);
}

static iree_status_t loom_template_providers_are_equivalent(
    loom_template_selection_state_t* state,
    const loom_func_provider_summary_t* lhs,
    const loom_func_provider_summary_t* rhs, bool* out_equivalent) {
  *out_equivalent = false;
  if (!loom_template_provider_is_materializable(lhs) ||
      !loom_template_provider_is_materializable(rhs)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->equivalence_cache.count; ++i) {
    const loom_template_provider_equivalence_t* entry =
        &state->equivalence_cache.entries[i];
    const bool forward = entry->lhs.symbol_id == lhs->symbol.symbol_id &&
                         entry->rhs.symbol_id == rhs->symbol.symbol_id;
    const bool reverse = entry->lhs.symbol_id == rhs->symbol.symbol_id &&
                         entry->rhs.symbol_id == lhs->symbol.symbol_id;
    if (!forward && !reverse) continue;
    *out_equivalent = entry->equivalent;
    return iree_ok_status();
  }

  bool equivalent = false;
  IREE_RETURN_IF_ERROR(loom_symbol_definitions_equivalent(
      state->module, lhs->symbol, rhs->symbol, state->pass->arena,
      &equivalent));
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(state->pass->arena, state->equivalence_cache.count,
                            state->equivalence_cache.count + 1,
                            sizeof(*state->equivalence_cache.entries),
                            &state->equivalence_cache.capacity,
                            (void**)&state->equivalence_cache.entries));
  state->equivalence_cache.entries[state->equivalence_cache.count++] =
      (loom_template_provider_equivalence_t){
          .lhs = lhs->symbol,
          .rhs = rhs->symbol,
          .equivalent = equivalent,
      };
  *out_equivalent = equivalent;
  return iree_ok_status();
}

typedef struct loom_template_selection_apply_target_t {
  // Authored target witness retained for diagnostics and reports.
  loom_symbol_ref_t witness;

  // Function target facts used for provider compatibility.
  const loom_target_facts_t* facts;
} loom_template_selection_apply_target_t;

typedef struct loom_template_selection_application_facts_t {
  // Function-scoped SSA facts projected with the function target facts.
  const loom_value_fact_table_t* values;

  // Facts established only along the lexical path to this func.apply.
  loom_condition_fact_set_t path;
} loom_template_selection_application_facts_t;

static iree_status_t loom_template_selection_lookup_target_facts(
    loom_template_selection_state_t* state, loom_symbol_ref_t target_ref,
    const loom_target_facts_t** out_target) {
  *out_target = NULL;
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &state->fact_table, state->module, target_ref, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts == NULL) {
    return iree_ok_status();
  }
  *out_target = target_facts->projection;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_resolve_apply_target(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    loom_template_selection_apply_target_t* out_target) {
  *out_target = (loom_template_selection_apply_target_t){
      .witness = loom_symbol_ref_null(),
  };
  if (!context || !context->source_symbol ||
      !context->source_symbol->defining_op) {
    return iree_ok_status();
  }
  loom_func_like_t source_function =
      loom_func_like_cast(state->module, context->source_symbol->defining_op);
  if (!loom_func_like_isa(source_function)) {
    return iree_ok_status();
  }
  out_target->witness = loom_func_like_target(source_function);
  const loom_symbol_ref_t source_ref = loom_func_like_callee(source_function);
  IREE_ASSERT(loom_symbol_ref_is_valid(source_ref));
  IREE_ASSERT(source_ref.module_id == 0);
  IREE_ASSERT(source_ref.symbol_id < state->module->symbols.count);
  const loom_target_function_version_t* function_version =
      loom_target_function_version_snapshot_at(&state->target_versions,
                                               source_ref.symbol_id);
  if (function_version != NULL) {
    out_target->facts = function_version->function_target_facts;
    return iree_ok_status();
  }
  return loom_template_selection_lookup_target_facts(state, out_target->witness,
                                                     &out_target->facts);
}

static iree_status_t loom_template_selection_provider_target_feasibility(
    loom_template_selection_state_t* state,
    const loom_template_selection_apply_target_t* apply_target,
    const loom_func_provider_summary_t* provider,
    loom_template_provider_feasibility_t* out_feasibility) {
  *out_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT;
  if (!loom_symbol_ref_is_valid(provider->target_symbol)) {
    *out_feasibility = LOOM_TEMPLATE_PROVIDER_MATCH;
    return iree_ok_status();
  }
  if (loom_symbol_ref_is_valid(apply_target->witness) &&
      provider->target_symbol.module_id == apply_target->witness.module_id &&
      provider->target_symbol.symbol_id == apply_target->witness.symbol_id) {
    *out_feasibility = LOOM_TEMPLATE_PROVIDER_MATCH;
    return iree_ok_status();
  }
  if (apply_target->facts == NULL) {
    *out_feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    return iree_ok_status();
  }
  const loom_target_facts_t* target_requirement = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_lookup_target_facts(
      state, provider->target_symbol, &target_requirement));
  if (target_requirement == NULL) {
    *out_feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    return iree_ok_status();
  }
  *out_feasibility = loom_target_facts_satisfy_identity_requirement(
                         apply_target->facts, target_requirement)
                         ? LOOM_TEMPLATE_PROVIDER_MATCH
                         : LOOM_TEMPLATE_PROVIDER_REJECT;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_append_report_detail(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_template_selection_entry_t* entry,
    const loom_template_selection_apply_target_t* apply_target,
    iree_host_size_t provider_count, uint32_t target_identity_match_count,
    uint32_t target_identity_unresolved_count, uint32_t possible_count,
    uint32_t best_match_count, int64_t highest_provider_priority) {
  if (!state->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[17];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"),
      loom_template_selection_outcome(entry, highest_provider_priority));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("function"),
      loom_template_selection_context_symbol_name(state, context));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("apply_op"), loom_op_name(state->module, entry->apply_op));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("contract"), entry->contract);
  if (apply_target->facts != NULL) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("target"),
        loom_target_facts_identity_name(apply_target->facts));
  } else if (loom_symbol_ref_is_valid(apply_target->witness)) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("target"),
        loom_template_selection_symbol_name(
            state->module, apply_target->witness, IREE_SV("<invalid>")));
  }
  if (entry->selected_provider) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("selected_provider"), entry->selected_provider->name);
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("selected_priority"), entry->selected_provider->priority);
  }
  if (entry->unresolved_provider) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("unresolved_provider"), entry->unresolved_provider->name);
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("unresolved_priority"), entry->unresolved_provider->priority);
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("unresolved_reason"),
        loom_template_selection_unresolved_reason_code(
            entry->unresolved_reason));
  }
  if (entry->unresolved_target_condition) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("unresolved_condition"),
        loom_template_selection_target_condition_name(
            state, entry->unresolved_target_condition));
  }
  if (provider_count > 0) {
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("highest_provider_priority"), highest_provider_priority);
  }
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("provider_count"), (uint64_t)provider_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("target_identity_match_count"), target_identity_match_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("target_identity_unresolved_count"),
      target_identity_unresolved_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("possible_count"), possible_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("best_match_count"), best_match_count);
  return loom_pass_report_append_detail(
      state->pass, IREE_SV("template-selection"), fields, field_count);
}

//===----------------------------------------------------------------------===//
// Provider feasibility
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_types_match(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider, bool* out_match) {
  *out_match = false;
  loom_value_slice_t operands = loom_func_apply_operands(apply_op);
  loom_value_slice_t results = loom_func_apply_results(apply_op);
  if (operands.count != provider->argument_count ||
      results.count != provider->result_count) {
    return iree_ok_status();
  }
  loom_type_value_remap_t signature_remap = {
      .source_values =
          provider->func_facts ? provider->func_facts->argument_ids : NULL,
      .target_values = operands.values,
      .count = operands.count,
  };

  for (uint16_t i = 0; i < operands.count; ++i) {
    if (operands.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply operand value %u is outside the "
                              "module value table",
                              (uint32_t)operands.values[i]);
    }
    loom_type_t operand_type =
        loom_module_value_type(state->module, operands.values[i]);
    if (!loom_type_equal_after_value_remap(state->module,
                                           provider->argument_types[i],
                                           operand_type, &signature_remap)) {
      return iree_ok_status();
    }
  }

  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply result value %u is outside the "
                              "module value table",
                              (uint32_t)results.values[i]);
    }
    loom_type_t result_type =
        loom_module_value_type(state->module, results.values[i]);
    if (!loom_type_equal_after_value_remap(state->module,
                                           provider->result_types[i],
                                           result_type, &signature_remap)) {
      return iree_ok_status();
    }
  }

  *out_match = true;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_collect_application_path_facts(
    loom_condition_query_t* condition_query,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_condition_fact_set_t* out_path, bool* out_complete) {
  *out_complete = true;
  const loom_op_t* child = apply_op;
  for (const loom_op_t* ancestor = apply_op->parent_op; ancestor;
       child = ancestor, ancestor = ancestor->parent_op) {
    if (!loom_scf_if_isa(ancestor) || child->parent_block == NULL) continue;
    const loom_region_t* child_region = child->parent_block->parent_region;
    bool assumed_truth = false;
    if (child_region == loom_scf_if_then_region(ancestor)) {
      assumed_truth = true;
    } else if (child_region != loom_scf_if_else_region(ancestor)) {
      continue;
    }
    bool edge_complete = false;
    IREE_RETURN_IF_ERROR(loom_condition_facts_query_into(
        condition_query, value_facts, loom_scf_if_condition(ancestor),
        assumed_truth, out_path, &edge_complete));
    *out_complete &= edge_complete;
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_grow_application_path_scratch(
    loom_template_selection_state_t* state, iree_host_size_t new_capacity) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->pass->arena, state->application_path_scratch.capacity,
      new_capacity, sizeof(*state->application_path_scratch.relations),
      &state->application_path_scratch.capacity,
      (void**)&state->application_path_scratch.relations));
  return iree_ok_status();
}

static iree_status_t loom_template_selection_prepare_application_path_facts(
    loom_template_selection_state_t* state,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_template_selection_application_facts_t* out_facts) {
  if (state->application_path_scratch.capacity == 0) {
    IREE_RETURN_IF_ERROR(
        loom_template_selection_grow_application_path_scratch(state, 16));
  }

  for (;;) {
    loom_condition_fact_set_initialize(
        state->application_path_scratch.relations,
        state->application_path_scratch.capacity, &out_facts->path);
    bool path_complete = false;
    IREE_RETURN_IF_ERROR(loom_template_selection_collect_application_path_facts(
        &state->condition_query, value_facts, apply_op, &out_facts->path,
        &path_complete));
    if (path_complete) {
      return iree_ok_status();
    }
    const iree_host_size_t old_capacity =
        state->application_path_scratch.capacity;
    const iree_host_size_t new_capacity = old_capacity * 2;
    if (new_capacity <= old_capacity) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "template selection application path fact capacity overflow");
    }
    IREE_RETURN_IF_ERROR(loom_template_selection_grow_application_path_scratch(
        state, new_capacity));
  }
}

static iree_status_t loom_template_selection_prepare_application_facts(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op,
    const loom_template_selection_apply_target_t* apply_target,
    loom_template_selection_application_facts_t* out_facts) {
  *out_facts = (loom_template_selection_application_facts_t){0};
  loom_condition_fact_set_initialize(NULL, 0, &out_facts->path);
  if (!context || !context->source_symbol ||
      !context->source_symbol->defining_op) {
    return iree_ok_status();
  }
  loom_func_like_t source_function =
      loom_func_like_cast(state->module, context->source_symbol->defining_op);
  if (!loom_func_like_isa(source_function) ||
      !loom_func_like_body(source_function)) {
    return iree_ok_status();
  }
  loom_value_fact_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      state->pass, state->module,
      loom_pass_value_fact_scope_function_for_target(source_function,
                                                     apply_target->facts),
      &table));
  out_facts->values = table;
  return loom_template_selection_prepare_application_path_facts(
      state, table, apply_op, out_facts);
}

static bool loom_template_selection_remap_provider_value(
    const loom_op_t* apply_op, const loom_func_provider_summary_t* provider,
    loom_value_id_t provider_value_id, loom_value_id_t* out_apply_value_id) {
  if (!provider->func_facts) return false;

  loom_value_slice_t operands = loom_func_apply_operands(apply_op);
  for (uint16_t i = 0;
       i < provider->func_facts->argument_count && i < operands.count; ++i) {
    if (provider->func_facts->argument_ids[i] == provider_value_id) {
      *out_apply_value_id = operands.values[i];
      return true;
    }
  }

  loom_value_slice_t results = loom_func_apply_results(apply_op);
  for (uint16_t i = 0;
       i < provider->func_facts->result_count && i < results.count; ++i) {
    if (provider->func_facts->result_ids[i] == provider_value_id) {
      *out_apply_value_id = results.values[i];
      return true;
    }
  }

  return false;
}

static void loom_template_predicate_arg_initialize(
    loom_template_predicate_arg_t* out_arg) {
  *out_arg = (loom_template_predicate_arg_t){
      .kind = LOOM_TEMPLATE_PREDICATE_ARG_INVALID,
      .value_id = LOOM_VALUE_ID_INVALID,
      .facts = loom_value_facts_unknown(),
  };
}

static bool loom_template_selection_resolve_application_value_arg(
    const loom_template_selection_state_t* state,
    const loom_template_selection_application_facts_t* application_facts,
    loom_value_id_t value_id, loom_template_predicate_arg_t* out_arg) {
  if (value_id >= state->module->values.count) return false;
  out_arg->kind = LOOM_TEMPLATE_PREDICATE_ARG_VALUE;
  out_arg->value_id = value_id;
  if (application_facts->values) {
    out_arg->facts =
        loom_value_fact_table_lookup(application_facts->values, value_id);
    (void)loom_condition_fact_set_apply_to_value_facts(
        &application_facts->path, application_facts->values, value_id,
        &out_arg->facts);
  }
  return true;
}

static bool loom_template_selection_resolve_provider_predicate_arg(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    const loom_template_selection_application_facts_t* application_facts,
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_template_predicate_arg_t* out_arg) {
  loom_template_predicate_arg_initialize(out_arg);
  if (argument_index >= predicate->arg_count) return false;

  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_arg->kind = LOOM_TEMPLATE_PREDICATE_ARG_CONST;
      out_arg->constant = predicate->args[argument_index];
      out_arg->facts = loom_value_facts_exact_i64(out_arg->constant);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) return false;
      loom_value_id_t provider_value_id = (loom_value_id_t)raw_value_id;
      loom_value_id_t apply_value_id = LOOM_VALUE_ID_INVALID;
      if (!loom_template_selection_remap_provider_value(
              apply_op, provider, provider_value_id, &apply_value_id)) {
        return false;
      }
      return loom_template_selection_resolve_application_value_arg(
          state, application_facts, apply_value_id, out_arg);
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static bool loom_template_selection_resolve_application_predicate_arg(
    const loom_template_selection_state_t* state,
    const loom_template_selection_application_facts_t* application_facts,
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_template_predicate_arg_t* out_arg) {
  loom_template_predicate_arg_initialize(out_arg);
  if (argument_index >= predicate->arg_count) return false;

  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[argument_index]) {
    case LOOM_PRED_ARG_CONST:
      out_arg->kind = LOOM_TEMPLATE_PREDICATE_ARG_CONST;
      out_arg->constant = predicate->args[argument_index];
      out_arg->facts = loom_value_facts_exact_i64(out_arg->constant);
      return true;
    case LOOM_PRED_ARG_VALUE: {
      const int64_t raw_value_id = predicate->args[argument_index];
      if (raw_value_id < 0 || raw_value_id > UINT32_MAX) return false;
      return loom_template_selection_resolve_application_value_arg(
          state, application_facts, (loom_value_id_t)raw_value_id, out_arg);
    }
    case LOOM_PRED_ARG_NONE:
    default:
      return false;
  }
}

static loom_template_provider_feasibility_t
loom_template_selection_feasibility_from_bool(bool value) {
  return value ? LOOM_TEMPLATE_PROVIDER_MATCH : LOOM_TEMPLATE_PROVIDER_REJECT;
}

static bool loom_template_predicate_arg_exact_i64(
    const loom_template_predicate_arg_t* arg, int64_t* out_value) {
  if (arg->kind == LOOM_TEMPLATE_PREDICATE_ARG_CONST) {
    *out_value = arg->constant;
    return true;
  }
  return loom_value_facts_as_exact_i64(arg->facts, out_value);
}

static bool loom_template_predicate_relation_kind(
    uint8_t predicate_kind, loom_symbolic_integer_relation_t* out_relation) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_PREDICATE_LT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_PREDICATE_GE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_template_predicate_arg_as_condition_operand(
    const loom_template_predicate_arg_t* arg,
    loom_condition_integer_operand_t* out_operand) {
  *out_operand = (loom_condition_integer_operand_t){0};
  switch (arg->kind) {
    case LOOM_TEMPLATE_PREDICATE_ARG_CONST:
      out_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT;
      out_operand->constant = arg->constant;
      return true;
    case LOOM_TEMPLATE_PREDICATE_ARG_VALUE:
      out_operand->kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE;
      out_operand->value_id = arg->value_id;
      return true;
    default:
      return false;
  }
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_relation(
    const loom_template_predicate_arg_t* lhs,
    const loom_template_predicate_arg_t* rhs, uint8_t predicate_kind,
    const loom_template_selection_application_facts_t* application_facts) {
  loom_condition_integer_relation_t queried_relation = {0};
  if (loom_template_predicate_relation_kind(predicate_kind,
                                            &queried_relation.relation) &&
      loom_template_predicate_arg_as_condition_operand(
          lhs, &queried_relation.left) &&
      loom_template_predicate_arg_as_condition_operand(
          rhs, &queried_relation.right)) {
    bool relation_result = false;
    if (loom_condition_fact_set_proves_integer_relation(
            &application_facts->path, application_facts->values,
            &queried_relation, &relation_result)) {
      return loom_template_selection_feasibility_from_bool(relation_result);
    }
  }

  if (lhs->kind == LOOM_TEMPLATE_PREDICATE_ARG_VALUE &&
      rhs->kind == LOOM_TEMPLATE_PREDICATE_ARG_VALUE &&
      lhs->value_id == rhs->value_id) {
    switch ((loom_predicate_kind_t)predicate_kind) {
      case LOOM_PREDICATE_EQ:
      case LOOM_PREDICATE_LE:
      case LOOM_PREDICATE_GE:
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      case LOOM_PREDICATE_NE:
      case LOOM_PREDICATE_LT:
      case LOOM_PREDICATE_GT:
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      default:
        return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }

  int64_t lhs_exact = 0;
  int64_t rhs_exact = 0;
  if (loom_template_predicate_arg_exact_i64(lhs, &lhs_exact) &&
      loom_template_predicate_arg_exact_i64(rhs, &rhs_exact)) {
    switch ((loom_predicate_kind_t)predicate_kind) {
      case LOOM_PREDICATE_EQ:
        return loom_template_selection_feasibility_from_bool(lhs_exact ==
                                                             rhs_exact);
      case LOOM_PREDICATE_NE:
        return loom_template_selection_feasibility_from_bool(lhs_exact !=
                                                             rhs_exact);
      case LOOM_PREDICATE_LT:
        return loom_template_selection_feasibility_from_bool(lhs_exact <
                                                             rhs_exact);
      case LOOM_PREDICATE_LE:
        return loom_template_selection_feasibility_from_bool(lhs_exact <=
                                                             rhs_exact);
      case LOOM_PREDICATE_GT:
        return loom_template_selection_feasibility_from_bool(lhs_exact >
                                                             rhs_exact);
      case LOOM_PREDICATE_GE:
        return loom_template_selection_feasibility_from_bool(lhs_exact >=
                                                             rhs_exact);
      default:
        return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }

  if (loom_value_facts_is_float(lhs->facts) ||
      loom_value_facts_is_float(rhs->facts)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
      if (lhs->facts.range_hi < rhs->facts.range_lo ||
          rhs->facts.range_hi < lhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    case LOOM_PREDICATE_NE:
      if (lhs->facts.range_hi < rhs->facts.range_lo ||
          rhs->facts.range_hi < lhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    case LOOM_PREDICATE_LT:
      if (lhs->facts.range_hi < rhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      }
      if (lhs->facts.range_lo >= rhs->facts.range_hi) {
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    case LOOM_PREDICATE_LE:
      if (lhs->facts.range_hi <= rhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      }
      if (lhs->facts.range_lo > rhs->facts.range_hi) {
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    case LOOM_PREDICATE_GT:
      if (lhs->facts.range_lo > rhs->facts.range_hi) {
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      }
      if (lhs->facts.range_hi <= rhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    case LOOM_PREDICATE_GE:
      if (lhs->facts.range_lo >= rhs->facts.range_hi) {
        return LOOM_TEMPLATE_PROVIDER_MATCH;
      }
      if (lhs->facts.range_hi < rhs->facts.range_lo) {
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      }
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    default:
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_multiple(
    const loom_template_predicate_arg_t* value_arg, int64_t divisor) {
  if (divisor <= 0 || loom_value_facts_is_float(value_arg->facts)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  int64_t exact_value = 0;
  if (loom_template_predicate_arg_exact_i64(value_arg, &exact_value)) {
    return loom_template_selection_feasibility_from_bool(
        exact_value % divisor == 0);
  }
  if (loom_value_facts_divisible_by(value_arg->facts, divisor)) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  if (value_arg->facts.range_lo > 0 && value_arg->facts.range_hi < divisor) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  if (value_arg->facts.range_hi < 0 && value_arg->facts.range_lo > -divisor) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  return LOOM_TEMPLATE_PROVIDER_MAYBE;
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_pow2(
    const loom_template_predicate_arg_t* value_arg) {
  if (loom_value_facts_is_float(value_arg->facts)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  if (loom_value_facts_is_power_of_two(value_arg->facts)) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  int64_t exact_value = 0;
  if (loom_template_predicate_arg_exact_i64(value_arg, &exact_value)) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  if (value_arg->facts.range_hi < 1) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  return LOOM_TEMPLATE_PROVIDER_MAYBE;
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_range(
    const loom_template_predicate_arg_t* value_arg, int64_t lower_bound,
    int64_t upper_bound) {
  if (lower_bound > upper_bound) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  if (loom_value_facts_is_float(value_arg->facts)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  if (value_arg->facts.range_lo >= lower_bound &&
      value_arg->facts.range_hi <= upper_bound) {
    return LOOM_TEMPLATE_PROVIDER_MATCH;
  }
  if (value_arg->facts.range_hi < lower_bound ||
      value_arg->facts.range_lo > upper_bound) {
    return LOOM_TEMPLATE_PROVIDER_REJECT;
  }
  return LOOM_TEMPLATE_PROVIDER_MAYBE;
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_resolved_predicate(
    const loom_template_selection_application_facts_t* application_facts,
    const loom_predicate_t* predicate, loom_template_predicate_arg_t* args) {
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
      return loom_template_selection_evaluate_relation(
          &args[0], &args[1], predicate->kind, application_facts);
    case LOOM_PREDICATE_MUL: {
      int64_t divisor = 0;
      if (!loom_template_predicate_arg_exact_i64(&args[1], &divisor)) {
        return LOOM_TEMPLATE_PROVIDER_MAYBE;
      }
      return loom_template_selection_evaluate_multiple(&args[0], divisor);
    }
    case LOOM_PREDICATE_MIN:
      return loom_template_selection_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_GE, application_facts);
    case LOOM_PREDICATE_MAX:
      return loom_template_selection_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_LE, application_facts);
    case LOOM_PREDICATE_POW2:
      return loom_template_selection_evaluate_pow2(&args[0]);
    case LOOM_PREDICATE_RANGE: {
      int64_t lower_bound = 0;
      int64_t upper_bound = 0;
      if (!loom_template_predicate_arg_exact_i64(&args[1], &lower_bound) ||
          !loom_template_predicate_arg_exact_i64(&args[2], &upper_bound)) {
        return LOOM_TEMPLATE_PROVIDER_MAYBE;
      }
      return loom_template_selection_evaluate_range(&args[0], lower_bound,
                                                    upper_bound);
    }
    default:
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
}

static bool loom_template_selection_predicate_arity_is_valid(
    const loom_predicate_t* predicate) {
  const uint8_t expected_argument_count =
      loom_predicate_kind_argument_count(predicate->kind);
  return expected_argument_count != UINT8_MAX &&
         predicate->arg_count == expected_argument_count &&
         predicate->arg_count <= IREE_ARRAYSIZE(predicate->args);
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_provider_predicate(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    const loom_template_selection_application_facts_t* application_facts,
    const loom_predicate_t* predicate) {
  if (!loom_template_selection_predicate_arity_is_valid(predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_template_predicate_arg_t args[3];
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (!loom_template_selection_resolve_provider_predicate_arg(
            state, apply_op, provider, application_facts, predicate, i,
            &args[i])) {
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return loom_template_selection_evaluate_resolved_predicate(application_facts,
                                                             predicate, args);
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_application_predicate(
    const loom_template_selection_state_t* state,
    const loom_template_selection_application_facts_t* application_facts,
    const loom_predicate_t* predicate) {
  if (!loom_template_selection_predicate_arity_is_valid(predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_template_predicate_arg_t args[3];
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (!loom_template_selection_resolve_application_predicate_arg(
            state, application_facts, predicate, i, &args[i])) {
      return LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return loom_template_selection_evaluate_resolved_predicate(application_facts,
                                                             predicate, args);
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_predicates(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    const loom_template_selection_application_facts_t* application_facts) {
  loom_template_provider_feasibility_t feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  for (uint16_t i = 0; i < provider->predicate_count; ++i) {
    const loom_template_provider_feasibility_t predicate_feasibility =
        loom_template_selection_evaluate_provider_predicate(
            state, apply_op, provider, application_facts,
            &provider->predicates[i]);
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      return LOOM_TEMPLATE_PROVIDER_REJECT;
    }
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return feasibility;
}

static bool loom_template_selection_apply_has_ancestor(
    const loom_op_t* apply_op, loom_op_kind_t ancestor_kind) {
  for (const loom_op_t* ancestor = apply_op->parent_op; ancestor;
       ancestor = ancestor->parent_op) {
    if (ancestor->kind == ancestor_kind) return true;
  }
  return false;
}

// A template body acquires the physical ancestors of its application site only
// after selection and inlining.
static bool loom_template_selection_has_deferred_caller_context(
    const loom_symbol_liveness_contributor_context_t* context) {
  return context && context->source_symbol &&
         context->source_symbol->defining_op &&
         loom_func_template_isa(context->source_symbol->defining_op);
}

static loom_template_provider_feasibility_t
loom_template_selection_provider_context_feasibility(
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, const loom_func_provider_summary_t* provider) {
  loom_template_provider_feasibility_t feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  const bool has_deferred_caller_context =
      loom_template_selection_has_deferred_caller_context(context);
  for (uint16_t i = 0; i < provider->required_caller_ancestor_count; ++i) {
    if (!loom_template_selection_apply_has_ancestor(
            apply_op, provider->required_caller_ancestors[i])) {
      if (!has_deferred_caller_context) return LOOM_TEMPLATE_PROVIDER_REJECT;
      feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  for (uint16_t i = 0; i < provider->forbidden_caller_ancestor_count; ++i) {
    if (loom_template_selection_apply_has_ancestor(
            apply_op, provider->forbidden_caller_ancestors[i])) {
      return LOOM_TEMPLATE_PROVIDER_REJECT;
    }
    if (has_deferred_caller_context) {
      feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return feasibility;
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_target_condition_query_value(
    const loom_template_selection_state_t* state,
    const loom_target_condition_t* condition,
    const loom_template_selection_application_facts_t* application_facts,
    loom_value_id_t value_id) {
  if (application_facts->values == NULL ||
      condition->descriptor->project_query_predicate == NULL) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  loom_value_fact_contextual_query_origin_t origin = {0};
  if (!loom_value_fact_table_query_contextual_query_origin(
          application_facts->values, state->module, value_id, &origin) ||
      origin.family_kind != loom_attr_as_parameterized_kind(condition->value)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }

  loom_predicate_t predicate = {0};
  if (!condition->descriptor->project_query_predicate(
          condition->value, origin.key, value_id, &predicate)) {
    return LOOM_TEMPLATE_PROVIDER_MAYBE;
  }
  return loom_template_selection_evaluate_application_predicate(
      state, application_facts, &predicate);
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_target_condition_queries(
    const loom_template_selection_state_t* state,
    const loom_target_condition_t* condition,
    const loom_template_selection_application_facts_t* application_facts) {
  bool has_match = false;
  bool has_reject = false;
  for (iree_host_size_t i = 0;
       i < application_facts->path.integer_relation_count; ++i) {
    const loom_condition_integer_relation_t* relation =
        &application_facts->path.integer_relations[i];
    for (uint8_t operand_index = 0; operand_index < 2; ++operand_index) {
      const loom_condition_integer_operand_t operand =
          operand_index == 0 ? relation->left : relation->right;
      if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) continue;
      const loom_template_provider_feasibility_t feasibility =
          loom_template_selection_evaluate_target_condition_query_value(
              state, condition, application_facts, operand.value_id);
      has_match |= feasibility == LOOM_TEMPLATE_PROVIDER_MATCH;
      has_reject |= feasibility == LOOM_TEMPLATE_PROVIDER_REJECT;
    }
  }
  if (has_match && has_reject) return LOOM_TEMPLATE_PROVIDER_MAYBE;
  if (has_match) return LOOM_TEMPLATE_PROVIDER_MATCH;
  if (has_reject) return LOOM_TEMPLATE_PROVIDER_REJECT;
  return LOOM_TEMPLATE_PROVIDER_MAYBE;
}

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_target_conditions(
    const loom_template_selection_state_t* state,
    const loom_func_provider_summary_t* provider,
    const loom_target_facts_t* target_facts,
    const loom_template_selection_application_facts_t* application_facts,
    const loom_target_condition_t** out_unresolved_condition) {
  *out_unresolved_condition = NULL;
  loom_template_provider_feasibility_t feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  for (uint16_t i = 0; i < provider->target_condition_count; ++i) {
    const loom_target_condition_t* condition = &provider->target_conditions[i];
    const loom_target_condition_outcome_t outcome =
        loom_target_condition_evaluate(condition->descriptor, condition->value,
                                       target_facts);
    switch (outcome) {
      case LOOM_TARGET_CONDITION_MATCH:
        break;
      case LOOM_TARGET_CONDITION_UNKNOWN:
      case LOOM_TARGET_CONDITION_UNBOUND: {
        const loom_template_provider_feasibility_t query_feasibility =
            loom_template_selection_evaluate_target_condition_queries(
                state, condition, application_facts);
        if (query_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
          return LOOM_TEMPLATE_PROVIDER_REJECT;
        }
        if (query_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
          feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
          if (*out_unresolved_condition == NULL) {
            *out_unresolved_condition = condition;
          }
        }
        break;
      }
      case LOOM_TARGET_CONDITION_REJECT:
        return LOOM_TEMPLATE_PROVIDER_REJECT;
      default:
        IREE_ASSERT_UNREACHABLE("target condition returned an invalid outcome");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return feasibility;
}

static iree_status_t loom_template_selection_classify_provider(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, const loom_func_provider_summary_t* provider,
    const loom_template_selection_apply_target_t* apply_target,
    const loom_template_selection_application_facts_t* application_facts,
    loom_template_provider_classification_t* out_classification) {
  *out_classification = (loom_template_provider_classification_t){
      .feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
      .target_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT,
  };
  IREE_RETURN_IF_ERROR(loom_template_selection_provider_target_feasibility(
      state, apply_target, provider, &out_classification->target_feasibility));
  if (out_classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return iree_ok_status();
  }

  bool types_match = false;
  IREE_RETURN_IF_ERROR(loom_template_selection_types_match(
      state, apply_op, provider, &types_match));
  if (!types_match) return iree_ok_status();
  loom_template_provider_feasibility_t context_feasibility =
      loom_template_selection_provider_context_feasibility(context, apply_op,
                                                           provider);
  if (context_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return iree_ok_status();
  }

  const loom_target_condition_t* unresolved_target_condition = NULL;
  const loom_template_provider_feasibility_t target_condition_feasibility =
      loom_template_selection_evaluate_target_conditions(
          state, provider, apply_target->facts, application_facts,
          &unresolved_target_condition);
  if (target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    return iree_ok_status();
  }

  loom_template_provider_feasibility_t predicate_feasibility =
      LOOM_TEMPLATE_PROVIDER_MATCH;
  if (provider->predicate_count > 0) {
    predicate_feasibility = loom_template_selection_evaluate_predicates(
        state, apply_op, provider, application_facts);
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      return iree_ok_status();
    }
  }

  out_classification->feasibility =
      out_classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE ||
              context_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE ||
              target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE ||
              predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE
          ? LOOM_TEMPLATE_PROVIDER_MAYBE
          : LOOM_TEMPLATE_PROVIDER_MATCH;
  if (out_classification->target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    out_classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY;
  } else if (context_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    out_classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_CALLER_CONTEXT;
  } else if (target_condition_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    out_classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION;
    out_classification->unresolved_target_condition =
        unresolved_target_condition;
  } else if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    out_classification->unresolved_reason =
        LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE;
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_provider_live(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_func_provider_summary_t* provider) {
  ++state->statistics->provider_edges;
  return loom_symbol_liveness_mark_symbol_ref(context, provider->symbol);
}

// Keeps rejected candidate evidence live until the final selection boundary
// emits the concrete target or placement diagnostic.
static iree_status_t loom_template_selection_preserve_rejected_providers(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    loom_func_provider_slice_t providers) {
  if (state->mode != LOOM_TEMPLATE_SELECTION_MODE_EARLY) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
        state, context, &providers.providers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_append_entry(
    loom_template_selection_state_t* state,
    loom_template_selection_entry_t** out_entry) {
  if (state->entry_count >= state->entry_capacity) {
    iree_host_size_t old_capacity = state->entry_capacity;
    iree_host_size_t new_capacity = old_capacity > 0 ? old_capacity * 2 : 16;
    if (new_capacity < old_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template selection entry capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->pass->arena, old_capacity, new_capacity, sizeof(*state->entries),
        &state->entry_capacity, (void**)&state->entries));
  }
  loom_template_selection_entry_t* entry = &state->entries[state->entry_count];
  memset(entry, 0, sizeof(*entry));
  ++state->entry_count;
  *out_entry = entry;
  return iree_ok_status();
}

static void loom_template_selection_record_blocker(
    loom_template_selection_state_t* state,
    loom_template_selection_entry_t* entry,
    loom_template_selection_blocker_t blocker) {
  entry->blocker = blocker;
  ++state->statistics->unresolved_sites;
  switch (blocker) {
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER:
      ++state->statistics->no_provider_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH:
      ++state->statistics->target_mismatch_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED:
      ++state->statistics->rejected_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS:
      ++state->statistics->missing_fact_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS:
      ++state->statistics->ambiguous_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION:
      ++state->statistics->materialization_blocked_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_NONE:
    default:
      break;
  }
}

static iree_status_t loom_template_selection_mark_match_priority(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    const loom_template_selection_apply_target_t* apply_target,
    const loom_template_selection_application_facts_t* application_facts,
    int64_t priority) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_classification_t classification = {0};
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, apply_target, application_facts,
        &classification));
    if (classification.feasibility != LOOM_TEMPLATE_PROVIDER_MATCH ||
        provider->priority != priority) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_unresolved_candidates(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    const loom_template_selection_apply_target_t* apply_target,
    const loom_template_selection_application_facts_t* application_facts,
    bool has_match, int64_t match_priority) {
  if (state->mode != LOOM_TEMPLATE_SELECTION_MODE_EARLY) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_classification_t classification = {0};
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, apply_target, application_facts,
        &classification));
    if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) continue;
    if (has_match) {
      if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_MATCH &&
          provider->priority != match_priority) {
        continue;
      }
      if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE &&
          provider->priority < match_priority) {
        continue;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
}

static bool loom_template_selection_requires_application_facts(
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    const loom_target_facts_t* target_facts) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    if (providers.providers[i].predicate_count > 0) return true;
  }
  if (!loom_template_selection_apply_has_ancestor(apply_op, LOOM_OP_SCF_IF)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    for (uint16_t j = 0; j < provider->target_condition_count; ++j) {
      const loom_target_condition_t* condition =
          &provider->target_conditions[j];
      if (condition->descriptor->project_query_predicate == NULL) continue;
      const loom_target_condition_outcome_t outcome =
          loom_target_condition_evaluate(condition->descriptor,
                                         condition->value, target_facts);
      if (outcome == LOOM_TARGET_CONDITION_UNKNOWN ||
          outcome == LOOM_TARGET_CONDITION_UNBOUND) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_template_selection_analyze_apply(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op) {
  loom_string_id_t contract_id = loom_func_apply_contract(apply_op);
  if (contract_id == LOOM_STRING_ID_INVALID ||
      contract_id >= state->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func.apply has an invalid contract string id");
  }

  loom_template_selection_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_append_entry(state, &entry));
  entry->apply_op = (loom_op_t*)apply_op;
  entry->contract_id = contract_id;
  entry->contract =
      loom_template_selection_contract_name(state->module, contract_id);
  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED;

  ++state->statistics->apply_sites;

  loom_template_selection_apply_target_t apply_target = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_resolve_apply_target(
      state, context, &apply_target));
  loom_func_provider_slice_t providers =
      loom_func_provider_catalog_lookup(&state->catalog, contract_id);
  if (providers.count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER);
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        /*target_identity_match_count=*/0,
        /*target_identity_unresolved_count=*/0,
        /*possible_count=*/0, /*best_match_count=*/0,
        /*highest_provider_priority=*/INT64_MIN);
  }

  loom_template_selection_application_facts_t application_facts = {0};
  loom_condition_fact_set_initialize(NULL, 0, &application_facts.path);
  if (loom_template_selection_requires_application_facts(apply_op, providers,
                                                         apply_target.facts)) {
    IREE_RETURN_IF_ERROR(loom_template_selection_prepare_application_facts(
        state, context, apply_op, &apply_target, &application_facts));
  }

  bool has_match = false;
  bool has_maybe = false;
  bool has_distinct_best_match = false;
  int64_t best_match_priority = INT64_MIN;
  int64_t highest_provider_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_match_count = 0;
  uint32_t target_identity_match_count = 0;
  uint32_t target_identity_unresolved_count = 0;
  uint32_t possible_count = 0;
  const loom_func_provider_summary_t* best_match_provider = NULL;
  const loom_func_provider_summary_t* highest_maybe_provider = NULL;
  loom_template_provider_classification_t highest_maybe_classification = {0};

  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    if (provider->priority > highest_provider_priority) {
      highest_provider_priority = provider->priority;
    }
    loom_template_provider_classification_t classification = {0};
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, &apply_target, &application_facts,
        &classification));
    if (classification.target_feasibility == LOOM_TEMPLATE_PROVIDER_MATCH) {
      ++target_identity_match_count;
    } else if (classification.target_feasibility ==
               LOOM_TEMPLATE_PROVIDER_MAYBE) {
      ++target_identity_unresolved_count;
    }
    if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      continue;
    }
    ++possible_count;
    if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      has_maybe = true;
      if (highest_maybe_provider == NULL ||
          provider->priority > highest_maybe_priority) {
        highest_maybe_priority = provider->priority;
        highest_maybe_provider = provider;
        highest_maybe_classification = classification;
      }
      continue;
    }

    if (!has_match || provider->priority > best_match_priority) {
      has_match = true;
      best_match_priority = provider->priority;
      best_match_count = 1;
      has_distinct_best_match = false;
      best_match_provider = provider;
    } else if (provider->priority == best_match_priority) {
      ++best_match_count;
      if (!has_distinct_best_match) {
        bool equivalent = false;
        IREE_RETURN_IF_ERROR(loom_template_providers_are_equivalent(
            state, best_match_provider, provider, &equivalent));
        has_distinct_best_match = !equivalent;
      }
    }
  }

  if (has_maybe) {
    entry->unresolved_provider = highest_maybe_provider;
    entry->unresolved_reason = highest_maybe_classification.unresolved_reason;
    entry->unresolved_target_condition =
        highest_maybe_classification.unresolved_target_condition;
  }

  if (target_identity_match_count == 0 &&
      target_identity_unresolved_count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH);
    IREE_RETURN_IF_ERROR(loom_template_selection_preserve_rejected_providers(
        state, context, providers));
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  if (possible_count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED);
    IREE_RETURN_IF_ERROR(loom_template_selection_preserve_rejected_providers(
        state, context, providers));
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  const bool unresolved_blocks_match =
      state->mode == LOOM_TEMPLATE_SELECTION_MODE_EARLY && has_maybe &&
      (!has_match || highest_maybe_priority >= best_match_priority);
  if (!has_match || unresolved_blocks_match) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS);
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_unresolved_candidates(
        state, context, apply_op, providers, &apply_target, &application_facts,
        has_match, best_match_priority));
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  if (has_distinct_best_match) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS);
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_match_priority(
        state, context, apply_op, providers, &apply_target, &application_facts,
        best_match_priority));
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  entry->selected_provider = best_match_provider;
  IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
      state, context, best_match_provider));
  if (!loom_template_provider_is_materializable(best_match_provider)) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION);
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_SELECT;
  entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_NONE;
  if (best_match_provider->priority < highest_provider_priority) {
    ++state->statistics->fallback_selected_sites;
  }
  ++state->statistics->selected_sites;
  return loom_template_selection_append_report_detail(
      state, context, entry, &apply_target, providers.count,
      target_identity_match_count, target_identity_unresolved_count,
      possible_count, best_match_count, highest_provider_priority);
}

static iree_status_t loom_template_selection_visit_reachable_demand(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_func_contract_demand_t* demand) {
  return loom_template_selection_analyze_apply(
      (loom_template_selection_state_t*)user_data, context, demand->apply_op);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_emit_blockers(
    loom_template_selection_state_t* state) {
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      continue;
    }
    if (entry->blocker == LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS &&
        entry->unresolved_target_condition != NULL) {
      const loom_func_provider_summary_t* provider = entry->unresolved_provider;
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_op_name(state->module, entry->apply_op)),
          loom_param_string(state->pass->info->name),
          loom_param_string(entry->contract),
          loom_param_string(provider->name),
          loom_param_string(loom_template_selection_target_condition_name(
              state, entry->unresolved_target_condition)),
      };
      loom_diagnostic_related_op_t related_op = {
          .label = IREE_SV("unresolved provider condition"),
          .op = provider->function.op,
          .field_ref = loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              provider->function.vtable->requires_attr_index),
      };
      loom_diagnostic_emission_t emission = {
          .op = entry->apply_op,
          .error = LOOM_ERR_LOWERING_048,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = &related_op,
          .related_op_count = 1,
      };
      IREE_RETURN_IF_ERROR(
          iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, entry->apply_op)),
        loom_param_string(state->pass->info->name),
        loom_param_string(entry->contract),
        loom_param_string(loom_template_selection_blocker_code(entry->blocker)),
    };
    const loom_func_provider_summary_t* related_provider =
        entry->selected_provider;
    if (related_provider == NULL &&
        entry->blocker == LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS) {
      related_provider = entry->unresolved_provider;
    }
    loom_diagnostic_related_op_t related_op = {
        .label = entry->selected_provider ? IREE_SV("selected provider")
                                          : IREE_SV("unresolved provider"),
        .op = related_provider ? related_provider->function.op : NULL,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .op = entry->apply_op,
        .error = LOOM_ERR_LOWERING_045,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_op.op ? &related_op : NULL,
        .related_op_count = related_op.op ? 1 : 0,
    };
    IREE_RETURN_IF_ERROR(
        iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_copy_result_types(
    loom_template_selection_state_t* state, const loom_op_t* apply_op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  loom_value_slice_t results = loom_func_apply_results(apply_op);
  if (results.count == 0) return iree_ok_status();

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, results.count,
                                sizeof(*result_types), (void**)&result_types));
  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func.apply result value %u is outside the "
                              "module value table",
                              (uint32_t)results.values[i]);
    }
    result_types[i] = loom_module_value_type(state->module, results.values[i]);
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_rewrite_entry(
    loom_template_selection_state_t* state, loom_rewriter_t* rewriter,
    const loom_template_selection_entry_t* entry) {
  loom_value_slice_t operands = loom_func_apply_operands(entry->apply_op);
  loom_value_slice_t results = loom_func_apply_results(entry->apply_op);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_copy_result_types(
      state, entry->apply_op, &result_types));

  loom_func_call_build_flags_t build_flags =
      LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY;
  uint8_t purity = loom_func_apply_purity(entry->apply_op);
  if (purity != 0) {
    build_flags |= LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
  }
  uint8_t temperature = loom_func_apply_temperature(entry->apply_op);
  if (temperature != 0) {
    build_flags |= LOOM_FUNC_CALL_BUILD_FLAG_HAS_TEMPERATURE;
  }

  loom_builder_set_before(&rewriter->builder, entry->apply_op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      &rewriter->builder, build_flags, purity, temperature,
      LOOM_INLINE_POLICY_INLINE, entry->selected_provider->symbol,
      operands.values, operands.count, result_types, results.count,
      loom_op_tied_results(entry->apply_op), entry->apply_op->tied_result_count,
      entry->apply_op->location, &call_op));
  loom_value_slice_t call_results = loom_func_call_results(call_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, entry->apply_op, call_results.values, call_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, entry->apply_op, call_results.values, call_results.count));
  loom_pass_mark_changed(state->pass);
  return iree_ok_status();
}

static iree_status_t loom_template_selection_execute_rewrites(
    loom_template_selection_state_t* state) {
  bool has_selected_entry = false;
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    if (state->entries[i].action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      has_selected_entry = true;
      break;
    }
  }
  if (!has_selected_entry) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, state->module, state->pass->arena));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->entry_count && iree_status_is_ok(status); ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action != LOOM_TEMPLATE_SELECTION_ACTION_SELECT) continue;
    status = loom_template_selection_rewrite_entry(state, &rewriter, entry);
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_build_liveness(
    loom_template_selection_state_t* state) {
  loom_symbol_liveness_contributor_t contributor = {
      .visit_contract_demand = loom_template_selection_visit_reachable_demand,
      .user_data = state,
  };
  loom_symbol_liveness_options_t options = {
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_pruning_symbol_is_root,
      .root_query_user_data = &state->pruning_options,
      .contributors = &contributor,
      .contributor_count = 1,
  };
  return loom_symbol_liveness_compute(state->module, &state->dependencies,
                                      &options, state->pass->arena,
                                      &state->liveness);
}

iree_status_t loom_template_selection_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  loom_template_selection_state_t state = {
      .pass = pass,
      .statistics = loom_template_selection_statistics(pass),
      .module = module,
      .mode = loom_template_selection_mode(pass),
      .reports_enabled = loom_pass_report_is_enabled(pass),
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_condition_query_initialize(module, pass->arena, &state.condition_query);
  loom_symbol_fact_table_initialize(&state.fact_table, pass->arena);
  loom_func_provider_catalog_initialize(&state.catalog, pass->arena);

  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(target_capability),
      pass->arena, &state.target_versions));
  IREE_RETURN_IF_ERROR(loom_func_provider_catalog_build_local(
      &state.catalog, module, &state.fact_table));
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(module, pass->arena,
                                                          &state.dependencies));
  IREE_RETURN_IF_ERROR(loom_template_selection_build_liveness(&state));

  if (state.mode == LOOM_TEMPLATE_SELECTION_MODE_FINAL) {
    IREE_RETURN_IF_ERROR(loom_template_selection_emit_blockers(&state));
    if (loom_pass_has_error_diagnostics(pass)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_template_selection_execute_rewrites(&state));

  loom_symbol_pruning_result_t pruning_result = {0};
  IREE_RETURN_IF_ERROR(loom_symbol_pruning_erase_unreachable(
      module, &state.liveness, &state.pruning_options, pass->arena,
      &pruning_result));
  if (pruning_result.symbol_count > 0) {
    loom_pass_mark_changed(pass);
    state.statistics->symbols_pruned += pruning_result.symbol_count;
  }

  if (!pass->changed) {
    return iree_ok_status();
  }
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
