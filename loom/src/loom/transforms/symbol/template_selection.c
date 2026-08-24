// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_selection.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/condition.h"
#include "loom/target/pass_environment.h"
#include "loom/transforms/symbol/symbol_pruning.h"
#include "loom/transforms/symbol/template_applicability.h"
#include "loom/util/bstring.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

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
    "Number of live template.apply sites analyzed.")                      \
  V(statistics_type, exact_call_sites, "exact-call-sites",                \
    "Number of live authored template.call sites analyzed.")              \
  V(statistics_type, selected_sites, "selected-sites",                    \
    "Number of template.apply sites resolved to exact template calls.")   \
  V(statistics_type, fallback_selected_sites, "fallback-selected-sites",  \
    "Number of selected sites that used a lower-priority "                \
    "provider while a higher-priority candidate existed.")                \
  V(statistics_type, unresolved_sites, "unresolved-sites",                \
    "Number of live template.apply sites left unresolved.")               \
  V(statistics_type, no_provider_sites, "no-provider-sites",              \
    "Number of unresolved sites with no provider for the "                \
    "requested family.")                                                  \
  V(statistics_type, target_mismatch_sites, "target-mismatch-sites",      \
    "Number of unresolved sites whose provider target identities "        \
    "are all disproven.")                                                 \
  V(statistics_type, rejected_sites, "rejected-sites",                    \
    "Number of unresolved sites whose providers were rejected by "        \
    "signature, target conditions, or value predicates.")                 \
  V(statistics_type, family_rejected_sites, "family-rejected-sites",      \
    "Number of unresolved sites rejected by the template family "         \
    "applicability contract.")                                            \
  V(statistics_type, missing_fact_sites, "missing-fact-sites",            \
    "Number of unresolved sites with no proven provider because target, " \
    "or value facts remain unknown.")                                     \
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
        "Select template providers for live template.apply family demands."),
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
      if (!option->present) {
        continue;
      }
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
  if (!pass->decoded_options) {
    return LOOM_TEMPLATE_SELECTION_MODE_EARLY;
  }
  for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
    const loom_pass_decoded_option_t* option =
        &pass->decoded_options->options[i];
    if (!option->present) {
      continue;
    }
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
  LOOM_TEMPLATE_SELECTION_BLOCKER_FAMILY_REJECTED = 7,
  LOOM_TEMPLATE_SELECTION_BLOCKER_EXACT_CALL_REJECTED = 8,
} loom_template_selection_blocker_t;

typedef enum loom_template_contract_role_e {
  LOOM_TEMPLATE_CONTRACT_NONE = 0,
  LOOM_TEMPLATE_CONTRACT_FAMILY = 1,
  LOOM_TEMPLATE_CONTRACT_PROVIDER = 2,
} loom_template_contract_role_t;

typedef struct loom_template_selection_entry_t {
  // Live template.apply or authored template.call operation.
  loom_op_t* application_op;

  // Template family demanded by apply_op.
  loom_symbol_ref_t family;

  // Borrowed template family symbol name.
  iree_string_view_t family_name;

  // Selected provider when action is SELECT or materialization is blocked.
  const loom_template_provider_summary_t* selected_provider;

  // Direct provider referenced by an authored template.call, or null.
  loom_symbol_ref_t exact_provider;

  // Highest-priority provider whose applicability remains unproven, or NULL
  // when the family declaration contract itself remains unproven.
  const loom_template_provider_summary_t* unresolved_provider;

  // First unresolved target condition on the family or provider contract.
  const loom_target_condition_t* unresolved_target_condition;

  // Selection action for this apply.
  loom_template_selection_action_t action;

  // Reason an unresolved apply could not be selected.
  loom_template_selection_blocker_t blocker;

  // First unresolved family or provider requirement category.
  loom_template_provider_unresolved_reason_t unresolved_reason;

  // Family or provider contract responsible for the blocker.
  loom_template_contract_role_t blocker_contract;
} loom_template_selection_entry_t;

typedef struct loom_template_selection_state_t {
  // Active pass invocation, or NULL for a read-only query.
  loom_pass_t* pass;

  // Typed statistics storage for the current pass invocation.
  loom_template_selection_statistics_t* statistics;

  // Arena owning this selection computation.
  iree_arena_allocator_t* arena;

  // Standalone value-fact owner for a read-only query, or NULL for a pass.
  loom_pass_value_fact_owner_t* value_fact_owner;

  // Module being transformed.
  loom_module_t* module;

  // Reusable condition traversal state for application path facts.
  loom_condition_query_t condition_query;

  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // Target-refined versions observed against the current module symbols.
  const loom_target_function_version_snapshot_t* target_versions;

  // Symbol facts backing the provider catalog.
  loom_symbol_fact_table_t fact_table;

  // Provider catalog keyed by template family symbol.
  const loom_template_provider_catalog_t* catalog;

  // Concrete symbol references for this module snapshot.
  loom_symbol_reference_table_t references;

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

  // External provider origins selected by exact reachable applications.
  struct {
    // Dense origins in first-selection order.
    iree_host_size_t* values;

    // Number of unique selected origins.
    iree_host_size_t count;

    // Allocated value capacity.
    iree_host_size_t capacity;

    // Dense membership bitmap indexed by external origin ordinal.
    uint64_t* membership_bits;

    // Exclusive upper bound for valid external origin ordinals.
    iree_host_size_t origin_count;
  } selected_origins;

  // Reusable branch-relation storage for the apply site being classified.
  struct {
    // Relations implied by structured control-flow ancestors.
    loom_condition_integer_relation_t* relations;

    // Allocated relation count.
    iree_host_size_t capacity;
  } application_path_scratch;
} loom_template_selection_state_t;

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
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION:
      return IREE_SV("target_condition");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE:
      return IREE_SV("value_predicate");
    case LOOM_TEMPLATE_PROVIDER_UNRESOLVED_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t loom_template_selection_contract_role_code(
    loom_template_contract_role_t role) {
  switch (role) {
    case LOOM_TEMPLATE_CONTRACT_FAMILY:
      return IREE_SV("family");
    case LOOM_TEMPLATE_CONTRACT_PROVIDER:
      return IREE_SV("provider");
    case LOOM_TEMPLATE_CONTRACT_NONE:
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
    case LOOM_TEMPLATE_SELECTION_BLOCKER_FAMILY_REJECTED:
      return IREE_SV("family_rejected");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_EXACT_CALL_REJECTED:
      return IREE_SV("exact_call_rejected");
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
    case LOOM_TEMPLATE_SELECTION_BLOCKER_FAMILY_REJECTED:
      return IREE_SV("family_rejected");
    case LOOM_TEMPLATE_SELECTION_BLOCKER_EXACT_CALL_REJECTED:
      return IREE_SV("exact_call_rejected");
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
    const loom_template_provider_summary_t* provider) {
  return provider->kind == LOOM_TEMPLATE_PROVIDER_KIND_DEF &&
         provider->has_body && loom_symbol_ref_is_valid(provider->symbol);
}

static bool loom_template_provider_is_external_materialization(
    const loom_template_provider_summary_t* provider) {
  return provider->kind == LOOM_TEMPLATE_PROVIDER_KIND_DEF &&
         provider->has_body && provider->origin_ordinal != IREE_HOST_SIZE_MAX;
}

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

static iree_status_t loom_template_selection_load_contract_from_facts(
    loom_template_selection_state_t* state,
    const loom_func_symbol_facts_t* facts,
    loom_template_applicability_contract_t* out_contract) {
  *out_contract = (loom_template_applicability_contract_t){
      .module = state->module,
      .target_symbol = facts->target_symbol,
      .argument_ids = facts->argument_ids,
      .result_ids = facts->result_ids,
      .predicates = facts->predicates,
      .target_conditions = facts->target_conditions,
      .argument_count = facts->argument_count,
      .result_count = facts->result_count,
      .predicate_count = facts->predicate_count,
      .target_condition_count = facts->target_condition_count,
  };
  return loom_template_selection_lookup_target_facts(
      state, facts->target_symbol, &out_contract->target_facts);
}

static iree_status_t loom_template_selection_lookup_function_facts(
    loom_template_selection_state_t* state, loom_symbol_ref_t symbol_ref,
    const loom_func_symbol_facts_t** out_facts) {
  *out_facts = NULL;
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &state->fact_table, state->module, symbol_ref, &base_facts));
  const loom_func_symbol_facts_t* facts =
      loom_func_symbol_facts_cast(base_facts);
  *out_facts = facts;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_load_family_contract(
    loom_template_selection_state_t* state, loom_symbol_ref_t family_ref,
    loom_template_applicability_contract_t* out_contract) {
  const loom_func_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_template_selection_lookup_function_facts(state, family_ref, &facts));
  if (facts == NULL || facts->base.symbol_kind != LOOM_SYMBOL_TEMPLATE_DECL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template family has no declaration contract");
  }
  return loom_template_selection_load_contract_from_facts(state, facts,
                                                          out_contract);
}

static iree_status_t loom_template_selection_load_exact_provider_contract(
    loom_template_selection_state_t* state, loom_symbol_ref_t provider_ref,
    const loom_func_symbol_facts_t** out_facts,
    loom_template_applicability_contract_t* out_contract) {
  *out_facts = NULL;
  *out_contract = (loom_template_applicability_contract_t){0};
  const loom_func_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_lookup_function_facts(
      state, provider_ref, &facts));
  if (facts == NULL ||
      (facts->base.symbol_kind != LOOM_SYMBOL_TEMPLATE_DEF &&
       facts->base.symbol_kind != LOOM_SYMBOL_TEMPLATE_UKERNEL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template.call callee has no template provider contract");
  }
  *out_facts = facts;
  return loom_template_selection_load_contract_from_facts(state, facts,
                                                          out_contract);
}

static iree_status_t loom_template_selection_resolve_application_target(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    loom_template_applicability_target_t* out_target) {
  *out_target = (loom_template_applicability_target_t){
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
      loom_target_function_version_snapshot_at(state->target_versions,
                                               source_ref.symbol_id);
  if (function_version != NULL) {
    out_target->facts = function_version->function_target_facts;
    return iree_ok_status();
  }
  return loom_template_selection_lookup_target_facts(state, out_target->witness,
                                                     &out_target->facts);
}

static iree_status_t loom_template_selection_append_report_detail(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_template_selection_entry_t* entry,
    const loom_template_applicability_target_t* apply_target,
    iree_host_size_t provider_count, uint32_t target_identity_match_count,
    uint32_t target_identity_unresolved_count, uint32_t possible_count,
    uint32_t best_match_count, int64_t highest_provider_priority) {
  if (!state->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[19];
  uint16_t field_count = 0;
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("outcome"),
      loom_template_selection_outcome(entry, highest_provider_priority));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("function"),
      loom_template_selection_context_symbol_name(state, context));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("application_op"),
      loom_op_name(state->module, entry->application_op));
  fields[field_count++] = loom_pass_report_detail_string_field(
      IREE_SV("family"), entry->family_name);
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
  if (loom_symbol_ref_is_valid(entry->exact_provider)) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("exact_provider"),
        loom_template_selection_symbol_name(
            state->module, entry->exact_provider, IREE_SV("<invalid>")));
  }
  if (entry->unresolved_provider) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("unresolved_provider"), entry->unresolved_provider->name);
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("unresolved_priority"), entry->unresolved_provider->priority);
  }
  if (entry->unresolved_reason != LOOM_TEMPLATE_PROVIDER_UNRESOLVED_NONE) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("unresolved_contract"),
        loom_template_selection_contract_role_code(entry->blocker_contract));
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
// Application facts
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_collect_application_path_facts(
    loom_condition_query_t* condition_query,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_condition_fact_set_t* out_path, bool* out_complete) {
  *out_complete = true;
  const loom_op_t* child = apply_op;
  for (const loom_op_t* ancestor = apply_op->parent_op; ancestor;
       child = ancestor, ancestor = ancestor->parent_op) {
    if (!loom_scf_if_isa(ancestor) || child->parent_block == NULL) {
      continue;
    }
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
    loom_template_selection_state_t* state, iree_host_size_t minimum_capacity) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->application_path_scratch.capacity, minimum_capacity,
      sizeof(*state->application_path_scratch.relations),
      &state->application_path_scratch.capacity,
      (void**)&state->application_path_scratch.relations));
  return iree_ok_status();
}

static iree_status_t loom_template_selection_prepare_application_path_facts(
    loom_template_selection_state_t* state,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_template_applicability_facts_t* out_facts) {
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
    IREE_RETURN_IF_ERROR(loom_template_selection_grow_application_path_scratch(
        state, state->application_path_scratch.capacity));
  }
}

static iree_status_t loom_template_selection_prepare_application_facts(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op,
    const loom_template_applicability_target_t* apply_target,
    loom_template_applicability_facts_t* out_facts) {
  *out_facts = (loom_template_applicability_facts_t){0};
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
  const loom_pass_value_fact_scope_t scope =
      loom_pass_value_fact_scope_function_for_target(source_function,
                                                     apply_target->facts);
  if (state->pass != NULL) {
    IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
        state->pass, state->module, scope, &table));
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_acquire(
        state->value_fact_owner, state->module, scope, &table));
  }
  out_facts->values = table;
  return loom_template_selection_prepare_application_path_facts(
      state, table, apply_op, out_facts);
}

static iree_status_t loom_template_selection_mark_provider_live(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_template_provider_summary_t* provider) {
  if (!loom_symbol_ref_is_valid(provider->symbol)) {
    return iree_ok_status();
  }
  ++state->statistics->provider_edges;
  return loom_symbol_liveness_mark_symbol_ref(context, provider->symbol);
}

static void loom_template_selection_record_selected_origin(
    loom_template_selection_state_t* state,
    const loom_template_provider_summary_t* provider) {
  if (provider->origin_ordinal == IREE_HOST_SIZE_MAX) {
    return;
  }
  IREE_ASSERT_LT(provider->origin_ordinal,
                 state->selected_origins.origin_count);
  const iree_host_size_t word_index = provider->origin_ordinal >> 6;
  const uint64_t mask = UINT64_C(1) << (provider->origin_ordinal & 63u);
  if ((state->selected_origins.membership_bits[word_index] & mask) != 0) {
    return;
  }
  IREE_ASSERT_LT(state->selected_origins.count,
                 state->selected_origins.capacity);
  state->selected_origins.membership_bits[word_index] |= mask;
  state->selected_origins.values[state->selected_origins.count++] =
      provider->origin_ordinal;
}

// Keeps rejected candidate evidence live until the final selection boundary
// emits the concrete selection diagnostic.
static iree_status_t loom_template_selection_preserve_rejected_providers(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    loom_template_provider_slice_t providers) {
  if (state->mode != LOOM_TEMPLATE_SELECTION_MODE_EARLY) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
        state, context, &providers.providers[i]));
  }
  return iree_ok_status();
}

static loom_template_selection_entry_t* loom_template_selection_append_entry(
    loom_template_selection_state_t* state) {
  IREE_ASSERT_LT(state->entry_count, state->entry_capacity);
  loom_template_selection_entry_t* entry = &state->entries[state->entry_count];
  memset(entry, 0, sizeof(*entry));
  entry->exact_provider = loom_symbol_ref_null();
  ++state->entry_count;
  return entry;
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
    case LOOM_TEMPLATE_SELECTION_BLOCKER_FAMILY_REJECTED:
      ++state->statistics->family_rejected_sites;
      break;
    case LOOM_TEMPLATE_SELECTION_BLOCKER_EXACT_CALL_REJECTED:
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
    const loom_op_t* apply_op, loom_template_provider_slice_t providers,
    const loom_template_applicability_target_t* apply_target,
    const loom_template_applicability_facts_t* application_facts,
    int64_t priority) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_template_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_classification_t classification = {0};
    loom_template_applicability_classify_provider(
        state->module, apply_op, provider, apply_target, application_facts,
        &classification);
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
    const loom_op_t* apply_op, loom_template_provider_slice_t providers,
    const loom_template_applicability_target_t* apply_target,
    const loom_template_applicability_facts_t* application_facts,
    bool has_match, int64_t match_priority) {
  if (state->mode != LOOM_TEMPLATE_SELECTION_MODE_EARLY) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_template_provider_summary_t* provider = &providers.providers[i];
    loom_template_provider_classification_t classification = {0};
    loom_template_applicability_classify_provider(
        state->module, apply_op, provider, apply_target, application_facts,
        &classification);
    if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      continue;
    }
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
    const loom_op_t* apply_op,
    const loom_template_applicability_contract_t* family_contract,
    loom_template_provider_slice_t providers,
    const loom_target_facts_t* target_facts) {
  if (loom_template_applicability_requires_application_facts(
          apply_op, family_contract, target_facts)) {
    return true;
  }
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_template_applicability_contract_t provider_contract =
        loom_template_applicability_provider_contract(&providers.providers[i]);
    if (loom_template_applicability_requires_application_facts(
            apply_op, &provider_contract, target_facts)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_template_selection_analyze_apply(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op) {
  const loom_symbol_ref_t family = loom_template_apply_family(apply_op);
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template.apply has an invalid family symbol");
  }

  loom_template_selection_entry_t* entry =
      loom_template_selection_append_entry(state);
  entry->application_op = (loom_op_t*)apply_op;
  entry->family = family;
  entry->family_name = loom_template_selection_symbol_name(
      state->module, family, IREE_SV("<invalid>"));
  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED;

  ++state->statistics->apply_sites;

  loom_template_applicability_target_t apply_target = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_resolve_application_target(
      state, context, &apply_target));
  loom_template_applicability_contract_t family_contract = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_load_family_contract(
      state, family, &family_contract));
  loom_template_provider_slice_t providers =
      loom_template_provider_catalog_lookup(state->catalog, family);
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

  loom_template_applicability_facts_t application_facts = {0};
  loom_condition_fact_set_initialize(NULL, 0, &application_facts.path);
  if (loom_template_selection_requires_application_facts(
          apply_op, &family_contract, providers, apply_target.facts)) {
    IREE_RETURN_IF_ERROR(loom_template_selection_prepare_application_facts(
        state, context, apply_op, &apply_target, &application_facts));
  }

  loom_template_provider_classification_t family_classification = {0};
  loom_template_applicability_classify_contract(
      state->module, apply_op, &family_contract, &apply_target,
      &application_facts, &family_classification);
  if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
    int64_t highest_provider_priority = INT64_MIN;
    for (iree_host_size_t i = 0; i < providers.count; ++i) {
      highest_provider_priority =
          iree_max(highest_provider_priority, providers.providers[i].priority);
    }
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_FAMILY;
    const loom_template_selection_blocker_t blocker =
        family_classification.target_feasibility ==
                LOOM_TEMPLATE_PROVIDER_REJECT
            ? LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH
            : LOOM_TEMPLATE_SELECTION_BLOCKER_FAMILY_REJECTED;
    loom_template_selection_record_blocker(state, entry, blocker);
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        /*target_identity_match_count=*/0,
        /*target_identity_unresolved_count=*/0,
        /*possible_count=*/0, /*best_match_count=*/0,
        highest_provider_priority);
  }
  if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
    int64_t highest_provider_priority = INT64_MIN;
    for (iree_host_size_t i = 0; i < providers.count; ++i) {
      highest_provider_priority =
          iree_max(highest_provider_priority, providers.providers[i].priority);
    }
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_FAMILY;
    entry->unresolved_reason = family_classification.unresolved_reason;
    entry->unresolved_target_condition =
        family_classification.unresolved_target_condition;
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS);
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_unresolved_candidates(
        state, context, apply_op, providers, &apply_target, &application_facts,
        /*has_match=*/false, /*match_priority=*/INT64_MIN));
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        /*target_identity_match_count=*/0,
        /*target_identity_unresolved_count=*/0,
        /*possible_count=*/0, /*best_match_count=*/0,
        highest_provider_priority);
  }

  bool has_match = false;
  bool has_maybe = false;
  int64_t highest_provider_priority = INT64_MIN;
  int64_t best_match_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_match_count = 0;
  uint32_t target_identity_match_count = 0;
  uint32_t target_identity_unresolved_count = 0;
  uint32_t possible_count = 0;
  const loom_template_provider_summary_t* best_match_provider = NULL;
  const loom_template_provider_summary_t* highest_maybe_provider = NULL;
  loom_template_provider_classification_t highest_maybe_classification = {0};

  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_template_provider_summary_t* provider = &providers.providers[i];
    highest_provider_priority =
        iree_max(highest_provider_priority, provider->priority);
    loom_template_provider_classification_t classification = {0};
    loom_template_applicability_classify_provider(
        state->module, apply_op, provider, &apply_target, &application_facts,
        &classification);
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
      best_match_provider = provider;
    } else if (provider->priority == best_match_priority) {
      ++best_match_count;
    }
  }

  if (has_maybe) {
    entry->unresolved_provider = highest_maybe_provider;
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_PROVIDER;
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

  if (best_match_count > 1) {
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
  if (!loom_template_provider_is_materializable(best_match_provider) &&
      !loom_template_provider_is_external_materialization(
          best_match_provider)) {
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_PROVIDER;
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION);
    return loom_template_selection_append_report_detail(
        state, context, entry, &apply_target, providers.count,
        target_identity_match_count, target_identity_unresolved_count,
        possible_count, best_match_count, highest_provider_priority);
  }

  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_SELECT;
  entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_NONE;
  loom_template_selection_record_selected_origin(state, best_match_provider);
  if (best_match_provider->priority < highest_provider_priority) {
    ++state->statistics->fallback_selected_sites;
  }
  ++state->statistics->selected_sites;
  return loom_template_selection_append_report_detail(
      state, context, entry, &apply_target, providers.count,
      target_identity_match_count, target_identity_unresolved_count,
      possible_count, best_match_count, highest_provider_priority);
}

static iree_status_t loom_template_selection_analyze_exact_call(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* call_op) {
  ++state->statistics->exact_call_sites;

  const loom_symbol_ref_t provider_ref = loom_template_call_callee(call_op);
  if (!loom_symbol_ref_is_valid(provider_ref) || provider_ref.module_id != 0 ||
      provider_ref.symbol_id >= state->module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template.call has an invalid provider symbol");
  }

  const loom_func_symbol_facts_t* provider_facts = NULL;
  loom_template_applicability_contract_t provider_contract = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_load_exact_provider_contract(
      state, provider_ref, &provider_facts, &provider_contract));
  const loom_symbol_ref_t family = provider_facts->template_family;
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= state->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template.call provider has an invalid template family");
  }

  loom_template_applicability_contract_t family_contract = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_load_family_contract(
      state, family, &family_contract));
  loom_template_applicability_target_t apply_target = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_resolve_application_target(
      state, context, &apply_target));

  loom_template_applicability_facts_t application_facts = {0};
  loom_condition_fact_set_initialize(NULL, 0, &application_facts.path);
  if (loom_template_applicability_requires_application_facts(
          call_op, &family_contract, apply_target.facts) ||
      loom_template_applicability_requires_application_facts(
          call_op, &provider_contract, apply_target.facts)) {
    IREE_RETURN_IF_ERROR(loom_template_selection_prepare_application_facts(
        state, context, call_op, &apply_target, &application_facts));
  }

  loom_template_provider_classification_t family_classification = {0};
  loom_template_applicability_classify_contract(
      state->module, call_op, &family_contract, &apply_target,
      &application_facts, &family_classification);
  loom_template_provider_classification_t provider_classification = {0};
  if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_MATCH) {
    loom_template_applicability_classify_contract(
        state->module, call_op, &provider_contract, &apply_target,
        &application_facts, &provider_classification);
  }

  const loom_template_contract_role_t blocker_contract =
      family_classification.feasibility != LOOM_TEMPLATE_PROVIDER_MATCH
          ? LOOM_TEMPLATE_CONTRACT_FAMILY
      : provider_classification.feasibility != LOOM_TEMPLATE_PROVIDER_MATCH
          ? LOOM_TEMPLATE_CONTRACT_PROVIDER
          : LOOM_TEMPLATE_CONTRACT_NONE;
  if (blocker_contract == LOOM_TEMPLATE_CONTRACT_NONE) {
    return iree_ok_status();
  }

  const loom_template_provider_classification_t* blocker_classification =
      blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
          ? &family_classification
          : &provider_classification;
  loom_template_selection_entry_t* entry =
      loom_template_selection_append_entry(state);
  entry->application_op = (loom_op_t*)call_op;
  entry->family = family;
  entry->family_name = loom_template_selection_symbol_name(
      state->module, family, IREE_SV("<invalid>"));
  entry->exact_provider = provider_ref;
  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_UNRESOLVED;
  entry->blocker_contract = blocker_contract;
  entry->unresolved_reason = blocker_classification->unresolved_reason;
  entry->unresolved_target_condition =
      blocker_classification->unresolved_target_condition;
  const loom_template_selection_blocker_t blocker =
      blocker_classification->feasibility == LOOM_TEMPLATE_PROVIDER_REJECT
          ? LOOM_TEMPLATE_SELECTION_BLOCKER_EXACT_CALL_REJECTED
          : LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS;
  loom_template_selection_record_blocker(state, entry, blocker);
  return loom_template_selection_append_report_detail(
      state, context, entry, &apply_target, /*provider_count=*/1,
      provider_classification.target_feasibility == LOOM_TEMPLATE_PROVIDER_MATCH
          ? 1
          : 0,
      provider_classification.target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE
          ? 1
          : 0,
      provider_classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT ? 0
                                                                           : 1,
      /*best_match_count=*/0, provider_facts->priority);
}

static iree_status_t loom_template_selection_analyze_exact_calls(
    loom_template_selection_state_t* state) {
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &state->references.occurrences[i];
    if (occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
        occurrence->user_op == NULL ||
        !loom_template_call_isa(occurrence->user_op) ||
        occurrence->source_symbol_id >= state->module->symbols.count ||
        !loom_symbol_liveness_is_live(&state->liveness,
                                      occurrence->source_symbol_id)) {
      continue;
    }
    loom_symbol_liveness_contributor_context_t context = {
        .module = state->module,
        .references = &state->references,
        .arena = state->arena,
        .source_symbol_id = occurrence->source_symbol_id,
        .source_symbol =
            &state->module->symbols.entries[occurrence->source_symbol_id],
    };
    IREE_RETURN_IF_ERROR(loom_template_selection_analyze_exact_call(
        state, &context, occurrence->user_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_visit_reachable_demand(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  return loom_template_selection_analyze_apply(
      (loom_template_selection_state_t*)user_data, context, demand->apply_op);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static loom_func_like_t loom_template_selection_blocker_contract_function(
    const loom_template_selection_state_t* state,
    const loom_template_selection_entry_t* entry) {
  if (entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY) {
    const loom_symbol_t* family_symbol =
        &state->module->symbols.entries[entry->family.symbol_id];
    return loom_func_like_cast(state->module, family_symbol->defining_op);
  }
  if (entry->blocker_contract != LOOM_TEMPLATE_CONTRACT_PROVIDER) {
    return (loom_func_like_t){0};
  }
  if (loom_symbol_ref_is_valid(entry->exact_provider)) {
    const loom_symbol_t* provider_symbol =
        &state->module->symbols.entries[entry->exact_provider.symbol_id];
    return loom_func_like_cast(state->module, provider_symbol->defining_op);
  }
  const loom_template_provider_summary_t* provider =
      entry->unresolved_provider ? entry->unresolved_provider
                                 : entry->selected_provider;
  return provider ? provider->function : (loom_func_like_t){0};
}

static iree_string_view_t loom_template_selection_blocker_contract_name(
    const loom_template_selection_state_t* state,
    const loom_template_selection_entry_t* entry) {
  if (entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY) {
    return entry->family_name;
  }
  if (loom_symbol_ref_is_valid(entry->exact_provider)) {
    return loom_template_selection_symbol_name(
        state->module, entry->exact_provider, IREE_SV("<invalid>"));
  }
  const loom_template_provider_summary_t* provider =
      entry->unresolved_provider ? entry->unresolved_provider
                                 : entry->selected_provider;
  return provider ? provider->name : IREE_SV("<unknown>");
}

static iree_status_t loom_template_selection_emit_blockers(
    loom_template_selection_state_t* state) {
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action == LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      continue;
    }
    if (entry->blocker == LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS &&
        entry->unresolved_target_condition != NULL) {
      const loom_func_like_t unresolved_contract =
          loom_template_selection_blocker_contract_function(state, entry);
      const iree_string_view_t unresolved_contract_name =
          loom_template_selection_blocker_contract_name(state, entry);
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_op_name(state->module, entry->application_op)),
          loom_param_string(state->pass->info->name),
          loom_param_string(entry->family_name),
          loom_param_string(unresolved_contract_name),
          loom_param_string(loom_template_selection_target_condition_name(
              state, entry->unresolved_target_condition)),
      };
      loom_diagnostic_related_op_t related_op = {
          .label = entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
                       ? IREE_SV("unresolved family condition")
                       : IREE_SV("unresolved provider condition"),
          .op = loom_func_like_isa(unresolved_contract) ? unresolved_contract.op
                                                        : NULL,
          .field_ref =
              loom_func_like_isa(unresolved_contract)
                  ? loom_diagnostic_field_ref(
                        LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                        unresolved_contract.vtable->requires_attr_index)
                  : loom_diagnostic_field_ref_none(),
      };
      loom_diagnostic_emission_t emission = {
          .op = entry->application_op,
          .error = LOOM_ERR_LOWERING_048,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = related_op.op ? &related_op : NULL,
          .related_op_count = related_op.op ? 1 : 0,
      };
      IREE_RETURN_IF_ERROR(
          iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, entry->application_op)),
        loom_param_string(state->pass->info->name),
        loom_param_string(entry->family_name),
        loom_param_string(loom_template_selection_blocker_code(entry->blocker)),
    };
    const loom_func_like_t blocker_contract =
        loom_template_selection_blocker_contract_function(state, entry);
    loom_diagnostic_related_op_t related_op = {
        .label = entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
                     ? IREE_SV("family contract")
                     : IREE_SV("provider contract"),
        .op = loom_func_like_isa(blocker_contract) ? blocker_contract.op : NULL,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .op = entry->application_op,
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
  loom_value_slice_t results = loom_template_apply_results(apply_op);
  if (results.count == 0) {
    return iree_ok_status();
  }

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, results.count,
                                sizeof(*result_types), (void**)&result_types));
  for (uint16_t i = 0; i < results.count; ++i) {
    if (results.values[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "template.apply result value %u is outside the "
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
  loom_value_slice_t operands =
      loom_template_apply_operands(entry->application_op);
  loom_value_slice_t results =
      loom_template_apply_results(entry->application_op);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_template_selection_copy_result_types(
      state, entry->application_op, &result_types));

  loom_template_call_build_flags_t build_flags = 0;
  uint8_t purity = loom_template_apply_purity(entry->application_op);
  if (purity != 0) {
    build_flags |= LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_PURITY;
  }
  uint8_t temperature = loom_template_apply_temperature(entry->application_op);
  if (temperature != 0) {
    build_flags |= LOOM_TEMPLATE_CALL_BUILD_FLAG_HAS_TEMPERATURE;
  }

  loom_builder_set_before(&rewriter->builder, entry->application_op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_template_call_build(
      &rewriter->builder, build_flags, purity, temperature,
      entry->selected_provider->symbol, operands.values, operands.count,
      result_types, results.count, loom_op_tied_results(entry->application_op),
      entry->application_op->tied_result_count, entry->application_op->location,
      &call_op));
  loom_value_slice_t call_results = loom_template_call_results(call_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, entry->application_op, call_results.values, call_results.count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, entry->application_op, call_results.values,
      call_results.count));
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
  if (!has_selected_entry) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, state->module, state->pass->arena));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->entry_count && iree_status_is_ok(status); ++i) {
    const loom_template_selection_entry_t* entry = &state->entries[i];
    if (entry->action != LOOM_TEMPLATE_SELECTION_ACTION_SELECT) {
      continue;
    }
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
      .visit_template_demand = loom_template_selection_visit_reachable_demand,
      .user_data = state,
  };
  loom_symbol_liveness_options_t options = {
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_pruning_symbol_is_root,
      .root_query_user_data = &state->pruning_options,
      .contributors = &contributor,
      .contributor_count = 1,
  };
  return loom_symbol_liveness_compute(state->module, &state->references,
                                      &options, state->arena, &state->liveness);
}

static iree_status_t loom_template_selection_allocate_entries(
    loom_template_selection_state_t* state) {
  iree_host_size_t exact_call_count = 0;
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &state->references.occurrences[i];
    if (occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL &&
        occurrence->user_op != NULL &&
        loom_template_call_isa(occurrence->user_op)) {
      ++exact_call_count;
    }
  }
  if (!iree_host_size_checked_add(state->references.template_demands.count,
                                  exact_call_count, &state->entry_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template application entry count overflow");
  }
  if (state->entry_capacity == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(state->arena, state->entry_capacity,
                                   sizeof(*state->entries),
                                   (void**)&state->entries);
}

static iree_status_t loom_template_selection_allocate_selected_origins(
    loom_template_selection_state_t* state, iree_host_size_t origin_count) {
  state->selected_origins.origin_count = origin_count;
  for (iree_host_size_t i = 0; i < state->catalog->provider_count; ++i) {
    const loom_template_provider_summary_t* provider =
        &state->catalog->providers[i];
    if (provider->origin_ordinal == IREE_HOST_SIZE_MAX) {
      continue;
    }
    if (provider->origin_ordinal >= origin_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template provider origin ordinal %zu exceeds origin count %zu",
          provider->origin_ordinal, origin_count);
    }
    ++state->selected_origins.capacity;
  }
  if (state->selected_origins.capacity == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->selected_origins.capacity,
                                sizeof(*state->selected_origins.values),
                                (void**)&state->selected_origins.values));
  iree_host_size_t rounded_origin_count = 0;
  if (!iree_host_size_checked_add(origin_count, 63, &rounded_origin_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template provider origin bitmap overflow");
  }
  const iree_host_size_t word_count = rounded_origin_count / 64;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      iree_arena_allocator(state->arena), word_count,
      sizeof(*state->selected_origins.membership_bits),
      (void**)&state->selected_origins.membership_bits));
  return iree_ok_status();
}

static iree_status_t loom_template_selection_compute(
    loom_template_selection_state_t* state, iree_host_size_t origin_count) {
  if (state->catalog->module != state->module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template provider catalog and selection module differ");
  }
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(
      state->module, state->arena, &state->references));
  IREE_RETURN_IF_ERROR(loom_template_selection_allocate_entries(state));
  IREE_RETURN_IF_ERROR(
      loom_template_selection_allocate_selected_origins(state, origin_count));
  IREE_RETURN_IF_ERROR(loom_template_selection_build_liveness(state));
  return loom_template_selection_analyze_exact_calls(state);
}

iree_status_t loom_template_selection_query(
    loom_module_t* module,
    const loom_template_selection_query_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    loom_template_selection_query_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->catalog);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_template_selection_query_result_t){0};

  loom_template_selection_statistics_t statistics = {0};
  loom_target_function_version_snapshot_t empty_function_versions = {0};
  loom_pass_value_fact_owner_t value_fact_owner = {0};
  loom_pass_value_fact_owner_initialize(block_pool, &value_fact_owner);
  loom_template_selection_state_t state = {
      .statistics = &statistics,
      .arena = arena,
      .value_fact_owner = &value_fact_owner,
      .module = module,
      .mode = options->mode,
      .target_versions = options->function_versions ? options->function_versions
                                                    : &empty_function_versions,
      .catalog = options->catalog,
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_condition_query_initialize(module, arena, &state.condition_query);
  loom_symbol_fact_table_initialize(&state.fact_table, arena);

  iree_status_t status =
      loom_template_selection_compute(&state, options->origin_count);
  loom_pass_value_fact_owner_deinitialize(&value_fact_owner);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  out_result->selected_origins.values = state.selected_origins.values;
  out_result->selected_origins.count = state.selected_origins.count;
  out_result->unresolved_site_count = statistics.unresolved_sites;
  if (options->mode == LOOM_TEMPLATE_SELECTION_MODE_FINAL &&
      statistics.unresolved_sites > 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%zu reachable template applications remain "
                            "unresolved",
                            statistics.unresolved_sites);
  }
  return iree_ok_status();
}

iree_status_t loom_template_selection_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  loom_target_function_version_snapshot_t target_versions = {0};
  loom_template_provider_catalog_t catalog = {0};
  loom_template_selection_state_t state = {
      .pass = pass,
      .statistics = loom_template_selection_statistics(pass),
      .arena = pass->arena,
      .module = module,
      .mode = loom_template_selection_mode(pass),
      .target_versions = &target_versions,
      .catalog = &catalog,
      .reports_enabled = loom_pass_report_is_enabled(pass),
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_condition_query_initialize(module, pass->arena, &state.condition_query);
  loom_symbol_fact_table_initialize(&state.fact_table, pass->arena);
  loom_template_provider_catalog_initialize(&catalog, pass->arena);

  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(target_capability),
      pass->arena, &target_versions));
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_build_local(
      &catalog, module, &state.fact_table));
  IREE_RETURN_IF_ERROR(
      loom_template_selection_compute(&state, /*origin_count=*/0));

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
