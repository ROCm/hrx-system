// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_selection.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/analysis/cfg_condition_facts.h"
#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/local_value_domain.h"
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
#include "loom/transforms/symbol/inline_callables.h"
#include "loom/transforms/symbol/symbol_pruning.h"
#include "loom/transforms/symbol/template_applicability.h"
#include "loom/transforms/symbol/template_application.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/transforms/symbol/template_rewrite.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Options and statistics
//===----------------------------------------------------------------------===//

typedef struct loom_template_selection_pass_state_t {
  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

  // True when mode was explicitly provided.
  bool has_mode_option;

  // Expands applicable calls with the common inliner in this transaction.
  bool inline_calls;

  // True when rewrite was explicitly provided.
  bool has_rewrite_option;
} loom_template_selection_pass_state_t;

static const loom_pass_option_def_t kTemplateSelectionOptions[] = {
    {IREE_SVL("mode"),
     IREE_SVL("Selection mode: early preserves unresolved applies, final "
              "emits diagnostics for every unresolved live apply.")},
    {IREE_SVL("rewrite"),
     IREE_SVL("Rewrite form: call preserves selected source calls; inline "
              "expands applicable calls before discarding decisions.")},
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
  if (iree_string_view_equal(name, IREE_SV("rewrite"))) {
    if (state->has_rewrite_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate option 'rewrite' for pass "
                              "'select-templates'");
    }
    if (!iree_string_view_equal(value, IREE_SV("call")) &&
        !iree_string_view_equal(value, IREE_SV("inline"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "select-templates option 'rewrite' expected "
                              "'call' or 'inline', got '%.*s'",
                              (int)value.size, value.data);
    }
    state->inline_calls = iree_string_view_equal(value, IREE_SV("inline"));
    state->has_rewrite_option = true;
    return iree_ok_status();
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
      IREE_RETURN_IF_ERROR(loom_template_selection_parse_option(
          state, option->schema->name,
          option->schema->enum_values[option->enum_value_index].value));
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

typedef struct loom_template_selection_entry_t {
  // Applicability result owned by this selection transaction.
  loom_template_application_result_t application;
  // Original containing symbol, retained across application-to-call rewriting.
  loom_symbol_id_t source_symbol_id;
} loom_template_selection_entry_t;

typedef struct loom_template_selection_cfg_facts_t {
  // Graph borrowed from the active function's value-fact scope.
  const loom_cfg_graph_t* graph;
  // Complete condition relations indexed by block and stable edge.
  loom_cfg_condition_relation_table_t conditions;
  // Next CFG region in the same function fact scope.
  struct loom_template_selection_cfg_facts_t* next;
} loom_template_selection_cfg_facts_t;

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

  // Invocation-owned entry demands consumed by symbol liveness.
  const loom_function_version_list_t* function_versions;

  // Symbol facts backing the provider catalog.
  loom_symbol_fact_table_t fact_table;

  // Provider catalog keyed by template family symbol.
  const loom_template_provider_catalog_t* catalog;

  // Ranked decision models for demanded provider-bearing families.
  loom_template_decision_model_catalog_t decision_models;

  // Concrete symbol references for this module snapshot.
  loom_symbol_reference_table_t references;

  // Symbol-pruning policy shared with the liveness root classifier.
  loom_symbol_pruning_options_t pruning_options;

  // Caller-owned query roots supplementing ordinary symbol roots.
  struct {
    // Borrowed module-local symbol IDs in the current snapshot.
    const loom_symbol_id_t* values;
    // Number of entries in |values|.
    iree_host_size_t count;
  } root_symbol_ids;

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

  // True when this transaction also owns call expansion.
  bool inline_calls;

  // Arena-owned plan borrowing the selection snapshot until rewriting ends.
  loom_inline_callables_plan_t* inline_plan;

  // Bit per family still demanded by an unresolved live application. Allocated
  // only when inline rewriting encounters a blocked application.
  uint64_t* demanded_families;

  // External provider origins required by reachable applications.
  struct {
    // Dense origins in first-requirement order.
    iree_host_size_t* values;

    // Number of unique required origins.
    iree_host_size_t count;

    // Allocated value capacity.
    iree_host_size_t capacity;

    // Dense membership bitmap indexed by external origin ordinal.
    uint64_t* membership_bits;

    // Exclusive upper bound for valid external origin ordinals.
    iree_host_size_t origin_count;
  } required_origins;

  // Reusable structured branch relations for the apply site being classified.
  struct {
    // Relations implied by enclosing structured control flow.
    loom_condition_integer_relation_t* relations;

    // Allocated relation count.
    iree_host_size_t relation_capacity;

    // Reusable complete-result wrapper over relations.
    loom_condition_derivation_t local_derivation;

    // Lexical scope nodes for indexed CFG views and local relations.
    loom_condition_fact_scope_t* scopes;

    // Allocated scope-node count.
    iree_host_size_t scope_capacity;
  } application_path_scratch;

  // Reusable ranked-selection storage sized to the largest family.
  loom_template_application_scratch_t decision_scratch;

  // Applications share the active function/target facts during immutable
  // selection. Only this transaction changes the fact owner before rewriting.
  struct {
    // Function of the most recent fact acquisition, including value-only uses.
    loom_op_t* function;
    // Target context used by that function's fact computation.
    const loom_target_facts_t* target_facts;
    // Borrowed active fact scope, shared by all applications in the function.
    loom_value_fact_table_t* values;
    // Function-local value index borrowed by retained CFG relation tables.
    loom_local_value_domain_t value_domain;
    // Exact CFG forwarding identities borrowed by retained relation tables.
    loom_cfg_value_identity_table_t value_identities;
    // CFG entry facts, populated on the first path-dependent application.
    loom_template_selection_cfg_facts_t* cfg_facts;
  } application_scope;
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
  return loom_string_table_get(&module->strings, symbol->name_id);
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
  return loom_string_table_get(&state->module->strings,
                               context->source_symbol->name_id);
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

static iree_string_view_t loom_template_selection_outcome(
    const loom_template_application_result_t* entry,
    int64_t highest_provider_priority) {
  if (entry->blocker == LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
    if (entry->selected_provider &&
        entry->selected_provider->priority < highest_provider_priority) {
      return IREE_SV("fallback_selected");
    }
    return IREE_SV("selected");
  }
  switch (entry->blocker) {
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NO_PROVIDER:
      return IREE_SV("no_provider");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH:
      return IREE_SV("target_mismatch");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_ALL_REJECTED:
      return IREE_SV("rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_FAMILY_REJECTED:
      return IREE_SV("family_rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_EXACT_CALL_REJECTED:
      return IREE_SV("exact_call_rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS:
      return IREE_SV("missing_facts");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_AMBIGUOUS:
      return IREE_SV("ambiguous");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MATERIALIZATION:
      return IREE_SV("materialization_blocked");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE:
    default:
      return IREE_SV("unresolved");
  }
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
    const loom_template_application_result_t* entry,
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
        loom_template_application_condition_name(
            state->module, entry->unresolved_target_condition));
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

typedef struct loom_template_selection_cfg_builder_t {
  // Dominators built from the fact owner's graph snapshots.
  loom_dominance_info_t dominance;
  // Region records allocated in the same transient fact scope.
  loom_template_selection_cfg_facts_t* entries;
} loom_template_selection_cfg_builder_t;

static iree_status_t loom_template_selection_add_cfg_graph(
    void* user_data, const loom_cfg_graph_t* graph) {
  loom_template_selection_cfg_builder_t* builder = user_data;
  IREE_RETURN_IF_ERROR(
      loom_dominance_info_add_cfg_graph(&builder->dominance, graph));
  loom_template_selection_cfg_facts_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(builder->dominance.arena,
                                           sizeof(*entry), (void**)&entry));
  *entry = (loom_template_selection_cfg_facts_t){
      .graph = graph,
      .next = builder->entries,
  };
  builder->entries = entry;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_prepare_cfg_facts(
    loom_template_selection_state_t* state,
    const loom_value_fact_table_t* value_facts) {
  if (state->application_scope.cfg_facts ||
      value_facts->regions.cfg_count == 0) {
    return iree_ok_status();
  }
  const loom_func_like_t function =
      loom_func_like_cast(state->module, state->application_scope.function);
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      state->module, loom_func_like_body(function),
      value_facts->transient_arena, &state->application_scope.value_domain));
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_table_initialize(
      &state->application_scope.value_domain, value_facts->transient_arena,
      &state->application_scope.value_identities));
  loom_template_selection_cfg_builder_t builder = {
      .dominance =
          {
              .module = state->module,
              .arena = value_facts->transient_arena,
          },
  };
  IREE_RETURN_IF_ERROR(loom_value_fact_table_enumerate_cfg_graphs(
      value_facts, (loom_value_fact_cfg_graph_callback_t){
                       .user_data = &builder,
                       .fn = loom_template_selection_add_cfg_graph,
                   }));
  // Complete the shared canonical identity table before publishing any
  // relation domain that borrows it.
  iree_status_t status = iree_ok_status();
  for (loom_template_selection_cfg_facts_t* entry = builder.entries;
       entry && iree_status_is_ok(status); entry = entry->next) {
    const loom_value_fact_cfg_region_t* region =
        loom_value_fact_table_lookup_cfg_region(value_facts,
                                                entry->graph->region);
    IREE_ASSERT(region != NULL);
    status = loom_cfg_value_identity_table_update(
        &state->application_scope.value_identities, region, &builder.dominance,
        value_facts->transient_arena);
  }
  // All regions' dominators and identities are available before entry
  // relations are derived, including values used by nested regions.
  for (loom_template_selection_cfg_facts_t* entry = builder.entries;
       entry && iree_status_is_ok(status); entry = entry->next) {
    status = loom_cfg_condition_relation_table_compute(
        state->module, entry->graph, value_facts, &builder.dominance,
        &state->application_scope.value_domain,
        &state->application_scope.value_identities,
        value_facts->transient_arena, &entry->conditions);
  }
  if (iree_status_is_ok(status)) {
    state->application_scope.cfg_facts = builder.entries;
  }
  return status;
}

static iree_status_t loom_template_selection_collect_application_path_facts(
    const loom_template_selection_cfg_facts_t* cfg_facts,
    loom_condition_query_t* condition_query,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_condition_derivation_t* local_derivation,
    loom_condition_fact_scope_t* scope_storage, iree_host_size_t scope_capacity,
    const loom_condition_fact_scope_t** out_scope, bool* out_complete) {
  *out_scope = NULL;
  *out_complete = true;
  iree_host_size_t scope_count = 0;
  for (const loom_op_t* child = apply_op; child && child->parent_block;
       child = child->parent_op) {
    const loom_block_t* block = child->parent_block;
    const loom_region_t* child_region = block->parent_region;
    for (const loom_template_selection_cfg_facts_t* entry = cfg_facts; entry;
         entry = entry->next) {
      if (entry->graph->region != child_region) {
        continue;
      }
      const loom_cfg_condition_relation_view_t* view =
          loom_cfg_condition_relation_table_block(&entry->conditions,
                                                  block->region_index);
      if (view != NULL) {
        IREE_ASSERT_LT(scope_count, scope_capacity);
        loom_condition_fact_scope_initialize_indexed(
            NULL, &entry->conditions, view, &scope_storage[scope_count++]);
      }
      break;
    }
    const loom_op_t* ancestor = child->parent_op;
    if (!ancestor || !loom_scf_if_isa(ancestor)) {
      continue;
    }
    bool assumed_truth = false;
    if (child_region == loom_scf_if_then_region(ancestor)) {
      assumed_truth = true;
    } else if (child_region != loom_scf_if_else_region(ancestor)) {
      continue;
    }
    bool edge_complete = false;
    IREE_RETURN_IF_ERROR(loom_condition_facts_query_into(
        condition_query, value_facts, loom_scf_if_condition(ancestor),
        assumed_truth, &local_derivation->integer_facts, &edge_complete));
    *out_complete &= edge_complete;
  }
  for (iree_host_size_t i = scope_count; i > 0; --i) {
    scope_storage[i - 1].parent = *out_scope;
    *out_scope = &scope_storage[i - 1];
  }
  if (local_derivation->integer_facts.integer_relation_count != 0) {
    IREE_ASSERT_LT(scope_count, scope_capacity);
    loom_condition_fact_scope_initialize_local(*out_scope, local_derivation,
                                               &scope_storage[scope_count]);
    *out_scope = &scope_storage[scope_count];
  }
  return iree_ok_status();
}

static iree_status_t
loom_template_selection_grow_application_path_relation_scratch(
    loom_template_selection_state_t* state, iree_host_size_t minimum_capacity) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->application_path_scratch.relation_capacity,
      minimum_capacity, sizeof(*state->application_path_scratch.relations),
      &state->application_path_scratch.relation_capacity,
      (void**)&state->application_path_scratch.relations));
  return iree_ok_status();
}

static iree_status_t loom_template_selection_grow_application_path_scopes(
    loom_template_selection_state_t* state, iree_host_size_t minimum_capacity) {
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->application_path_scratch.scope_capacity,
      minimum_capacity, sizeof(*state->application_path_scratch.scopes),
      &state->application_path_scratch.scope_capacity,
      (void**)&state->application_path_scratch.scopes));
  return iree_ok_status();
}

static iree_host_size_t loom_template_selection_application_scope_bound(
    const loom_op_t* apply_op) {
  iree_host_size_t scope_count = 1;
  for (const loom_op_t* child = apply_op; child && child->parent_block;
       child = child->parent_op) {
    ++scope_count;
  }
  return scope_count;
}

static iree_status_t loom_template_selection_prepare_application_path_facts(
    loom_template_selection_state_t* state,
    const loom_value_fact_table_t* value_facts, const loom_op_t* apply_op,
    loom_template_applicability_facts_t* out_facts) {
  IREE_RETURN_IF_ERROR(
      loom_template_selection_prepare_cfg_facts(state, value_facts));
  if (state->application_path_scratch.relation_capacity == 0) {
    IREE_RETURN_IF_ERROR(
        loom_template_selection_grow_application_path_relation_scratch(state,
                                                                       16));
  }
  const iree_host_size_t scope_bound =
      loom_template_selection_application_scope_bound(apply_op);
  if (scope_bound > state->application_path_scratch.scope_capacity) {
    IREE_RETURN_IF_ERROR(loom_template_selection_grow_application_path_scopes(
        state, scope_bound));
  }

  for (;;) {
    state->application_path_scratch.local_derivation =
        (loom_condition_derivation_t){0};
    loom_condition_fact_set_initialize(
        state->application_path_scratch.relations,
        state->application_path_scratch.relation_capacity,
        &state->application_path_scratch.local_derivation.integer_facts);
    bool path_complete = false;
    IREE_RETURN_IF_ERROR(loom_template_selection_collect_application_path_facts(
        state->application_scope.cfg_facts, &state->condition_query,
        value_facts, apply_op,
        &state->application_path_scratch.local_derivation,
        state->application_path_scratch.scopes,
        state->application_path_scratch.scope_capacity, &out_facts->path,
        &path_complete));
    if (path_complete) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_grow_application_path_relation_scratch(
            state, state->application_path_scratch.relation_capacity));
  }
}

static void loom_template_selection_reset_application_scope(
    loom_template_selection_state_t* state) {
  loom_local_value_domain_release(&state->application_scope.value_domain);
  state->application_scope.function = NULL;
  state->application_scope.target_facts = NULL;
  state->application_scope.values = NULL;
  state->application_scope.value_identities =
      (loom_cfg_value_identity_table_t){0};
  state->application_scope.cfg_facts = NULL;
}

static iree_status_t loom_template_selection_prepare_application_facts(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op,
    const loom_template_applicability_target_t* apply_target,
    loom_template_decision_fact_requirements_t requirements,
    loom_template_applicability_facts_t* out_facts) {
  *out_facts = (loom_template_applicability_facts_t){0};
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
  if (state->application_scope.function != source_function.op ||
      state->application_scope.target_facts != apply_target->facts) {
    loom_template_selection_reset_application_scope(state);
    const loom_pass_value_fact_scope_t scope =
        loom_pass_value_fact_scope_function_for_target(source_function,
                                                       apply_target->facts);
    loom_value_fact_table_t* values = NULL;
    if (state->pass != NULL) {
      IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
          state->pass, state->module, scope, &values));
    } else {
      IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_acquire(
          state->value_fact_owner, state->module, scope, &values));
    }
    state->application_scope.function = source_function.op;
    state->application_scope.target_facts = apply_target->facts;
    state->application_scope.values = values;
  }
  const loom_value_fact_table_t* table = state->application_scope.values;
  out_facts->values = table;
  if (!iree_any_bit_set(requirements,
                        LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_PATH)) {
    return iree_ok_status();
  }
  return loom_template_selection_prepare_application_path_facts(
      state, table, apply_op, out_facts);
}

static void loom_template_selection_record_required_origin(
    loom_template_selection_state_t* state,
    const loom_template_provider_summary_t* provider) {
  if (provider->origin_ordinal == IREE_HOST_SIZE_MAX) {
    return;
  }
  IREE_ASSERT_LT(provider->origin_ordinal,
                 state->required_origins.origin_count);
  const iree_host_size_t word_index = provider->origin_ordinal >> 6;
  const uint64_t mask = UINT64_C(1) << (provider->origin_ordinal & 63u);
  if ((state->required_origins.membership_bits[word_index] & mask) != 0) {
    return;
  }
  IREE_ASSERT_LT(state->required_origins.count,
                 state->required_origins.capacity);
  state->required_origins.membership_bits[word_index] |= mask;
  state->required_origins.values[state->required_origins.count++] =
      provider->origin_ordinal;
}

static iree_status_t loom_template_selection_mark_provider_live(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_template_provider_summary_t* provider) {
  loom_template_selection_record_required_origin(state, provider);
  if (!loom_symbol_ref_is_valid(provider->symbol)) {
    return iree_ok_status();
  }
  ++state->statistics->provider_edges;
  return loom_symbol_liveness_mark_symbol_ref(context, provider->symbol);
}

static loom_template_application_result_t* loom_template_selection_append_entry(
    loom_template_selection_state_t* state, loom_symbol_id_t source_symbol_id) {
  IREE_ASSERT_LT(state->entry_count, state->entry_capacity);
  loom_template_selection_entry_t* entry = &state->entries[state->entry_count];
  entry->source_symbol_id = source_symbol_id;
  ++state->entry_count;
  return &entry->application;
}

static void loom_template_selection_record_blocker(
    loom_template_selection_state_t* state,
    loom_template_application_blocker_t blocker) {
  ++state->statistics->unresolved_sites;
  switch (blocker) {
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NO_PROVIDER:
      ++state->statistics->no_provider_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH:
      ++state->statistics->target_mismatch_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_ALL_REJECTED:
      ++state->statistics->rejected_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_FAMILY_REJECTED:
      ++state->statistics->family_rejected_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_EXACT_CALL_REJECTED:
      ++state->statistics->rejected_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS:
      ++state->statistics->missing_fact_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_AMBIGUOUS:
      ++state->statistics->ambiguous_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MATERIALIZATION:
      ++state->statistics->materialization_blocked_sites;
      break;
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE:
    default:
      break;
  }
}

static iree_status_t loom_template_selection_analyze_apply(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  const loom_op_t* apply_op = demand->apply_op;
  const loom_symbol_ref_t family = loom_template_apply_family(apply_op);
  loom_template_application_result_t* entry =
      loom_template_selection_append_entry(state, context->source_symbol_id);
  ++state->statistics->apply_sites;

  loom_template_applicability_target_t apply_target = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_resolve_application_target(
      state, context, &apply_target));
  const loom_template_decision_model_t* model =
      loom_template_decision_model_lookup(&state->decision_models, family);

  loom_template_applicability_facts_t application_facts = {0};
  const loom_template_decision_fact_requirements_t fact_requirements =
      model ? loom_template_decision_model_application_fact_requirements(model,
                                                                         demand)
            : 0;
  if (fact_requirements != 0) {
    IREE_RETURN_IF_ERROR(loom_template_selection_prepare_application_facts(
        state, context, apply_op, &apply_target, fact_requirements,
        &application_facts));
  }
  const loom_template_decision_site_t site = {
      .application_op = apply_op,
      .application_target = &apply_target,
      .application_facts = &application_facts,
  };
  const loom_decision_program_resolution_policy_t resolution_policy =
      state->mode == LOOM_TEMPLATE_SELECTION_MODE_EARLY
          ? LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED
          : LOOM_DECISION_PROGRAM_SELECT_PROVEN;
  loom_template_application_select(state->module, model, &site,
                                   resolution_policy, &state->decision_scratch,
                                   entry);
  for (uint32_t i = 0; i < state->decision_scratch.live_provider_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
        state, context,
        loom_template_decision_model_provider(
            model, state->decision_scratch.live_provider_ordinals[i])));
  }
  if (entry->blocker == LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
    loom_template_selection_record_required_origin(state,
                                                   entry->selected_provider);
    if (entry->selected_provider->priority < model->highest_provider_priority) {
      ++state->statistics->fallback_selected_sites;
    }
    ++state->statistics->selected_sites;
  } else {
    loom_template_selection_record_blocker(state, entry->blocker);
    if (state->inline_calls) {
      if (!state->demanded_families) {
        IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
            iree_arena_allocator(state->arena),
            (state->module->symbols.count + 63) / 64,
            sizeof(*state->demanded_families),
            (void**)&state->demanded_families));
      }
      state->demanded_families[family.symbol_id >> 6] |=
          UINT64_C(1) << (family.symbol_id & 63);
    }
  }

  const loom_template_decision_evidence_summary_t* summary =
      &state->decision_scratch.summary;
  return loom_template_selection_append_report_detail(
      state, context, entry, &apply_target, model ? model->providers.count : 0,
      summary->target_identity_match_count,
      summary->target_identity_unresolved_count, summary->possible_count,
      summary->best_match_count,
      model ? model->highest_provider_priority : INT64_MIN);
}

static iree_status_t loom_template_selection_analyze_exact_call(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* call_op, bool* out_eligible) {
  *out_eligible = false;
  ++state->statistics->exact_call_sites;

  loom_template_application_call_t call = {0};
  IREE_RETURN_IF_ERROR(loom_template_application_load_call(
      state->module, call_op, &state->fact_table, &call));
  loom_template_applicability_target_t apply_target = {0};
  IREE_RETURN_IF_ERROR(loom_template_selection_resolve_application_target(
      state, context, &apply_target));

  loom_template_applicability_facts_t application_facts = {0};
  if (loom_template_applicability_requires_application_facts(
          call_op, &call.family_contract, apply_target.facts) ||
      loom_template_applicability_requires_application_facts(
          call_op, &call.provider_contract, apply_target.facts)) {
    IREE_RETURN_IF_ERROR(loom_template_selection_prepare_application_facts(
        state, context, call_op, &apply_target,
        LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES |
            LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_PATH,
        &application_facts));
  }

  const loom_template_decision_site_t site = {
      .application_op = call_op,
      .application_target = &apply_target,
      .application_facts = &application_facts,
  };
  loom_template_application_result_t result = {0};
  loom_template_provider_classification_t provider_classification = {0};
  loom_template_application_check_call(state->module, &call, &site,
                                       &provider_classification, &result);
  if (result.blocker == LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
    *out_eligible = true;
    return iree_ok_status();
  }
  loom_template_application_result_t* entry =
      loom_template_selection_append_entry(state, context->source_symbol_id);
  *entry = result;
  loom_template_selection_record_blocker(state, entry->blocker);
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
      /*best_match_count=*/0, call.provider_facts->priority);
}

static iree_status_t loom_template_selection_check_inline_eligibility(
    void* user_data, loom_symbol_id_t source_symbol_id, loom_op_t* call_op,
    bool* out_eligible) {
  loom_template_selection_state_t* state = user_data;
  loom_symbol_liveness_contributor_context_t context = {
      .module = state->module,
      .references = &state->references,
      .arena = state->arena,
      .source_symbol_id = source_symbol_id,
      .source_symbol = &state->module->symbols.entries[source_symbol_id],
  };
  return loom_template_selection_analyze_exact_call(state, &context, call_op,
                                                    out_eligible);
}

static iree_status_t loom_template_selection_analyze_exact_calls(
    loom_template_selection_state_t* state) {
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(&state->references, i);
    if (occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
        occurrence->user_op == NULL ||
        !loom_template_call_isa(occurrence->user_op) ||
        occurrence->source_symbol_id >= state->module->symbols.count ||
        !loom_symbol_liveness_is_live(&state->liveness,
                                      occurrence->source_symbol_id)) {
      continue;
    }
    bool eligible = false;
    IREE_RETURN_IF_ERROR(loom_template_selection_check_inline_eligibility(
        state, occurrence->source_symbol_id, (loom_op_t*)occurrence->user_op,
        &eligible));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_visit_reachable_demand(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  return loom_template_selection_analyze_apply(
      (loom_template_selection_state_t*)user_data, context, demand);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_emit_blockers(
    loom_template_selection_state_t* state) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < state->entry_count && iree_status_is_ok(status); ++i) {
    const loom_template_application_result_t* entry =
        &state->entries[i].application;
    if (entry->blocker == LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
      continue;
    }
    status = loom_template_application_emit_blocker(
        state->module, entry, state->pass->info->name,
        state->pass->diagnostic_emitter);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static iree_status_t loom_template_selection_rewrite_entry(
    loom_template_selection_state_t* state, loom_rewriter_t* rewriter,
    const loom_template_selection_entry_t* selection) {
  const loom_template_application_result_t* entry = &selection->application;
  const loom_value_slice_t operands =
      loom_template_apply_operands(entry->application_op);
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_template_rewrite_apply_as_exact_call(
      rewriter, entry->application_op, entry->selected_provider->symbol,
      operands.values, &call_op));
  if (state->inline_plan) {
    loom_inline_callables_plan_append(
        state->inline_plan, selection->source_symbol_id,
        entry->selected_provider->symbol.symbol_id, call_op);
  }
  loom_pass_mark_changed(state->pass);
  return iree_ok_status();
}

static iree_status_t loom_template_selection_execute_rewrites(
    loom_template_selection_state_t* state) {
  bool has_selected_entry = false;
  for (iree_host_size_t i = 0; i < state->entry_count; ++i) {
    if (state->entries[i].application.blocker ==
        LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
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
    if (entry->application.blocker != LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE) {
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
      .function_versions = state->function_versions,
      .root_symbol_ids =
          {
              .values = state->root_symbol_ids.values,
              .count = state->root_symbol_ids.count,
          },
  };
  return loom_symbol_liveness_compute(state->module, &state->references,
                                      &options, state->arena, &state->liveness);
}

static iree_status_t loom_template_selection_allocate_entries(
    loom_template_selection_state_t* state) {
  iree_host_size_t exact_call_count = 0;
  for (iree_host_size_t i = 0; i < state->references.occurrence_count; ++i) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(&state->references, i);
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

static iree_status_t loom_template_selection_allocate_decision_scratch(
    loom_template_selection_state_t* state) {
  const uint32_t maximum_choice_count =
      state->decision_models.maximum_choice_count;
  if (maximum_choice_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, maximum_choice_count,
      sizeof(*state->decision_scratch.live_provider_ordinals),
      (void**)&state->decision_scratch.live_provider_ordinals));
  if (!state->reports_enabled) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(
      state->arena, maximum_choice_count,
      sizeof(*state->decision_scratch.provider_evidence),
      (void**)&state->decision_scratch.provider_evidence);
}

static iree_status_t loom_template_selection_allocate_required_origins(
    loom_template_selection_state_t* state, iree_host_size_t origin_count) {
  state->required_origins.origin_count = origin_count;
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
    ++state->required_origins.capacity;
  }
  if (state->required_origins.capacity == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->required_origins.capacity,
                                sizeof(*state->required_origins.values),
                                (void**)&state->required_origins.values));
  iree_host_size_t rounded_origin_count = 0;
  if (!iree_host_size_checked_add(origin_count, 63, &rounded_origin_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template provider origin bitmap overflow");
  }
  const iree_host_size_t word_count = rounded_origin_count / 64;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      iree_arena_allocator(state->arena), word_count,
      sizeof(*state->required_origins.membership_bits),
      (void**)&state->required_origins.membership_bits));
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
  IREE_RETURN_IF_ERROR(loom_template_decision_model_catalog_build(
      state->module, &state->fact_table, &state->references, state->catalog,
      state->arena, &state->decision_models));
  IREE_RETURN_IF_ERROR(loom_template_selection_allocate_entries(state));
  IREE_RETURN_IF_ERROR(
      loom_template_selection_allocate_decision_scratch(state));
  IREE_RETURN_IF_ERROR(
      loom_template_selection_allocate_required_origins(state, origin_count));
  IREE_RETURN_IF_ERROR(loom_template_selection_build_liveness(state));
  return iree_ok_status();
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
      .root_symbol_ids =
          {
              .values = options->root_symbol_ids.values,
              .count = options->root_symbol_ids.count,
          },
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_condition_query_initialize(module, /*value_domain=*/NULL, arena,
                                  &state.condition_query);
  loom_symbol_fact_table_initialize(&state.fact_table, arena);

  iree_status_t status =
      loom_template_selection_compute(&state, options->origin_count);
  if (iree_status_is_ok(status)) {
    status = loom_template_selection_analyze_exact_calls(&state);
  }
  loom_template_selection_reset_application_scope(&state);
  loom_pass_value_fact_owner_deinitialize(&value_fact_owner);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  out_result->required_origins.values = state.required_origins.values;
  out_result->required_origins.count = state.required_origins.count;
  out_result->unresolved_site_count = statistics.unresolved_sites;
  if (options->mode == LOOM_TEMPLATE_SELECTION_MODE_FINAL &&
      statistics.unresolved_sites > 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%" PRId64
                            " reachable template applications remain "
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
      .function_versions =
          loom_target_pass_capability_function_versions(target_capability),
      .catalog = &catalog,
      .reports_enabled = loom_pass_report_is_enabled(pass),
      .inline_calls = pass->state &&
                      ((const loom_template_selection_pass_state_t*)pass->state)
                          ->inline_calls,
      .pruning_options =
          {
              .flags = LOOM_SYMBOL_PRUNING_RETAIN_TARGET_SOURCE_ENTRIES,
          },
  };
  loom_condition_query_initialize(module, /*value_domain=*/NULL, pass->arena,
                                  &state.condition_query);
  loom_symbol_fact_table_initialize(&state.fact_table, pass->arena);
  loom_template_provider_catalog_initialize(&catalog, pass->arena);

  iree_status_t status = loom_target_function_version_snapshot_build(
      module, loom_target_pass_capability_function_versions(target_capability),
      pass->arena, &target_versions);
  if (iree_status_is_ok(status)) {
    status = loom_template_provider_catalog_build_local(&catalog, module,
                                                        &state.fact_table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_template_selection_compute(&state, /*origin_count=*/0);
  }

  if (iree_status_is_ok(status) && state.inline_calls) {
    const loom_inline_callables_plan_options_t options = {
        .references = &state.references,
        .target_versions = &target_versions,
        .live_symbols = state.liveness.live_symbols,
        .demanded_families = state.demanded_families,
        .template_eligibility =
            {
                .fn = loom_template_selection_check_inline_eligibility,
                .user_data = &state,
            },
        .additional_call_capacity = state.statistics->selected_sites,
    };
    status = loom_inline_callables_plan_create(pass, module, &options,
                                               &state.inline_plan);
  } else if (iree_status_is_ok(status)) {
    status = loom_template_selection_analyze_exact_calls(&state);
  }
  loom_template_selection_reset_application_scope(&state);
  IREE_RETURN_IF_ERROR(status);

  if (state.mode == LOOM_TEMPLATE_SELECTION_MODE_FINAL) {
    IREE_RETURN_IF_ERROR(loom_template_selection_emit_blockers(&state));
    if (loom_pass_has_error_diagnostics(pass)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_template_selection_execute_rewrites(&state));
  if (state.inline_plan) {
    IREE_RETURN_IF_ERROR(loom_inline_callables_plan_execute(state.inline_plan));
    if (loom_pass_has_error_diagnostics(pass)) {
      return iree_ok_status();
    }
  }

  loom_symbol_pruning_result_t pruning_result = {0};
  IREE_RETURN_IF_ERROR(loom_symbol_pruning_erase_unreachable(
      module, &state.liveness, &state.pruning_options,
      loom_target_pass_capability_function_version_owner(target_capability),
      pass->arena, &pruning_result));
  if (pruning_result.symbol_count > 0) {
    loom_pass_mark_changed(pass);
    state.statistics->symbols_pruned += pruning_result.symbol_count;
  }

  if (!pass->changed) {
    return iree_ok_status();
  }
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
