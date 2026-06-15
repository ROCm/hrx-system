// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_selection.h"

#include <stdint.h>
#include <string.h>

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
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/selection.h"
#include "loom/transforms/symbol/symbol_pruning.h"

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

#define LOOM_TEMPLATE_SELECTION_STATISTICS(V, statistics_type)           \
  V(statistics_type, apply_sites, "apply-sites",                         \
    "Number of live func.apply sites analyzed.")                         \
  V(statistics_type, selected_sites, "selected-sites",                   \
    "Number of func.apply sites resolved to inline calls.")              \
  V(statistics_type, fallback_selected_sites, "fallback-selected-sites", \
    "Number of selected sites that used a lower-priority "               \
    "provider while a higher-priority candidate existed.")               \
  V(statistics_type, unresolved_sites, "unresolved-sites",               \
    "Number of live func.apply sites left unresolved.")                  \
  V(statistics_type, no_provider_sites, "no-provider-sites",             \
    "Number of unresolved sites with no provider for the "               \
    "requested contract.")                                               \
  V(statistics_type, target_mismatch_sites, "target-mismatch-sites",     \
    "Number of unresolved sites with no target-applicable "              \
    "provider.")                                                         \
  V(statistics_type, rejected_sites, "rejected-sites",                   \
    "Number of unresolved sites whose providers were rejected "          \
    "by signature or predicates.")                                       \
  V(statistics_type, missing_fact_sites, "missing-fact-sites",           \
    "Number of unresolved sites blocked by unknown predicate "           \
    "facts.")                                                            \
  V(statistics_type, ambiguous_sites, "ambiguous-sites",                 \
    "Number of unresolved sites with multiple best providers.")          \
  V(statistics_type, materialization_blocked_sites,                      \
    "materialization-blocked-sites",                                     \
    "Number of unresolved sites with a selected provider that "          \
    "could not be materialized.")                                        \
  V(statistics_type, provider_edges, "provider-edges",                   \
    "Number of apply-generated provider liveness edges.")                \
  V(statistics_type, symbols_pruned, "symbols-pruned",                   \
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

  // Selection action for this apply.
  loom_template_selection_action_t action;

  // Reason an unresolved apply could not be selected.
  loom_template_selection_blocker_t blocker;
} loom_template_selection_entry_t;

typedef struct loom_template_selection_state_t {
  // Active pass invocation.
  loom_pass_t* pass;

  // Typed statistics storage for the current pass invocation.
  loom_template_selection_statistics_t* statistics;

  // Module being transformed.
  loom_module_t* module;

  // Early or final selection behavior.
  loom_template_selection_mode_t mode;

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

static bool loom_template_selection_symbol_refs_equal(loom_symbol_ref_t lhs,
                                                      loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static loom_symbol_ref_t loom_template_selection_apply_target(
    const loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context) {
  if (!context || !context->source_symbol ||
      !context->source_symbol->defining_op) {
    return loom_symbol_ref_null();
  }
  loom_func_like_t source_function =
      loom_func_like_cast(state->module, context->source_symbol->defining_op);
  if (!loom_func_like_isa(source_function)) {
    return loom_symbol_ref_null();
  }
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(state->pass);
  return loom_target_effective_target_ref(
      loom_func_like_target(source_function), target_capability);
}

static bool loom_template_selection_provider_applies_to_target(
    loom_symbol_ref_t apply_target,
    const loom_func_provider_summary_t* provider) {
  if (!loom_symbol_ref_is_valid(provider->target_symbol)) {
    return true;
  }
  if (!loom_symbol_ref_is_valid(apply_target)) {
    return false;
  }
  return loom_template_selection_symbol_refs_equal(apply_target,
                                                   provider->target_symbol);
}

static iree_status_t loom_template_selection_append_report_detail(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_template_selection_entry_t* entry,
    loom_symbol_ref_t apply_target, iree_host_size_t provider_count,
    uint32_t target_applicable_count, uint32_t possible_count,
    uint32_t best_exact_count, int64_t highest_provider_priority) {
  if (!state->reports_enabled) {
    return iree_ok_status();
  }

  loom_pass_report_detail_field_t fields[12];
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
  if (loom_symbol_ref_is_valid(apply_target)) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("target"),
        loom_template_selection_symbol_name(state->module, apply_target,
                                            IREE_SV("<invalid>")));
  }
  if (entry->selected_provider) {
    fields[field_count++] = loom_pass_report_detail_string_field(
        IREE_SV("selected_provider"), entry->selected_provider->name);
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("selected_priority"), entry->selected_provider->priority);
  }
  if (provider_count > 0) {
    fields[field_count++] = loom_pass_report_detail_int64_field(
        IREE_SV("highest_provider_priority"), highest_provider_priority);
  }
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("provider_count"), (uint64_t)provider_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("target_applicable_count"), target_applicable_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("possible_count"), possible_count);
  fields[field_count++] = loom_pass_report_detail_uint64_field(
      IREE_SV("best_exact_count"), best_exact_count);
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
    if (!loom_type_equal_after_value_remap(provider->argument_types[i],
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
    if (!loom_type_equal_after_value_remap(provider->result_types[i],
                                           result_type, &signature_remap)) {
      return iree_ok_status();
    }
  }

  *out_match = true;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_apply_fact_table(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_value_fact_table_t** out_table) {
  *out_table = NULL;
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
      loom_pass_value_fact_scope_function(source_function), &table));
  *out_table = table;
  return iree_ok_status();
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

static bool loom_template_selection_resolve_predicate_arg(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    const loom_value_fact_table_t* fact_table,
    const loom_predicate_t* predicate, uint8_t argument_index,
    loom_template_predicate_arg_t* out_arg) {
  *out_arg = (loom_template_predicate_arg_t){
      .kind = LOOM_TEMPLATE_PREDICATE_ARG_INVALID,
      .value_id = LOOM_VALUE_ID_INVALID,
      .facts = loom_value_facts_unknown(),
  };
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
      if (apply_value_id >= state->module->values.count) return false;
      out_arg->kind = LOOM_TEMPLATE_PREDICATE_ARG_VALUE;
      out_arg->value_id = apply_value_id;
      if (fact_table) {
        out_arg->facts =
            loom_value_fact_table_lookup(fact_table, apply_value_id);
      }
      return true;
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

static loom_template_provider_feasibility_t
loom_template_selection_evaluate_relation(
    const loom_template_predicate_arg_t* lhs,
    const loom_template_predicate_arg_t* rhs, uint8_t predicate_kind) {
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

static iree_status_t loom_template_selection_evaluate_predicate(
    const loom_template_selection_state_t* state, const loom_op_t* apply_op,
    const loom_func_provider_summary_t* provider,
    const loom_value_fact_table_t* fact_table,
    const loom_predicate_t* predicate,
    loom_template_provider_feasibility_t* out_feasibility) {
  *out_feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
  loom_template_predicate_arg_t args[3];
  uint8_t expected_argument_count =
      loom_predicate_kind_argument_count(predicate->kind);
  if (expected_argument_count == UINT8_MAX ||
      predicate->arg_count != expected_argument_count ||
      predicate->arg_count > IREE_ARRAYSIZE(args)) {
    return iree_ok_status();
  }

  for (uint8_t i = 0; i < predicate->arg_count && i < IREE_ARRAYSIZE(args);
       ++i) {
    if (!loom_template_selection_resolve_predicate_arg(
            state, apply_op, provider, fact_table, predicate, i, &args[i])) {
      return iree_ok_status();
    }
  }

  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
      *out_feasibility = loom_template_selection_evaluate_relation(
          &args[0], &args[1], predicate->kind);
      return iree_ok_status();
    case LOOM_PREDICATE_MUL: {
      int64_t divisor = 0;
      if (!loom_template_predicate_arg_exact_i64(&args[1], &divisor)) {
        return iree_ok_status();
      }
      *out_feasibility =
          loom_template_selection_evaluate_multiple(&args[0], divisor);
      return iree_ok_status();
    }
    case LOOM_PREDICATE_MIN:
      *out_feasibility = loom_template_selection_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_GE);
      return iree_ok_status();
    case LOOM_PREDICATE_MAX:
      *out_feasibility = loom_template_selection_evaluate_relation(
          &args[0], &args[1], LOOM_PREDICATE_LE);
      return iree_ok_status();
    case LOOM_PREDICATE_POW2:
      *out_feasibility = loom_template_selection_evaluate_pow2(&args[0]);
      return iree_ok_status();
    case LOOM_PREDICATE_RANGE: {
      int64_t lower_bound = 0;
      int64_t upper_bound = 0;
      if (!loom_template_predicate_arg_exact_i64(&args[1], &lower_bound) ||
          !loom_template_predicate_arg_exact_i64(&args[2], &upper_bound)) {
        return iree_ok_status();
      }
      *out_feasibility = loom_template_selection_evaluate_range(
          &args[0], lower_bound, upper_bound);
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_template_selection_evaluate_predicates(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, const loom_func_provider_summary_t* provider,
    loom_template_provider_feasibility_t* out_feasibility) {
  const loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(
      loom_template_selection_apply_fact_table(state, context, &fact_table));

  *out_feasibility = LOOM_TEMPLATE_PROVIDER_MATCH;
  for (uint16_t i = 0; i < provider->predicate_count; ++i) {
    loom_template_provider_feasibility_t predicate_feasibility =
        LOOM_TEMPLATE_PROVIDER_MAYBE;
    IREE_RETURN_IF_ERROR(loom_template_selection_evaluate_predicate(
        state, apply_op, provider, fact_table, &provider->predicates[i],
        &predicate_feasibility));
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      *out_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT;
      return iree_ok_status();
    }
    if (predicate_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      *out_feasibility = LOOM_TEMPLATE_PROVIDER_MAYBE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_classify_provider(
    loom_template_selection_state_t* state,
    const loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, const loom_func_provider_summary_t* provider,
    loom_template_provider_feasibility_t* out_feasibility) {
  *out_feasibility = LOOM_TEMPLATE_PROVIDER_REJECT;
  bool types_match = false;
  IREE_RETURN_IF_ERROR(loom_template_selection_types_match(
      state, apply_op, provider, &types_match));
  if (!types_match) return iree_ok_status();

  if (provider->predicate_count > 0) {
    return loom_template_selection_evaluate_predicates(
        state, context, apply_op, provider, out_feasibility);
  }

  *out_feasibility = LOOM_TEMPLATE_PROVIDER_MATCH;
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_provider_live(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_func_provider_summary_t* provider) {
  ++state->statistics->provider_edges;
  return loom_symbol_liveness_mark_symbol_ref(context, provider->symbol);
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

static iree_status_t loom_template_selection_mark_exact_priority(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    int64_t priority) {
  const loom_symbol_ref_t apply_target =
      loom_template_selection_apply_target(state, context);
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    if (!loom_template_selection_provider_applies_to_target(apply_target,
                                                            provider)) {
      continue;
    }
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, &feasibility));
    if (feasibility != LOOM_TEMPLATE_PROVIDER_MATCH ||
        provider->priority != priority) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
}

static iree_status_t loom_template_selection_mark_missing_fact_candidates(
    loom_template_selection_state_t* state,
    loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* apply_op, loom_func_provider_slice_t providers,
    bool has_exact, int64_t exact_priority) {
  const loom_symbol_ref_t apply_target =
      loom_template_selection_apply_target(state, context);
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    if (!loom_template_selection_provider_applies_to_target(apply_target,
                                                            provider)) {
      continue;
    }
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, &feasibility));
    if (feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) continue;
    if (has_exact) {
      if (feasibility == LOOM_TEMPLATE_PROVIDER_MATCH &&
          provider->priority != exact_priority) {
        continue;
      }
      if (feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE &&
          provider->priority < exact_priority) {
        continue;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_template_selection_mark_provider_live(state, context, provider));
  }
  return iree_ok_status();
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

  const loom_symbol_ref_t apply_target =
      loom_template_selection_apply_target(state, context);
  loom_func_provider_slice_t providers =
      loom_func_provider_catalog_lookup(&state->catalog, contract_id);
  if (providers.count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_NO_PROVIDER);
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        /*target_applicable_count=*/0, /*possible_count=*/0,
        /*best_exact_count=*/0, /*highest_provider_priority=*/INT64_MIN);
  }

  bool has_exact = false;
  bool has_maybe = false;
  int64_t best_exact_priority = INT64_MIN;
  int64_t highest_provider_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_exact_count = 0;
  uint32_t target_applicable_count = 0;
  uint32_t possible_count = 0;
  const loom_func_provider_summary_t* best_exact_provider = NULL;

  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const loom_func_provider_summary_t* provider = &providers.providers[i];
    if (provider->priority > highest_provider_priority) {
      highest_provider_priority = provider->priority;
    }
    if (!loom_template_selection_provider_applies_to_target(apply_target,
                                                            provider)) {
      continue;
    }
    ++target_applicable_count;
    loom_template_provider_feasibility_t feasibility =
        LOOM_TEMPLATE_PROVIDER_REJECT;
    IREE_RETURN_IF_ERROR(loom_template_selection_classify_provider(
        state, context, apply_op, provider, &feasibility));
    if (feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
      continue;
    }
    ++possible_count;
    if (feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      has_maybe = true;
      if (provider->priority > highest_maybe_priority) {
        highest_maybe_priority = provider->priority;
      }
      continue;
    }

    if (!has_exact || provider->priority > best_exact_priority) {
      has_exact = true;
      best_exact_priority = provider->priority;
      best_exact_count = 1;
      best_exact_provider = provider;
    } else if (provider->priority == best_exact_priority) {
      ++best_exact_count;
    }
  }

  if (target_applicable_count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_TARGET_MISMATCH);
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        target_applicable_count, possible_count, best_exact_count,
        highest_provider_priority);
  }

  if (possible_count == 0) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_ALL_REJECTED);
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        target_applicable_count, possible_count, best_exact_count,
        highest_provider_priority);
  }

  if (!has_exact ||
      (has_maybe && highest_maybe_priority >= best_exact_priority)) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MISSING_FACTS);
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_missing_fact_candidates(
        state, context, apply_op, providers, has_exact, best_exact_priority));
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        target_applicable_count, possible_count, best_exact_count,
        highest_provider_priority);
  }

  if (best_exact_count > 1) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_AMBIGUOUS);
    IREE_RETURN_IF_ERROR(loom_template_selection_mark_exact_priority(
        state, context, apply_op, providers, best_exact_priority));
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        target_applicable_count, possible_count, best_exact_count,
        highest_provider_priority);
  }

  entry->selected_provider = best_exact_provider;
  IREE_RETURN_IF_ERROR(loom_template_selection_mark_provider_live(
      state, context, best_exact_provider));
  if (!loom_template_provider_is_materializable(best_exact_provider)) {
    loom_template_selection_record_blocker(
        state, entry, LOOM_TEMPLATE_SELECTION_BLOCKER_MATERIALIZATION);
    return loom_template_selection_append_report_detail(
        state, context, entry, apply_target, providers.count,
        target_applicable_count, possible_count, best_exact_count,
        highest_provider_priority);
  }

  entry->action = LOOM_TEMPLATE_SELECTION_ACTION_SELECT;
  entry->blocker = LOOM_TEMPLATE_SELECTION_BLOCKER_NONE;
  if (best_exact_provider->priority < highest_provider_priority) {
    ++state->statistics->fallback_selected_sites;
  }
  ++state->statistics->selected_sites;
  return loom_template_selection_append_report_detail(
      state, context, entry, apply_target, providers.count,
      target_applicable_count, possible_count, best_exact_count,
      highest_provider_priority);
}

static iree_status_t loom_template_selection_visit_reachable_op(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_op_t* op) {
  if (!loom_func_apply_isa(op)) return iree_ok_status();
  return loom_template_selection_analyze_apply(
      (loom_template_selection_state_t*)user_data, context, op);
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
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, entry->apply_op)),
        loom_param_string(state->pass->info->name),
        loom_param_string(entry->contract),
        loom_param_string(loom_template_selection_blocker_code(entry->blocker)),
    };
    loom_op_t* related_provider_op =
        entry->selected_provider ? entry->selected_provider->function.op : NULL;
    loom_diagnostic_related_op_t related_op = {
        .label = IREE_SV("selected provider"),
        .op = related_provider_op,
        .field_ref = loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .op = entry->apply_op,
        .error = LOOM_ERR_LOWERING_045,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_provider_op ? &related_op : NULL,
        .related_op_count = related_provider_op ? 1 : 0,
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
      LOOM_FUNC_INLINE_POLICY_INLINE, entry->selected_provider->symbol,
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
      .visit_op = loom_template_selection_visit_reachable_op,
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
  loom_symbol_fact_table_initialize(&state.fact_table, pass->arena);
  loom_func_provider_catalog_initialize(&state.catalog, pass->arena);

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
  return loom_target_pass_compact_symbols_preserving_target_ref(
      pass, module, pass->arena, NULL);
}
