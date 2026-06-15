// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_selection.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"

static iree_status_t loom_low_source_selection_lookup_func_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_low_source_selection_lookup_target_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_ref_t target_ref,
    const loom_target_symbol_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  if (target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_ref, &base_facts));
  *out_target_facts = loom_target_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_string_view_t loom_low_source_selection_target_bundle_name(
    const loom_target_bundle_t* target_bundle) {
  if (target_bundle == NULL || iree_string_view_is_empty(target_bundle->name)) {
    return IREE_SV("<unnamed>");
  }
  return target_bundle->name;
}

static iree_status_t loom_low_source_selection_emit_target_conflict(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts,
    const loom_target_bundle_t* authored_target,
    const loom_target_bundle_t* selected_target) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(
          loom_low_source_selection_target_bundle_name(authored_target)),
      loom_param_string(
          loom_low_source_selection_target_bundle_name(selected_target)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = func_facts->func_op,
      .error = LOOM_ERR_TARGET_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_low_source_selection_apply_target_selection(
    const loom_low_source_selection_options_t* options,
    const loom_func_symbol_facts_t* func_facts, bool* inout_contract_valid,
    loom_low_source_selection_t* selection) {
  if (!*inout_contract_valid) {
    selection->target_data = NULL;
    return iree_ok_status();
  }
  if (loom_target_selection_is_empty(options->target_selection)) {
    selection->target_data = NULL;
    return iree_ok_status();
  }
  if (options->target_selection.bundle == NULL) {
    selection->target_data = options->target_selection.data;
    return iree_ok_status();
  }
  if (!loom_target_function_contract_bundles_compatible(
          &selection->target_bundle_storage.bundle,
          options->target_selection.bundle)) {
    *inout_contract_valid = false;
    selection->target_data = NULL;
    IREE_RETURN_IF_ERROR(loom_low_source_selection_emit_target_conflict(
        options->diagnostic_emitter, func_facts,
        &selection->target_bundle_storage.bundle,
        options->target_selection.bundle));
    return iree_ok_status();
  }
  loom_target_function_contract_apply_compatible_selection(
      options->target_selection.bundle, &selection->target_bundle_storage);
  selection->target_bundle = &selection->target_bundle_storage.bundle;
  selection->target_data = options->target_selection.data;
  return iree_ok_status();
}

typedef uint8_t loom_low_source_selection_filter_t;

#define LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION ((uint8_t)1u << 0)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_IMPORT_DECL ((uint8_t)1u << 1)

static iree_status_t loom_low_source_selection_try_symbol(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_symbol_fact_table_t* fact_table,
    loom_low_source_selection_filter_t filter, loom_symbol_id_t symbol_id,
    bool* out_compatible, loom_low_source_selection_t* out_selection) {
  *out_compatible = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_lookup_func_facts(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts) {
    return iree_ok_status();
  }
  loom_low_source_selection_kind_t kind = 0;
  if (func_facts->has_body) {
    kind = LOOM_LOW_SOURCE_SELECTION_FUNCTION;
  } else if (func_facts->imports) {
    kind = LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL;
  } else {
    return iree_ok_status();
  }
  if (kind == LOOM_LOW_SOURCE_SELECTION_FUNCTION &&
      !iree_all_bits_set(filter, LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION)) {
    return iree_ok_status();
  }
  if (kind == LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL &&
      !iree_all_bits_set(filter,
                         LOOM_LOW_SOURCE_SELECTION_FILTER_IMPORT_DECL)) {
    return iree_ok_status();
  }
  loom_symbol_ref_t target_ref = func_facts->target_symbol;
  if (!loom_symbol_ref_is_valid(target_ref)) {
    target_ref = options->target_ref;
  }
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  const loom_target_symbol_facts_t* target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_lookup_target_facts(
      module, fact_table, target_ref, &target_facts));
  if (target_facts == NULL) {
    return iree_ok_status();
  }
  bool contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_from_bundle(
      module, func_facts, target_facts->name, &target_facts->storage.bundle,
      options->diagnostic_emitter, &contract_valid,
      &out_selection->target_bundle_storage));
  if (!contract_valid) {
    return iree_ok_status();
  }
  out_selection->target_bundle = &out_selection->target_bundle_storage.bundle;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_apply_target_selection(
      options, func_facts, &contract_valid, out_selection));
  if (!contract_valid) {
    return iree_ok_status();
  }
  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(
          options->policy_registry, out_selection->target_bundle);
  if (policy == NULL) {
    return iree_ok_status();
  }
  if (kind == LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL &&
      policy->import_decl_kind == 0) {
    return iree_ok_status();
  }

  out_selection->kind = kind;
  out_selection->func = loom_func_like_cast(module, func_facts->func_op);
  out_selection->target_ref = target_ref;
  out_selection->policy = policy;
  *out_compatible = true;
  return iree_ok_status();
}

static void loom_low_source_selection_assign(
    const loom_low_source_selection_t* source,
    loom_low_source_selection_t* out_selection) {
  *out_selection = *source;
  loom_target_bundle_storage_rebind(&out_selection->target_bundle_storage);
  out_selection->target_bundle = &out_selection->target_bundle_storage.bundle;
}

static iree_status_t loom_low_select_source_symbols_with_filter(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_low_source_selection_filter_t filter, iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  *out_selection_list = (loom_low_source_selection_list_t){0};
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  loom_low_source_selection_t* selections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*selections), (void**)&selections));
  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    bool compatible = false;
    loom_low_source_selection_t candidate = {0};
    IREE_RETURN_IF_ERROR(loom_low_source_selection_try_symbol(
        module, options, &fact_table, filter, (loom_symbol_id_t)i, &compatible,
        &candidate));
    if (!compatible) {
      continue;
    }
    loom_low_source_selection_assign(&candidate, &selections[selection_count]);
    ++selection_count;
  }

  out_selection_list->values = selections;
  out_selection_list->count = selection_count;
  return iree_ok_status();
}

iree_status_t loom_low_select_source_symbols(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_source_symbols_with_filter(
      module, options,
      LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION |
          LOOM_LOW_SOURCE_SELECTION_FILTER_IMPORT_DECL,
      arena, out_selection_list);
}

iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_source_symbols_with_filter(
      module, options, LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION, arena,
      out_selection_list);
}
