// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_selection.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"

iree_diagnostic_emitter_t loom_low_source_selection_scope_diagnostics(
    const loom_low_source_selection_t* selection,
    iree_diagnostic_emitter_t target, loom_diagnostic_export_scope_t* scope) {
  const loom_target_bundle_t* bundle =
      loom_low_source_selection_target_bundle(selection);
  loom_diagnostic_export_scope_initialize(bundle->export_plan->name, target,
                                          scope);
  return loom_diagnostic_export_scope_emitter(scope);
}

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

static iree_string_view_t loom_low_source_selection_symbol_ref_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static bool loom_low_source_selection_u32_topology_differs(uint32_t lhs,
                                                           uint32_t rhs) {
  return lhs != 0 && rhs != 0 && lhs != rhs;
}

static bool loom_low_source_selection_snapshots_differ(
    const loom_target_snapshot_t* lhs, const loom_target_snapshot_t* rhs) {
  if (lhs == NULL || rhs == NULL) {
    return false;
  }
  return loom_low_source_selection_u32_topology_differs(lhs->subgroup_size,
                                                        rhs->subgroup_size) ||
         loom_low_source_selection_u32_topology_differs(
             lhs->max_flat_workgroup_size, rhs->max_flat_workgroup_size) ||
         loom_low_source_selection_u32_topology_differs(
             lhs->max_workgroup_size.x, rhs->max_workgroup_size.x) ||
         loom_low_source_selection_u32_topology_differs(
             lhs->max_workgroup_size.y, rhs->max_workgroup_size.y) ||
         loom_low_source_selection_u32_topology_differs(
             lhs->max_workgroup_size.z, rhs->max_workgroup_size.z);
}

static iree_status_t loom_low_source_selection_find_candidate_targets(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena, loom_low_source_selection_t* selection) {
  if (selection->kind == LOOM_LOW_SOURCE_SELECTION_REPRESENTATION ||
      !options->collect_target_candidates ||
      selection->target_source != LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION) {
    return iree_ok_status();
  }
  loom_low_source_selection_report_t* report = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*report), (void**)&report));
  *report = (loom_low_source_selection_report_t){
      .target_symbol_name = loom_low_source_selection_symbol_ref_name(
          module, selection->target_ref),
  };
  selection->report = report;
  const loom_target_bundle_t* selected_bundle =
      loom_target_facts_bundle(selection->target_facts);
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_ref_t candidate_ref = {
        .module_id = 0,
        .symbol_id = (loom_symbol_id_t)i,
    };
    if (candidate_ref.symbol_id == selection->target_ref.symbol_id) {
      continue;
    }
    const loom_symbol_facts_base_t* base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
        fact_table, module, candidate_ref, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    if (target_facts == NULL) {
      continue;
    }
    if (!loom_target_function_contract_bundles_compatible(
            &target_facts->projection->storage.bundle, selected_bundle)) {
      continue;
    }
    if (!loom_low_source_selection_snapshots_differ(
            target_facts->projection->storage.bundle.snapshot,
            selected_bundle->snapshot)) {
      continue;
    }
    if (report->candidates.count == 0) {
      report->candidates.symbol_name =
          loom_low_source_selection_symbol_ref_name(module,
                                                    target_facts->symbol);
      report->candidates.bundle =
          loom_target_facts_bundle(target_facts->projection);
    }
    if (report->candidates.count != UINT32_MAX) {
      ++report->candidates.count;
    }
  }
  return iree_ok_status();
}

typedef uint8_t loom_low_source_selection_filter_t;

#define LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION ((uint8_t)1u << 0)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_DECLARATION ((uint8_t)1u << 1)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP ((uint8_t)1u << 2)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_LOW_OP ((uint8_t)1u << 3)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_EXECUTION ((uint8_t)1u << 4)

static iree_status_t loom_low_source_selection_try_symbol(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_symbol_fact_table_t* fact_table,
    const loom_target_function_version_snapshot_t* target_versions,
    loom_low_source_selection_filter_t filter, loom_symbol_id_t symbol_id,
    iree_arena_allocator_t* arena, bool* out_compatible,
    loom_low_source_selection_t* out_selection) {
  *out_compatible = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_lookup_func_facts(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts) {
    return iree_ok_status();
  }
  const bool is_source = func_facts->func_op->kind == LOOM_OP_FUNC_DEF ||
                         func_facts->func_op->kind == LOOM_OP_KERNEL_DEF ||
                         func_facts->func_op->kind == LOOM_OP_FUNC_DECL;
  const bool is_low = loom_low_func_def_isa(func_facts->func_op) ||
                      loom_low_func_decl_isa(func_facts->func_op);
  const loom_low_source_selection_filter_t operation_filter =
      is_source ? LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP
                : (is_low ? LOOM_LOW_SOURCE_SELECTION_FILTER_LOW_OP : 0);
  const loom_low_source_selection_filter_t dialect_filter =
      filter & (LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP |
                LOOM_LOW_SOURCE_SELECTION_FILTER_LOW_OP);
  if (dialect_filter != 0 &&
      !iree_any_bit_set(dialect_filter, operation_filter)) {
    return iree_ok_status();
  }
  if (func_facts->has_body
          ? !iree_any_bit_set(filter, LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION)
          : !iree_any_bit_set(filter,
                              LOOM_LOW_SOURCE_SELECTION_FILTER_DECLARATION)) {
    return iree_ok_status();
  }
  const loom_low_source_selection_kind_t kind =
      is_low ? LOOM_LOW_SOURCE_SELECTION_REPRESENTATION
             : (func_facts->has_body ? LOOM_LOW_SOURCE_SELECTION_FUNCTION
                                     : LOOM_LOW_SOURCE_SELECTION_DECLARATION);
  const loom_func_like_t function =
      loom_func_like_cast(module, func_facts->func_op);
  loom_function_version_t* version_handle =
      loom_target_function_version_snapshot_handle_at(target_versions,
                                                      symbol_id);
  const loom_target_function_version_t* target_version =
      loom_target_function_version_const_cast(version_handle);
  const loom_symbol_ref_t target_ref = func_facts->target_symbol;
  const loom_target_binding_source_t target_source =
      target_version != NULL ? LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION
                             : LOOM_TARGET_BINDING_SOURCE_AUTHORED;
  const loom_target_facts_t* target_facts = NULL;
  if (target_version != NULL) {
    target_facts = target_version->function_target_facts;
  } else {
    if (!loom_symbol_ref_is_valid(target_ref)) {
      return iree_ok_status();
    }
    bool contract_valid = false;
    IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_facts(
        module, fact_table, func_facts, options->diagnostic_emitter, arena,
        &contract_valid, &target_facts));
    if (!contract_valid) {
      return iree_ok_status();
    }
    if (is_source &&
        iree_any_bit_set(filter, LOOM_LOW_SOURCE_SELECTION_FILTER_EXECUTION)) {
      IREE_RETURN_IF_ERROR(loom_target_facts_builder_select_execution(
          target_facts, arena, &target_facts));
    }
  }
  const loom_target_bundle_t* target_bundle =
      loom_target_facts_bundle(target_facts);
  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(options->policy_registry,
                                                       target_bundle);
  if (policy == NULL) {
    return iree_ok_status();
  }
  if (func_facts->imports && policy->import_decl_kind == 0) {
    return iree_ok_status();
  }

  out_selection->kind = kind;
  out_selection->func = function;
  out_selection->function_name = func_facts->name;
  out_selection->version_handle = version_handle;
  out_selection->target_source = target_source;
  out_selection->target_ref = target_ref;
  out_selection->target_facts = target_facts;
  out_selection->policy = policy;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_find_candidate_targets(
      module, fact_table, options, arena, out_selection));
  *out_compatible = true;
  return iree_ok_status();
}

static iree_status_t loom_low_select_symbols_with_filter(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_low_source_selection_filter_t filter, iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  *out_selection_list = (loom_low_source_selection_list_t){0};
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  loom_target_function_version_snapshot_t target_versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, options->function_versions, arena, &target_versions));
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
        module, options, &fact_table, &target_versions, filter,
        (loom_symbol_id_t)i, arena, &compatible, &candidate));
    if (!compatible) {
      continue;
    }
    selections[selection_count] = candidate;
    ++selection_count;
  }

  out_selection_list->values = selections;
  out_selection_list->count = selection_count;
  return iree_ok_status();
}

iree_status_t loom_low_select_lowering_symbols(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_symbols_with_filter(
      module, options,
      LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION |
          LOOM_LOW_SOURCE_SELECTION_FILTER_DECLARATION |
          LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP |
          LOOM_LOW_SOURCE_SELECTION_FILTER_LOW_OP |
          LOOM_LOW_SOURCE_SELECTION_FILTER_EXECUTION,
      arena, out_selection_list);
}

iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_symbols_with_filter(
      module, options,
      LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION |
          LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP,
      arena, out_selection_list);
}

iree_status_t loom_low_select_target_bound_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_symbols_with_filter(
      module, options, LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION, arena,
      out_selection_list);
}

static bool loom_low_source_selection_has_finalizer_policy(
    const loom_low_source_selection_list_t* selection_list,
    const loom_low_lower_policy_t* policy, iree_host_size_t limit) {
  for (iree_host_size_t i = 0; i < limit; ++i) {
    if (selection_list->values[i].policy == policy) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_source_selection_finalize_policies(
    loom_module_t* module, loom_low_source_selection_list_t* selection_list,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  // All source plans have consumed their selection bindings. Compact distinct
  // finalizer policies into that storage without overtaking the read cursor.
  iree_host_size_t policy_count = 0;
  for (iree_host_size_t i = 0; i < selection_list->count; ++i) {
    if (selection_list->values[i].kind ==
        LOOM_LOW_SOURCE_SELECTION_REPRESENTATION) {
      continue;
    }
    const loom_low_lower_policy_t* policy = selection_list->values[i].policy;
    if (policy->finalize_module.fn == NULL) {
      continue;
    }
    if (loom_low_source_selection_has_finalizer_policy(selection_list, policy,
                                                       policy_count)) {
      continue;
    }
    selection_list->values[policy_count++].policy = policy;
  }
  for (iree_host_size_t i = 0; i < policy_count; ++i) {
    const loom_low_lower_policy_t* policy = selection_list->values[i].policy;
    IREE_RETURN_IF_ERROR(
        policy->finalize_module.fn(policy->finalize_module.user_data, module,
                                   module_state, scratch_arena));
  }
  return loom_low_lower_module_state_finalize(module_state, module);
}
