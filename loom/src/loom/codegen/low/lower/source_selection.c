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
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"

iree_status_t loom_low_prepare_source_module(
    loom_module_t* module, const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_lower_prepare_module_result_t* out_result) {
  *out_result = (loom_low_lower_prepare_module_result_t){
      .valid = true,
  };
  if (options->policy_registry == NULL ||
      options->policy_registry->entry_count == 0 ||
      module->symbols.count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t prepare_policy_capacity = 0;
  for (iree_host_size_t i = 0; i < options->policy_registry->entry_count; ++i) {
    const loom_low_lower_policy_t* policy =
        options->policy_registry->entries[i].policy;
    if (policy != NULL && policy->prepare_module.fn != NULL) {
      ++prepare_policy_capacity;
    }
  }
  if (prepare_policy_capacity == 0) return iree_ok_status();

  const loom_low_lower_policy_t** prepare_policies = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, prepare_policy_capacity,
                                                 sizeof(*prepare_policies),
                                                 (void**)&prepare_policies));
  iree_host_size_t prepare_policy_count = 0;
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_TARGET)) {
      continue;
    }
    const loom_symbol_facts_base_t* base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
        &fact_table, module, (loom_symbol_id_t)i, &base_facts));
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(base_facts);
    if (target_facts == NULL) continue;
    const loom_low_lower_policy_t* policy =
        loom_low_lower_policy_registry_lookup_for_bundle(
            options->policy_registry,
            loom_target_facts_bundle(target_facts->projection));
    if (policy == NULL || policy->prepare_module.fn == NULL) continue;

    bool seen = false;
    for (iree_host_size_t j = 0; j < prepare_policy_count; ++j) {
      if (prepare_policies[j] == policy) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      IREE_ASSERT_LT(prepare_policy_count, prepare_policy_capacity);
      prepare_policies[prepare_policy_count++] = policy;
    }
  }

  for (iree_host_size_t i = 0; i < prepare_policy_count; ++i) {
    loom_low_lower_prepare_module_result_t policy_result = {0};
    const loom_low_lower_policy_t* policy = prepare_policies[i];
    IREE_RETURN_IF_ERROR(policy->prepare_module.fn(
        policy->prepare_module.user_data, module, options->diagnostic_emitter,
        arena, &policy_result));
    out_result->changed |= policy_result.changed;
    if (!policy_result.valid) {
      out_result->valid = false;
      break;
    }
  }
  return iree_ok_status();
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
  return module->strings.entries[symbol->name_id];
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

static void loom_low_source_selection_set_candidate_target(
    const loom_module_t* module, const loom_target_symbol_facts_t* target_facts,
    loom_low_source_selection_t* selection) {
  const loom_target_bundle_t* bundle =
      loom_target_facts_bundle(target_facts->projection);
  selection->candidate_target_symbol_name =
      loom_low_source_selection_symbol_ref_name(module, target_facts->symbol);
  selection->candidate_target_bundle_name = bundle->name;
  selection->candidate_target_snapshot_name = bundle->snapshot->name;
  selection->candidate_target_config_name = bundle->config->name;
  selection->candidate_target_subgroup_size = bundle->snapshot->subgroup_size;
}

static iree_status_t loom_low_source_selection_find_candidate_targets(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_low_source_selection_options_t* options,
    loom_low_source_selection_t* selection) {
  if (!options->collect_target_candidates ||
      selection->target_source != LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION) {
    return iree_ok_status();
  }
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
    if (selection->candidate_target_count == 0) {
      loom_low_source_selection_set_candidate_target(module, target_facts,
                                                     selection);
    }
    if (selection->candidate_target_count != UINT32_MAX) {
      ++selection->candidate_target_count;
    }
  }
  return iree_ok_status();
}

typedef uint8_t loom_low_source_selection_filter_t;

#define LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION ((uint8_t)1u << 0)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_IMPORT_DECL ((uint8_t)1u << 1)
#define LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP ((uint8_t)1u << 2)

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
  if (iree_all_bits_set(filter, LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP) &&
      func_facts->func_op->kind != LOOM_OP_FUNC_DEF &&
      func_facts->func_op->kind != LOOM_OP_KERNEL_DEF &&
      func_facts->func_op->kind != LOOM_OP_FUNC_DECL) {
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
  }
  const loom_target_bundle_t* target_bundle =
      loom_target_facts_bundle(target_facts);
  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(options->policy_registry,
                                                       target_bundle);
  if (policy == NULL) {
    return iree_ok_status();
  }
  if (kind == LOOM_LOW_SOURCE_SELECTION_IMPORT_DECL &&
      policy->import_decl_kind == 0 &&
      !iree_any_bit_set(policy->flags,
                        LOOM_LOW_LOWER_POLICY_FLAG_MODULE_IMPORTS)) {
    return iree_ok_status();
  }

  out_selection->kind = kind;
  out_selection->func = function;
  out_selection->function_name = func_facts->name;
  out_selection->version_handle = version_handle;
  out_selection->target_source = target_source;
  out_selection->target_ref = target_ref;
  out_selection->target_facts = target_facts;
  out_selection->target_symbol_name =
      loom_low_source_selection_symbol_ref_name(module, target_ref);
  out_selection->policy = policy;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_find_candidate_targets(
      module, fact_table, options, out_selection));
  *out_compatible = true;
  return iree_ok_status();
}

static void loom_low_source_selection_assign(
    const loom_low_source_selection_t* source,
    loom_low_source_selection_t* out_selection) {
  *out_selection = *source;
}

static iree_status_t loom_low_select_source_symbols_with_filter(
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
          LOOM_LOW_SOURCE_SELECTION_FILTER_IMPORT_DECL |
          LOOM_LOW_SOURCE_SELECTION_FILTER_SOURCE_OP,
      arena, out_selection_list);
}

iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  return loom_low_select_source_symbols_with_filter(
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
  return loom_low_select_source_symbols_with_filter(
      module, options, LOOM_LOW_SOURCE_SELECTION_FILTER_FUNCTION, arena,
      out_selection_list);
}

static bool loom_low_source_selection_policy_seen_before(
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
    loom_module_t* module,
    const loom_low_source_selection_list_t* selection_list,
    loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena) {
  for (iree_host_size_t i = 0; i < selection_list->count; ++i) {
    const loom_low_lower_policy_t* policy = selection_list->values[i].policy;
    if (policy == NULL || policy->finalize_module.fn == NULL) {
      continue;
    }
    if (loom_low_source_selection_policy_seen_before(selection_list, policy,
                                                     i)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        policy->finalize_module.fn(policy->finalize_module.user_data, module,
                                   module_state, scratch_arena));
  }
  return iree_ok_status();
}
