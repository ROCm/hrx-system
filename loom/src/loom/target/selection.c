// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/selection.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"

static bool loom_target_pass_capability_satisfies_requirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  (void)capability;
  (void)requirement;
  return false;
}

const loom_pass_environment_capability_type_t loom_target_pass_capability_type =
    {
        .name = IREE_SVL("target"),
        .satisfies_requirement =
            loom_target_pass_capability_satisfies_requirement,
};

loom_target_pass_capability_t loom_target_pass_capability_make(
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref) {
  return (loom_target_pass_capability_t){
      .base =
          {
              .type = &loom_target_pass_capability_type,
          },
      .target_selection = target_selection,
      .target_ref = target_ref,
  };
}

const loom_target_pass_capability_t*
loom_target_pass_capability_from_environment(
    const loom_pass_environment_t* environment) {
  if (environment == NULL) {
    return NULL;
  }
  return (const loom_target_pass_capability_t*)loom_pass_environment_lookup(
      environment, &loom_target_pass_capability_type);
}

const loom_target_pass_capability_t* loom_target_pass_capability_from_pass(
    const loom_pass_t* pass) {
  return pass && pass->environment
             ? loom_target_pass_capability_from_environment(pass->environment)
             : NULL;
}

loom_target_selection_t loom_target_pass_capability_target_selection(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->target_selection
                    : loom_target_selection_empty();
}

loom_symbol_ref_t loom_target_pass_capability_target_ref(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->target_ref : loom_symbol_ref_null();
}

loom_symbol_ref_t loom_target_effective_target_ref(
    loom_symbol_ref_t authored_target_ref,
    const loom_target_pass_capability_t* capability) {
  if (loom_symbol_ref_is_valid(authored_target_ref)) {
    return authored_target_ref;
  }
  return loom_target_pass_capability_target_ref(capability);
}

bool loom_target_pass_capability_can_refine_target_bundle(
    const loom_target_pass_capability_t* capability,
    loom_symbol_ref_t effective_target_ref,
    const loom_target_bundle_t* authored_target_bundle) {
  if (!capability || !authored_target_bundle) return false;
  const loom_symbol_ref_t invocation_target_ref =
      loom_target_pass_capability_target_ref(capability);
  if (!loom_symbol_ref_is_valid(effective_target_ref) ||
      !loom_symbol_ref_is_valid(invocation_target_ref) ||
      effective_target_ref.module_id != invocation_target_ref.module_id ||
      effective_target_ref.symbol_id != invocation_target_ref.symbol_id) {
    return false;
  }
  const loom_target_selection_t target_selection =
      loom_target_pass_capability_target_selection(capability);
  return target_selection.bundle != NULL &&
         loom_target_function_contract_bundles_compatible(
             authored_target_bundle, target_selection.bundle);
}

static bool loom_target_function_symbol_id(const loom_module_t* module,
                                           loom_func_like_t function,
                                           loom_symbol_id_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_func_like_isa(function)) {
    return false;
  }
  loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return false;
  }
  *out_symbol_id = symbol_ref.symbol_id;
  return true;
}

static iree_status_t loom_target_function_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_target_record_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_ref_t target_ref,
    const loom_target_symbol_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_ref, &base_facts));
  *out_target_facts = loom_target_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

iree_status_t loom_target_pass_capability_resolve_function_bundle(
    const loom_pass_environment_t* environment, const loom_module_t* module,
    loom_func_like_t function, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* arena, bool* out_resolved,
    loom_target_bundle_storage_t* out_bundle_storage) {
  *out_resolved = false;
  *out_bundle_storage = (loom_target_bundle_storage_t){0};

  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_target_function_symbol_id(module, function, &symbol_id)) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_function_facts(module, &fact_table, symbol_id, &func_facts));
  if (!func_facts) {
    return iree_ok_status();
  }

  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_environment(environment);
  const loom_symbol_ref_t target_ref =
      loom_target_effective_target_ref(func_facts->target_symbol, capability);
  const loom_target_symbol_facts_t* target_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_record_facts(module, &fact_table, target_ref, &target_facts));
  if (!target_facts) {
    return iree_ok_status();
  }

  bool contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_from_bundle(
      module, func_facts, target_facts->name, &target_facts->storage.bundle,
      diagnostic_emitter, &contract_valid, out_bundle_storage));
  if (!contract_valid) {
    return iree_ok_status();
  }

  const loom_target_selection_t target_selection =
      loom_target_pass_capability_target_selection(capability);
  if (target_selection.bundle != NULL) {
    if (!loom_target_function_contract_bundles_compatible(
            &out_bundle_storage->bundle, target_selection.bundle)) {
      *out_bundle_storage = (loom_target_bundle_storage_t){0};
      return iree_ok_status();
    }
    loom_target_function_contract_apply_compatible_selection(
        target_selection.bundle, out_bundle_storage);
  }

  *out_resolved = true;
  return iree_ok_status();
}

iree_status_t loom_target_pass_compact_symbols_preserving_target_ref(
    const loom_pass_t* pass, loom_module_t* module,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_removed_count) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_symbol_ref_t target_ref =
      loom_target_pass_capability_target_ref(capability);
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return loom_module_compact_symbols(module, scratch_arena,
                                       out_removed_count);
  }
  return loom_module_compact_symbols_preserving_symbol_refs(
      module, &target_ref, 1, scratch_arena, out_removed_count);
}
