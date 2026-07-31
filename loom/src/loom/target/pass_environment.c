// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pass_environment.h"

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
    const loom_target_environment_t* target_environment,
    const loom_target_specialization_context_t* specialization_context,
    const loom_function_version_list_t* function_versions) {
  return (loom_target_pass_capability_t){
      .base =
          {
              .type = &loom_target_pass_capability_type,
          },
      .target_environment = target_environment,
      .specialization_context = specialization_context,
      .function_versions = function_versions,
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

const loom_target_environment_t* loom_target_pass_capability_target_environment(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->target_environment : NULL;
}

const loom_target_specialization_context_t*
loom_target_pass_capability_specialization_context(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->specialization_context : NULL;
}

const loom_function_version_list_t*
loom_target_pass_capability_function_versions(
    const loom_target_pass_capability_t* capability) {
  return capability ? capability->function_versions : NULL;
}

const loom_target_profile_t* loom_target_pass_capability_specialization_profile(
    const loom_target_pass_capability_t* capability,
    const loom_module_t* module, loom_func_like_t function) {
  return capability ? loom_target_specialization_context_lookup(
                          loom_target_pass_capability_specialization_context(
                              capability),
                          module, function)
                    : NULL;
}

static bool loom_target_function_symbol_id(const loom_module_t* module,
                                           loom_func_like_t function,
                                           loom_symbol_id_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_func_like_isa(function)) {
    return false;
  }
  const loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return false;
  }
  *out_symbol_id = symbol_ref.symbol_id;
  return true;
}

iree_status_t loom_target_pass_capability_resolve_function_bundle(
    const loom_pass_environment_t* environment, const loom_module_t* module,
    loom_func_like_t function, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* arena, bool* out_resolved,
    loom_target_bundle_storage_t* out_bundle_storage) {
  (void)environment;
  *out_resolved = false;
  *out_bundle_storage = (loom_target_bundle_storage_t){0};

  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_target_function_symbol_id(module, function, &symbol_id)) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(&fact_table, module,
                                                     symbol_id, &base_facts));
  const loom_func_symbol_facts_t* func_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (func_facts == NULL ||
      !loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    return iree_ok_status();
  }

  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &fact_table, module, func_facts->target_symbol, &target_base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(target_base_facts);
  if (target_facts == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "verified function target has no indexed target facts");
    IREE_BUILTIN_UNREACHABLE();
  }

  bool contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_from_bundle(
      module, func_facts, target_facts->name,
      &target_facts->projection->storage.bundle, diagnostic_emitter,
      &contract_valid, out_bundle_storage));
  if (!contract_valid) {
    return iree_ok_status();
  }

  *out_resolved = true;
  return iree_ok_status();
}
