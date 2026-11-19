// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_specialization.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/target/facts.h"
#include "loom/target/function_version_projection.h"
#include "loom/target/profile.h"

iree_status_t loom_target_specialize_module(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_function_version_owner_t versions;
  loom_function_version_owner_initialize(&arena, &versions);
  iree_status_t status = loom_target_specialize_functions(
      environment, *inout_module, requests, bindings, diagnostic_emitter,
      &versions, out_error_count);
  if (iree_status_is_ok(status) && *out_error_count == 0 &&
      versions.list.count != 0) {
    loom_module_t* projected_module = NULL;
    status = loom_target_function_versions_project_module(
        *inout_module, &versions.list, block_pool, allocator, NULL,
        &projected_module);
    if (iree_status_is_ok(status)) {
      loom_module_free(*inout_module);
      *inout_module = projected_module;
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_target_specialize_module_in_place(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    uint32_t* out_error_count) {
  loom_function_version_owner_t versions;
  loom_function_version_owner_initialize(arena, &versions);
  IREE_RETURN_IF_ERROR(loom_target_specialize_functions(
      environment, module, requests, bindings, diagnostic_emitter, &versions,
      out_error_count));
  if (*out_error_count != 0 || versions.list.count == 0) {
    return iree_ok_status();
  }
  return loom_target_function_versions_bind_module(module, &versions.list,
                                                   arena);
}

static bool loom_target_deployment_function_uses_profile(
    const loom_module_t* module, loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return false;
  }
  if (loom_func_like_is_kernel_entry(function)) {
    return true;
  }
  const loom_symbol_ref_t authored_target = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(authored_target)) {
    return false;
  }
  IREE_ASSERT_EQ(authored_target.module_id, 0u);
  IREE_ASSERT_LT(authored_target.symbol_id, module->symbols.count);
  return loom_target_decl_isa(
      module->symbols.entries[authored_target.symbol_id].defining_op);
}

// Returns whether a function's authored target can be specialized to
// |profile_facts|. Targetless functions and target declarations impose no
// concrete requirement at this point.
static iree_status_t loom_target_deployment_function_accepts_profile(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* profile_facts,
    loom_symbol_fact_table_t* symbol_facts, bool* out_accepts_profile) {
  *out_accepts_profile = true;
  const loom_symbol_ref_t authored_target = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(authored_target)) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(authored_target.module_id, 0u);
  IREE_ASSERT_LT(authored_target.symbol_id, module->symbols.count);
  if (loom_target_decl_isa(
          module->symbols.entries[authored_target.symbol_id].defining_op)) {
    return iree_ok_status();
  }

  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, authored_target, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "verified concrete function target has no target facts");
    IREE_BUILTIN_UNREACHABLE();
  }
  *out_accepts_profile = loom_target_facts_satisfy_specialization_requirement(
      profile_facts, target_facts->projection);
  return iree_ok_status();
}

typedef enum loom_target_deployment_specialization_policy_e {
  LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_STRICT = 0,
  LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_COMMAND_PROJECTION = 1,
} loom_target_deployment_specialization_policy_t;

static iree_status_t loom_target_deployment_function_should_specialize(
    const loom_module_t* module, loom_func_like_t function,
    loom_target_deployment_specialization_policy_t policy,
    const loom_target_facts_t* profile_facts,
    loom_symbol_fact_table_t* symbol_facts, bool* out_should_specialize) {
  *out_should_specialize = false;
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  if (!loom_func_like_is_kernel_entry(function)) {
    *out_should_specialize =
        loom_target_deployment_function_uses_profile(module, function);
    return iree_ok_status();
  }
  if (policy == LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_STRICT) {
    *out_should_specialize = true;
    return iree_ok_status();
  }
  return loom_target_deployment_function_accepts_profile(
      module, function, profile_facts, symbol_facts, out_should_specialize);
}

static iree_status_t
loom_target_specialize_module_deployment_entries_with_policy(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    loom_target_deployment_specialization_policy_t policy,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(target_profile);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_error_count = 0;

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_target_facts_t* profile_facts = NULL;
  loom_symbol_fact_table_t symbol_facts = {0};
  iree_status_t status = iree_ok_status();
  if (policy ==
      LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_COMMAND_PROJECTION) {
    status = loom_target_profile_project_facts(target_profile, &arena,
                                               &profile_facts);
    loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  }

  iree_host_size_t request_count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!iree_status_is_ok(status)) {
      break;
    }
    bool should_specialize = false;
    status = loom_target_deployment_function_should_specialize(
        module, loom_func_like_cast(module, op), policy, profile_facts,
        &symbol_facts, &should_specialize);
    request_count += should_specialize ? 1u : 0u;
  }

  loom_target_specialization_request_t* requests = NULL;
  if (iree_status_is_ok(status) && request_count != 0) {
    status = iree_arena_allocate_array(&arena, request_count, sizeof(*requests),
                                       (void**)&requests);
  }
  if (iree_status_is_ok(status) && request_count != 0) {
    iree_host_size_t request_ordinal = 0;
    loom_block_for_each_op(loom_module_block(module), op) {
      bool should_specialize = false;
      status = loom_target_deployment_function_should_specialize(
          module, loom_func_like_cast(module, op), policy, profile_facts,
          &symbol_facts, &should_specialize);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (!should_specialize) {
        continue;
      }
      requests[request_ordinal++] = (loom_target_specialization_request_t){
          .function_ref =
              loom_func_like_callee(loom_func_like_cast(module, op)),
          .target_profile = target_profile,
      };
    }
    IREE_ASSERT(!iree_status_is_ok(status) || request_ordinal == request_count);
  }
  if (iree_status_is_ok(status) && request_count != 0) {
    status = loom_target_specialize_module_in_place(
        environment, module,
        (loom_target_specialization_request_list_t){
            .values = requests,
            .count = request_count,
        },
        (loom_target_declaration_binding_list_t){0}, diagnostic_emitter, &arena,
        out_error_count);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_target_specialize_module_deployment_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    uint32_t* out_error_count) {
  return loom_target_specialize_module_deployment_entries_with_policy(
      environment, target_profile,
      LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_STRICT, diagnostic_emitter,
      block_pool, module, out_error_count);
}

iree_status_t loom_target_specialize_module_command_projection_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    uint32_t* out_error_count) {
  return loom_target_specialize_module_deployment_entries_with_policy(
      environment, target_profile,
      LOOM_TARGET_DEPLOYMENT_SPECIALIZATION_POLICY_COMMAND_PROJECTION,
      diagnostic_emitter, block_pool, module, out_error_count);
}

iree_status_t loom_target_validate_module_deployment_kernel_entry_refs(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_symbol_ref_t* kernel_refs, iree_host_size_t kernel_count,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(kernel_refs || kernel_count == 0);
  IREE_ASSERT_ARGUMENT(target_profile);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_error_count = 0;
  if (kernel_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_target_facts_t* profile_facts = NULL;
  iree_status_t status =
      loom_target_profile_project_facts(target_profile, &arena, &profile_facts);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, &arena);
  loom_target_specialization_request_t* incompatible_requests = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&arena, kernel_count,
                                       sizeof(*incompatible_requests),
                                       (void**)&incompatible_requests);
  }

  iree_host_size_t incompatible_count = 0;
  for (iree_host_size_t i = 0; i < kernel_count && iree_status_is_ok(status);
       ++i) {
    const loom_symbol_ref_t kernel_ref = kernel_refs[i];
    IREE_ASSERT(loom_symbol_ref_is_valid(kernel_ref));
    IREE_ASSERT_EQ(kernel_ref.module_id, 0u);
    IREE_ASSERT_LT(kernel_ref.symbol_id, module->symbols.count);
    const loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[kernel_ref.symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_is_kernel_entry(function));
    bool accepts_profile = false;
    status = loom_target_deployment_function_accepts_profile(
        module, function, profile_facts, &symbol_facts, &accepts_profile);
    if (iree_status_is_ok(status) && !accepts_profile) {
      incompatible_requests[incompatible_count++] =
          (loom_target_specialization_request_t){
              .function_ref = kernel_ref,
              .target_profile = target_profile,
          };
    }
  }

  if (iree_status_is_ok(status) && incompatible_count != 0) {
    loom_function_version_owner_t unused_versions;
    loom_function_version_owner_initialize(&arena, &unused_versions);
    status = loom_target_specialize_functions(
        environment, module,
        (loom_target_specialization_request_list_t){
            .values = incompatible_requests,
            .count = incompatible_count,
        },
        (loom_target_declaration_binding_list_t){0}, diagnostic_emitter,
        &unused_versions, out_error_count);
    IREE_ASSERT(!iree_status_is_ok(status) ||
                *out_error_count == incompatible_count);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
