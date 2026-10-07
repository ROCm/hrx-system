// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/pass_environment.h"

static iree_status_t loom_codegen_resolve_cleanup_canonicalizer_context(
    void* user_data, const loom_pass_t* pass, const loom_module_t* module,
    loom_func_like_t function,
    loom_cleanup_canonicalizer_context_t* out_context) {
  (void)user_data;
  bool target_resolved = false;
  IREE_RETURN_IF_ERROR(loom_target_pass_resolve_function_facts(
      pass, module, function, &target_resolved, &out_context->target_facts));
  if (!target_resolved) {
    out_context->target_facts = NULL;
  }
  const loom_target_math_pass_capability_t* math_capability =
      loom_target_math_pass_capability_from_pass(pass);
  out_context->math_policy = loom_target_math_policy_registry_lookup_for_bundle(
      loom_target_math_pass_capability_policy_registry(math_capability),
      loom_target_facts_bundle(out_context->target_facts));
  return iree_ok_status();
}

static loom_pass_environment_t
loom_codegen_pass_environment_storage_initialize_with_target(
    const loom_codegen_pass_environment_options_t* options,
    loom_target_pass_capability_t target_capability,
    loom_codegen_pass_environment_storage_t* out_storage) {
  out_storage->target_capability = target_capability;
  out_storage->low_capability = loom_low_pass_capability_make(
      options->descriptor_registry, options->lower_policy_registry,
      options->legality_provider_list, options->legalizer_registry,
      options->compile_report);
  out_storage->math_capability = loom_target_math_pass_capability_make(
      options->math_policy_registry, options->compile_report);
  out_storage->cleanup_capability = loom_cleanup_pass_capability_make(
      options->cleanup_pattern_registry,
      (loom_cleanup_canonicalizer_context_resolver_t){
          .fn = loom_codegen_resolve_cleanup_canonicalizer_context,
      });
  iree_host_size_t capability_count = 0;
  out_storage->capabilities[capability_count++] =
      &out_storage->target_capability.base;
  out_storage->capabilities[capability_count++] =
      &out_storage->low_capability.base;
  out_storage->capabilities[capability_count++] =
      &out_storage->math_capability.base;
  out_storage->capabilities[capability_count++] =
      &out_storage->cleanup_capability.base;
  if (options->launch_config_capability != NULL) {
    out_storage->capabilities[capability_count++] =
        options->launch_config_capability;
  }
  out_storage->environment =
      loom_pass_environment_make(out_storage->capabilities, capability_count);
  return out_storage->environment;
}

loom_pass_environment_t loom_codegen_pass_environment_storage_initialize(
    const loom_codegen_pass_environment_options_t* options,
    const loom_function_version_list_t* function_versions,
    loom_codegen_pass_environment_storage_t* out_storage) {
  return loom_codegen_pass_environment_storage_initialize_with_target(
      options,
      loom_target_pass_capability_make(options->target_environment,
                                       function_versions),
      out_storage);
}

loom_pass_environment_t
loom_codegen_pass_environment_storage_initialize_mutable(
    const loom_codegen_pass_environment_options_t* options,
    loom_function_version_owner_t* function_version_owner,
    loom_codegen_pass_environment_storage_t* out_storage) {
  return loom_codegen_pass_environment_storage_initialize_with_target(
      options,
      loom_target_pass_capability_make_mutable(options->target_environment,
                                               function_version_owner),
      out_storage);
}
