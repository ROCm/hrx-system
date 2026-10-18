// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/interop.h"

#include "context.h"
#include "loom/pass/builtin_registry.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "target.h"

loomc_status_t loomc_target_environment_create_from_provider_set(
    const loom_target_provider_set_t* provider_set, loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set_internal(
      provider_set, allocator, out_target_environment);
}

loomc_status_t loomc_target_environment_get_native_pass_registry(
    const loomc_target_environment_t* target_environment,
    loom_pass_registry_storage_t* out_storage) {
  if (out_storage == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_storage must not be NULL");
  }
  const loom_target_environment_t* environment =
      loomc_target_environment_loom_target_environment(target_environment);
  return loomc_status_from_iree(
      loom_pass_registry_storage_initialize_with_builtins(
          environment ? loom_target_environment_pass_registry(environment)
                      : NULL,
          out_storage));
}

loomc_status_t loomc_module_get_interop_view(
    loomc_module_t* module, loomc_allocator_t allocator,
    loomc_module_interop_view_t* out_view, loomc_result_t** out_result) {
  if (out_view != NULL) {
    *out_view = (loomc_module_interop_view_t){0};
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (module == NULL || out_view == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module, out_view, and out_result must not be NULL");
  }
  if (loomc_module_const_loom_module(module) == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK,
      loomc_context_source_retention(loomc_module_context(module)), allocator,
      &result));
  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(loomc_module_context(module));
  loomc_status_t status =
      target_environment != NULL
          ? loomc_module_admit(module, target_environment, result)
          : loomc_module_admit_structural(module, result);
  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_view = (loomc_module_interop_view_t){
          .module = loomc_module_const_loom_module(module),
          .source_table = loomc_module_source_table(module),
      };
    }
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

loomc_module_mutable_interop_view_t loomc_module_get_mutable_interop_view(
    loomc_module_t* module) {
  if (module == NULL || loomc_module_loom_module(module) == NULL) {
    return (loomc_module_mutable_interop_view_t){0};
  }
  loomc_module_invalidate_verification(module);
  loomc_module_invalidate_compilation(module);
  return (loomc_module_mutable_interop_view_t){
      .module = loomc_module_loom_module(module),
      .source_table = loomc_module_source_table(module),
  };
}

const loom_target_environment_t* loomc_target_environment_get_interop_view(
    const loomc_target_environment_t* target_environment) {
  return loomc_target_environment_loom_target_environment(target_environment);
}

const loom_target_profile_t* loomc_target_profile_get_interop_view(
    const loomc_target_profile_t* target_profile) {
  return loomc_target_profile_loom_target_profile(target_profile);
}
