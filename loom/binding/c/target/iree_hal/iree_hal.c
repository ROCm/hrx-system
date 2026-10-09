// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include "diagnostic.h"
#include "loom/sanitizer/runtime_requirements.h"
#include "loomc/iree.h"
#include "module.h"
#include "option_chain.h"
#include "result.h"

loomc_status_t loomc_iree_hal_module_query_runtime_features(
    const loomc_module_t* module,
    const loomc_sanitizer_options_t* sanitizer_options,
    iree_hal_device_runtime_feature_flags_t* out_runtime_features) {
  if (out_runtime_features == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_runtime_features must not be NULL");
  }
  *out_runtime_features = IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }

  loom_sanitizer_options_t resolved_sanitizer_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_sanitizer_options_resolve(
      sanitizer_options, &resolved_sanitizer_options));
  loom_sanitizer_runtime_requirements_t requirements =
      LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_sanitizer_runtime_requirements_query(
          loomc_module_const_loom_module(module), &resolved_sanitizer_options,
          &requirements)));

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  if (iree_any_bit_set(requirements,
                       LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK)) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  }
  if (iree_any_bit_set(requirements,
                       LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW)) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN;
  }
  if (iree_any_bit_set(requirements,
                       LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW)) {
    runtime_features |= IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN;
  }
  *out_runtime_features = runtime_features;
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_validate_provider(
    const loomc_iree_hal_target_provider_t* provider) {
  if (provider == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL target provider must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_iree_hal_validate_string_view(provider->name));
  if (provider->select_target == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL target provider must define select_target");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_validate_options(
    const loomc_iree_hal_target_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL target options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL target options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL target options structure_size is too small");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_iree_hal_validate_string_view(options->identifier));
  if (options->device == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL profile options require a device");
  }
  if (options->provider_count != 0 && options->providers == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL target provider_count is non-zero but providers is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->provider_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_iree_hal_validate_provider(options->providers[i]));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_create_failed_result(
    loomc_allocator_t allocator, loomc_string_view_t code,
    loomc_status_t diagnostic_status, loomc_result_t** out_result) {
  *out_result = NULL;
  loomc_result_t* result = NULL;
  loomc_status_t status =
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                          LOOMC_SOURCE_RETENTION_EXACT, allocator, &result);
  if (loomc_status_is_ok(status)) {
    status = loomc_result_fail_status_diagnostic_consume(
        result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, diagnostic_status);
    diagnostic_status = loomc_ok_status();
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  } else {
    loomc_result_release(result);
  }
  loomc_status_free(diagnostic_status);
  return status;
}

static loomc_status_t loomc_iree_hal_create_unsupported_result(
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  return loomc_iree_hal_create_failed_result(
      allocator, loomc_make_cstring_view("IREE_HAL/TARGET"),
      loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                        "no IREE HAL target provider supports request"),
      out_result);
}

loomc_status_t loomc_target_select_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  if (out_selection == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_selection and out_result must not be NULL");
  }
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_iree_hal_validate_options(options));

  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0;
       i < options->provider_count && loomc_status_is_ok(status); ++i) {
    const loomc_iree_hal_target_provider_t* provider = options->providers[i];
    bool supported = false;
    loomc_iree_hal_target_selection_t selection = {0};
    loomc_result_t* result = NULL;
    status = provider->select_target(provider->user_data, target_environment,
                                     options, allocator, &supported, &selection,
                                     &result);
    if (!loomc_status_is_ok(status)) {
      loomc_target_profile_release(selection.target_profile);
      loomc_result_release(result);
      break;
    }
    if (!supported) {
      if (selection.target_profile != NULL ||
          selection.executable_target != NULL || result != NULL) {
        loomc_target_profile_release(selection.target_profile);
        loomc_result_release(result);
        status = loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "unsupported IREE HAL target provider returned outputs");
      }
      continue;
    }
    if (result == NULL) {
      loomc_target_profile_release(selection.target_profile);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL target provider did not return a result");
      break;
    }
    if (loomc_result_succeeded(result) &&
        (selection.target_profile == NULL ||
         selection.executable_target == NULL)) {
      loomc_target_profile_release(selection.target_profile);
      loomc_result_release(result);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL target provider succeeded without a complete "
          "selection");
      break;
    }
    if (!loomc_result_succeeded(result) &&
        (selection.target_profile != NULL ||
         selection.executable_target != NULL)) {
      loomc_target_profile_release(selection.target_profile);
      loomc_result_release(result);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL target provider returned a failed result with "
          "a selection");
      break;
    }
    *out_selection = selection;
    *out_result = result;
    return loomc_ok_status();
  }
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  return loomc_iree_hal_create_unsupported_result(allocator, out_result);
}
