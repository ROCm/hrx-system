// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include "diagnostic.h"
#include "loomc/iree.h"
#include "result.h"

static loomc_status_t loomc_iree_hal_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_validate_provider(
    const loomc_iree_hal_profile_provider_t* provider) {
  if (provider == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL profile provider must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_iree_hal_validate_string_view(provider->name));
  if (provider->create_profile == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL profile provider must define create_profile");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_iree_hal_validate_options(
    const loomc_iree_hal_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL profile options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL profile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL profile options structure_size is too small");
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
        "IREE HAL profile provider_count is non-zero but providers is NULL");
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
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result);
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
                        "no IREE HAL target profile provider supports device"),
      out_result);
}

loomc_status_t loomc_target_profile_create_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result) {
  if (out_profile == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile and out_result must not be NULL");
  }
  *out_profile = NULL;
  *out_result = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_iree_hal_validate_options(options));

  loomc_status_t status = loomc_ok_status();
  for (loomc_host_size_t i = 0;
       i < options->provider_count && loomc_status_is_ok(status); ++i) {
    const loomc_iree_hal_profile_provider_t* provider = options->providers[i];
    bool supported = false;
    loomc_target_profile_t* profile = NULL;
    loomc_result_t* result = NULL;
    status = provider->create_profile(provider->user_data, target_environment,
                                      options, allocator, &supported, &profile,
                                      &result);
    if (!loomc_status_is_ok(status)) {
      loomc_target_profile_release(profile);
      loomc_result_release(result);
      break;
    }
    if (!supported) {
      if (profile != NULL || result != NULL) {
        loomc_target_profile_release(profile);
        loomc_result_release(result);
        status = loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "unsupported IREE HAL profile provider returned outputs");
      }
      continue;
    }
    if (result == NULL) {
      loomc_target_profile_release(profile);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL profile provider did not return a result");
      break;
    }
    if (loomc_result_succeeded(result) && profile == NULL) {
      loomc_result_release(result);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL profile provider succeeded without a profile");
      break;
    }
    if (!loomc_result_succeeded(result) && profile != NULL) {
      loomc_target_profile_release(profile);
      loomc_result_release(result);
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "supported IREE HAL profile provider returned a failed result with "
          "a profile");
      break;
    }
    *out_profile = profile;
    *out_result = result;
    return loomc_ok_status();
  }
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  return loomc_iree_hal_create_unsupported_result(allocator, out_result);
}
