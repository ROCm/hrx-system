// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amd/xdna/iree_hal.h"

#include "diagnostic.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loomc/iree.h"
#include "result.h"
#include "target.h"

static loomc_status_t loomc_xdna_iree_hal_fail_status(loomc_result_t* result,
                                                      loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("XDNA/IREE_HAL"), status);
}

static loomc_status_t loomc_xdna_iree_hal_fail_cstring(loomc_result_t* result,
                                                       loomc_status_code_t code,
                                                       const char* message) {
  return loomc_xdna_iree_hal_fail_status(result,
                                         loomc_make_status(code, message));
}

static const loom_aie2p_target_profile_t*
loomc_xdna_iree_hal_cast_array_profile(
    const loomc_target_profile_t* target_profile) {
  const loom_target_profile_t* native_profile =
      loomc_target_profile_loom_target_profile(target_profile);
  const loom_aie2p_target_profile_t* profile =
      loom_aie2p_target_profile_cast(native_profile);
  return profile != NULL && profile->kind == LOOM_AIE2P_TARGET_KIND_ARRAY &&
                 profile->device_profile != NULL
             ? profile
             : NULL;
}

static bool loomc_xdna_iree_hal_device_is_supported(
    const loomc_iree_hal_target_options_t* options) {
  if (options->target_profile != NULL) {
    return loomc_xdna_iree_hal_cast_array_profile(options->target_profile) !=
           NULL;
  }
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return false;
  }
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("xdna"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = options->physical_device_affinity,
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  return result.outcome !=
         IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH;
}

static loomc_status_t loomc_xdna_iree_hal_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_string_view_t target_key, loomc_result_t* result,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_executable_target = NULL;
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("xdna"),
      .target_key = target_key,
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = physical_device_affinity,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return loomc_xdna_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device has no compatible exact XDNA target");
  }
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return loomc_xdna_iree_hal_fail_cstring(
        result, LOOMC_STATUS_FAILED_PRECONDITION,
        "IREE HAL device advertises ambiguous exact XDNA targets; select a "
        "physical-device affinity");
  }
  *out_executable_target = target_result.target;
  return loomc_ok_status();
}

static loomc_status_t loomc_xdna_iree_hal_select_target(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (options->target_profile != NULL) {
    LOOMC_RETURN_IF_ERROR(loomc_target_profile_validate_environment(
        options->target_profile, target_environment));
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                            LOOMC_SOURCE_RETENTION_EXACT,
                                            allocator, &result));
  loomc_target_profile_t* target_profile = NULL;
  const iree_hal_executable_target_t* executable_target = NULL;
  loomc_status_t status = loomc_ok_status();

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    status = loomc_xdna_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts");
  }

  const loom_aie2p_target_profile_t* native_profile = NULL;
  iree_string_view_t requested_target_key = iree_string_view_empty();
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      options->target_profile != NULL) {
    native_profile =
        loomc_xdna_iree_hal_cast_array_profile(options->target_profile);
    if (native_profile == NULL) {
      status = loomc_xdna_iree_hal_fail_cstring(
          result, LOOMC_STATUS_INVALID_ARGUMENT,
          "XDNA IREE HAL target selection requires an AIE2P array profile");
    } else {
      requested_target_key =
          iree_make_cstring_view(native_profile->device_profile->key);
    }
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_xdna_iree_hal_select_executable_target(
        device_spec, options->physical_device_affinity, requested_target_key,
        result, &executable_target);
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      options->target_profile == NULL) {
    iree_status_t iree_status = loom_aie2p_target_profile_select(
        executable_target->target_key, &native_profile);
    if (!iree_status_is_ok(iree_status)) {
      status = loomc_xdna_iree_hal_fail_status(
          result, loomc_status_from_iree(iree_status));
    }
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    if (options->target_profile != NULL) {
      loomc_target_profile_retain(options->target_profile);
      target_profile = options->target_profile;
    } else {
      const loomc_string_view_t identifier =
          loomc_string_view_is_empty(options->identifier)
              ? loomc_string_view_from_iree(executable_target->target_key)
              : options->identifier;
      status = loomc_target_profile_create(
          target_environment, identifier,
          (loom_target_profile_t*)&native_profile->base, NULL, allocator,
          &target_profile);
    }
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_selection = (loomc_iree_hal_target_selection_t){
          .target_profile = target_profile,
          .executable_target = executable_target,
      };
      target_profile = NULL;
    }
    *out_result = result;
    result = NULL;
  }
  loomc_target_profile_release(target_profile);
  loomc_result_release(result);
  return status;
}

static loomc_status_t loomc_xdna_iree_hal_provider_select_target(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  (void)user_data;
  *out_supported = loomc_xdna_iree_hal_device_is_supported(options);
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (!*out_supported) {
    return loomc_ok_status();
  }
  return loomc_xdna_iree_hal_select_target(
      target_environment, options, allocator, out_selection, out_result);
}

const loomc_iree_hal_target_provider_t* loomc_xdna_iree_hal_target_provider(
    void) {
  static const loomc_iree_hal_target_provider_t provider = {
      .name = {"amd.xdna.iree_hal", 17},
      .user_data = NULL,
      .select_target = loomc_xdna_iree_hal_provider_select_target,
  };
  return &provider;
}
