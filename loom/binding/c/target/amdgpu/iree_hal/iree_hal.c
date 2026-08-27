// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu/iree_hal.h"

#include "diagnostic.h"
#include "iree/hal/executable/amdgpu/target_id.h"
#include "loomc/iree.h"
#include "result.h"

static loomc_status_t loomc_amdgpu_iree_hal_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_amdgpu_iree_hal_validate_options(
    const loomc_amdgpu_iree_hal_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU IREE HAL options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU IREE HAL options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU IREE HAL options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "AMDGPU IREE HAL option extensions are not supported");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_amdgpu_iree_hal_validate_string_view(options->identifier));
  if (options->device == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU IREE HAL options require a device");
  }
  return loomc_ok_status();
}

static loomc_amdgpu_target_feature_state_t
loomc_amdgpu_iree_hal_map_feature_state(
    iree_hal_amdgpu_target_feature_state_t state) {
  switch (state) {
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY:
      return LOOMC_AMDGPU_TARGET_FEATURE_ANY;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED:
      return LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF:
      return LOOMC_AMDGPU_TARGET_FEATURE_OFF;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON:
      return LOOMC_AMDGPU_TARGET_FEATURE_ON;
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU target feature state");
      return LOOMC_AMDGPU_TARGET_FEATURE_ANY;
  }
}

static loomc_status_t loomc_amdgpu_iree_hal_fail_status(loomc_result_t* result,
                                                        loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("AMDGPU/IREE_HAL"), status);
}

static loomc_status_t loomc_amdgpu_iree_hal_fail_cstring(
    loomc_result_t* result, loomc_status_code_t code, const char* message) {
  return loomc_amdgpu_iree_hal_fail_status(result,
                                           loomc_make_status(code, message));
}

static bool loomc_amdgpu_iree_hal_is_profile_diagnostic(loomc_status_t status) {
  switch (loomc_status_code(status)) {
    case LOOMC_STATUS_INVALID_ARGUMENT:
    case LOOMC_STATUS_NOT_FOUND:
    case LOOMC_STATUS_FAILED_PRECONDITION:
    case LOOMC_STATUS_OUT_OF_RANGE:
    case LOOMC_STATUS_UNIMPLEMENTED:
    case LOOMC_STATUS_UNAVAILABLE:
      return true;
    default:
      return false;
  }
}

static loomc_status_t loomc_amdgpu_iree_hal_query_identity(
    const loomc_amdgpu_iree_hal_profile_options_t* options,
    loomc_result_t* result, loomc_amdgpu_target_identity_t* out_identity) {
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts");
  }

  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = options->physical_device_affinity,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not advertise an exact AMDGPU target");
  }
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_FAILED_PRECONDITION,
        "IREE HAL device advertises ambiguous exact AMDGPU targets; select a "
        "physical-device affinity");
  }

  iree_hal_amdgpu_target_identity_t hal_identity = {0};
  loomc_status_t status =
      loomc_status_from_iree(iree_hal_amdgpu_target_identity_parse_artifact_key(
          target_result.target->target_key, &hal_identity));
  if (!loomc_status_is_ok(status)) {
    return loomc_amdgpu_iree_hal_fail_status(result, status);
  }
  if (hal_identity.kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_FAILED_PRECONDITION,
        "IREE HAL exact AMDGPU target contains a generic target key");
  }

  *out_identity = (loomc_amdgpu_target_identity_t){
      .target = loomc_string_view_from_iree(hal_identity.target),
      .amdhsa_features =
          {
              .sramecc = loomc_amdgpu_iree_hal_map_feature_state(
                  hal_identity.amdhsa_features.sramecc),
              .xnack = loomc_amdgpu_iree_hal_map_feature_state(
                  hal_identity.amdhsa_features.xnack),
          },
  };
  return loomc_ok_status();
}

loomc_status_t loomc_target_profile_create_amdgpu_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_iree_hal_profile_options_t* options,
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
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_iree_hal_validate_options(options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
  loomc_target_profile_t* profile = NULL;
  loomc_amdgpu_target_identity_t identity = {0};
  loomc_status_t status =
      loomc_amdgpu_iree_hal_query_identity(options, result, &identity);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loomc_amdgpu_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .identifier = options->identifier,
        .identity = identity,
    };
    status = loomc_target_profile_create_amdgpu(
        target_environment, &profile_options, allocator, &profile);
    if (!loomc_status_is_ok(status) &&
        loomc_amdgpu_iree_hal_is_profile_diagnostic(status)) {
      status = loomc_amdgpu_iree_hal_fail_status(result, status);
    }
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_profile = profile;
      profile = NULL;
    }
    *out_result = result;
    result = NULL;
  }
  loomc_target_profile_release(profile);
  loomc_result_release(result);
  return status;
}

static bool loomc_amdgpu_iree_hal_device_is_supported(
    const loomc_iree_hal_profile_options_t* options) {
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return false;
  }
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .physical_device_affinity = options->physical_device_affinity,
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  return result.outcome !=
         IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH;
}

static loomc_status_t loomc_amdgpu_iree_hal_provider_create_profile(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, bool* out_supported,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result) {
  (void)user_data;
  *out_supported = loomc_amdgpu_iree_hal_device_is_supported(options);
  *out_profile = NULL;
  *out_result = NULL;
  if (!*out_supported) {
    return loomc_ok_status();
  }

  const loomc_amdgpu_iree_hal_profile_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
      .next = options->next,
      .identifier = options->identifier,
      .device = options->device,
      .physical_device_affinity = options->physical_device_affinity,
  };
  return loomc_target_profile_create_amdgpu_iree_hal(
      target_environment, &amdgpu_options, allocator, out_profile, out_result);
}

const loomc_iree_hal_profile_provider_t* loomc_amdgpu_iree_hal_profile_provider(
    void) {
  static const loomc_iree_hal_profile_provider_t provider = {
      .name = {"amdgpu.iree_hal", 15},
      .user_data = NULL,
      .create_profile = loomc_amdgpu_iree_hal_provider_create_profile,
  };
  return &provider;
}
