// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cpu/iree_hal.h"

#include "diagnostic.h"
#include "iree/hal/drivers/task/device_spec.h"
#include "loom/target/selection.h"
#include "loomc/iree.h"
#include "result.h"
#include "target.h"

static loomc_status_t loomc_cpu_iree_hal_fail_status(loomc_result_t* result,
                                                     loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("CPU/IREE_HAL"), status);
}

static loomc_status_t loomc_cpu_iree_hal_fail_cstring(loomc_result_t* result,
                                                      loomc_status_code_t code,
                                                      const char* message) {
  return loomc_cpu_iree_hal_fail_status(result,
                                        loomc_make_status(code, message));
}

static const loom_target_provider_t* loomc_cpu_iree_hal_lookup_profile_provider(
    const loomc_target_environment_t* target_environment,
    const loomc_target_profile_t* target_profile) {
  const loom_target_profile_t* native_profile =
      loomc_target_profile_loom_target_profile(target_profile);
  if (native_profile == NULL || native_profile->type == NULL) {
    return NULL;
  }
  const loom_target_environment_t* native_environment =
      loomc_target_environment_loom_target_environment(target_environment);
  const loom_target_provider_t* provider =
      loom_target_environment_lookup_profile_provider(native_environment,
                                                      native_profile->type);
  return provider != NULL && provider->select_cpu_profile != NULL ? provider
                                                                  : NULL;
}

static bool loomc_cpu_iree_hal_device_is_supported(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options) {
  if (options->target_profile != NULL) {
    return loomc_cpu_iree_hal_lookup_profile_provider(
               target_environment, options->target_profile) != NULL;
  }
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  return device_spec != NULL &&
         iree_hal_cpu_device_spec_find_facet(device_spec) != NULL;
}

static loomc_status_t loomc_cpu_iree_hal_select_target(
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
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, LOOMC_SOURCE_RETENTION_EXACT, allocator, &result));
  loomc_target_profile_t* target_profile = NULL;
  const iree_hal_executable_target_t* executable_target = NULL;
  loomc_status_t status = loomc_ok_status();

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    status = loomc_cpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts");
  }

  iree_hal_cpu_device_spec_t cpu_spec = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const iree_hal_device_spec_facet_t* facet =
        iree_hal_cpu_device_spec_find_facet(device_spec);
    if (facet == NULL) {
      status = loomc_cpu_iree_hal_fail_cstring(
          result, LOOMC_STATUS_UNAVAILABLE,
          "IREE HAL device does not expose native CPU facts");
    } else {
      iree_status_t iree_status =
          iree_hal_cpu_device_spec_decode_facet(facet, &cpu_spec);
      if (!iree_status_is_ok(iree_status)) {
        status = loomc_cpu_iree_hal_fail_status(
            result, loomc_status_from_iree(iree_status));
      }
    }
  }

  loom_target_profile_selection_t selected_profile = {0};
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loom_target_profile_t* requested_profile =
        loomc_target_profile_loom_target_profile(options->target_profile);
    iree_status_t iree_status = loom_target_environment_select_cpu_profile(
        loomc_target_environment_loom_target_environment(target_environment),
        &cpu_spec.cpu_data, NULL, requested_profile, &selected_profile,
        iree_allocator_from_loomc(allocator));
    if (!iree_status_is_ok(iree_status)) {
      status = loomc_cpu_iree_hal_fail_status(
          result, loomc_status_from_iree(iree_status));
    }
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("cpu"),
        .target_key =
            iree_cpu_architecture_name(cpu_spec.cpu_data.architecture),
        .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
        .physical_device_affinity = options->physical_device_affinity,
    };
    const iree_hal_executable_target_selection_result_t target_result =
        iree_hal_device_spec_select_executable_target(device_spec, &selection);
    if (target_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      status = loomc_cpu_iree_hal_fail_cstring(
          result, LOOMC_STATUS_UNAVAILABLE,
          "IREE HAL device has no compatible native CPU loader target");
    } else if (target_result.outcome ==
               IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
      status = loomc_cpu_iree_hal_fail_cstring(
          result, LOOMC_STATUS_FAILED_PRECONDITION,
          "IREE HAL device advertises ambiguous native CPU loader targets; "
          "select a physical-device affinity");
    } else {
      executable_target = target_result.target;
    }
  }

  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    if (options->target_profile != NULL) {
      loomc_target_profile_retain(options->target_profile);
      target_profile = options->target_profile;
    } else {
      const loomc_string_view_t identifier =
          loomc_string_view_is_empty(options->identifier)
              ? loomc_string_view_from_iree(
                    selected_profile.profile->target_bundle->name)
              : options->identifier;
      status = loomc_target_profile_create(
          target_environment, identifier, selected_profile.profile,
          selected_profile.destroy, allocator, &target_profile);
      selected_profile = (loom_target_profile_selection_t){0};
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
  loom_target_profile_selection_release(&selected_profile,
                                        iree_allocator_from_loomc(allocator));
  loomc_target_profile_release(target_profile);
  loomc_result_release(result);
  return status;
}

static loomc_status_t loomc_cpu_iree_hal_provider_select_target(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  (void)user_data;
  *out_supported =
      loomc_cpu_iree_hal_device_is_supported(target_environment, options);
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (!*out_supported) {
    return loomc_ok_status();
  }
  return loomc_cpu_iree_hal_select_target(target_environment, options,
                                          allocator, out_selection, out_result);
}

const loomc_iree_hal_target_provider_t* loomc_cpu_iree_hal_target_provider(
    void) {
  static const loomc_iree_hal_target_provider_t provider = {
      .name = {"cpu.iree_hal", 12},
      .user_data = NULL,
      .select_target = loomc_cpu_iree_hal_provider_select_target,
  };
  return &provider;
}
