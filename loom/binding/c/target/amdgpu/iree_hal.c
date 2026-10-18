// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu/iree_hal.h"

#include "diagnostic.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loomc/iree.h"
#include "result.h"
#include "target.h"

static loomc_status_t loomc_amdgpu_iree_hal_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_amdgpu_iree_hal_validate_options(
    const loomc_amdgpu_iree_hal_target_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU IREE HAL target options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_TARGET_OPTIONS) {
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

static bool loomc_amdgpu_iree_hal_target_is_modeled(
    iree_string_view_t target_key) {
  iree_string_view_t target_name = target_key;
  iree_string_view_split(target_key, ':', &target_name, NULL);
  return loom_amdgpu_target_info_find_target(target_name) != NULL;
}

static loomc_status_t loomc_amdgpu_iree_hal_select_automatic_candidate(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_kind_t target_kind, loomc_result_t* result,
    loomc_amdgpu_target_identity_t* out_identity,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  *out_executable_target = NULL;

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  const iree_hal_executable_target_t* selected_target = NULL;
  loomc_amdgpu_target_identity_t selected_identity = {0};
  bool selected_target_is_ambiguous = false;
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &executable_spec->targets[i];
    if (!iree_string_view_equal(candidate->family, IREE_SV("amdgpu")) ||
        candidate->kind != target_kind ||
        (physical_device_affinity != 0 &&
         !iree_all_bits_set(candidate->physical_device_affinity,
                            physical_device_affinity)) ||
        !loomc_amdgpu_iree_hal_target_is_modeled(candidate->target_key)) {
      continue;
    }

    loomc_amdgpu_target_identity_t candidate_identity = {0};
    loomc_status_t status = loomc_amdgpu_target_identity_parse_artifact_key(
        loomc_string_view_from_iree(candidate->target_key),
        &candidate_identity);
    if (!loomc_status_is_ok(status)) {
      return loomc_amdgpu_iree_hal_fail_status(result, status);
    }
    const loom_amdgpu_target_info_t* compiler_target =
        loom_amdgpu_target_info_find_target(
            iree_string_view_from_loomc(candidate_identity.target));
    IREE_ASSERT(compiler_target != NULL);
    const iree_hal_executable_target_kind_t compiler_target_kind =
        loom_amdgpu_target_info_is_generic(compiler_target)
            ? IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC
            : IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
    if (candidate->kind != compiler_target_kind) {
      return loomc_amdgpu_iree_hal_fail_status(
          result,
          loomc_status_from_iree(iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "AMDGPU device target '%.*s' kind does not match its target key",
              (int)candidate->target_key.size, candidate->target_key.data)));
    }
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(compiler_target);
    IREE_ASSERT(processor != NULL);
    if (!loom_amdgpu_processor_properties_support_hsaco(
            &processor->properties)) {
      continue;
    }

    if (selected_target == NULL ||
        candidate->priority > selected_target->priority) {
      selected_target = candidate;
      selected_identity = candidate_identity;
      selected_target_is_ambiguous = false;
    } else if (candidate->priority == selected_target->priority) {
      selected_target_is_ambiguous = true;
    }
  }

  if (selected_target_is_ambiguous) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_FAILED_PRECONDITION,
        "IREE HAL device advertises ambiguous Loom-supported AMDGPU targets; "
        "select a physical-device affinity");
  }
  if (selected_target != NULL) {
    *out_identity = selected_identity;
    *out_executable_target = selected_target;
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_amdgpu_iree_hal_query_identity(
    const loomc_amdgpu_iree_hal_target_options_t* options,
    loomc_result_t* result, loomc_amdgpu_target_identity_t* out_identity,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  *out_executable_target = NULL;
  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts");
  }

  loomc_status_t status = loomc_amdgpu_iree_hal_select_automatic_candidate(
      device_spec, options->physical_device_affinity,
      IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, result, out_identity,
      out_executable_target);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      *out_executable_target == NULL) {
    status = loomc_amdgpu_iree_hal_select_automatic_candidate(
        device_spec, options->physical_device_affinity,
        IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, result, out_identity,
        out_executable_target);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
      *out_executable_target == NULL) {
    status = loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device has no Loom-supported native AMDGPU target");
  }
  return status;
}

static loomc_status_t loomc_amdgpu_iree_hal_select_profile_target(
    const loomc_amdgpu_iree_hal_target_options_t* options,
    loomc_allocator_t allocator, loomc_result_t* result,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_executable_target = NULL;
  const loom_target_profile_t* base_profile =
      loomc_target_profile_loom_target_profile(options->target_profile);
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(base_profile);
  if (profile == NULL) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU IREE HAL target selection requires an AMDGPU profile");
  }

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_amdgpu_iree_hal_fail_cstring(
        result, LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable device facts");
  }

  iree_string_builder_t target_key_builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &target_key_builder);
  iree_status_t iree_status =
      loom_amdgpu_artifact_key_append(&profile->identity, &target_key_builder);
  if (iree_status_is_ok(iree_status)) {
    const iree_hal_executable_target_selection_t target_selection = {
        .family = IREE_SV("amdgpu"),
        .target_key = iree_string_builder_view(&target_key_builder),
        .kind_flags =
            loom_amdgpu_target_info_is_generic(profile->identity.target)
                ? IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC
                : IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
        .physical_device_affinity = options->physical_device_affinity,
    };
    const iree_hal_executable_target_selection_result_t target_result =
        iree_hal_device_spec_select_executable_target(device_spec,
                                                      &target_selection);
    if (target_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      iree_status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "IREE HAL device cannot load forced AMDGPU target '%.*s'",
          (int)target_selection.target_key.size,
          target_selection.target_key.data);
    } else if (target_result.outcome ==
               IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
      iree_status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "IREE HAL device advertises ambiguous forced AMDGPU targets '%.*s'; "
          "select a physical-device affinity",
          (int)target_selection.target_key.size,
          target_selection.target_key.data);
    } else {
      *out_executable_target = target_result.target;
    }
  }
  iree_string_builder_deinitialize(&target_key_builder);
  if (!iree_status_is_ok(iree_status)) {
    return loomc_amdgpu_iree_hal_fail_status(
        result, loomc_status_from_iree(iree_status));
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_select_amdgpu_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_iree_hal_target_options_t* options,
    loomc_allocator_t allocator,
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
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_iree_hal_validate_options(options));
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
  if (options->target_profile != NULL) {
    status = loomc_amdgpu_iree_hal_select_profile_target(
        options, allocator, result, &executable_target);
    if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
      loomc_target_profile_retain(options->target_profile);
      target_profile = options->target_profile;
    }
  } else {
    loomc_amdgpu_target_identity_t identity = {0};
    status = loomc_amdgpu_iree_hal_query_identity(options, result, &identity,
                                                  &executable_target);
    if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
      const loomc_amdgpu_profile_options_t profile_options = {
          .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
          .structure_size = sizeof(profile_options),
          .identifier = options->identifier,
          .identity = identity,
      };
      status = loomc_target_profile_create_amdgpu(
          target_environment, &profile_options, allocator, &target_profile);
      if (!loomc_status_is_ok(status) &&
          loomc_amdgpu_iree_hal_is_profile_diagnostic(status)) {
        status = loomc_amdgpu_iree_hal_fail_status(result, status);
      }
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

static bool loomc_amdgpu_iree_hal_device_is_supported(
    const loomc_iree_hal_target_options_t* options) {
  if (options->target_profile != NULL) {
    return loom_amdgpu_target_profile_cast(
               loomc_target_profile_loom_target_profile(
                   options->target_profile)) != NULL;
  }
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

static loomc_status_t loomc_amdgpu_iree_hal_provider_select_target(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  (void)user_data;
  *out_supported = loomc_amdgpu_iree_hal_device_is_supported(options);
  *out_selection = (loomc_iree_hal_target_selection_t){0};
  *out_result = NULL;
  if (!*out_supported) {
    return loomc_ok_status();
  }

  const loomc_amdgpu_iree_hal_target_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
      .next = options->next,
      .identifier = options->identifier,
      .device = options->device,
      .physical_device_affinity = options->physical_device_affinity,
      .target_profile = options->target_profile,
  };
  return loomc_target_select_amdgpu_iree_hal(target_environment,
                                             &amdgpu_options, allocator,
                                             out_selection, out_result);
}

const loomc_iree_hal_target_provider_t* loomc_amdgpu_iree_hal_target_provider(
    void) {
  static const loomc_iree_hal_target_provider_t provider = {
      .name = {"amdgpu.iree_hal", 15},
      .user_data = NULL,
      .select_target = loomc_amdgpu_iree_hal_provider_select_target,
  };
  return &provider;
}
