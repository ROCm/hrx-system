// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/target/selection.h"

enum {
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT = 100,
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC = 50,
};

static iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_identity_t* identity,
    iree_hal_executable_target_kind_t kind, uint32_t priority,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  char target_key_storage[128] = {0};
  iree_host_size_t target_key_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_format_artifact_key(
      identity, sizeof(target_key_storage), target_key_storage,
      &target_key_length));
  const iree_hal_executable_target_t target = {
      .family = IREE_SV("amdgpu"),
      .target_key =
          iree_make_string_view(target_key_storage, target_key_length),
      .kind = kind,
      .priority = priority,
      .physical_device_affinity = physical_device_affinity,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_add_executable_target(builder, &target);
}

iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_spec_builder_add_executable_target(
          builder, exact_identity, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT,
          physical_device_affinity));

  iree_hal_amdgpu_target_identity_t code_object_identity;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_project_code_object(
      exact_identity, &code_object_identity));
  if (iree_string_view_equal(exact_identity->processor,
                             code_object_identity.processor) &&
      exact_identity->amdhsa_features.sramecc ==
          code_object_identity.amdhsa_features.sramecc &&
      exact_identity->amdhsa_features.xnack ==
          code_object_identity.amdhsa_features.xnack) {
    return iree_ok_status();
  }
  return iree_hal_amdgpu_device_spec_builder_add_executable_target(
      builder, &code_object_identity, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC,
      physical_device_affinity);
}

static iree_hal_executable_target_kind_t iree_hal_amdgpu_executable_target_kind(
    const iree_hal_amdgpu_target_identity_t* identity) {
  return identity->kind == IREE_HAL_AMDGPU_TARGET_KIND_GENERIC
             ? IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC
             : IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
}

iree_status_t iree_hal_amdgpu_device_spec_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (iree_hal_executable_target_selection_result_t){
      .outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
      .target = NULL,
      .target_ordinal = IREE_HOST_SIZE_MAX,
  };

  iree_hal_amdgpu_target_identity_t artifact_identity;
  iree_status_t status = iree_hal_amdgpu_target_identity_parse_artifact_key(
      artifact_target_key, &artifact_identity);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status, "parsing AMDGPU executable artifact target '%.*s'",
        (int)artifact_target_key.size, artifact_target_key.data);
  }
  const iree_hal_executable_target_kind_t artifact_target_kind =
      iree_hal_amdgpu_executable_target_kind(&artifact_identity);

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  const iree_hal_executable_target_t* selected_target = NULL;
  iree_host_size_t selected_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &executable_spec->targets[i];
    if (!iree_string_view_equal(candidate->family, IREE_SV("amdgpu")) ||
        candidate->kind != artifact_target_kind) {
      continue;
    }
    if (physical_device_affinity != 0 &&
        !iree_all_bits_set(candidate->physical_device_affinity,
                           physical_device_affinity)) {
      continue;
    }

    iree_hal_amdgpu_target_identity_t candidate_identity;
    status = iree_hal_amdgpu_target_identity_parse_artifact_key(
        candidate->target_key, &candidate_identity);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(
          status, "parsing AMDGPU device target '%.*s'",
          (int)candidate->target_key.size, candidate->target_key.data);
    }
    if (iree_hal_amdgpu_executable_target_kind(&candidate_identity) !=
        candidate->kind) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU device target '%.*s' kind does not match its target key",
          (int)candidate->target_key.size, candidate->target_key.data);
    }
    if (iree_hal_amdgpu_target_identity_check_compatible(&artifact_identity,
                                                         &candidate_identity) !=
        IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      continue;
    }

    if (selected_target == NULL ||
        candidate->priority > selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED;
      selected_target = candidate;
      selected_ordinal = i;
    } else if (candidate->priority == selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS;
    }
  }

  if (out_result->outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    out_result->target = selected_target;
    out_result->target_ordinal = selected_ordinal;
  }
  return iree_ok_status();
}
