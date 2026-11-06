// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/target/selection.h"

#include "iree/hal/driver/amd/gpu/target/identity.h"

static iree_status_t iree_hal_amd_gpu_device_spec_add_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amd_gpu_target_identity_t* identity,
    iree_hal_executable_target_kind_t kind, uint32_t priority,
    iree_hal_physical_device_affinity_t affinity) {
  char key[128];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_format_artifact_key(
      identity, sizeof(key), key, &length));
  const iree_hal_executable_target_t target = {
      .family = IREE_SV("amdgpu"),
      .target_key = iree_make_string_view(key, length),
      .kind = kind,
      .priority = priority,
      .physical_device_affinity = affinity,
  };
  return iree_hal_device_spec_builder_add_executable_target(builder, &target);
}

iree_status_t iree_hal_amd_gpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amd_gpu_target_identity_t* identity,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_device_spec_add_target(
      builder, identity, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT, 100,
      physical_device_affinity));
  iree_hal_amd_gpu_target_identity_t projected;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_project_code_object(
      identity, &projected));
  if (iree_hal_amd_gpu_target_identity_equal(identity, &projected)) {
    return iree_ok_status();
  }
  return iree_hal_amd_gpu_device_spec_add_target(
      builder, &projected, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC, 50,
      physical_device_affinity);
}

iree_status_t iree_hal_amd_gpu_device_spec_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result) {
  *out_result = (iree_hal_executable_target_selection_result_t){
      .outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
      .target_ordinal = IREE_HOST_SIZE_MAX,
  };
  iree_hal_amd_gpu_target_identity_t artifact;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      artifact_target_key, &artifact));
  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  const iree_hal_executable_target_t* selected = NULL;
  iree_host_size_t selected_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
    const iree_hal_executable_target_t* candidate = &executables->targets[i];
    if (!iree_string_view_equal(candidate->family, IREE_SV("amdgpu")) ||
        candidate->kind != IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT ||
        (physical_device_affinity &&
         !iree_all_bits_set(candidate->physical_device_affinity,
                            physical_device_affinity))) {
      continue;
    }
    iree_hal_amd_gpu_target_identity_t device;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        candidate->target_key, &device));
    if (iree_hal_amd_gpu_target_identity_check_compatible(&artifact, &device) !=
        IREE_HAL_AMD_GPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      continue;
    }
    if (!selected || candidate->priority > selected->priority) {
      selected = candidate;
      selected_ordinal = i;
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED;
    } else if (candidate->priority == selected->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS;
    }
  }
  if (out_result->outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    out_result->target = selected;
    out_result->target_ordinal = selected_ordinal;
  }
  return iree_ok_status();
}
