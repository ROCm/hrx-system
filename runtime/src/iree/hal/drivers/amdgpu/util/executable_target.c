// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/executable_target.h"

enum {
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT = 100,
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC = 50,
};

static iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_id_t* target_id,
    iree_hal_executable_target_kind_t kind, uint32_t priority,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  char target_key_storage[128] = {0};
  iree_host_size_t target_key_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_format(target_id, sizeof(target_key_storage),
                                       target_key_storage, &target_key_length));
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
    const iree_hal_amdgpu_target_id_t* exact_target_id,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_spec_builder_add_executable_target(
          builder, exact_target_id, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT,
          physical_device_affinity));

  iree_hal_amdgpu_target_id_t code_object_target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_lookup_code_object_target(
      exact_target_id, &code_object_target_id));
  if (iree_string_view_equal(exact_target_id->processor,
                             code_object_target_id.processor) &&
      exact_target_id->sramecc == code_object_target_id.sramecc &&
      exact_target_id->xnack == code_object_target_id.xnack) {
    return iree_ok_status();
  }
  return iree_hal_amdgpu_device_spec_builder_add_executable_target(
      builder, &code_object_target_id, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC,
      physical_device_affinity);
}
