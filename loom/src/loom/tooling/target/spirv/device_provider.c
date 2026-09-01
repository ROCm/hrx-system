// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/device_provider.h"

#include "loom/target/arch/spirv/facts.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/spirv/artifact_provider.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

static iree_status_t loom_spirv_device_provider_select_target(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_device_target_t){0};

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(runtime->device);
  if (device_spec == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan HAL device does not expose immutable device facts");
  }
  const iree_hal_executable_target_selection_t target_selection = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec,
                                                    &target_selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan HAL device does not support the vulkan1.3+bda SPIR-V target");
  } else if (target_result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan HAL device reports ambiguous vulkan1.3+bda SPIR-V targets");
  }

  loom_spirv_vulkan_hal_profile_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_vulkan_hal_profile_query(runtime->device, &facts));

  iree_hal_vulkan_cooperative_matrix_property_t* matrix_rows = NULL;
  iree_host_size_t matrix_row_count = 0;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(
          facts.flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    status = loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
        runtime->device, allocator, &matrix_rows, &matrix_row_count);
  }
  loom_spirv_vulkan_hal_target_profile_storage_t* profile_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, sizeof(*profile_storage),
                                   (void**)&profile_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_vulkan_hal_target_profile_storage_initialize(
        &facts, matrix_rows, matrix_row_count, allocator, profile_storage);
  }
  iree_allocator_free(allocator, matrix_rows);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, profile_storage);
    return status;
  }

  out_target->executable_target = target_result.target;
  out_target->artifact_target.target_profile = &profile_storage->profile.base;
  out_target->artifact_target.target_key = target_result.target->target_key;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_device_provider_select_compatible_target_from_facts(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  if (target_requirement != NULL &&
      loom_spirv_target_facts_cast(target_requirement) == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "SPIR-V device provider cannot satisfy target family '%.*s'",
        (int)target_requirement->fact_type->name.size,
        target_requirement->fact_type->name.data);
  }
  return loom_spirv_device_provider_select_target(provider, runtime, allocator,
                                                  out_target);
}

static void loom_spirv_device_provider_deinitialize_target(
    const loom_device_provider_t* provider, loom_device_target_t* target,
    iree_allocator_t allocator) {
  (void)provider;
  if (target == NULL) {
    return;
  }
  if (target->artifact_target.target_profile != NULL) {
    loom_spirv_vulkan_hal_target_profile_storage_t* storage =
        (loom_spirv_vulkan_hal_target_profile_storage_t*)
            target->artifact_target.target_profile;
    loom_spirv_vulkan_hal_target_profile_storage_deinitialize(storage,
                                                              allocator);
    iree_allocator_free(allocator, storage);
  }
  *target = (loom_device_target_t){0};
}

const loom_device_provider_t loom_spirv_vulkan_device_provider = {
    .artifact_provider = &loom_spirv_vulkan_artifact_provider,
    .driver_name = IREE_SVL("vulkan"),
    .select_target = loom_spirv_device_provider_select_target,
    .select_compatible_target =
        loom_spirv_device_provider_select_compatible_target_from_facts,
    .deinitialize_target = loom_spirv_device_provider_deinitialize_target,
};
