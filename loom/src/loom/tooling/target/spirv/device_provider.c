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
    const loom_run_hal_runtime_t* runtime, loom_device_target_t* out_target) {
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
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_initialize_target_bundle(
      &facts, &out_target->target_bundle_storage));

  out_target->executable_target = target_result.target;
  return iree_ok_status();
}

static iree_status_t
loom_spirv_device_provider_select_compatible_target_from_facts(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement,
    loom_device_target_t* out_target) {
  if (target_requirement != NULL &&
      loom_spirv_target_facts_cast(target_requirement) == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "SPIR-V device provider cannot satisfy target family '%.*s'",
        (int)target_requirement->fact_type->name.size,
        target_requirement->fact_type->name.data);
  }
  return loom_spirv_device_provider_select_target(provider, runtime,
                                                  out_target);
}

static iree_status_t loom_spirv_device_provider_project_target_facts(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, const loom_device_target_t* target,
    iree_arena_allocator_t* arena, loom_target_facts_t* out_facts) {
  (void)provider;
  (void)target;
  loom_spirv_target_facts_t* spirv_facts =
      (loom_spirv_target_facts_t*)out_facts;
  return loom_spirv_vulkan_hal_profile_project_target_facts(runtime->device,
                                                            arena, spirv_facts);
}

const loom_device_provider_t loom_spirv_vulkan_device_provider = {
    .artifact_provider = &loom_spirv_vulkan_artifact_provider,
    .driver_name = IREE_SVL("vulkan"),
    .select_target = loom_spirv_device_provider_select_target,
    .select_compatible_target =
        loom_spirv_device_provider_select_compatible_target_from_facts,
    .project_target_facts = loom_spirv_device_provider_project_target_facts,
};
