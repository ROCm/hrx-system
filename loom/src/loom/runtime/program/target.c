// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/target.h"

#include "loomc/iree.h"

// Public emitter names paired with the loader families selected by LoomC.
// CPU artifacts are architecture-specific; a CPU device does not imply x86.
// This is a cold emission-policy table, not a second device/profile selection.
static const struct {
  // HAL loader family owning the target-key grammar.
  iree_string_view_t family;
  // Required key, or empty when the emitter covers the whole family.
  iree_string_view_t target_key;
  // Public LoomC artifact format consumed by that loader.
  iree_string_view_t artifact_format;
  // Control-flow form accepted by target-low emission.
  loomc_target_control_flow_lowering_t control_flow_lowering;
} loom_runtime_target_formats[] = {
    {IREE_SVL("amdgpu"), IREE_SVL(""), IREE_SVL("amdgpu-hsaco"),
     LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG},
    {IREE_SVL("spirv"), IREE_SVL("vulkan1.3+bda"), IREE_SVL("spirv"),
     LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW},
    {IREE_SVL("cpu"), IREE_SVL("x86_64"), IREE_SVL("x86-hal"),
     LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG},
    {IREE_SVL("xdna"), IREE_SVL(""), IREE_SVL("xdna"),
     LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG},
};

iree_status_t loom_runtime_target_select(
    loomc_target_environment_t* environment,
    const loomc_iree_hal_target_provider_t* const* providers,
    iree_host_size_t provider_count, iree_hal_device_t* device,
    iree_hal_queue_t* queue, loom_runtime_compilation_report_t* report,
    iree_allocator_t host_allocator, loom_runtime_target_t* out_target) {
  *out_target = (loom_runtime_target_t){0};
  const loomc_iree_hal_target_options_t options = {
      .device = device,
      .queue = queue,
      .providers = providers,
      .provider_count = provider_count,
  };
  loomc_iree_hal_target_selection_t selection = {0};
  loomc_result_t* result = NULL;
  const loomc_status_t operation = loomc_target_select_iree_hal(
      environment, &options, loomc_allocator_from_iree(host_allocator),
      &selection, &result);
  IREE_RETURN_IF_ERROR(
      loom_runtime_compilation_result(operation, result, report));
  const iree_hal_executable_target_t* target = selection.executable_target;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_runtime_target_formats);
       ++i) {
    if (iree_string_view_equal(target->family,
                               loom_runtime_target_formats[i].family) &&
        (iree_string_view_is_empty(loom_runtime_target_formats[i].target_key) ||
         iree_string_view_equal(target->target_key,
                                loom_runtime_target_formats[i].target_key))) {
      *out_target = (loom_runtime_target_t){
          .selection = selection,
          .artifact_format = loom_runtime_target_formats[i].artifact_format,
          .control_flow_lowering =
              loom_runtime_target_formats[i].control_flow_lowering,
      };
      return iree_ok_status();
    }
  }
  loomc_target_profile_release(selection.target_profile);
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "no serving artifact format for HAL target %.*s:%.*s",
                          (int)target->family.size, target->family.data,
                          (int)target->target_key.size,
                          target->target_key.data);
}
