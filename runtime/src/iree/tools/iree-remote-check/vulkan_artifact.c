// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-remote-check/vulkan_artifact.h"

#include <string.h>

#include "iree/tools/iree-remote-check/vulkan_kernel_artifact_c.h"

iree_status_t iree_remote_check_select_vulkan_artifact(
    const iree_hal_device_spec_t* device_spec,
    iree_remote_check_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_artifact);
  memset(out_artifact, 0, sizeof(*out_artifact));

  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      .physical_device_affinity = 0,
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote device has ambiguous targets for Vulkan BDA SPIR-V");
  }
  if (result.outcome != IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_ok_status();
  }

  const iree_file_toc_t* files =
      iree_remote_check_vulkan_kernel_artifact_create();
  if (iree_remote_check_vulkan_kernel_artifact_size() != 1 ||
      files[0].size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "embedded Vulkan SPIR-V artifact is missing");
  }

  out_artifact->executable_target = result.target;
  out_artifact->file_name = iree_make_cstring_view(files[0].name);
  out_artifact->artifact_target_key = IREE_SV("vulkan1.3+bda");
  out_artifact->executable_data = iree_make_const_byte_span(
      (const uint8_t*)files[0].data, (iree_host_size_t)files[0].size);
  out_artifact->entry_point = IREE_SV("main");
  out_artifact->dispatch_config = iree_hal_make_static_dispatch_config(4, 1, 1);
  return iree_ok_status();
}
