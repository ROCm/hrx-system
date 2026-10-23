// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/cts/testdata.h"

#include <cstdio>

namespace iree::hal::amd::gpu::cts {

const iree::hal::cts::ExecutableTarget* SelectExecutableTarget(
    const std::vector<iree::hal::cts::ExecutableTarget>& targets,
    const iree_hal_amd_gpu_target_identity_t& device_identity) {
  const iree::hal::cts::ExecutableTarget* selected = nullptr;
  for (const auto& target : targets) {
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_CHECK_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        iree_make_cstring_view(target.target_key), &identity));
    if (iree_hal_amd_gpu_target_identity_check_compatible(&identity,
                                                          &device_identity) !=
        IREE_HAL_AMD_GPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      continue;
    }
    selected = &target;
    if (identity.kind == IREE_HAL_AMD_GPU_TARGET_KIND_EXACT) {
      break;
    }
  }
  return selected;
}

iree_status_t LookupExecutableData(
    const char* collection, const amdf_gpu_endpoint_info_t& endpoint_info,
    iree_string_view_t file_name, iree_const_byte_span_t* out_data) {
  *out_data = iree_const_byte_span_empty();
  char processor_name[32];
  std::snprintf(processor_name, sizeof(processor_name), "gfx%u%u%x",
                endpoint_info.gfx_ip.major, endpoint_info.gfx_ip.minor,
                endpoint_info.gfx_ip.stepping);
  iree_hal_amd_gpu_target_identity_t device_identity;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_parse_processor(
      iree_make_cstring_view(processor_name), &device_identity));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_resolve_physical_target(
      endpoint_info.asic_revision, &device_identity));
  const auto targets =
      iree::hal::cts::CtsRegistry::ListExecutableTargets(collection);
  const auto* target = SelectExecutableTarget(targets, device_identity);
  if (!target) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "CTS collection '%s' has no configured image compatible with '%.*s'",
        collection, static_cast<int>(device_identity.target.size),
        device_identity.target.data);
  }
  *out_data = target->data_fn(file_name);
  if (!out_data->data) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "CTS collection '%s' target '%s' has no '%.*s'",
                            collection, target->target_key,
                            static_cast<int>(file_name.size), file_name.data);
  }
  return iree_ok_status();
}

}  // namespace iree::hal::amd::gpu::cts
