// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_
#define IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_

#include <vector>

#include "amdf/gpu.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amd/gpu/target/code_object.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

namespace iree::hal::amd::gpu::cts {

// Selects a compatible compiled target, preferring exact over generic code.
// Returns a borrowed entry in |targets|, or nullptr when none is compatible.
const iree::hal::cts::ExecutableTarget* SelectExecutableTarget(
    const std::vector<iree::hal::cts::ExecutableTarget>& targets,
    const iree_hal_amd_gpu_target_identity_t& device_identity);

// Resolves the native physical identity and looks up a logical program in a
// linked CTS collection. The returned bytes borrow immutable embedded storage.
// A missing compatible image or program is a configuration error, not a skip.
iree_status_t LookupExecutableData(
    const char* collection, const amdf_gpu_endpoint_info_t& endpoint_info,
    iree_string_view_t file_name, iree_const_byte_span_t* out_data);

// Selects a compiler-produced code object using its ELF identity. Each table
// contains variants of one logical program; filenames have no selection role.
// The embedding tool defines its table type in each generated corpus header,
// so the template keeps independently compiled corpora out of this dependency.
template <typename File>
iree_status_t LookupCodeObject(const File* files, size_t count,
                               iree_string_view_t target_key,
                               iree_const_byte_span_t* out_data) {
  *out_data = iree_const_byte_span_empty();
  iree_hal_amd_gpu_target_identity_t device_identity;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      target_key, &device_identity));
  for (size_t i = 0; i < count; ++i) {
    const auto bytes = iree_make_const_byte_span(files[i].data, files[i].size);
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_code_object_identity_from_elf(bytes, &identity));
    if (iree_hal_amd_gpu_target_identity_check_compatible(&identity,
                                                          &device_identity) !=
        IREE_HAL_AMD_GPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      continue;
    }
    *out_data = bytes;
    if (identity.kind == IREE_HAL_AMD_GPU_TARGET_KIND_EXACT) {
      break;
    }
  }
  if (!out_data->data) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CTS has no configured code object for '%.*s'",
                            static_cast<int>(target_key.size), target_key.data);
  }
  return iree_ok_status();
}

}  // namespace iree::hal::amd::gpu::cts

#endif  // IREE_HAL_DRIVERS_AMD_GPU_CTS_TESTDATA_H_
