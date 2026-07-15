// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/kmt_api.h"

namespace iree::hal::amdxdna::mcdm {

constexpr iree_host_size_t kContextBlobNameCapacity = 64;

struct ContextBlobInfo {
  char kernel_name[kContextBlobNameCapacity] = {};
  uint32_t column_width = 0;
  uint32_t start_column = 0;
  uint32_t pdi_count = 0;
  char pdi_name[kContextBlobNameCapacity] = {};
  uint64_t dpu_kernel_id = 0;
  // Base CU/kernel names from IP_LAYOUT, in CU-mask bit order. Names are
  // normalized by dropping the xclbin instance suffix after ':'.
  uint32_t kernel_name_count = 0;
  char* kernel_names = nullptr;
  uint32_t pdi_name_count = 0;
  char* pdi_names = nullptr;
  uint64_t* dpu_kernel_ids = nullptr;
  iree_allocator_t allocator = iree_allocator_null();
};

bool BuildContextPrivateDataFromXclbin(const uint8_t* xclbin,
                                       size_t xclbin_size, uint32_t process_id,
                                       iree_allocator_t allocator,
                                       iree_byte_span_t* out_blob,
                                       ContextBlobInfo* out_info,
                                       Error* out_error);

bool BuildCompactContextPrivateDataFromXclbin(
    const uint8_t* xclbin, size_t xclbin_size, uint32_t process_id,
    const Buffer& context_private_buffer, iree_allocator_t allocator,
    iree_byte_span_t* out_blob, ContextBlobInfo* out_info, Error* out_error);

// Builds the context-private packet selected by the device's negotiated MCDM
// contract. Any supporting BO remains owned by the caller on success.
bool BuildContextPrivateDataForDevice(
    const KmtApi& api, const Device& device, const uint8_t* xclbin,
    size_t xclbin_size, uint32_t process_id, iree_allocator_t allocator,
    iree_byte_span_t* out_blob, ContextBlobInfo* out_info,
    Buffer* out_context_private_buffer, Error* out_error);

const char* ContextBlobInfoKernelName(const ContextBlobInfo* info,
                                      uint32_t index);
const char* ContextBlobInfoPdiName(const ContextBlobInfo* info, uint32_t index);
void ContextBlobInfoDeinitialize(ContextBlobInfo* info);

}  // namespace iree::hal::amdxdna::mcdm

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_CONTEXT_BLOB_H_
