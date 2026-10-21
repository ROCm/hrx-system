// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_program.h"

#include "iree/hal/drivers/amdgpu/util/sdma_emitter.h"

iree_status_t iree_hal_amdgpu_sdma_program_measure(
    const iree_hal_amdgpu_sdma_program_params_t* params,
    uint32_t* out_dword_count) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_dword_count);
  if (params->capabilities.profile != IREE_HAL_AMDGPU_SDMA_PROFILE_GFX12_0 ||
      params->capabilities.max_copy_bytes !=
          IREE_HAL_AMDGPU_SDMA_COPY_MAX_BYTES) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported SDMA packet profile");
  }
  bool timestamps = params->start_timestamp_address != 0;
  if ((params->copy_count && !params->copies) || !params->completion_address ||
      (params->completion_address & 3) || (params->wait_address & 3) ||
      (timestamps != (params->end_timestamp_address != 0)) ||
      (params->start_timestamp_address & 31) ||
      (params->end_timestamp_address & 31)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid SDMA program storage");
  }
  if (timestamps && !params->capabilities.supports_timestamps)
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SDMA timestamps unavailable");
  uint64_t count =
      2 * IREE_HAL_AMDGPU_SDMA_GCR_DWORDS + IREE_HAL_AMDGPU_SDMA_FENCE_DWORDS;
  if (timestamps) count += 2 * IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS;
  if (params->wait_address) count += IREE_HAL_AMDGPU_SDMA_POLL_DWORDS;
  for (iree_host_size_t i = 0; i < params->copy_count; ++i) {
    const iree_hal_amdgpu_sdma_copy_t* copy = &params->copies[i];
    if (!copy->length) continue;
    if (!copy->source || !copy->target ||
        copy->source > UINT64_MAX - (copy->length - 1) ||
        copy->target > UINT64_MAX - (copy->length - 1))
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "invalid SDMA copy address range");
    uint64_t packets =
        1 + (copy->length - 1) / params->capabilities.max_copy_bytes;
    // Restrict programs to a uint32 byte size, matching native queue capacity.
    if (packets > (UINT32_MAX / 4 - count) / IREE_HAL_AMDGPU_SDMA_COPY_DWORDS)
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "SDMA program size overflow");
    count += packets * IREE_HAL_AMDGPU_SDMA_COPY_DWORDS;
  }
  *out_dword_count = (uint32_t)count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_sdma_program_emit(
    const iree_hal_amdgpu_sdma_program_params_t* params, uint32_t capacity,
    uint32_t* dwords, uint32_t* out_dword_count) {
  uint32_t count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_sdma_program_measure(params, &count));
  if (!dwords || !out_dword_count || capacity < count)
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "insufficient SDMA program storage");
  uint32_t* p = dwords;
  if (params->wait_address)
    p += iree_hal_amdgpu_sdma_emit_poll32(IREE_HAL_AMDGPU_SDMA_POLL_DWORDS, p,
                                          params->wait_address,
                                          params->wait_value);
  if (params->start_timestamp_address)
    p += iree_hal_amdgpu_sdma_emit_timestamp(
        IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS, p,
        params->start_timestamp_address);
  p += iree_hal_amdgpu_sdma_emit_gcr(IREE_HAL_AMDGPU_SDMA_GCR_DWORDS, p, true);
  for (iree_host_size_t i = 0; i < params->copy_count; ++i) {
    const iree_hal_amdgpu_sdma_copy_t* copy = &params->copies[i];
    uint64_t offset = 0;
    while (offset < copy->length) {
      uint64_t remaining = copy->length - offset;
      uint32_t length = remaining > params->capabilities.max_copy_bytes
                            ? params->capabilities.max_copy_bytes
                            : (uint32_t)remaining;
      p += iree_hal_amdgpu_sdma_emit_copy(IREE_HAL_AMDGPU_SDMA_COPY_DWORDS, p,
                                          copy->source + offset,
                                          copy->target + offset, length);
      offset += length;
    }
  }
  p += iree_hal_amdgpu_sdma_emit_gcr(IREE_HAL_AMDGPU_SDMA_GCR_DWORDS, p, false);
  if (params->end_timestamp_address)
    p += iree_hal_amdgpu_sdma_emit_timestamp(
        IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS, p,
        params->end_timestamp_address);
  p += iree_hal_amdgpu_sdma_emit_fence32(IREE_HAL_AMDGPU_SDMA_FENCE_DWORDS, p,
                                         params->completion_address,
                                         params->completion_value);
  IREE_ASSERT((uint32_t)(p - dwords) == count);
  *out_dword_count = count;
  return iree_ok_status();
}
