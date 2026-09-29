// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_PROGRAM_H_

#include "iree/hal/drivers/amdgpu/util/sdma_capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif

// Addresses are device virtual addresses. Recording never dereferences them.
typedef struct iree_hal_amdgpu_sdma_copy_t {
  uint64_t source;
  uint64_t target;
  uint64_t length;
} iree_hal_amdgpu_sdma_copy_t;

typedef struct iree_hal_amdgpu_sdma_program_params_t {
  iree_hal_amdgpu_sdma_capabilities_t capabilities;
  iree_host_size_t copy_count;
  const iree_hal_amdgpu_sdma_copy_t* copies;
  // Optional DWORD equality wait before the acquire. Zero address omits it.
  // The producer must release its payload before writing the ready value.
  // Keep that value observable until the wait retires; later values do not
  // match.
  uint64_t wait_address;
  uint32_t wait_value;
  // Required DWORD completion after copy/release. This is a memory milestone,
  // NOT an HSA signal operation or a HAL semaphore notification.
  // The consumer must acquire before reading the copied payload.
  uint64_t completion_address;
  uint32_t completion_value;
  // Optional pair of 32-byte-aligned SDMA GET_GLOBAL destinations.
  uint64_t start_timestamp_address;
  uint64_t end_timestamp_address;
} iree_hal_amdgpu_sdma_program_params_t;

// Validates the complete description and calculates exact storage. Outputs
// remain unchanged on error. Zero-byte copies are ignored; large copies split.
iree_status_t iree_hal_amdgpu_sdma_program_measure(
    const iree_hal_amdgpu_sdma_program_params_t* params,
    uint32_t* out_dword_count);
// Emits into borrowed storage. No write on validation/capacity failure. The
// description and copy list must remain immutable throughout the call.
// Payload and synchronization storage must remain live until execution retires.
iree_status_t iree_hal_amdgpu_sdma_program_emit(
    const iree_hal_amdgpu_sdma_program_params_t* params, uint32_t capacity,
    uint32_t* dwords, uint32_t* out_dword_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_PROGRAM_H_
