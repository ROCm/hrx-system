// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BUFFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Qualified address source. The submission owns the buffer reference while
// queue-ordered commitment supplies the address after dependency admission.
typedef struct iree_hal_amd_gpu_buffer_reference_t {
  // Buffer whose lifetime is managed by the enclosing submission.
  iree_hal_buffer_t* buffer;
  // Byte offset relative to the buffer's logical beginning.
  iree_device_size_t offset;
  // Direct index into the immutable native binding layout.
  iree_hal_buffer_native_binding_slot_t slot;
} iree_hal_amd_gpu_buffer_reference_t;

// Qualifies the native interface without requiring a committed address.
// Public validation has established the range, usage, and family membership.
// Borrows buffer; the caller retains it only when taking capture ownership.
iree_status_t iree_hal_amd_gpu_buffer_reference_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_hal_amd_gpu_buffer_reference_t* out_reference);

// Resolves the qualified slot after allocation dependencies. A missing native
// commitment is an error; no topology or contract lookup is repeated here.
iree_status_t iree_hal_amd_gpu_buffer_reference_resolve(
    const iree_hal_amd_gpu_buffer_reference_t* reference,
    uint64_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_BUFFER_H_
