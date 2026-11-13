// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_BUFFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_BUFFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Qualified address source. A buffer reference resolves after queue-ordered
// commitment. Native backing already owned by an enclosing operation uses an
// absolute address; that operation keeps it live through native retirement.
typedef struct iree_hal_amd_gpu_buffer_reference_t {
  // Buffer retained by the submission, or NULL for borrowed native backing.
  iree_hal_buffer_t* buffer;
  // Buffer-relative byte offset, or absolute GPU address when buffer is NULL.
  iree_device_size_t offset;
  // Direct native binding index, unused for borrowed native backing.
  iree_hal_buffer_native_binding_slot_t slot;
} iree_hal_amd_gpu_buffer_reference_t;

// Borrows an already-qualified native address. The enclosing operation owns
// its backing and visibility contract through the submission's final callback.
static inline iree_hal_amd_gpu_buffer_reference_t
iree_hal_amd_gpu_buffer_reference_borrow(uint64_t address) {
  return (iree_hal_amd_gpu_buffer_reference_t){.offset = address};
}

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

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_BUFFER_H_
