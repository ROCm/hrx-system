// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_CODE_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_GPU_CODE_QUEUE_H_

#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_code_queue_t iree_hal_amd_gpu_code_queue_t;

// Owns one cold queue in the selected compute family and bounded executable
// command storage. Confirmed native writes bootstrap code without a resident
// shader or a transfer-engine queue. Loading callers serialize publication;
// application dependencies cannot block it. No worker is created.
// Partial construction remains in out_queue so the parent can join native
// ownership with free even when preparation fails.
iree_status_t iree_hal_amd_gpu_code_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_code_queue_t** out_queue);

// Publishes initialized code and invalidates instruction caches. Nonempty
// contents are copied through bounded native WRITE_DATA commands; their length
// is a multiple of four and fits memory. Empty contents publish code already
// written through its host mapping. Waits on this loading caller for checked
// native completion, independently of all application queues.
//
// Host input bytes are borrowed only until return, including on failure. If a
// native command was accepted without proven retirement, this owner takes
// memory's existing handles before the failed executable can roll back. Those
// handles remain owned until native stop; subsequent publication is rejected.
// |timeout| is converted once to an absolute deadline before submitting the
// first chunk and applies to the complete synchronous publication.
iree_status_t iree_hal_amd_gpu_code_queue_publish(
    iree_hal_amd_gpu_code_queue_t* queue, iree_timeout_t timeout,
    iree_hal_amd_fabric_memory_t* memory, iree_const_byte_span_t contents);

// Closes publication and wakes the loading caller without claiming native
// access cessation. NULL represents mapped PM4 code needing no private queue.
amdf_status_t iree_hal_amd_gpu_code_queue_close(
    iree_hal_amd_gpu_code_queue_t* queue, amdf_status_t failure);

// Proves the loading caller has relinquished its CPU publication borrow.
// Native access may remain unproven. NULL is idle.
bool iree_hal_amd_gpu_code_queue_is_host_idle(
    iree_hal_amd_gpu_code_queue_t* queue);

// Joins the loading publisher and stops native access while backing remains
// retained. False means a loading caller still owns publication; true supplies
// the sticky native stop result. The parent retains this owner until teardown.
bool iree_hal_amd_gpu_code_queue_try_inactivate(
    iree_hal_amd_gpu_code_queue_t* queue, amdf_status_t* out_status);

// Called after all loaders return. Success joins native notification tails
// and frees the owner; failure preserves it and its context. NULL is safe.
amdf_status_t iree_hal_amd_gpu_code_queue_free(
    iree_hal_amd_gpu_code_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_CODE_QUEUE_H_
