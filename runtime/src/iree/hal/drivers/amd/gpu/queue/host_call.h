// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_

#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/virtual_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed native queue services. Successful barrier completion dispatches
// host semaphore timepoints on the device's completion worker, after native
// execution and cache effects. Submission hints never trigger those points.
// Each accepted call retains the parent device through its terminal action.
typedef struct iree_hal_amd_gpu_host_call_executor_t {
  // Exact queue supplying native dependency and cache boundaries.
  iree_hal_queue_t* queue;
  // Device diagnostics, dominated by the retained parent.
  const iree_hal_amd_gpu_context_t* context;
  // Existing cold-memory owner for mapping and deferred native submission.
  iree_hal_memory_maintenance_t* service;
  // Queue-owned reusable CPU capture storage.
  iree_hal_amd_gpu_capture_pool_t* captures;
} iree_hal_amd_gpu_host_call_executor_t;

// Captures callback state and invokes it on the existing completion worker
// after a native release boundary. DEFERRED callbacks receive a private
// completion list; public completion follows the return-to-device acquire.
// NON_BLOCKING publishes at invocation and promises no ordering against later
// callback effects. Failure retains nothing and invokes no callback.
iree_status_t iree_hal_amd_gpu_host_call_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_host_call_t call, const uint64_t arguments[4],
    iree_hal_host_call_flags_t flags);

// Captures a mapping batch and executes native mapping on the existing service
// between native release/acquire boundaries. Captured buffer uses end before
// public completion. Borrowed-buffer mode preserves the caller's responsibility
// through that same completion boundary. No mapping worker is created.
iree_status_t iree_hal_amd_gpu_memory_mapping_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t mapping_count, const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_
