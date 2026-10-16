// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_TRANSFER_H_

#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native execution borrowed by the shared file pipeline. The operation
// retains the queue's device. Packet encoding and publication remain with the
// executor; this component owns I/O progress and captured payload lifetime.
typedef struct iree_hal_amd_gpu_file_executor_t {
  // Borrowed exact queue whose native transfer engine moves buffer bytes.
  iree_hal_queue_t* queue;
  // Borrowed device diagnostics, dominated by the retained device.
  const iree_hal_amd_gpu_context_t* context;
  // Existing device service handling staging and I/O continuations.
  iree_hal_memory_maintenance_t* service;
  // Queue-owned reusable CPU metadata storage.
  iree_hal_amd_gpu_capture_pool_t* captures;
  // Queue-owned host dependency associations, finalized on service.
  iree_async_semaphore_wait_context_t* waits;
  // Optional device-owned session, stable while operations are in flight.
  iree_hal_amd_gpu_profile_t* profile;
  // Exact stream index within the device-owned session.
  uint32_t queue_ordinal;
  // Size of one immutable native barrier value copied into each operation.
  // The executor defines its representation; the file pipeline never reads it.
  iree_host_size_t barrier_size;
  // Captures one transfer or, for NULL operation, a barrier. The caller has
  // resolved dependencies and prepared immutable cache recipes. Success owns
  // captured arguments through signals; failure retains no borrowed payload.
  // NULL before/after omits that boundary. Other values point to barrier_size
  // prepared bytes that remain live through this call.
  iree_status_t (*submit)(iree_hal_queue_t* queue,
                          const iree_hal_transfer_operation_t* operation,
                          iree_hal_semaphore_list_t signals, const void* before,
                          const void* after);
} iree_hal_amd_gpu_file_executor_t;

// The executor is queue-owned and outlives accepted operations.
// Captures a validated, nonempty queue_read/write transaction. Memory files
// use their owned host span directly; platform files use bounded ordinary host
// staging and their imported proactor. Device buffers need no CPU mapping.
// The before boundary precedes native buffer access. The after boundary and
// public signals follow every native copy, file request, and host effect.
// Captured buffer uses end before public signaling, permitting immediate
// source-pool destruction once completion is observed.
// Before/after each point to executor->barrier_size prepared bytes, captured
// before returning. Their representation and application belong to the
// executor. Failure retains nothing. Success transfers ownership to terminal
// signaling.
iree_status_t iree_hal_amd_gpu_file_transfer_submit(
    const iree_hal_amd_gpu_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    const void* before, const void* after);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_FILE_TRANSFER_H_
