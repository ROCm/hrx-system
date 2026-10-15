// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FILE_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FILE_TRANSFER_H_

#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native transfer admission borrowed by a cold file operation. The operation
// retains the queue's device; no worker or private execution queue is created.
typedef struct iree_hal_amd_gpu_pm4_file_executor_t {
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
  // Captures one transfer or, for NULL operation, a barrier. The caller has
  // resolved dependencies and prepared immutable cache recipes. Success owns
  // captured arguments through signals; failure retains no borrowed payload.
  iree_status_t (*submit)(iree_hal_queue_t* queue,
                          const iree_hal_transfer_operation_t* operation,
                          iree_hal_semaphore_list_t signals,
                          iree_hal_amd_gpu_pm4_barrier_t before,
                          iree_hal_amd_gpu_pm4_barrier_t after);
} iree_hal_amd_gpu_pm4_file_executor_t;

// Captures a validated, nonempty queue_read/write transaction. Memory files
// use their owned host span directly; platform files use bounded ordinary host
// staging and their imported proactor. Device buffers need no CPU mapping.
// The before boundary precedes native buffer access. The after boundary and
// public signals follow every native copy, file request, and host effect.
// Captured buffer uses end before public signaling, permitting immediate
// source-pool destruction once completion is observed.
// Failure retains nothing. Success transfers ownership to terminal signaling.
iree_status_t iree_hal_amd_gpu_pm4_file_transfer_submit(
    const iree_hal_amd_gpu_pm4_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_hal_amd_gpu_pm4_barrier_t before,
    iree_hal_amd_gpu_pm4_barrier_t after);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FILE_TRANSFER_H_
