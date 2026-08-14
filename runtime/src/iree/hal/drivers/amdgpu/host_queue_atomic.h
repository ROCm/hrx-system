// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_ATOMIC_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Atomic operation kind carried by direct and deferred host queue submissions.
typedef enum iree_hal_amdgpu_host_queue_atomic_operation_kind_e {
  IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT = 0,
  IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE = 1,
  IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW = 2,
} iree_hal_amdgpu_host_queue_atomic_operation_kind_t;

// Compact payload for one direct queue atomic operation.
typedef struct iree_hal_amdgpu_host_queue_atomic_operation_t {
  // Target buffer retained until the operation completes.
  iree_hal_buffer_t* target_buffer;
  // Target byte offset within |target_buffer|.
  iree_device_size_t target_offset;
  // Operation kind selecting the active parameter union member.
  iree_hal_amdgpu_host_queue_atomic_operation_kind_t kind;
  union {
    // Parameters for IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_WAIT.
    iree_hal_atomic_wait_params_t wait;
    // Parameters for IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_STORE.
    iree_hal_atomic_store_params_t store;
    // Parameters for IREE_HAL_AMDGPU_HOST_QUEUE_ATOMIC_OPERATION_RMW.
    iree_hal_atomic_rmw_params_t rmw;
  } params;
} iree_hal_amdgpu_host_queue_atomic_operation_t;

// Submits an atomic operation after wait resolution. Caller must hold
// queue->locks.submission_mutex. If temporary queue capacity is unavailable,
// |out_ready| is false and no queue state or ownership is mutated.
iree_status_t iree_hal_amdgpu_host_queue_submit_atomic(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Captures an atomic operation for later issue. Caller must hold
// queue->locks.submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_defer_atomic(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_atomic_operation_t* operation,
    iree_hal_amdgpu_pending_op_t** out_op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_ATOMIC_H_
