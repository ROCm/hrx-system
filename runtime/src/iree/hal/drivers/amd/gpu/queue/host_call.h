// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_

#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed native queue services. Entry retirement invokes the host callback
// after native execution and cache effects. Submission hints never invoke it.
// Accepted calls borrow their parent device through the invocation's last use.
typedef struct iree_hal_amd_gpu_host_call_executor_t {
  // Exact queue supplying native dependency and cache boundaries.
  iree_hal_queue_t* queue;
  // Borrowed diagnostics, dominated by the application's device ownership.
  const iree_hal_amd_gpu_context_t* context;
  // Queue-owned reusable CPU capture storage.
  iree_hal_amd_gpu_capture_pool_t* captures;
  // Marks accepted host work in the queue's existing shutdown accounting.
  void (*begin)(iree_hal_queue_t* queue);
  // Returns that borrow after the final capture access and wakes shutdown.
  void (*end)(iree_hal_queue_t* queue);

  // Native entry boundary, observed without an intermediate semaphore.
  // Rejection retains no operands and invokes no callback.
  iree_status_t (*enter)(iree_hal_queue_t* queue,
                         iree_hal_semaphore_list_t waits,
                         iree_hal_amd_gpu_retirement_callback_t callback,
                         const iree_hal_queue_barriers_t* barriers,
                         iree_host_size_t public_signal_count);
} iree_hal_amd_gpu_host_call_executor_t;

// The executor is queue-owned and outlives accepted operations.
// Captures callback state and invokes it on the existing completion worker
// after a native release boundary. DEFERRED callbacks take ownership of the
// original public output list. Later queue operations own their visibility
// boundaries; there is no additional return-to-device submission.
// NON_BLOCKING publishes at invocation and promises no ordering against later
// callback effects. Failure retains nothing and invokes no callback.
iree_status_t iree_hal_amd_gpu_host_call_submit(
    const iree_hal_amd_gpu_host_call_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_host_call_t call, const uint64_t arguments[4],
    iree_hal_host_call_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_HOST_CALL_H_
