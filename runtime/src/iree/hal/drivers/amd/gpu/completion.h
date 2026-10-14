// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// On success, consumes a bounded batch of authoritative completion records.
// Returns true only when another batch is immediately ready; outstanding GPU
// work alone is not ready work. Native events are hints, never completion
// facts. The callback runs only on the completion worker, without a
// worker-owned lock.
//
// A native wait failure is delivered once to every source before worker exit.
// The source records terminal queue failure; its return value is then ignored.
// Failure does not establish native quiescence or permit releasing GPU storage.
typedef bool (*iree_hal_amd_gpu_completion_fn_t)(void* user_data,
                                                 amdf_status_t wait_status);

// One queue's protocol-specific completion observation and host retirement.
// Sources are immutable and borrowed until completion deinitialization joins.
typedef struct iree_hal_amd_gpu_completion_source_t {
  // Bounded retirement or terminal native-failure delivery.
  iree_hal_amd_gpu_completion_fn_t fn;
  // Borrowed queue state, live through the callback's final access.
  void* user_data;
} iree_hal_amd_gpu_completion_source_t;

// One high-priority completion worker per device. Its fixed source array has
// no submission-time activation or registration traffic. The worker waits
// directly on the context's native event; GPU producers and host stop signals
// share that event. No timer, polling thread, or per-queue worker is created.
// The embedding owner joins accepted work before stopping this worker.
typedef struct iree_hal_amd_gpu_completion_t {
  // Borrowed context whose event and API table outlive the worker join.
  const iree_hal_amd_gpu_context_t* context;
  // Number of immutable provisioned queue sources.
  iree_host_size_t source_count;
  // Borrowed contiguous source array, live through the worker join.
  const iree_hal_amd_gpu_completion_source_t* sources;
  // Owned thread, or NULL after failed initialization or a successful join.
  iree_thread_t* thread;
  // Running, stop-requested, or stopped state shared only with teardown.
  iree_atomic_int32_t state;
} iree_hal_amd_gpu_completion_t;

// Starts the sole native event waiter with an explicit affinity and high
// priority hint. Sources must be ready before entry; callbacks may run before
// this returns. Allocation and thread startup are cold, fallible operations.
// Failure runs no callback and leaves out_completion safe to deinitialize.
iree_status_t iree_hal_amd_gpu_completion_initialize(
    const iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    const iree_hal_amd_gpu_completion_source_t* sources,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion);

// Stops and joins the final source access. Called by one external owner after
// accepted work and producers have joined, never from a source callback. This
// does not wait for GPU work or consume notification tails on the owner's
// behalf. A failed native stop wake returns an error and preserves the entire
// live owner for retry. A successful return permits releasing borrowed source
// state and the context, subject to their separate native producer lifetime
// joins.
iree_status_t iree_hal_amd_gpu_completion_deinitialize(
    iree_hal_amd_gpu_completion_t* completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_
