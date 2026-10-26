// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_STREAM_H_
#define LIBHRX_SRC_BINDING_HIP_STREAM_H_

#include "binding/hip/api.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

// Binding-private ownership and lifetime state behind a public HIP stream
// handle. The common stream remains the execution engine while this object
// provides a stable API identity that can outlive execution-context teardown.
struct hipStream_st {
  // Reference count including the live public-handle ownership.
  iree_atomic_ref_count_t ref_count;

  // Serializes access to attachment and execution-context membership.
  iree_slim_mutex_t mutex;

  // Common stream owning the exact HAL queue, or NULL after detachment.
  iree_hal_streaming_stream_t* stream;

  // Execution context whose membership owns one reference, or NULL.
  hipExecutionCtx_t execution_context;

  // Next stream in the execution context membership list.
  hipStream_t next_execution_context_stream;

  // Host allocator owning this handle allocation.
  iree_allocator_t host_allocator;
};

// Maps the sorted HAL scheduling-priority levels accepted by |queue_family|
// into a dense HIP range centered on normal priority. Either output may be
// NULL.
void iree_hip_queue_family_priority_range(
    const iree_hal_queue_family_t* queue_family, int* out_least_priority,
    int* out_greatest_priority);

// Selects the exact HAL scheduling-priority level corresponding to the
// clamped dense HIP |requested_priority|. Returns the canonical HIP priority
// in |out_hip_priority| when non-NULL.
iree_hal_queue_priority_t iree_hip_queue_family_select_priority(
    const iree_hal_queue_family_t* queue_family, int requested_priority,
    int* out_hip_priority);

// Creates and publishes a HIP stream that submits through the exact hardware
// |queue|. The common stream retains |queue| and records the clamped HIP
// |priority| for compatibility queries. |initial_wait_semaphores| orders the
// stream before its public handle becomes visible. |out_handle| is unchanged
// on failure.
iree_status_t iree_hip_stream_create(
    iree_hal_streaming_context_t* context, iree_hal_queue_t* queue,
    unsigned int flags, int priority,
    iree_hal_semaphore_list_t initial_wait_semaphores, hipStream_t* out_handle);

// Publishes |stream| as an opaque HIP stream handle. On success the handle
// assumes ownership of the caller's stream reference and |out_handle| receives
// its public identity. On failure ownership remains with the caller and
// |out_handle| is unchanged.
iree_status_t iree_hip_stream_publish(iree_hal_streaming_stream_t* stream,
                                      iree_allocator_t host_allocator,
                                      hipStream_t* out_handle);

// Looks up and retains a live explicit stream handle. |out_handle| is unchanged
// when the handle is not registered.
bool iree_hip_stream_lookup_retain(hipStream_t handle, hipStream_t* out_handle);

// Removes a live explicit stream handle and transfers its public ownership to
// the caller. |out_handle| is unchanged when the handle is not registered.
bool iree_hip_stream_take(hipStream_t handle, hipStream_t* out_handle);

// Retains the common stream and its context while |handle| remains attached.
// Both outputs are unchanged when the stream has been detached by context
// destruction.
bool iree_hip_stream_retain_attached(
    hipStream_t handle, iree_hal_streaming_stream_t** out_stream,
    iree_hal_streaming_context_t** out_context);

// Detaches |handle| and transfers its owning common-stream reference to the
// caller. When the common context remains live, |out_context| receives a
// retained reference; otherwise it receives NULL. Both outputs are unchanged
// when |handle| was already detached.
bool iree_hip_stream_detach(hipStream_t handle,
                            iree_hal_streaming_stream_t** out_stream,
                            iree_hal_streaming_context_t** out_context);

// Retains a HIP stream handle for the caller.
void iree_hip_stream_retain(hipStream_t handle);

// Releases a retained or transferred HIP stream handle.
void iree_hip_stream_release(hipStream_t handle);

// Reclaims empty public-handle tables after runtime teardown.
void iree_hip_stream_trim(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_STREAM_H_
