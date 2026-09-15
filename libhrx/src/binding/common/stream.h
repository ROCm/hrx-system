// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_STREAM_H_
#define IREE_EXPERIMENTAL_STREAMING_STREAM_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;

// Dispatch flags for kernel launches.
typedef enum iree_hal_streaming_dispatch_flag_bits_e {
  IREE_HAL_STREAMING_DISPATCH_FLAG_NONE = 0ull,
  // Launches a cooperative grid whose workgroups may synchronize globally.
  IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE = 1ull << 0,
  // Treats the parameter buffer as an array of pointers to argument values.
  IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY = 1ull << 1,
  // Treats the parameter buffer as a target-native argument byte image.
  IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED = 1ull << 2,
} iree_hal_streaming_dispatch_flags_t;

// Parameters describing one kernel launch.
typedef struct iree_hal_streaming_dispatch_params_t {
  // Grid dimensions measured in workgroups.
  uint32_t grid_dim[3];
  // Workgroup dimensions measured in workitems.
  uint32_t block_dim[3];
  // Dynamic workgroup-local memory available to each workgroup, in bytes.
  uint32_t shared_memory_bytes;
  // Parameter storage interpreted according to |flags|.
  void* buffer;
  // Length of |buffer| in bytes when it contains a packed byte image.
  size_t buffer_size;
  // Flags controlling parameter interpretation and dispatch behavior.
  iree_hal_streaming_dispatch_flags_t flags;
  // Exact workitem dimensions, or zeroes when every workgroup is full.
  uint32_t workitem_count[3];
} iree_hal_streaming_dispatch_params_t;

// One member of a kernel launch batch.
typedef struct iree_hal_streaming_kernel_launch_t {
  // Function symbol dispatched by this launch.
  iree_hal_streaming_symbol_t* symbol;
  // Launch parameters copied into the batch record.
  iree_hal_streaming_dispatch_params_t params;
  // Stream receiving the dispatch.
  iree_hal_streaming_stream_t* stream;
} iree_hal_streaming_kernel_launch_t;

// Selects the exact hardware queue on which a cooperative operation submitted
// to |stream| must execute. Lazily acquires and retains a queue with the same
// family, priority, and execution-resource set as the stream's ordinary queue.
// The returned pointer is borrowed from |stream| and unchanged on failure.
//
// Synchronization: caller must hold |stream->mutex|.
IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_stream_select_cooperative_queue_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t** out_queue);

// Retains the stream's context for one operation. Returns false after context
// teardown has detached the stream. The caller releases |*out_context|.
bool iree_hal_streaming_stream_retain_context(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_context_t** out_context);

// Orders future work on |stream| after work already enqueued on
// |source_stream|. Both streams are flushed on the calling thread, but the
// dependency itself is submitted to the device queue without waiting for
// completion. Both streams must belong to one context and must not be
// capturing.
iree_status_t iree_hal_streaming_stream_wait_stream(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* source_stream);

// Orders future work on |stream| after all work already enqueued on |sources|
// with one queue barrier. Source streams are flushed before their timeline
// points are captured; |stream| is flushed before the barrier is appended. All
// streams must belong to one context and must not be capturing.
iree_status_t iree_hal_streaming_stream_wait_streams(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* const* sources, iree_host_size_t source_count);

// Orders future work on |stream| after all |wait_semaphores| with one queue
// barrier. The stream is flushed before the barrier is appended and must not
// be capturing. The semaphore list is borrowed for the duration of the call.
iree_status_t iree_hal_streaming_stream_wait_semaphores(
    iree_hal_streaming_stream_t* stream,
    iree_hal_semaphore_list_t wait_semaphores);

// Enqueues a HAL host call at the current stream timeline point.
// Synchronization: flushes pending stream commands before enqueueing.
iree_status_t iree_hal_streaming_queue_host_call(
    iree_hal_streaming_stream_t* stream, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags);

// Enqueues one kernel launch on |stream| without waiting for completion.
// Pointer-array arguments are copied into an owned native argument image before
// the function returns. Graph capture is supported by the single-launch path.
iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream);

// Enqueues all |launches| as one host-side transaction. All pointer-array
// arguments are copied before any stream is mutated, then every participating
// stream is locked in stable identifier order until all dispatches have been
// appended. Other threads therefore cannot interleave stream operations inside
// the launch set. Every successfully recorded stream is submitted before the
// call returns, without waiting for completion. Repeated streams and graph
// capture are not supported, and each member must use pointer-array arguments.
iree_status_t iree_hal_streaming_launch_kernel_batch(
    iree_host_size_t launch_count,
    const iree_hal_streaming_kernel_launch_t* launches);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_STREAM_H_
