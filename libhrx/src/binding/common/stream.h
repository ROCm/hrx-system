// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_STREAM_H_
#define IREE_EXPERIMENTAL_STREAMING_STREAM_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;
typedef struct iree_hal_streaming_graph_t iree_hal_streaming_graph_t;
typedef struct iree_hal_streaming_graph_node_t iree_hal_streaming_graph_node_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;
typedef struct iree_hal_streaming_value_flush_timer_t
    iree_hal_streaming_value_flush_timer_t;
typedef struct iree_hal_streaming_value_wait_lane_t
    iree_hal_streaming_value_wait_lane_t;

typedef enum iree_hal_streaming_stream_flag_bits_e {
  IREE_HAL_STREAMING_STREAM_FLAG_NONE = 0ull,
  IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING = 1ull << 0,
} iree_hal_streaming_stream_flags_t;

// Stream membership lifecycle in its parent context registry.
typedef enum iree_hal_streaming_stream_registration_state_e {
  IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED = 0,
  IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED = 1,
  IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_DETACHING = 2,
} iree_hal_streaming_stream_registration_state_t;

// Stream capture status enum.
typedef enum iree_hal_streaming_capture_status_e {
  IREE_HAL_STREAMING_CAPTURE_STATUS_NONE = 0,
  IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE = 1,
  IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED = 2,
} iree_hal_streaming_capture_status_t;

// Stream capture mode.
typedef enum iree_hal_streaming_capture_mode_e {
  IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL = 0,
  IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL = 1,
  IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED = 2,
} iree_hal_streaming_capture_mode_t;

// Stream capture dependencies update mode.
typedef enum iree_hal_streaming_capture_dependencies_mode_e {
  // Replace the current dependencies with new ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET = 0,
  // Add new dependencies to existing ones.
  IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD = 1,
} iree_hal_streaming_capture_dependencies_mode_t;

// A source-stream timeline point that orders all later work on a stream.
typedef struct iree_hal_streaming_memory_reuse_dependency_t {
  // Stable identifier of the source stream that recorded the event.
  unsigned long long source_stream_id;
  // Source timeline value the event is known to follow.
  uint64_t source_timeline_value;
} iree_hal_streaming_memory_reuse_dependency_t;

// Stream for asynchronous execution.
typedef struct iree_hal_streaming_stream_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Parent context, unowned to keep stream/context ownership acyclic. Access is
  // serialized by |mutex| and operations retain it with
  // iree_hal_streaming_stream_retain_context before dereferencing it.
  iree_hal_streaming_context_t* context;
  // Membership in |context->streams|. REGISTERED -> DETACHING ->
  // UNREGISTERED transitions are serialized by context stream-list -> stream
  // locking, nested below the capture graph mutex when a session is active.
  iree_hal_streaming_stream_registration_state_t registration_state;

  // HIP stream creation flags.
  iree_hal_streaming_stream_flags_t flags;
  // HIP stream scheduling priority hint.
  int priority;
  // Stable process-wide stream identifier used by timeline dependencies.
  unsigned long long stream_id;

  // Command buffer for batching operations.
  iree_hal_command_buffer_t* command_buffer;
  // Outstanding bounded flush for a write-only value-operation batch, or NULL.
  // Protected by |mutex|. The timer owns a stream reference until its callback
  // clears this field, so stream destruction cannot race the callback.
  iree_hal_streaming_value_flush_timer_t* value_flush_timer;
  // Number of kernel launches recorded in |command_buffer|.
  uint32_t pending_launch_count;

  // Semaphore chain for synchronization.
  iree_hal_semaphore_t* timeline_semaphore;
  uint64_t pending_value;    // Last value a submission has been accepted for.
  uint64_t completed_value;  // Last value we've verified as completed

  // Exact hardware queue retained while the stream remains attached to its
  // context.
  iree_hal_queue_t* queue;

  // Lazily acquired cooperative realization of |queue| retaining its exact
  // family, priority, and execution-resource set. NULL until first use.
  iree_hal_queue_t* cooperative_queue;

  // Context-owned exclusive queue while native waits or their dependent work
  // remain in flight. Access and completion detachment hold |mutex|.
  iree_hal_streaming_value_wait_lane_t* value_wait_lane;

  // Event dependencies that establish safe cross-stream allocation reuse.
  iree_hal_streaming_memory_reuse_dependency_t* memory_reuse_dependencies;
  // Number of valid entries in |memory_reuse_dependencies|.
  iree_host_size_t memory_reuse_dependency_count;
  // Allocated entry capacity of |memory_reuse_dependencies|.
  iree_host_size_t memory_reuse_dependency_capacity;

  // Stream capture state.
  iree_hal_streaming_capture_status_t capture_status;
  iree_hal_streaming_capture_mode_t capture_mode;
  iree_hal_streaming_graph_t* capture_graph;
  // True when |capture_graph| is retained by this stream and must be released.
  bool capture_graph_owned;
  // True when this stream began the capture and is allowed to end it.
  bool capture_origin;
  unsigned long long capture_id;
  // Host thread that began this capture sequence.
  uintptr_t capture_owner_thread_id;
  iree_hal_streaming_graph_node_t** capture_dependencies;
  iree_host_size_t capture_dependency_count;
  iree_host_size_t capture_dependency_capacity;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;
} iree_hal_streaming_stream_t;

// Reserves the next value on |stream|'s timeline for one submission. Callers
// must hold |stream->mutex| and publish |*out_signal_value| to
// |stream->pending_value| only once the submission is accepted, so a rejected
// submission leaves the timeline where it was and hands the value out again.
//
// A value must name exactly one submission, and nothing catches a violation:
// queues publish their completions with a duplicate-tolerant advance, so the
// second submission's signal is a silent no-op and the timeline reaches the
// value when the first submission completes. Every reader treats the timeline
// reaching a value as "the submission that signals it has completed", so all of
// them report completion while the second submission is still running.
//
// |*out_wait_value| is the value the submission must wait on to stay behind the
// work in front of it, or 0 when the stream has never submitted, in which case
// callers drop the wait rather than waiting on value zero.
static inline iree_status_t iree_hal_streaming_stream_reserve_next_value_locked(
    iree_hal_streaming_stream_t* stream, uint64_t* out_wait_value,
    uint64_t* out_signal_value) {
  const uint64_t wait_value = stream->pending_value;
  if (IREE_UNLIKELY(wait_value >= IREE_HAL_SEMAPHORE_MAX_VALUE)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream timeline value overflow");
  }
  *out_wait_value = wait_value;
  *out_signal_value = wait_value + 1;
  return iree_ok_status();
}

// Updates capture status while keeping the owning context's capture-stream
// count in sync. Callers serialize access to the stream capture fields.
void iree_hal_streaming_stream_set_capture_status(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_capture_status_t new_status);

// Creates a stream that submits through the exact hardware |queue|. The stream
// retains the queue until it is detached from |context|.
// Synchronization: none (creates new stream).
iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context, iree_hal_queue_t* queue,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream);

// Synchronization: none (reference counting).
void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream);
void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream);

// Begins command buffer recording.
// Synchronization: none (begins recording).
iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream);

// Ensures a stream command buffer is recording while the caller holds
// stream->mutex. Use this when appending commands under the stream lock.
iree_status_t iree_hal_streaming_stream_begin_locked(
    iree_hal_streaming_stream_t* stream);

// Flushes pending commands.
// Synchronization: none (submits to queue, non-blocking).
iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream);

// Synchronization: none (queries stream status, non-blocking).
iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream, int* status);

// Synchronization: stream (blocks until stream idle).
iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream);
// Synchronizes stream work that the caller has already flushed/submitted.
iree_status_t iree_hal_streaming_stream_synchronize_flushed(
    iree_hal_streaming_stream_t* stream);

// Wait for already-submitted work on this stream to complete.
// Does NOT flush in-progress recordings - safe to call from other threads.
iree_status_t iree_hal_streaming_stream_wait_submitted(
    iree_hal_streaming_stream_t* stream);

// Waits for an event on a stream.
// Synchronization: none (enqueues wait operation, non-blocking).
iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    bool capture_external_wait);

// Returns whether work on |stream| is ordered after |source_timeline_value|
// from the stream identified by |source_stream_id|.
bool iree_hal_streaming_stream_has_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value);

// Synchronous host work executed at a reserved stream timeline point.
typedef iree_status_t (*iree_hal_streaming_host_operation_fn_t)(
    void* user_data);

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
  // Opaque function token preserved for binding-level graph parameter queries.
  void* binding_function;
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

// Submits the stream's pending command buffer, if any.
// Synchronization: caller must hold |stream->mutex|.
IREE_MUST_USE_RESULT iree_status_t
iree_hal_streaming_stream_flush_locked(iree_hal_streaming_stream_t* stream);

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

// Launches a host function on the stream.
// The function will be called with user_data when the stream reaches this
// point. The stream will be flushed before enqueueing the host call to ensure
// proper ordering with device operations.
// Synchronization: stream flush (flushes stream before enqueue).
iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data);

// Executes blocking host work in stream order. The caller waits for all prior
// work, invokes |fn|, and publishes success or failure to a timeline point
// reserved before the wait. Concurrent later submissions therefore remain
// ordered after the operation. Intended only for cold fallback paths that
// cannot be represented by one device queue. |fn| executes without the stream
// mutex held and must not submit to or wait on |stream| because its reserved
// point remains unsignaled until |fn| returns.
iree_status_t iree_hal_streaming_execute_host_operation(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_host_operation_fn_t fn, void* user_data);

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
