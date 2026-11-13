// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_
#define LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_

#include "buffer_table.h"
#include "common/capture_admission.h"
#include "common/completion.h"
#include "common/event.h"
#include "common/registry.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"
#include "libhrx/src/libhrx/event_timestamp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_deferred_device_free_t
    iree_hal_streaming_deferred_device_free_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_value_wait_lane_t
    iree_hal_streaming_value_wait_lane_t;
typedef struct iree_hal_streaming_value_wait_submission_t
    iree_hal_streaming_value_wait_submission_t;

typedef iree_host_size_t iree_hal_streaming_device_ordinal_t;

//===----------------------------------------------------------------------===//
// Context types
//===----------------------------------------------------------------------===//

// Scheduling policy.
typedef enum iree_hal_streaming_scheduling_mode_e {
  // Favor active progress for low-latency stream synchronization. Callers that
  // prefer to release the CPU select YIELD or BLOCKING_SYNC explicitly.
  IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO = 0,
  // Spin wait (busy wait).
  IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN,
  // Yield to OS scheduler.
  IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD,
  // Blocking synchronization.
  IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC,
} iree_hal_streaming_scheduling_mode_t;

// Context scheduling and behavior flags.
typedef struct iree_hal_streaming_context_flags_t {
  // Scheduling policy.
  iree_hal_streaming_scheduling_mode_t scheduling_mode;

  // Memory mapping: can map host memory.
  uint64_t map_host_memory : 1;
  // Memory mapping: resize local memory to max.
  uint64_t resize_local_mem_to_max : 1;
} iree_hal_streaming_context_flags_t;

// Context resource limits.
typedef struct iree_hal_streaming_limits_t {
  size_t stack_size;                        // Stack size per GPU thread.
  size_t printf_fifo_size;                  // Printf FIFO buffer size.
  size_t malloc_heap_size;                  // Device malloc heap size.
  size_t dev_runtime_sync_depth;            // Device runtime sync depth.
  size_t dev_runtime_pending_launch_count;  // Pending launch count.
  size_t max_l2_fetch_granularity;          // L2 cache fetch granularity.
  size_t persisting_l2_cache_size;          // Persistent L2 cache size.
} iree_hal_streaming_limits_t;

// Optional test instrumentation run while the final observer completion is
// serialized with context teardown by |value_wait_lane_mutex|.
typedef void (*iree_hal_streaming_value_wait_observer_finish_hook_t)(
    void* user_data);

// Stream context mapped to HAL device.
struct iree_hal_streaming_context_t {
  // Reference counting.
  iree_atomic_ref_count_t ref_count;

  // Associated device.
  iree_hal_device_t* device;
  iree_hal_streaming_device_ordinal_t device_ordinal;
  iree_hal_streaming_device_t* device_entry;

  // Owned primary streaming queue, preferring host-local command storage.
  // Released after stream teardown and before the retained device.
  iree_hal_queue_t* queue;

  // HAL resources.
  iree_hal_allocator_t* device_allocator;
  iree_status_t loop_status;

  // Facts converting the ticks this context's event records capture, or a
  // zeroed domain when the device advertises none. Constant for the context's
  // life: the device spec is immutable.
  hrx_event_timestamp_domain_t timestamp_domain;
  // Suballocator for the tick slots this context's event records write into.
  // Unused, and never grown, when |timestamp_domain| is zeroed.
  hrx_event_timestamp_pool_t timestamp_pool;

  // Encoded context flags, accessed atomically through the flag accessors.
  iree_atomic_uint32_t flags;

  // Default stream for this context (always created during context
  // initialization).
  iree_hal_streaming_stream_t* default_stream;

  // Next non-zero stream capture identifier assigned under |stream_list_mutex|.
  unsigned long long next_capture_id;

  // Peer access list.
  iree_hal_streaming_context_t** peer_contexts;
  iree_host_size_t peer_count;
  iree_host_size_t peer_capacity;

  // Buffer mapping table (pyre unified implementation).
  hrx_buffer_table_t buffer_table;

  // Stream-ordered frees available for dependency-aware reuse in this context.
  // Protected by |pending_free_mutex|.
  iree_hal_streaming_deferred_device_free_t* pending_free_head;

  // Serializes access to |pending_free_head| and terminal free callbacks.
  iree_slim_mutex_t pending_free_mutex;

  // Number of streams in this context with capture state other than NONE.
  iree_atomic_int32_t capture_stream_count;
  // Coordinates capture-state transitions with operations whose stream
  // ordering and capture disposition must be decided as one transaction.
  iree_hal_streaming_capture_admission_t capture_admission;

  // Idle exact queues available for an atomic wait submission.
  iree_hal_streaming_value_wait_lane_t* idle_value_wait_lanes;
  // Number of lanes in |idle_value_wait_lanes|.
  iree_host_size_t idle_value_wait_lane_count;
  // Exact queues still occupied by accepted atomic wait submissions.
  iree_hal_streaming_value_wait_lane_t* pending_value_wait_lanes;
  // Completion observers that have not yet delivered their final callback.
  iree_hal_streaming_value_wait_submission_t* active_value_wait_observers;
  // Rejected records whose observers completed during context shutdown.
  iree_hal_streaming_value_wait_submission_t* shutdown_value_wait_submissions;
  // Number of completion observers that have not entered their final
  // mutex-serialized completion. A zero predicate is valid only while holding
  // |value_wait_lane_mutex|, after the final notification post has returned.
  iree_atomic_int32_t active_value_wait_observer_count;
  // Wakes context teardown when observer callbacks finish.
  iree_notification_t value_wait_observer_notification;
  // Optional test instrumentation invoked after decrementing the active
  // observer count and before posting the notification. Guarded by
  // |value_wait_lane_mutex|.
  iree_hal_streaming_value_wait_observer_finish_hook_t
      value_wait_observer_finish_hook;
  void* value_wait_observer_finish_hook_user_data;
  // True once teardown has forbidden publication and begun cancelling
  // observers. Guarded by |value_wait_lane_mutex|.
  bool value_wait_lanes_shutting_down;
  // Diagnostic counters used to enforce bounded reclamation. Live includes
  // every published heap record until it is reclaimed, including terminal
  // failed records retained for queue-first teardown. Guarded by
  // |value_wait_lane_mutex|.
  iree_host_size_t live_value_wait_submission_count;
  iree_host_size_t peak_value_wait_submission_count;
  uint64_t value_wait_completion_query_count;
  uint64_t value_wait_record_visit_count;
  uint64_t value_wait_observer_removal_count;
  // Guards both value-wait lane lists, their completion records, observer
  // ownership, shutdown state, and the diagnostic counters above.
  iree_slim_mutex_t value_wait_lane_mutex;

  // Context resource limits.
  iree_hal_streaming_limits_t limits;

  // Synchronization.
  iree_slim_mutex_t mutex;

  // Host allocator.
  iree_allocator_t host_allocator;

  // Streams retained by the context until explicitly unregistered. Streams
  // retain no context reference, so this ownership is acyclic.
  iree_hal_streaming_stream_t** streams;
  // Number of retained streams in |streams|.
  iree_host_size_t stream_count;
  // Number of allocated entries in |streams|.
  iree_host_size_t stream_capacity;

  // Outstanding context wait timepoints inherited by newly registered
  // streams. Immutable once published and guarded by |stream_list_mutex|.
  iree_hal_fence_t* stream_wait_frontier;

  // Dedicated mutex for stream list access.
  iree_slim_mutex_t stream_list_mutex;

  // Timeline covering context-wide event records submitted on behalf of this
  // context and every binding scheduling domain layered over it.
  iree_hal_streaming_operation_timeline_t event_record_timeline;
  // Lazily acquired queue for context-wide event joins, owned until context
  // destruction. Independent of stream producers; guarded by
  // event_record_mutex.
  iree_hal_queue_t* event_record_queue;
  // Serializes event record submission and |event_record_timeline| updates.
  iree_slim_mutex_t event_record_mutex;

  // Lazily selected bulk queue, owned until joined context teardown. Context
  // copies and shared-primary streams use it with their existing ordering.
  iree_hal_queue_t* transfer_queue;
  // Guards construction and borrowing of |transfer_queue|. Native acquisition
  // happens outside stream locks; execution callbacks never take this mutex.
  iree_slim_mutex_t transfer_queue_mutex;

  // Global context list node pointers for cleanup tracking.
  // These are used to link all contexts in a global list for proper cleanup.
  // Guarded by the context list mutex.
  struct {
    iree_hal_streaming_context_t* next;
    iree_hal_streaming_context_t* prev;
  } context_list_entry;

  // Symbol map for compiler-generated host registration functions. Lazily
  // initialized on first use. Explicit module-management paths bypass it.
  iree_hal_streaming_context_symbol_map_t symbol_map;
};

static inline bool iree_hal_streaming_context_has_capture_streams(
    const iree_hal_streaming_context_t* context) {
  return iree_atomic_load(&context->capture_stream_count,
                          iree_memory_order_acquire) > 0;
}

static inline void iree_hal_streaming_context_enter_capture(
    iree_hal_streaming_context_t* context) {
  iree_atomic_fetch_add(&context->capture_stream_count, 1,
                        iree_memory_order_acq_rel);
}

static inline void iree_hal_streaming_context_leave_capture(
    iree_hal_streaming_context_t* context) {
  iree_atomic_fetch_sub(&context->capture_stream_count, 1,
                        iree_memory_order_acq_rel);
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Synchronization: none (creates new context).
iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context);

// Synchronization: none (reference counting).
void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context);
void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context);

// Attempts to form a reference without resurrecting a context whose final
// release has begun. Returns false when the reference count has reached zero.
bool iree_hal_streaming_context_try_retain(
    iree_hal_streaming_context_t* context);

// Loads one coherent snapshot of the caller-selected context flags.
iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context);

// Publishes caller-selected context flags atomically for subsequent waits.
void iree_hal_streaming_context_set_flags(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_flags_t flags);

// Resolves the current scheduling policy to the HAL's progress preference.
// The backend owns progression; this does not change completion guarantees.
iree_async_wait_flags_t iree_hal_streaming_context_wait_flags(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local access).
uintptr_t iree_hal_streaming_current_thread_token(void);

// Adopts a current context after registering thread-exit cleanup. A native
// thread-storage failure preserves the previous context. NULL clears it.
// Synchronization: none (thread-local modification).
iree_status_t iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context);

// Clears the current context without changing stacked context ownership.
void iree_hal_streaming_context_clear_current(void);

// Releases the calling thread's context references and stack storage. Used as
// the registry-owned TLS destructor and before global registry teardown.
void iree_hal_streaming_context_cleanup_thread(void* state);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context);

// Synchronization: none (thread-local stack operation).
iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context);

// Limit types for context resource limits.
typedef enum iree_hal_streaming_context_limit_e {
  IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE = 0,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY,
  IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE,
} iree_hal_streaming_context_limit_t;

// Synchronization: none (queries limit value).
iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value);

// Synchronization: none (sets limit value).
iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value);

// Synchronization: none (configures peer access).
iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Synchronization: none (disables peer access).
iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context);

// Registers a stream and retains it for the context's stream list. Callers
// must hold a reference to |context| across the call: context destruction
// zeroes the count under the list mutex and then walks the emptied extent and
// frees the array without holding it. A registration landing in that window
// writes into the array that walk is reading and about to free, and the
// reference it takes for the list outlives both. Any outstanding context-wide
// event waits are appended to |stream| before the call succeeds. A failure
// leaves the stream unregistered.
// Synchronization: thread-safe internal locking.
iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Takes a retained snapshot of all streams currently registered with
// |context|. The caller must release the snapshot with
// iree_hal_streaming_context_release_stream_snapshot. Both outputs are
// unchanged on failure.
// Synchronization: thread-safe internal locking.
iree_status_t iree_hal_streaming_context_snapshot_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams,
    iree_host_size_t* out_stream_count);

// Releases every retained stream in |streams| and frees the snapshot storage.
void iree_hal_streaming_context_release_stream_snapshot(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** streams, iree_host_size_t stream_count);

// Removes a registered stream and releases the stream-list reference. The
// stream's context pointer remains valid until its final release because every
// operation that can outlive removal retains the context independently. A
// missing stream is a no-op; public handle validity is owned by the binding's
// handle registry rather than this ownership list.
// Synchronization: thread-safe internal locking.
void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Extends immutable |previous_frontier| with the semaphore point
// (|semaphore|, |value|), pruning points that have already completed.
// |out_frontier| is unchanged on failure.
iree_status_t iree_hal_streaming_wait_frontier_extend(
    iree_hal_fence_t* previous_frontier, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_frontier);

// Records |event| after the captured tails of all streams currently registered
// with |context|. Each stream is flushed before its tail is captured and the
// fan-in record is submitted directly to the context's primary queue. The
// event must have been created by |context|.
iree_status_t iree_hal_streaming_context_record_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event);

// Orders all current and future streams registered with |context| after the
// point currently recorded on |event|. Current streams receive device-side
// barriers and later registrations inherit an immutable pending frontier; the
// call does not wait for host-visible completion. Events from other contexts
// and devices are accepted when the destination queues support their
// semaphores.
iree_status_t iree_hal_streaming_context_wait_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event);

iree_status_t iree_hal_streaming_context_allocate_capture_id(
    iree_hal_streaming_context_t* context, unsigned long long* out_capture_id);

// Returns true when another context is present in the global context list.
bool iree_hal_streaming_context_has_peer_contexts(
    iree_hal_streaming_context_t* context);

// Waits for all streams in the context to become idle.
// Synchronization: all streams in context (blocking wait).
iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout);

// Flushes pending command buffers in all streams in the context without
// waiting for completion.
iree_status_t iree_hal_streaming_context_flush(
    iree_hal_streaming_context_t* context);

// Flushes pending command buffers in every active context without waiting for
// completion.
iree_status_t iree_hal_streaming_context_flush_all(void);

// Synchronization: all streams (blocks until all streams idle).
// This flushes and waits for all streams and context-wide event records on the
// device.
iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context);

// Waits for every context-wide event record accepted before this call's
// internal timeline snapshot. Used by binding scheduling domains whose stream
// membership differs from the common context while sharing its primary scope.
iree_status_t iree_hal_streaming_context_synchronize_event_records(
    iree_hal_streaming_context_t* context);

// Synchronizes streams that participate in legacy default stream ordering.
// Non-blocking streams are excluded. The legacy default stream itself is always
// synchronized.
iree_status_t iree_hal_streaming_context_synchronize_legacy_default(
    iree_hal_streaming_context_t* context);

// Orders future work on |stream| after work already enqueued on each blocking,
// non-capturing stream in the context. Null entries, the legacy default stream,
// |stream| itself, and non-blocking or capturing streams are excluded.
iree_status_t iree_hal_streaming_context_wait_blocking_streams(
    iree_hal_streaming_context_t* context, iree_hal_streaming_stream_t* stream);

// Queries whether any stream participating in legacy default-stream ordering
// still has queued work. Non-blocking streams are excluded. On failure
// |out_completion_state| is unchanged.
iree_status_t iree_hal_streaming_context_query(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_completion_state_t* out_completion_state);

// Returns the current thread's borrowed streaming context, or NULL when no
// context has been selected.
iree_hal_streaming_context_t* iree_hal_streaming_context_current(void);

// Flushes and waits for every stream in every active context. Destructive
// operations use this when device pointers may be hidden from host-side
// resource tracking.
iree_status_t iree_hal_streaming_context_synchronize_all(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_
