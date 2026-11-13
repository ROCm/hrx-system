// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_STREAM_QUEUE_H_
#define LIBHRX_SRC_BINDING_COMMON_STREAM_QUEUE_H_

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;

// Lifecycle of one asynchronous completion record for an accepted value-wait
// submission.
typedef enum iree_hal_streaming_value_wait_submission_state_e {
  // The completion observer is active but queue acceptance is not yet known.
  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PREPARED = 0,
  // The queue operation was accepted and this record is owned by its lane.
  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED = 1,
  // The queue operation was rejected and no lane owns this record.
  IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_REJECTED = 2,
} iree_hal_streaming_value_wait_submission_state_t;

typedef struct iree_hal_streaming_value_wait_lane_t
    iree_hal_streaming_value_wait_lane_t;

// Terminal record for one accepted submission on a value-wait lane. Each
// submission owns an independent semaphore so a later submission failure
// cannot poison the completion proof for earlier blocked work on the queue.
typedef struct iree_hal_streaming_value_wait_submission_t {
  // Next accepted submission in lane order.
  struct iree_hal_streaming_value_wait_submission_t* next;
  // Previous accepted submission in lane order.
  struct iree_hal_streaming_value_wait_submission_t* prev;
  // Next completion observer registered in the context.
  struct iree_hal_streaming_value_wait_submission_t* observer_next;
  // Previous completion observer registered in the context.
  struct iree_hal_streaming_value_wait_submission_t* observer_prev;
  // Semaphore that becomes terminal only with this exact submission.
  iree_hal_semaphore_t* completion_semaphore;
  // Context borrowed while the record is prepared or published.
  iree_hal_streaming_context_t* context;
  // Stream retained until observer retirement, or NULL for standalone records.
  iree_hal_streaming_stream_t* stream;
  // Lane borrowed while the record is prepared or published.
  iree_hal_streaming_value_wait_lane_t* lane;
  // Proactor running the preaccepted asynchronous observer.
  iree_async_proactor_t* observer_proactor;
  // Wait operation submitted to |observer_proactor|.
  iree_async_semaphore_wait_operation_t observer_operation;
  // Async adapter for |completion_semaphore|.
  iree_async_semaphore_t* observer_semaphore;
  // Value on |observer_semaphore| that proves terminal completion.
  uint64_t observer_value;
  // Publication state guarded by the context value-wait lane mutex.
  iree_hal_streaming_value_wait_submission_state_t state;
  // True after the completion semaphore reaches a terminal state.
  bool is_terminal;
  // True when terminal completion carried a failure.
  bool has_failed;
  // True after context teardown requests observer cancellation.
  bool cancellation_requested;
  // True while the context owns an active observer for this record.
  bool observer_active;
} iree_hal_streaming_value_wait_submission_t;

// Context list containing a value-wait lane.
typedef enum iree_hal_streaming_value_wait_lane_list_state_e {
  IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE = 0,
  IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE = 1,
  IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING = 2,
} iree_hal_streaming_value_wait_lane_list_state_t;

// Exact queue kept exclusive to one logical stream while any of its externally
// controlled atomic waits or their dependent submissions remain in flight.
// Every later stream submission extends the lease; completed lanes are recycled
// across streams.
struct iree_hal_streaming_value_wait_lane_t {
  // Next lane in a context-owned idle or pending list.
  iree_hal_streaming_value_wait_lane_t* next;
  // Previous lane in a context-owned idle or pending list.
  iree_hal_streaming_value_wait_lane_t* prev;
  // List owning this lane, or NONE while temporarily acquired or detached.
  iree_hal_streaming_value_wait_lane_list_state_t list_state;
  // Dynamically acquired exact queue owned by this lane.
  iree_hal_queue_t* queue;
  // Queue family the lane realizes.
  const iree_hal_queue_family_t* family;
  // Scheduling priority the lane realizes.
  iree_hal_queue_priority_t priority;
  // Immutable execution-resource set the lane realizes.
  iree_hal_queue_execution_resource_list_t execution_resources;
  // Stable identifier of the stream whose ordered waits occupy this lane, or
  // zero while the lane is idle.
  unsigned long long owner_stream_id;
  // First accepted submission in enqueue order.
  iree_hal_streaming_value_wait_submission_t* submission_head;
  // Last accepted submission in enqueue order.
  iree_hal_streaming_value_wait_submission_t* submission_tail;
  // Number of unresolved records in the submission list.
  iree_host_size_t submission_count;
  // Failed terminal records retained until authoritative queue teardown.
  iree_hal_streaming_value_wait_submission_t* retired_failure_head;
  // Number of records in |retired_failure_head|.
  iree_host_size_t retired_failure_count;
  // True while an acquired lane must return to the pending list if the new
  // submission is rejected synchronously.
  bool restore_pending;
  // True after any tracked submission fails. Such a lane is destroyed, never
  // recycled, after every tracked submission is terminal.
  bool has_failed_submission;
  // Serializes the narrow queue-acceptance and publication transaction with
  // asynchronous completion or failure processing for this lane. Stream
  // flushes and queue teardown never run while this gate is held.
  iree_slim_mutex_t submission_mutex;
};

// Initializes context-owned value-wait lane state.
void iree_hal_streaming_value_wait_lanes_initialize(
    iree_hal_streaming_context_t* context);

// Prepares an independently observed completion record before queue
// acceptance. The caller must publish or reject the returned record.
iree_status_t iree_hal_streaming_prepare_value_wait_submission(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_value_wait_submission_t** out_submission);

// Publishes an accepted submission and returns its lane to the pending list.
void iree_hal_streaming_publish_pending_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane,
    iree_hal_streaming_value_wait_submission_t* submission);

// Rejects a prepared submission whose queue operation was not accepted.
void iree_hal_streaming_reject_value_wait_submission(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_submission_t* submission);

// Returns whether an acquired lane can accept another submission.
bool iree_hal_streaming_value_wait_lane_accepts_submission(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_value_wait_lane_t* lane);

// Returns an acquired lane to context ownership or destroys a failed lane.
void iree_hal_streaming_release_value_wait_lane(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_value_wait_lane_t* lane);

// Releases context-owned value-wait lanes after all context work is complete.
// Synchronization: none; the context must be exclusively owned.
void iree_hal_streaming_value_wait_lanes_deinitialize(
    iree_hal_streaming_context_t* context);

// Requirements for one stream submission's execution queue.
typedef enum iree_hal_streaming_submission_flag_bits_e {
  IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE = 0,
  // Starts an exclusive lease for externally controlled stream work.
  IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT = 1u << 0,
} iree_hal_streaming_submission_flag_bits_t;
typedef uint32_t iree_hal_streaming_submission_flags_t;

// Stack-owned acceptance transaction. Ordinary streams only populate the queue
// and public signal. An active exclusive lease adds an independent completion
// proof, prepared before queue acceptance and retired by the context proactor.
typedef struct iree_hal_streaming_submission_t {
  // Borrowed execution queue for this transaction.
  iree_hal_queue_t* queue;
  // Signals supplied to the final queue operation of this transaction.
  iree_hal_semaphore_list_t signals;
  // Public stream timeline and optional private completion semaphore.
  iree_hal_semaphore_t* signal_semaphores[2];
  // Payload values corresponding to |signal_semaphores|.
  uint64_t signal_values[2];
  // Prepared completion record while the lane's acceptance gate is held.
  iree_hal_streaming_value_wait_submission_t* completion;
} iree_hal_streaming_submission_t;

// Prepares queue selection and completion ownership before acceptance. Requires
// the stream mutex and a live context. |queue| is the stream's ordinary queue
// or its separately owned cooperative realization. Every successful prepare
// must be finished under the same stream lock; failure leaves no transaction
// live.
iree_status_t iree_hal_streaming_submission_prepare_locked(
    iree_hal_streaming_stream_t* stream, iree_hal_queue_t* queue,
    uint64_t signal_value, iree_hal_streaming_submission_flags_t flags,
    iree_hal_streaming_submission_t* out_submission);

// Publishes accepted completion ownership and the stream tail, or rejects the
// prepared proof. Call before queue_flush so a flush error cannot undo accepted
// work. The queue remains valid until the caller releases the stream mutex.
void iree_hal_streaming_submission_finish_locked(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_submission_t* submission, bool accepted);

// Appends a system release for outstanding native writes, if needed. The
// accepted release owns a new stream timeline point even if queue flushing
// subsequently fails. Rejection preserves the obligation and previous tail.
// Requires the stream mutex and a live context.
iree_status_t iree_hal_streaming_submission_release_system_locked(
    iree_hal_streaming_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_STREAM_QUEUE_H_
