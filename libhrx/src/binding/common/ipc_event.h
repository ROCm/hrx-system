// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_IPC_EVENT_H_
#define IREE_HAL_STREAMING_IPC_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE 32

typedef struct iree_hal_streaming_event_t iree_hal_streaming_event_t;
typedef struct iree_hal_streaming_ipc_event_t iree_hal_streaming_ipc_event_t;
typedef struct iree_hal_streaming_recorded_point_t
    iree_hal_streaming_recorded_point_t;

// Opaque state prepared before an IPC event record is submitted.
typedef void* iree_hal_streaming_ipc_event_record_state_t;

// Opaque state prepared before a queue wait on an IPC event is submitted.
typedef void* iree_hal_streaming_ipc_event_wait_state_t;

typedef struct iree_hal_streaming_ipc_event_ops_t {
  // Destroys |ipc_event| and every backend resource it owns.
  void (*destroy)(iree_hal_streaming_ipc_event_t* ipc_event);

  // Prepares one record generation before the queue operation is submitted.
  // On success |out_record_state| owns every private resource needed to commit
  // or abort that generation. On failure the implementation must release those
  // resources and leave |out_record_state| empty.
  iree_status_t (*begin_record)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      iree_hal_semaphore_t* recorded_semaphore, uint64_t recorded_value,
      iree_hal_streaming_ipc_event_record_state_t* out_record_state);

  // Commits and consumes a prepared generation after the queue accepted its
  // record. This operation is infallible.
  void (*commit_record)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      iree_hal_streaming_ipc_event_record_state_t record_state);

  // Aborts and consumes a prepared generation after the queue rejected its
  // record.
  void (*abort_record)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      iree_hal_streaming_ipc_event_record_state_t record_state);

  // Returns 0 when the current IPC generation is complete and 1 otherwise.
  // IPC transports report completion only: producer failure also completes a
  // generation and remains observable through the producer's local stream.
  iree_status_t (*query)(iree_hal_streaming_ipc_event_t* ipc_event,
                         int* out_status);

  // Blocks until the current IPC generation completes. Like query, this does
  // not carry producer status across the IPC event transport.
  iree_status_t (*synchronize)(iree_hal_streaming_ipc_event_t* ipc_event);

  // Reserves one wait before any destination queue operation is submitted. The
  // returned semaphore is retained for the caller, while |out_wait_state|
  // owns every resource needed to arm, activate, or abandon the wait. Reserve
  // must not observe or lock the source event generation. On failure it must
  // release its private resources and leave both outputs empty.
  iree_status_t (*reserve_wait)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      iree_hal_device_t* destination_device,
      iree_hal_semaphore_t** out_semaphore, uint64_t* out_value,
      iree_hal_streaming_ipc_event_wait_state_t* out_wait_state);

  // Binds a reserved wait to the generation visible at its queue position.
  // This operation must be allocation-free and infallible. Returns false when
  // the source has no carrier or its current generation is already complete;
  // the queue wait must then be omitted.
  bool (*arm_wait)(iree_hal_streaming_ipc_event_t* ipc_event,
                   iree_hal_streaming_ipc_event_wait_state_t wait_state);

  // Activates an armed reserved wait after the queue accepted its operation.
  // An unarmed lazy wait is simply consumed. The state is consumed even on
  // failure. A failure must also fail or signal the reserved semaphore so
  // already accepted queue work cannot remain blocked.
  iree_status_t (*commit_wait)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      iree_hal_streaming_ipc_event_wait_state_t wait_state);

  // Abandons a reserved or armed wait before its queue operation was accepted.
  void (*abort_wait)(iree_hal_streaming_ipc_event_t* ipc_event,
                     iree_hal_streaming_ipc_event_wait_state_t wait_state);

  // Exports the backend token for this event, creating it lazily if needed.
  iree_status_t (*export_token)(
      iree_hal_streaming_ipc_event_t* ipc_event,
      uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]);
} iree_hal_streaming_ipc_event_ops_t;

// Backend-neutral IPC event interface owned by a streaming event.
struct iree_hal_streaming_ipc_event_t {
  // Operations implemented by the binding-specific IPC event adapter.
  const iree_hal_streaming_ipc_event_ops_t* ops;
};

// Attaches |ipc_event| to |event| and transfers ownership on success. Must be
// called before |event| is published to other threads.
iree_status_t iree_hal_streaming_event_attach_ipc_event(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_t* ipc_event);

// Exports the fixed-size backend token from an IPC-enabled event.
iree_status_t iree_hal_streaming_event_export_ipc_token(
    iree_hal_streaming_event_t* event,
    uint8_t out_token[IREE_HAL_STREAMING_IPC_EVENT_TOKEN_SIZE]);

// Acquires the point a destination-device wait must name. Ordinary events use
// their recorded point; IPC events resolve a fresh proxy point. A zeroed point
// is valid and means that an unactivated source event requires no wait.
iree_status_t iree_hal_streaming_event_reserve_wait_point(
    iree_hal_streaming_event_t* event, iree_hal_device_t* destination_device,
    iree_hal_streaming_recorded_point_t* out_point,
    iree_hal_streaming_ipc_event_wait_state_t* out_wait_state);

// Arms a reserved IPC wait at its queue position. Returns false when the wait
// names an unactivated IPC event or an already-complete generation and its
// reserved proxy must be omitted.
bool iree_hal_streaming_event_arm_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state);

// Commits a reserved IPC wait. A NULL state belongs to the ordinary event hot
// path and is an immediate no-op.
iree_status_t iree_hal_streaming_event_commit_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state);

// Aborts a reserved IPC wait. A NULL state belongs to the ordinary event hot
// path and is an immediate no-op.
void iree_hal_streaming_event_abort_wait(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_ipc_event_wait_state_t wait_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_STREAMING_IPC_EVENT_H_
