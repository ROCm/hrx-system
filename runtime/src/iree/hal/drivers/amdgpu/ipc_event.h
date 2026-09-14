// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_H_
#define IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ref-counted carrier for a ROCr IPC signal used as a binary event.
typedef struct iree_hal_amdgpu_ipc_event_t iree_hal_amdgpu_ipc_event_t;

// Prepared local-record operation. Exactly one of
// iree_hal_amdgpu_ipc_event_record_commit or
// iree_hal_amdgpu_ipc_event_record_abort must consume each operation.
// Preparation and the matching commit or abort must run on the same thread.
typedef struct iree_hal_amdgpu_ipc_event_record_t
    iree_hal_amdgpu_ipc_event_record_t;

// Prepared destination wait. Exactly one of
// iree_hal_amdgpu_ipc_event_wait_commit or
// iree_hal_amdgpu_ipc_event_wait_abort must consume each operation.
typedef struct iree_hal_amdgpu_ipc_event_wait_t
    iree_hal_amdgpu_ipc_event_wait_t;

// Size of the process-independent ROCr IPC signal token.
#define IREE_HAL_AMDGPU_IPC_EVENT_TOKEN_SIZE 32

// Opaque process-independent ROCr IPC signal token.
//
// The bytes are transported by value and must not be interpreted or rewritten.
// A token is only valid while at least one process keeps the referenced ROCr
// signal alive.
typedef struct iree_hal_amdgpu_ipc_event_token_t {
  // Opaque ROCr hsa_amd_ipc_signal_t bytes.
  uint32_t data[IREE_HAL_AMDGPU_IPC_EVENT_TOKEN_SIZE / sizeof(uint32_t)];
} iree_hal_amdgpu_ipc_event_token_t;

// A fresh destination-device semaphore timepoint resolved by an IPC event.
typedef struct iree_hal_amdgpu_ipc_event_wait_point_t {
  // Semaphore retained for the caller; release with iree_hal_semaphore_release.
  iree_hal_semaphore_t* semaphore;
  // Timeline value signaled when the observed IPC event generation completes.
  uint64_t value;
} iree_hal_amdgpu_ipc_event_wait_point_t;

// Creates an IPC event backed by a ROCr IPC signal initialized to pending (1).
// The first accepted record resolves it to complete (0) after the producing
// stream reaches the recorded point.
//
// |device| must be an AMDGPU HAL logical device. Other device implementations
// are rejected with IREE_STATUS_INVALID_ARGUMENT.
iree_status_t iree_hal_amdgpu_ipc_event_create(
    iree_hal_device_t* device, iree_hal_amdgpu_ipc_event_t** out_event);

// Imports an IPC event from a by-value ROCr IPC signal |token|.
//
// A process keeps one canonical carrier per live token. A locally created owner
// is that token's canonical carrier; repeated imports share it without another
// attachment. Each successful import returns one retained reference. |device|
// must be an AMDGPU HAL logical device.
iree_status_t iree_hal_amdgpu_ipc_event_import(
    iree_hal_device_t* device, iree_hal_amdgpu_ipc_event_token_t token,
    iree_hal_amdgpu_ipc_event_t** out_event);

// Retains |event| for the caller.
void iree_hal_amdgpu_ipc_event_retain(iree_hal_amdgpu_ipc_event_t* event);

// Releases |event| and destroys it when the last reference is released. A
// registry-anchored carrier serializes its final caller release with imports;
// HSA signal destruction runs only after the carrier enters detaching state.
void iree_hal_amdgpu_ipc_event_release(iree_hal_amdgpu_ipc_event_t* event);

// Exports the process-independent ROCr IPC signal token for |event|. Returns
// IREE_STATUS_ABORTED after shutdown abandons a pending local record.
iree_status_t iree_hal_amdgpu_ipc_event_export(
    iree_hal_amdgpu_ipc_event_t* event,
    iree_hal_amdgpu_ipc_event_token_t* out_token);

// Queries whether the current event generation has completed (signal < 1).
// Producer failure also resolves the interoperable signal because the ROCr IPC
// token has no cross-process error channel. The query uses acquire semantics
// for preceding producer writes and never waits for process-local generation
// serialization. A genuinely pending local record cancelled during monitor
// shutdown leaves the signal at 1 and makes subsequent local queries return
// IREE_STATUS_ABORTED.
iree_status_t iree_hal_amdgpu_ipc_event_query(
    iree_hal_amdgpu_ipc_event_t* event, bool* out_reached);

// Blocks until the current event generation completes or monitor shutdown
// cancels a still-pending local wait. A carrier whose pending local record was
// abandoned by an earlier shutdown returns IREE_STATUS_ABORTED.
// Producer failure also resolves the interoperable signal because the ROCr IPC
// token has no cross-process error channel. The wait uses acquire semantics for
// preceding producer writes.
iree_status_t iree_hal_amdgpu_ipc_event_wait(
    iree_hal_amdgpu_ipc_event_t* event);

// Prepares a local record associated with |recorded_semaphore| at
// |recorded_value|.
//
// Preparation allocates and retains all state needed by commit, then serializes
// access to the carrier through the enqueue outcome. A rerecord waits for the
// prior locally recorded generation to complete, but cancels that wait if
// monitor shutdown begins. The ROCr signal is not rearmed until commit proves
// the new queue operation was accepted. The caller must enqueue work that
// signals the exact semaphore/value and then consume |out_record| with commit
// on success or abort on failure. If shutdown cancelled a genuinely pending
// prior local record, the carrier remains pending and future preparation
// returns IREE_STATUS_ABORTED rather than waiting for a retired timepoint.
iree_status_t iree_hal_amdgpu_ipc_event_record_prepare(
    iree_hal_amdgpu_ipc_event_t* event,
    iree_hal_semaphore_t* recorded_semaphore, uint64_t recorded_value,
    iree_hal_amdgpu_ipc_event_record_t** out_record);

// Commits a prepared local record after its queue operation was accepted.
//
// The prepared operation is consumed and its generation is published pending.
// The IPC monitor observes completion or failure of its ordinary HAL semaphore
// and SC-release stores 0 to the ROCr signal so an external waiter cannot hang
// on terminal producer failure. Shutdown cancellation does not store 0 while
// accepted producer work is still pending because its writes are not visible.
void iree_hal_amdgpu_ipc_event_record_commit(
    iree_hal_amdgpu_ipc_event_record_t* record);

// Aborts a prepared local record after its queue operation was rejected.
//
// The prepared operation is consumed without changing transport state. A
// rejected record is never visible as a pending generation.
void iree_hal_amdgpu_ipc_event_record_abort(
    iree_hal_amdgpu_ipc_event_record_t* record);

// Reserves a fresh flags-NONE semaphore on |destination_device|. Reservation
// does not inspect or lock an IPC event generation, so an entire graph can
// reserve every fallible wait resource before submitting its first block.
//
// |destination_device| must be an AMDGPU HAL logical device. The returned
// semaphore intentionally has no local-producer fast-path metadata and is safe
// to use as an ordinary wait in destination-device queue submissions. The
// owning source and destination devices must remain externally retained until
// every queue submission containing the returned semaphore has drained; wait
// operation references are not a device-teardown handoff.
iree_status_t iree_hal_amdgpu_ipc_event_wait_reserve(
    iree_hal_device_t* destination_device,
    iree_hal_amdgpu_ipc_event_wait_point_t* out_wait_point,
    iree_hal_amdgpu_ipc_event_wait_t** out_wait);

// Allocation-free, infallible binding of |wait| to |event|'s current
// process-local generation. Returns false when that generation is already
// complete and the caller must omit the reserved proxy from its queue wait.
// A pending generation remains locally stable through commit or abort.
//
// The interoperable ROCr signal is binary and carries no cross-process
// generation counter. Applications reusing an event from multiple processes
// must externally order rerecords after prior-generation consumers have passed
// their waits.
bool iree_hal_amdgpu_ipc_event_wait_arm(iree_hal_amdgpu_ipc_event_t* event,
                                        iree_hal_amdgpu_ipc_event_wait_t* wait);

// Commits an armed reserved wait after its proxy semaphore was accepted by a
// queue, or consumes an unarmed reservation whose proxy was omitted. Committed
// waits are polled until their generation completes. Monitor shutdown fails a
// still-pending proxy before releasing its destination-device reference.
iree_status_t iree_hal_amdgpu_ipc_event_wait_commit(
    iree_hal_amdgpu_ipc_event_wait_t* wait);

// Aborts a reserved or armed wait before its proxy was accepted by the queue.
void iree_hal_amdgpu_ipc_event_wait_abort(
    iree_hal_amdgpu_ipc_event_wait_t* wait);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_H_
