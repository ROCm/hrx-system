// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server bulk staging pool.
//
// Owns fixed host-backed HAL file slots used to bridge bulk transport payloads
// and asynchronous HAL queue file operations. The pool is deliberately unaware
// of protocol frame kinds or transfer state; callers attach per-slot state to
// owner-managed user storage.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_STAGING_POOL_H_
#define IREE_HAL_REMOTE_SERVER_BULK_STAGING_POOL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_bulk_staging_pool_t
    iree_hal_remote_server_bulk_staging_pool_t;
typedef struct iree_hal_remote_server_bulk_staging_slot_t
    iree_hal_remote_server_bulk_staging_slot_t;

// Called when an acquired slot's semaphore reaches |signal_value|.
//
// The callback consumes |status| and must release |slot| when the staged
// contents are no longer needed. The slot and its pool remain valid for the
// duration of the callback even if the owning session has released the pool.
typedef void (*iree_hal_remote_server_bulk_staging_slot_callback_fn_t)(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t signal_value, iree_status_t status);

// Staging pool creation options.
typedef struct iree_hal_remote_server_bulk_staging_pool_options_t {
  // Maximum acquired slot count.
  iree_host_size_t slot_count;

  // Byte length of each host-backed staging allocation.
  iree_host_size_t slot_length;

  // Byte length of owner-managed storage attached to each slot.
  iree_host_size_t user_storage_size;

  // Alignment of owner-managed storage. Zero selects max host alignment.
  iree_host_size_t user_storage_alignment;
} iree_hal_remote_server_bulk_staging_pool_options_t;

// Returns conservative default staging pool options.
static inline iree_hal_remote_server_bulk_staging_pool_options_t
iree_hal_remote_server_bulk_staging_pool_options_default(void) {
  iree_hal_remote_server_bulk_staging_pool_options_t options = {0};
  options.user_storage_alignment = iree_max_align_t;
  return options;
}

// Creates a staging pool with fixed host-backed slots.
iree_status_t iree_hal_remote_server_bulk_staging_pool_create(
    const iree_hal_remote_server_bulk_staging_pool_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_staging_pool_t** out_pool);

// Retains |pool| for the caller.
void iree_hal_remote_server_bulk_staging_pool_retain(
    iree_hal_remote_server_bulk_staging_pool_t* pool);

// Releases |pool| from the caller.
void iree_hal_remote_server_bulk_staging_pool_release(
    iree_hal_remote_server_bulk_staging_pool_t* pool);

// Returns the maximum acquired slot count.
iree_host_size_t iree_hal_remote_server_bulk_staging_pool_capacity(
    const iree_hal_remote_server_bulk_staging_pool_t* pool);

// Returns the current acquired slot count.
iree_host_size_t iree_hal_remote_server_bulk_staging_pool_count(
    const iree_hal_remote_server_bulk_staging_pool_t* pool);

// Tries to acquire one staging slot bound to |local_device|.
//
// Returns OK with NULL |out_slot| when no slot is available. The caller must
// release a non-NULL returned slot with
// iree_hal_remote_server_bulk_staging_slot_release.
iree_status_t iree_hal_remote_server_bulk_staging_pool_try_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot);

// Acquires one staging slot bound to |local_device|.
iree_status_t iree_hal_remote_server_bulk_staging_pool_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot);

// Releases |slot| back to its pool after its semaphore has reached
// |last_signal_value|.
void iree_hal_remote_server_bulk_staging_slot_release(
    iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t last_signal_value);

// Returns the pool that owns |slot|.
iree_hal_remote_server_bulk_staging_pool_t*
iree_hal_remote_server_bulk_staging_slot_pool(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the local device currently bound to |slot|.
iree_hal_device_t* iree_hal_remote_server_bulk_staging_slot_device(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the HAL memory file currently bound to |slot|.
iree_hal_file_t* iree_hal_remote_server_bulk_staging_slot_file(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the local semaphore currently bound to |slot|.
iree_hal_semaphore_t* iree_hal_remote_server_bulk_staging_slot_semaphore(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the byte-addressable host contents of |slot|.
iree_byte_span_t iree_hal_remote_server_bulk_staging_slot_contents(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns owner-managed user storage for |slot|.
iree_byte_span_t iree_hal_remote_server_bulk_staging_slot_user_storage(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the last payload value observed when |slot| was released.
uint64_t iree_hal_remote_server_bulk_staging_slot_last_signal_value(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Returns the next payload value to use for a staged operation.
uint64_t iree_hal_remote_server_bulk_staging_slot_next_signal_value(
    const iree_hal_remote_server_bulk_staging_slot_t* slot);

// Registers |callback| to run when |slot|'s semaphore reaches |signal_value|.
//
// The callback may run before this function returns if the semaphore is already
// satisfied. On success, the callback owns a pool reference until it returns.
iree_status_t iree_hal_remote_server_bulk_staging_slot_acquire_timepoint(
    iree_hal_remote_server_bulk_staging_slot_t* slot, uint64_t signal_value,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t callback,
    void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_STAGING_POOL_H_
