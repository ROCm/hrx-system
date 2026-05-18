// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk transfer scheduler.
//
// Owns the bounded active transfer set used by HAL remote bulk paths. The
// scheduler is deliberately unaware of file, buffer, or profile state: owners
// attach their state to scheduler-owned user storage and provide a
// deinitializer callback that is called exactly once when a transfer leaves the
// scheduler.

#ifndef IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_SCHEDULER_H_
#define IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_SCHEDULER_H_

#include "iree/base/api.h"
#include "iree/net/channel/bulk/transfer_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_bulk_transfer_scheduler_t
    iree_hal_remote_bulk_transfer_scheduler_t;

// Called before |transfer| is removed from the scheduler.
//
// The transfer and its user storage are still valid for the duration of the
// callback. The callback must not mutate the scheduler.
typedef void (*iree_hal_remote_bulk_transfer_deinitialize_fn_t)(
    void* user_data, iree_net_bulk_transfer_t* transfer);

// Returns true when |transfer| should be selected.
typedef bool (*iree_hal_remote_bulk_transfer_select_fn_t)(
    void* user_data, iree_net_bulk_transfer_t* transfer);

// Visits |transfer|.
typedef void (*iree_hal_remote_bulk_transfer_visit_fn_t)(
    void* user_data, iree_net_bulk_transfer_t* transfer);

// Scheduler callbacks.
typedef struct iree_hal_remote_bulk_transfer_scheduler_callbacks_t {
  // Optional callback invoked exactly once before each transfer is removed.
  iree_hal_remote_bulk_transfer_deinitialize_fn_t deinitialize;

  // User data passed to callbacks.
  void* user_data;
} iree_hal_remote_bulk_transfer_scheduler_callbacks_t;

// Scheduler creation options.
typedef struct iree_hal_remote_bulk_transfer_scheduler_options_t {
  // Maximum active transfers.
  iree_host_size_t capacity;

  // Byte length of owner-managed storage attached to each transfer.
  iree_host_size_t user_storage_size;

  // Alignment of owner-managed storage. Zero selects max host alignment.
  iree_host_size_t user_storage_alignment;

  // First locally allocated transfer ID. Zero selects the table default.
  uint64_t initial_transfer_id;

  // Locally allocated transfer ID increment. Zero selects the table default.
  uint64_t transfer_id_stride;
} iree_hal_remote_bulk_transfer_scheduler_options_t;

// Returns conservative default scheduler options.
static inline iree_hal_remote_bulk_transfer_scheduler_options_t
iree_hal_remote_bulk_transfer_scheduler_options_default(void) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options = {0};
  options.capacity = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY;
  options.user_storage_alignment = iree_max_align_t;
  return options;
}

// Allocates a scheduler.
iree_status_t iree_hal_remote_bulk_transfer_scheduler_allocate(
    const iree_hal_remote_bulk_transfer_scheduler_options_t* options,
    iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler);

// Frees a scheduler and deinitializes any active transfers first.
void iree_hal_remote_bulk_transfer_scheduler_free(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler);

// Returns the maximum active transfer count.
iree_host_size_t iree_hal_remote_bulk_transfer_scheduler_capacity(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler);

// Returns the current active transfer count.
iree_host_size_t iree_hal_remote_bulk_transfer_scheduler_count(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler);

// Returns true when at least one active transfer slot is available.
bool iree_hal_remote_bulk_transfer_scheduler_has_capacity(
    const iree_hal_remote_bulk_transfer_scheduler_t* scheduler);

// Inserts a peer-assigned transfer ID.
iree_status_t iree_hal_remote_bulk_transfer_scheduler_insert_peer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler, uint64_t transfer_id,
    uint64_t total_size, uint64_t user_value,
    iree_net_bulk_transfer_t** out_transfer);

// Allocates a locally assigned transfer ID.
iree_status_t iree_hal_remote_bulk_transfer_scheduler_allocate_local(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler, uint64_t total_size,
    uint64_t user_value, iree_net_bulk_transfer_t** out_transfer);

// Looks up an active transfer by ID. Returns NULL when not found.
iree_net_bulk_transfer_t* iree_hal_remote_bulk_transfer_scheduler_lookup(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler, uint64_t transfer_id);

// Releases |transfer| if it is active in |scheduler|.
//
// Returns true if the transfer was active and removed.
bool iree_hal_remote_bulk_transfer_scheduler_release(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer);

// Deinitializes and removes all active transfers.
void iree_hal_remote_bulk_transfer_scheduler_clear(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler);

// Visits all active transfers in unspecified order.
//
// The visitor must not mutate |scheduler|.
void iree_hal_remote_bulk_transfer_scheduler_visit(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_hal_remote_bulk_transfer_visit_fn_t visitor, void* user_data);

// Collects active transfer IDs selected by |select| into |out_transfer_ids|.
//
// This is a stable two-phase retry primitive: callers can collect IDs while
// visiting the scheduler and then perform operations that may release transfers
// after the visit ends. Returns true if all selected IDs fit; false when the
// caller-provided scratch storage filled before the visit completed.
bool iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_hal_remote_bulk_transfer_select_fn_t select, void* select_user_data,
    uint64_t* out_transfer_ids, iree_host_size_t capacity,
    iree_host_size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_SCHEDULER_H_
