// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_ALLOCATION_PREPARATION_H_
#define LIBHRX_SRC_BINDING_COMMON_ALLOCATION_PREPARATION_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"

#ifdef __cplusplus
extern "C" {
#endif

// Coordinates operation preparation with allocation teardown.
//
// Each operation acquires a lease while resolving and retaining the allocation
// resources needed for submission. Closing atomically rejects new leases and
// waits for existing leases to finish without serializing unrelated
// allocations. Once idle, a closing allocation may either be destroyed or
// reopened after a failed teardown transaction.
typedef struct iree_hal_streaming_allocation_preparation_t {
  // Serializes lease admission and closure state transitions.
  iree_slim_mutex_t mutex;

  // Wakes allocation teardown after the final admitted lease completes.
  iree_notification_t notification;

  // Number of admitted operations not yet recorded or rejected.
  iree_host_size_t active_count;

  // True while new operation preparation leases are rejected.
  bool is_closing;
} iree_hal_streaming_allocation_preparation_t;

// Initializes |out_preparation| with no active leases and admission open.
void iree_hal_streaming_allocation_preparation_initialize(
    iree_hal_streaming_allocation_preparation_t* out_preparation);

// Deinitializes an idle preparation object. No thread may be waiting on it.
void iree_hal_streaming_allocation_preparation_deinitialize(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Acquires an operation preparation lease, or returns false when closing.
bool iree_hal_streaming_allocation_preparation_try_acquire(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Releases one previously acquired operation preparation lease.
void iree_hal_streaming_allocation_preparation_release(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Closes admission before removing the allocation from public lookup.
void iree_hal_streaming_allocation_preparation_begin_close(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Leaves admission closed, whether it was open or already closed. Final
// wrapper teardown uses this after earlier transactions may already have
// removed the allocation from public lookup.
void iree_hal_streaming_allocation_preparation_ensure_closed(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Reopens an idle closing allocation after a failed teardown transaction.
void iree_hal_streaming_allocation_preparation_reopen(
    iree_hal_streaming_allocation_preparation_t* preparation);

// Waits until every lease admitted before closure has been released.
void iree_hal_streaming_allocation_preparation_await_idle(
    iree_hal_streaming_allocation_preparation_t* preparation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_ALLOCATION_PREPARATION_H_
