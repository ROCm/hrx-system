// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-capacity send-buffer reservation table for pending RDMA SEND payloads.
//
// begin_send() hands callers writable bytes from an RDMA-registered staging
// pool. This table owns those leases between begin_send() and commit_send() or
// abort_send(), returning generation-checked handles that fit in the generic
// carrier send handle. The carrier also uses the table as the pending-post FIFO
// once send() payloads are accepted under native SEND backpressure. Each
// reservation copies bounded span descriptors, optionally retains one staging
// lease, and continues borrowing caller-owned registered spans until
// completion. No memory is allocated after initialization.

#ifndef IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_
#define IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_

#include <stdint.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/rdma/connection_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_send_reservation_slot_t
    iree_net_rdma_send_reservation_slot_t;

typedef uint8_t iree_net_rdma_send_reservation_completion_t;
enum iree_net_rdma_send_reservation_completion_e {
  // Internal send with no user-visible completion.
  IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL = 0,
  // User-visible two-sided SEND completion.
  IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND = 1,
};

typedef struct iree_net_rdma_send_reservation_table_t {
  // Host allocator used for the slot allocation.
  iree_allocator_t host_allocator;

  // Allocated slot storage.
  iree_net_rdma_send_reservation_slot_t* slots;

  // Total number of slots in the table.
  uint32_t capacity;

  // Number of slots currently on the free list.
  uint32_t available_capacity;

  // Index of the first free slot, or UINT32_MAX when full.
  uint32_t free_head;

  // Index of the first committed reservation waiting to be posted.
  uint32_t pending_head;

  // Index of the last committed reservation waiting to be posted.
  uint32_t pending_tail;

  // Number of committed reservations waiting to be posted.
  uint32_t pending_count;
} iree_net_rdma_send_reservation_table_t;

typedef struct iree_net_rdma_send_reservation_t {
  // Optional registered staging storage retained by the reservation.
  iree_async_buffer_lease_t buffer_lease;

  // RDMA spans borrowed until completion or backed by buffer_lease.
  iree_async_span_t spans[IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE];

  // Number of valid entries in spans.
  iree_host_size_t span_count;

  // Number of bytes to send from the reservation payload.
  iree_host_size_t byte_length;

  // Completion behavior for the eventual posted send.
  iree_net_rdma_send_reservation_completion_t completion;

  // User data forwarded to user-visible completions.
  uint64_t user_data;
} iree_net_rdma_send_reservation_t;

// Initializes |out_table| with |capacity| fixed reservation slots.
//
// Callers must serialize table access; the table intentionally contains no
// locks so carriers can choose the synchronization boundary that matches their
// queue model.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_net_rdma_send_reservation_table_t* out_table);

// Deinitializes |table| and releases any leases still reserved by callers.
IREE_API_EXPORT void iree_net_rdma_send_reservation_table_deinitialize(
    iree_net_rdma_send_reservation_table_t* table);

// Returns the number of slots available for new reservations.
IREE_API_EXPORT uint32_t
iree_net_rdma_send_reservation_table_available_capacity(
    const iree_net_rdma_send_reservation_table_t* table);

// Returns the number of committed reservations waiting to be posted.
IREE_API_EXPORT uint32_t iree_net_rdma_send_reservation_table_pending_count(
    const iree_net_rdma_send_reservation_table_t* table);

// Acquires one send reservation and copies |spans| into it.
//
// The table stores only the span descriptors. Caller-owned referenced memory
// must remain valid until the carrier completion fires, matching
// iree_net_carrier_send()'s normal async I/O contract.
//
// When |buffer_lease| is non-NULL and live, ownership transfers to the table
// and the caller's lease is cleared. The lease is returned by resolve() or
// released by abort().
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_acquire(
    iree_net_rdma_send_reservation_table_t* table, iree_async_span_list_t spans,
    iree_async_buffer_lease_t* buffer_lease,
    iree_net_rdma_send_reservation_completion_t completion, uint64_t user_data,
    iree_net_carrier_send_handle_t* out_handle);

// Returns a borrowed view of |handle| without releasing its slot.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_peek(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle,
    iree_net_rdma_send_reservation_t* out_reservation);

// Marks |handle| committed and appends it to the pending-post FIFO.
//
// After commit, the handle is consumed by the table. Callers must not resolve
// or abort it directly; the carrier must post it through
// resolve_pending_front() or release it through abort_all().
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_commit(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle);

// Returns a borrowed view of the pending FIFO front without releasing it.
IREE_API_EXPORT iree_status_t
iree_net_rdma_send_reservation_table_peek_pending_front(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t* out_handle,
    iree_net_rdma_send_reservation_t* out_reservation);

// Resolves the pending FIFO front, releases its slot, and returns the
// reservation to the caller.
IREE_API_EXPORT iree_status_t
iree_net_rdma_send_reservation_table_resolve_pending_front(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t* out_handle,
    iree_net_rdma_send_reservation_t* out_reservation);

// Resolves uncommitted |handle|, releases its slot, and returns the
// reservation.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_resolve(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle,
    iree_net_rdma_send_reservation_t* out_reservation);

// Resolves the next active reservation at or after |*inout_cursor|.
//
// Initialize |*inout_cursor| to 0 and call until |*out_found| is false. The
// caller owns the returned buffer_lease, when present, and must release it.
// Committed reservations are removed from the pending FIFO before returning.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_resolve_next(
    iree_net_rdma_send_reservation_table_t* table, uint32_t* inout_cursor,
    iree_net_rdma_send_reservation_t* out_reservation, bool* out_found);

// Aborts |handle| and releases its retained buffer lease.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_abort(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle);

// Aborts every active reservation and releases all retained buffer leases.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_abort_all(
    iree_net_rdma_send_reservation_table_t* table, uint32_t* out_aborted_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_
