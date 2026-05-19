// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed send-buffer reservation table for RDMA begin_send/commit_send.
//
// begin_send() hands callers writable bytes from an RDMA-registered staging
// pool. This table owns those leases between begin_send() and commit_send() or
// abort_send(), returning generation-checked handles that fit in the generic
// carrier send handle. No memory is allocated after initialization.

#ifndef IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_
#define IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_send_reservation_slot_t
    iree_net_rdma_send_reservation_slot_t;

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
} iree_net_rdma_send_reservation_table_t;

typedef struct iree_net_rdma_send_reservation_t {
  // Buffer lease containing the staged send bytes.
  iree_async_buffer_lease_t buffer_lease;

  // Number of bytes written by the caller into buffer_lease.
  iree_host_size_t byte_length;
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

// Acquires one reservation slot and transfers |buffer_lease| into it.
//
// On success |buffer_lease| is cleared, |out_handle| receives a
// generation-checked carrier send handle, and the table owns the lease until
// resolve() or abort() is called.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_acquire(
    iree_net_rdma_send_reservation_table_t* table,
    iree_async_buffer_lease_t* buffer_lease, iree_host_size_t byte_length,
    iree_net_carrier_send_handle_t* out_handle);

// Resolves |handle|, releases its slot, and returns the reservation to caller.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_resolve(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle,
    iree_net_rdma_send_reservation_t* out_reservation);

// Aborts |handle| and releases its retained buffer lease.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_abort(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_SEND_RESERVATION_TABLE_H_
