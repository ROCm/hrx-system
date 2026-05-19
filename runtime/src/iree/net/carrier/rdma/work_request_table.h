// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded work-request correlation table for RDMA wr_id values.
//
// RDMA completions return only the 64-bit wr_id from the posted work request.
// The carrier uses this table to allocate fixed in-flight slots, encode a
// generation-checked wr_id, and recover operation metadata when the CQE
// arrives. No memory is allocated after initialization.

#ifndef IREE_NET_CARRIER_RDMA_WORK_REQUEST_TABLE_H_
#define IREE_NET_CARRIER_RDMA_WORK_REQUEST_TABLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_work_request_slot_t
    iree_net_rdma_work_request_slot_t;

typedef struct iree_net_rdma_work_request_table_t {
  // Host allocator used for the slot allocation.
  iree_allocator_t host_allocator;

  // Allocated slot storage.
  iree_net_rdma_work_request_slot_t* slots;

  // Total number of slots in the table.
  uint32_t capacity;

  // Number of slots currently on the free list.
  uint32_t available_capacity;

  // Index of the first free slot, or UINT32_MAX when full.
  uint32_t free_head;
} iree_net_rdma_work_request_table_t;

// Operation class stored with an in-flight work request.
typedef uint8_t iree_net_rdma_work_request_operation_t;
enum iree_net_rdma_work_request_operation_e {
  IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE = 0u,
  IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND = 1u,
  IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV = 2u,
  IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE = 3u,
  IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ = 4u,
};

typedef struct iree_net_rdma_work_request_completion_t {
  // Operation class originally supplied during acquire.
  iree_net_rdma_work_request_operation_t operation;

  // User data originally supplied during acquire.
  uint64_t user_data;

  // Byte length originally supplied during acquire.
  iree_host_size_t byte_length;
} iree_net_rdma_work_request_completion_t;

// Initializes |out_table| with |capacity| fixed slots.
//
// Callers must serialize table access; the table intentionally contains no
// locks so carrier post/completion paths can choose the correct synchronization
// boundary for their queue model.
IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_net_rdma_work_request_table_t* out_table);

// Deinitializes |table| and releases its slot storage.
IREE_API_EXPORT void iree_net_rdma_work_request_table_deinitialize(
    iree_net_rdma_work_request_table_t* table);

// Returns the number of slots available for new work requests.
IREE_API_EXPORT uint32_t iree_net_rdma_work_request_table_available_capacity(
    const iree_net_rdma_work_request_table_t* table);

// Acquires one slot and returns the wr_id to place in the native WR.
IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_acquire(
    iree_net_rdma_work_request_table_t* table,
    iree_net_rdma_work_request_operation_t operation, uint64_t user_data,
    iree_host_size_t byte_length, uint64_t* out_wr_id);

// Completes the work request identified by |wr_id| and releases its slot.
IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_complete(
    iree_net_rdma_work_request_table_t* table, uint64_t wr_id,
    iree_net_rdma_work_request_completion_t* out_completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_WORK_REQUEST_TABLE_H_
