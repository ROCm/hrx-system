// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server bulk channel router.
//
// Bridges the transport-level iree_net_bulk_channel callback ABI to
// server-owned bulk operations. The router owns callback status ownership and
// transport event normalization; operation callbacks own HAL/file/profile
// behavior and must perform their own locking.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_ROUTER_H_
#define IREE_HAL_REMOTE_SERVER_BULK_ROUTER_H_

#include "iree/base/api.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_bulk_router_t
    iree_hal_remote_server_bulk_router_t;

// Called when a START frame is received.
typedef iree_status_t (*iree_hal_remote_server_bulk_router_start_fn_t)(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags);

// Called when a DATA frame is received.
typedef iree_status_t (*iree_hal_remote_server_bulk_router_data_fn_t)(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease);

// Called when a COMPLETE frame is received.
typedef iree_status_t (*iree_hal_remote_server_bulk_router_complete_fn_t)(
    void* user_data, uint64_t transfer_id);

// Called when an ABORT frame is received.
typedef iree_status_t (*iree_hal_remote_server_bulk_router_abort_fn_t)(
    void* user_data, uint64_t transfer_id);

// Called when the bulk transport reports an error. Consumes |status|.
typedef void (*iree_hal_remote_server_bulk_router_transport_error_fn_t)(
    void* user_data, iree_status_t status);

// Called when a send operation completes. Consumes |status|.
typedef void (*iree_hal_remote_server_bulk_router_send_complete_fn_t)(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Called when peer receive credit is replenished.
typedef void (*iree_hal_remote_server_bulk_router_credit_fn_t)(void* user_data);

// Retains callback user data for asynchronous send-completion handling.
typedef void (*iree_hal_remote_server_bulk_router_retain_fn_t)(void* user_data);

// Releases callback user data retained for asynchronous send-completion
// handling.
typedef void (*iree_hal_remote_server_bulk_router_release_fn_t)(
    void* user_data);

// Operation callbacks supplied by the server session.
typedef struct iree_hal_remote_server_bulk_router_operations_t {
  // Optional callback retaining |user_data| across send completion dispatch.
  iree_hal_remote_server_bulk_router_retain_fn_t retain;

  // Optional callback releasing |user_data| after send completion dispatch.
  iree_hal_remote_server_bulk_router_release_fn_t release;

  // Required START handler.
  iree_hal_remote_server_bulk_router_start_fn_t start;

  // Required DATA handler.
  iree_hal_remote_server_bulk_router_data_fn_t data;

  // Required COMPLETE handler.
  iree_hal_remote_server_bulk_router_complete_fn_t complete;

  // Required ABORT handler.
  iree_hal_remote_server_bulk_router_abort_fn_t abort;

  // Required transport error handler. Consumes the received status.
  iree_hal_remote_server_bulk_router_transport_error_fn_t transport_error;

  // Required send completion handler. Consumes the received status.
  iree_hal_remote_server_bulk_router_send_complete_fn_t send_complete;

  // Required credit replenishment handler.
  iree_hal_remote_server_bulk_router_credit_fn_t credit;
} iree_hal_remote_server_bulk_router_operations_t;

// Routes transport callbacks to server bulk operations.
typedef struct iree_hal_remote_server_bulk_router_t {
  // Operation callbacks invoked by this router.
  iree_hal_remote_server_bulk_router_operations_t operations;

  // User data passed to operation callbacks.
  void* user_data;
} iree_hal_remote_server_bulk_router_t;

// Initializes |out_router| in caller-owned storage.
void iree_hal_remote_server_bulk_router_initialize(
    iree_hal_remote_server_bulk_router_operations_t operations, void* user_data,
    iree_hal_remote_server_bulk_router_t* out_router);

// Deinitializes |router| without touching the operation user data.
void iree_hal_remote_server_bulk_router_deinitialize(
    iree_hal_remote_server_bulk_router_t* router);

// Returns net bulk channel callbacks using |router| as callback user data.
//
// The returned callbacks assume the router storage outlives the bulk channel.
// Callback threading follows the underlying iree_net_bulk_channel contract.
// Operation callbacks are invoked without taking any locks in the router.
iree_net_bulk_channel_callbacks_t iree_hal_remote_server_bulk_router_callbacks(
    iree_hal_remote_server_bulk_router_t* router);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_ROUTER_H_
