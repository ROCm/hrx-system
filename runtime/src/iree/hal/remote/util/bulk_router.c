// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_router.h"

#include <string.h>

void iree_hal_remote_bulk_router_initialize(
    iree_hal_remote_bulk_router_operations_t operations, void* user_data,
    iree_hal_remote_bulk_router_t* out_router) {
  IREE_ASSERT_ARGUMENT(out_router);
  IREE_ASSERT_ARGUMENT(operations.start);
  IREE_ASSERT_ARGUMENT(operations.data);
  IREE_ASSERT_ARGUMENT(operations.complete);
  IREE_ASSERT_ARGUMENT(operations.abort);
  IREE_ASSERT_ARGUMENT(operations.transport_error);
  IREE_ASSERT_ARGUMENT(operations.send_complete);
  IREE_ASSERT_ARGUMENT(operations.credit);
  memset(out_router, 0, sizeof(*out_router));
  out_router->operations = operations;
  out_router->user_data = user_data;
}

void iree_hal_remote_bulk_router_deinitialize(
    iree_hal_remote_bulk_router_t* router) {
  if (!router) return;
  memset(router, 0, sizeof(*router));
}

static iree_status_t iree_hal_remote_bulk_router_on_start(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags) {
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  return router->operations.start(router->user_data, transfer_id, total_size,
                                  flags);
}

static iree_status_t iree_hal_remote_bulk_router_on_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  return router->operations.data(router->user_data, transfer_id, chunk_offset,
                                 sequence, flags, chunk_data, lease);
}

static iree_status_t iree_hal_remote_bulk_router_on_complete(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  return router->operations.complete(router->user_data, transfer_id);
}

static iree_status_t iree_hal_remote_bulk_router_on_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  (void)abort_data;
  (void)lease;
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  return router->operations.abort(router->user_data, transfer_id);
}

static void iree_hal_remote_bulk_router_on_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  router->operations.transport_error(router->user_data, status);
}

static void iree_hal_remote_bulk_router_on_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  iree_hal_remote_bulk_router_operations_t operations = router->operations;
  void* callback_user_data = router->user_data;
  if (operations.retain) {
    operations.retain(callback_user_data);
  }

  iree_status_t transport_status = iree_ok_status();
  if (operation_user_data != 0 && !iree_status_is_ok(status)) {
    transport_status = iree_status_clone(status);
  }
  operations.send_complete(callback_user_data, operation_user_data, status);
  if (!iree_status_is_ok(transport_status)) {
    operations.transport_error(callback_user_data, transport_status);
  }

  if (operations.release) {
    operations.release(callback_user_data);
  }
}

static void iree_hal_remote_bulk_router_on_credit(
    void* user_data, uint32_t credit_delta, uint32_t available_credit_count) {
  (void)credit_delta;
  (void)available_credit_count;
  iree_hal_remote_bulk_router_t* router =
      (iree_hal_remote_bulk_router_t*)user_data;
  router->operations.credit(router->user_data);
}

iree_net_bulk_channel_callbacks_t iree_hal_remote_bulk_router_callbacks(
    iree_hal_remote_bulk_router_t* router) {
  IREE_ASSERT_ARGUMENT(router);
  iree_net_bulk_channel_callbacks_t callbacks = {
      .on_start = iree_hal_remote_bulk_router_on_start,
      .on_data = iree_hal_remote_bulk_router_on_data,
      .on_complete = iree_hal_remote_bulk_router_on_complete,
      .on_abort = iree_hal_remote_bulk_router_on_abort,
      .on_transport_error = iree_hal_remote_bulk_router_on_transport_error,
      .on_send_complete = iree_hal_remote_bulk_router_on_send_complete,
      .on_credit = iree_hal_remote_bulk_router_on_credit,
      .user_data = router,
  };
  return callbacks;
}
