// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk.h"

#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/bulk/bulk_channel.h"

// Header buffers used by the bulk channel frame sender. Bulk DATA payloads are
// not copied into this pool; only the 40-byte frame headers are retained until
// send completion.
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT 128
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE 128

typedef struct iree_hal_remote_client_bulk_endpoint_context_t {
  // Owning device receiving the bulk channel.
  iree_hal_remote_client_device_t* device;

  // Queue channel created before bulk endpoint provisioning. Ownership is
  // transferred to the device only after the bulk channel is ready.
  iree_net_queue_channel_t* queue_channel;
} iree_hal_remote_client_bulk_endpoint_context_t;

static iree_status_t iree_hal_remote_client_device_on_bulk_start(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags) {
  (void)user_data;
  (void)transfer_id;
  (void)total_size;
  (void)flags;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote client bulk START is not implemented yet");
}

static iree_status_t iree_hal_remote_client_device_on_bulk_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)chunk_offset;
  (void)sequence;
  (void)flags;
  (void)chunk_data;
  (void)lease;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote client bulk DATA is not implemented yet");
}

static iree_status_t iree_hal_remote_client_device_on_bulk_complete(
    void* user_data, uint64_t transfer_id) {
  (void)user_data;
  (void)transfer_id;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote client bulk COMPLETE is not implemented yet");
}

static iree_status_t iree_hal_remote_client_device_on_bulk_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)abort_data;
  (void)lease;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote client bulk ABORT is not implemented yet");
}

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_device_store_state(
      device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
  if (device->options.error_callback.fn) {
    device->options.error_callback.fn(device->options.error_callback.user_data,
                                      status);
  } else {
    iree_status_ignore(status);
  }
}

static void iree_hal_remote_client_device_on_bulk_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  (void)operation_user_data;
  if (iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }
  iree_hal_remote_client_device_on_bulk_transport_error(user_data, status);
}

static void iree_hal_remote_client_device_on_bulk_credit(
    void* user_data, uint32_t credit_delta, uint32_t available_credit_count) {
  (void)user_data;
  (void)credit_delta;
  (void)available_credit_count;
}

static void iree_hal_remote_client_device_on_bulk_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_client_bulk_endpoint_context_t* context =
      (iree_hal_remote_client_bulk_endpoint_context_t*)user_data;
  iree_hal_remote_client_device_t* device = context->device;
  iree_net_queue_channel_t* queue_channel = context->queue_channel;
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, context);
  context = NULL;

  iree_async_buffer_pool_t* header_pool = NULL;
  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    iree_status_ignore(status);
    iree_net_queue_channel_release(queue_channel);
  } else {
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_create_queue_header_pool(
          IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT,
          IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE, host_allocator,
          &header_pool);
    }
    if (iree_status_is_ok(status)) {
      iree_net_bulk_channel_callbacks_t callbacks = {
          .on_start = iree_hal_remote_client_device_on_bulk_start,
          .on_data = iree_hal_remote_client_device_on_bulk_data,
          .on_complete = iree_hal_remote_client_device_on_bulk_complete,
          .on_abort = iree_hal_remote_client_device_on_bulk_abort,
          .on_transport_error =
              iree_hal_remote_client_device_on_bulk_transport_error,
          .on_send_complete =
              iree_hal_remote_client_device_on_bulk_send_complete,
          .on_credit = iree_hal_remote_client_device_on_bulk_credit,
          .user_data = device,
      };
      iree_async_buffer_pool_t* channel_header_pool = header_pool;
      header_pool = NULL;  // Ownership transfers to create, including failure.
      status = iree_net_bulk_channel_create(endpoint, /*options=*/NULL,
                                            channel_header_pool, callbacks,
                                            host_allocator, &bulk_channel);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(bulk_channel);
    }

    if (iree_status_is_ok(status)) {
      iree_net_queue_channel_t* old_queue_channel =
          iree_hal_remote_client_device_publish_queue_channel(device,
                                                              queue_channel);
      queue_channel = NULL;
      iree_net_queue_channel_detach(old_queue_channel);
      iree_net_queue_channel_release(old_queue_channel);

      iree_net_bulk_channel_t* old_bulk_channel =
          iree_hal_remote_client_device_publish_bulk_channel(device,
                                                             bulk_channel);
      bulk_channel = NULL;
      iree_net_bulk_channel_detach(old_bulk_channel);
      iree_net_bulk_channel_release(old_bulk_channel);

      iree_hal_remote_client_device_store_state(
          device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
      iree_hal_remote_client_device_complete_connect(device, iree_ok_status());
    } else {
      iree_net_bulk_channel_release(bulk_channel);
      iree_async_buffer_pool_free(header_pool);
      iree_net_queue_channel_release(queue_channel);
      iree_hal_remote_client_device_store_state(
          device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
      iree_hal_remote_client_device_complete_connect(device, status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_remote_client_device_open_bulk_endpoint(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* queue_channel) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_channel);

  iree_hal_remote_client_bulk_endpoint_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      device->host_allocator, sizeof(*context), (void**)&context));
  context->device = device;
  context->queue_channel = queue_channel;

  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_hal_remote_client_device_on_bulk_endpoint_ready,
      .user_data = context,
  };
  iree_status_t status =
      iree_net_session_open_endpoint(device->session, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(device->host_allocator, context);
  }
  return status;
}
