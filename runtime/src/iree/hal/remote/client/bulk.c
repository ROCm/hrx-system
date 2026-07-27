// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk.h"

#include "iree/hal/remote/client/bulk_download_receiver.h"
#include "iree/hal/remote/client/bulk_profile_receiver.h"
#include "iree/hal/remote/client/bulk_upload_sender.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/bulk/chunk_pool.h"
#include "iree/net/channel/bulk/transfer_table.h"

// Header buffers used by the bulk channel frame sender. Bulk DATA payloads are
// not copied into this pool; only the 40-byte frame headers are retained until
// send completion.
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT 128
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE 128

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status);

static void iree_hal_remote_client_bulk_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer);

static void iree_hal_remote_client_bulk_transfer_deinitialize(
    iree_hal_remote_client_bulk_transfer_t* transfer) {
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ:
      iree_hal_remote_client_bulk_upload_sender_deinitialize_transfer(
          &transfer->file_read);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE:
      iree_hal_remote_client_bulk_download_receiver_deinitialize_transfer(
          &transfer->file_write);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE:
      iree_hal_remote_client_bulk_profile_receiver_free_transfer(
          transfer->profile_receive);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_client_bulk_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_client_bulk_transfer_deinitialize(
      iree_hal_remote_client_bulk_transfer_storage(transfer));
  iree_net_bulk_transfer_table_remove(table, transfer_id);
}

static void iree_hal_remote_client_bulk_deinitialize_transfer(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_client_bulk_transfer_deinitialize(
      iree_hal_remote_client_bulk_transfer_storage(transfer));
}

iree_status_t iree_hal_remote_client_device_initialize_bulk_transfers(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_session_transfer_options_t options =
      iree_hal_remote_client_bulk_session_transfer_options_default();
  options.transfer_user_storage_size =
      sizeof(iree_hal_remote_client_bulk_transfer_t);
  options.transfer_user_storage_alignment =
      iree_alignof(iree_hal_remote_client_bulk_transfer_t);
  options.chunk_user_storage_size = iree_max(
      iree_hal_remote_client_bulk_upload_sender_chunk_storage_size(),
      iree_hal_remote_client_bulk_download_receiver_chunk_storage_size());
  options.chunk_user_storage_alignment = iree_max(
      iree_hal_remote_client_bulk_upload_sender_chunk_storage_alignment(),
      iree_hal_remote_client_bulk_download_receiver_chunk_storage_alignment());
  return iree_hal_remote_client_bulk_session_initialize_transfers(
      &device->bulk_session, &options, device->host_allocator);
}

static void iree_hal_remote_client_bulk_deinitialize_transfers_locked(
    void* user_data, iree_net_bulk_transfer_table_t* transfers) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_bulk_profile_receiver_release_all_locked(device);
  if (transfers) {
    iree_net_bulk_transfer_table_visit(
        transfers, iree_hal_remote_client_bulk_deinitialize_transfer, NULL);
    iree_net_bulk_transfer_table_clear(transfers);
  }
}

void iree_hal_remote_client_device_deinitialize_bulk_transfers(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_session_deinitialize_transfers(
      &device->bulk_session,
      iree_hal_remote_client_bulk_deinitialize_transfers_locked, device);
}

typedef struct iree_hal_remote_client_bulk_failure_context_t {
  // Preserved device failure cloned for local transfer waiters.
  iree_status_t status;

  // Transfer IDs that have no admitted async or transport ownership.
  iree_hal_remote_client_bulk_transfer_id_list_t releasable;
} iree_hal_remote_client_bulk_failure_context_t;

static void iree_hal_remote_client_bulk_mark_failed(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_failure_context_t* context =
      (iree_hal_remote_client_bulk_failure_context_t*)user_data;
  iree_hal_remote_client_bulk_transfer_t* transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  bool releasable = false;
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ:
      releasable =
          iree_hal_remote_client_bulk_upload_sender_mark_terminal_locked(
              &transfer->file_read);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE:
      transfer->file_write.flags |=
          IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL;
      iree_hal_remote_client_bulk_download_receiver_fail_locked(
          &transfer->file_write, iree_status_clone(context->status));
      releasable = transfer->file_write.pending_write_count == 0;
      break;
    default:
      break;
  }
  if (releasable && context->releasable.transfer_count <
                        IREE_ARRAYSIZE(context->releasable.transfer_ids)) {
    context->releasable.transfer_ids[context->releasable.transfer_count++] =
        iree_net_bulk_transfer_id(table_transfer);
  }
}

void iree_hal_remote_client_bulk_fail_transfers(
    iree_hal_remote_client_device_t* device, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT(!iree_status_is_ok(status));

  iree_hal_profile_sink_t* profile_sink = NULL;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  if (!iree_status_is_ok(device->bulk_session.terminal_status)) {
    iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
    iree_status_free(status);
    return;
  }
  device->bulk_session.terminal_status = status;

  // Profile transfers have no external async ownership and can be released
  // immediately. File transfers retain only descriptors that admitted work
  // still identifies by transfer ID.
  profile_sink = device->bulk_session.profile_sink;
  device->bulk_session.profile_sink = NULL;
  iree_hal_remote_client_bulk_profile_receiver_release_all_locked(device);

  iree_hal_remote_client_bulk_failure_context_t context;
  memset(&context, 0, sizeof(context));
  context.status = device->bulk_session.terminal_status;
  iree_net_bulk_transfer_table_visit(device->bulk_session.transfers,
                                     iree_hal_remote_client_bulk_mark_failed,
                                     &context);
  for (iree_host_size_t i = 0; i < context.releasable.transfer_count; ++i) {
    iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
        device->bulk_session.transfers, context.releasable.transfer_ids[i]);
    if (transfer) {
      iree_hal_remote_client_bulk_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  iree_hal_profile_sink_release(profile_sink);
}

void iree_hal_remote_client_bulk_cancel_transfer(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  if (!transfer_id) return;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    bool releasable = false;
    switch (bulk_transfer->kind) {
      case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ:
        releasable =
            iree_hal_remote_client_bulk_upload_sender_mark_terminal_locked(
                &bulk_transfer->file_read);
        break;
      case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE:
        bulk_transfer->file_write.flags |=
            IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL;
        iree_hal_remote_client_bulk_download_receiver_fail_locked(
            &bulk_transfer->file_write,
            iree_make_status(IREE_STATUS_CANCELLED,
                             "remote bulk transfer cancelled"));
        releasable = bulk_transfer->file_write.pending_write_count == 0;
        break;
      default:
        releasable = true;
        break;
    }
    if (releasable) {
      iree_hal_remote_client_bulk_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
}

iree_status_t iree_hal_remote_client_bulk_upload_file_read(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  iree_status_t status =
      iree_hal_remote_client_bulk_upload_sender_upload(device, transfer_id);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(
        device, iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_upload_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  return iree_hal_remote_client_bulk_upload_file_read(device, transfer_id);
}

void iree_hal_remote_client_bulk_end_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
}

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
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote client bulk START flags: 0x%02x", flags);
  }

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_status_t status =
      iree_hal_remote_client_bulk_session_check_active_locked(
          &device->bulk_session);
  iree_net_bulk_transfer_t* transfer = NULL;
  if (iree_status_is_ok(status)) {
    transfer = iree_net_bulk_transfer_table_lookup(
        device->bulk_session.transfers, transfer_id);
  }
  if (iree_status_is_ok(status) && !transfer) {
    status = iree_hal_remote_client_bulk_profile_receiver_begin_locked(
        device, transfer_id, total_size);
  } else if (iree_status_is_ok(status) &&
             iree_net_bulk_transfer_total_size(transfer) != total_size) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote client bulk START size mismatch for transfer_id=%" PRIu64,
        transfer_id);
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      iree_hal_remote_client_bulk_download_receiver_fail_locked(
          &bulk_transfer->file_write, iree_status_clone(status));
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_t* bulk_channel =
        iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
    if (bulk_channel) {
      status = iree_net_bulk_channel_refresh_credit(bulk_channel,
                                                    /*operation_user_data=*/0);
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote client bulk DATA flags: 0x%02x",
                            flags);
  }

  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  bool handled = false;
  iree_status_t status = iree_hal_remote_client_bulk_download_receiver_on_data(
      device, bulk_channel, transfer_id, chunk_offset, sequence, flags,
      chunk_data, lease, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  status = iree_hal_remote_client_bulk_profile_receiver_on_data(
      device, bulk_channel, transfer_id, chunk_offset, flags, chunk_data,
      &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "remote client bulk DATA received for unsupported "
                          "transfer_id=%" PRIu64,
                          transfer_id);
}

static iree_status_t iree_hal_remote_client_device_on_bulk_complete(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);

  bool handled = false;
  iree_status_t status =
      iree_hal_remote_client_bulk_download_receiver_on_complete(
          device, bulk_channel, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  status = iree_hal_remote_client_bulk_profile_receiver_on_complete(
      device, bulk_channel, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "remote client bulk COMPLETE received for "
                          "unsupported transfer_id=%" PRIu64,
                          transfer_id);
}

static iree_status_t iree_hal_remote_client_device_on_bulk_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  (void)abort_data;
  (void)lease;

  bool handled = false;
  iree_status_t status = iree_hal_remote_client_bulk_download_receiver_on_abort(
      device, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  status = iree_hal_remote_client_bulk_profile_receiver_on_abort(
      device, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
  return iree_ok_status();
}

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_device_fail(device, status);
}

static void iree_hal_remote_client_device_on_bulk_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    if (!iree_status_is_ok(status)) {
      iree_hal_remote_client_device_on_bulk_transport_error(user_data, status);
    }
    return;
  }

  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  iree_status_t transport_status =
      iree_hal_remote_client_bulk_upload_sender_send_complete(
          device, bulk_channel, operation_user_data, status);
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(user_data,
                                                          transport_status);
  }
}

static void iree_hal_remote_client_device_on_bulk_credit(
    void* user_data, uint32_t credit_delta, uint32_t available_credit_count) {
  (void)credit_delta;
  if (available_credit_count == 0) return;

  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_status_t status =
      iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
          device, bulk_channel);
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(user_data, status);
  }
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
    iree_status_free(status);
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
      const uint32_t initial_chunk_credit =
          (uint32_t)iree_net_bulk_chunk_pool_capacity(
              device->bulk_session.receive_chunks);
      status = iree_net_bulk_channel_send_credit(
          bulk_channel, initial_chunk_credit, /*operation_user_data=*/0);
    }

    if (iree_status_is_ok(status)) {
      if (iree_hal_remote_client_device_try_commit_connected(
              device, queue_channel, bulk_channel)) {
        queue_channel = NULL;
        bulk_channel = NULL;
      }
    } else {
      iree_hal_remote_client_device_fail(device, status);
    }
    iree_net_bulk_channel_release(bulk_channel);
    iree_async_buffer_pool_release(header_pool);
    iree_net_queue_channel_release(queue_channel);
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
