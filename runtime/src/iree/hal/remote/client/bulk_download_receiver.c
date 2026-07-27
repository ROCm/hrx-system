// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_download_receiver.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/channel/bulk/chunk_pool.h"
#include "iree/net/channel/bulk/transfer_table.h"

typedef struct iree_hal_remote_client_file_write_chunk_t {
  // Device retained while the async file write is active.
  iree_hal_remote_client_device_t* device;

  // Owning chunk pool descriptor retaining the received DATA lease.
  iree_net_bulk_chunk_t* chunk;

  // Bulk transfer ID this chunk belongs to.
  uint64_t transfer_id;

  // Target file byte offset for this chunk.
  uint64_t file_offset;

  // Number of bytes expected for this chunk.
  iree_host_size_t chunk_length;

  // Number of bytes written across partial async file write completions.
  iree_host_size_t file_progress;

  // Async file retained while partial writes may be resubmitted.
  iree_async_file_t* async_file;

  // Async file write operation.
  iree_async_file_write_operation_t write_op;
} iree_hal_remote_client_file_write_chunk_t;

iree_host_size_t
iree_hal_remote_client_bulk_download_receiver_chunk_storage_size(void) {
  return sizeof(iree_hal_remote_client_file_write_chunk_t);
}

iree_host_size_t
iree_hal_remote_client_bulk_download_receiver_chunk_storage_alignment(void) {
  return iree_alignof(iree_hal_remote_client_file_write_chunk_t);
}

static iree_hal_remote_client_file_write_chunk_t*
iree_hal_remote_client_file_write_chunk_storage(iree_net_bulk_chunk_t* chunk) {
  return (iree_hal_remote_client_file_write_chunk_t*)
      iree_net_bulk_chunk_user_storage(chunk)
          .data;
}

static void iree_hal_remote_client_file_write_chunk_release(
    iree_hal_remote_client_file_write_chunk_t* chunk_context) {
  if (!chunk_context) return;
  iree_hal_remote_client_device_t* device = chunk_context->device;
  iree_net_bulk_chunk_t* chunk = chunk_context->chunk;
  iree_async_file_release(chunk_context->async_file);
  memset(chunk_context, 0, sizeof(*chunk_context));
  if (chunk) {
    iree_net_bulk_chunk_release(device->bulk_session.receive_chunks, chunk);
  }
  iree_hal_device_release((iree_hal_device_t*)device);
}

void iree_hal_remote_client_bulk_download_receiver_deinitialize_transfer(
    iree_hal_remote_client_file_write_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_tracker_deinitialize(
      &transfer->receive_tracker);
  iree_hal_file_release(transfer->target_file);
}

static void iree_hal_remote_client_bulk_download_receiver_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  const uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(transfer);
  iree_hal_remote_client_bulk_download_receiver_deinitialize_transfer(
      &bulk_transfer->file_write);
  memset(bulk_transfer, 0, sizeof(*bulk_transfer));
  iree_net_bulk_transfer_table_remove(table, transfer_id);
}

static void iree_hal_remote_client_bulk_download_receiver_cancel_transfer(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      iree_hal_remote_client_bulk_download_receiver_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
}

static void iree_hal_remote_client_file_write_notify_locked(
    iree_hal_remote_client_file_write_transfer_t* transfer,
    iree_status_t status) {
  iree_hal_remote_client_bulk_completion_callback_t callback =
      transfer->completion_callback;
  memset(&transfer->completion_callback, 0,
         sizeof(transfer->completion_callback));
  if (callback.fn) {
    callback.fn(callback.user_data, status);
  } else {
    iree_status_free(status);
  }
}

void iree_hal_remote_client_bulk_download_receiver_fail_locked(
    iree_hal_remote_client_file_write_transfer_t* transfer,
    iree_status_t status) {
  iree_hal_remote_client_file_write_notify_locked(transfer, status);
}

iree_status_t iree_hal_remote_client_bulk_begin_file_write(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* target_file,
    const iree_hal_remote_client_file_view_t* target_file_view,
    uint64_t target_offset, iree_device_size_t length,
    uint64_t* out_transfer_id) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_ASSERT_ARGUMENT(target_file_view);
  IREE_ASSERT_ARGUMENT(out_transfer_id);
  *out_transfer_id = 0;

  const uint64_t target_length = (uint64_t)length;
  uint64_t target_end = target_offset;
  const bool target_range_overflow = target_offset > UINT64_MAX - target_length;
  if (!target_range_overflow) {
    target_end = target_offset + target_length;
  }
  if (target_range_overflow || target_offset > target_file_view->length ||
      target_length > target_file_view->length - target_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "remote queue_write target range [%" PRIu64 ", %" PRIu64
        ") exceeds client file length %" PRIu64,
        target_offset, target_end, target_file_view->length);
  }

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = NULL;
  iree_status_t status =
      iree_hal_remote_client_bulk_session_check_active_locked(
          &device->bulk_session);
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_transfer_table_allocate_transfer(
        device->bulk_session.transfers, (uint64_t)length, /*user_value=*/0,
        &transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind = IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE;
    bulk_transfer->file_write.target_file = target_file;
    iree_hal_file_retain(bulk_transfer->file_write.target_file);
    bulk_transfer->file_write.target_file_view = *target_file_view;
    bulk_transfer->file_write.target_offset = target_offset;
    bulk_transfer->file_write.total_length = (uint64_t)length;
    status = iree_hal_remote_bulk_transfer_tracker_initialize(
        (uint64_t)length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
        device->host_allocator, &bulk_transfer->file_write.receive_tracker);
    if (iree_status_is_ok(status)) {
      *out_transfer_id = iree_net_bulk_transfer_id(transfer);
    } else {
      iree_hal_remote_client_bulk_download_receiver_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

iree_status_t iree_hal_remote_client_bulk_begin_buffer_map_read(
    iree_hal_remote_client_device_t* device, iree_byte_span_t target_bytes,
    iree_hal_remote_client_bulk_completion_callback_t callback,
    uint64_t* out_transfer_id) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_transfer_id);
  *out_transfer_id = 0;

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = NULL;
  iree_status_t status =
      iree_hal_remote_client_bulk_session_check_active_locked(
          &device->bulk_session);
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_transfer_table_allocate_transfer(
        device->bulk_session.transfers, (uint64_t)target_bytes.data_length,
        /*user_value=*/0, &transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind = IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE;
    bulk_transfer->file_write.target_file = NULL;
    bulk_transfer->file_write.target_file_view =
        (iree_hal_remote_client_file_view_t){
            .kind = IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION,
            .access = IREE_HAL_MEMORY_ACCESS_WRITE,
            .length = (uint64_t)target_bytes.data_length,
            .host_allocation = target_bytes,
        };
    bulk_transfer->file_write.target_offset = 0;
    bulk_transfer->file_write.total_length = (uint64_t)target_bytes.data_length;
    bulk_transfer->file_write.completion_callback = callback;
    status = iree_hal_remote_bulk_transfer_tracker_initialize(
        (uint64_t)target_bytes.data_length,
        IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH, device->host_allocator,
        &bulk_transfer->file_write.receive_tracker);
    if (iree_status_is_ok(status)) {
      *out_transfer_id = iree_net_bulk_transfer_id(transfer);
    } else {
      iree_hal_remote_client_bulk_download_receiver_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

static bool iree_hal_remote_client_file_write_is_ready_locked(
    const iree_hal_remote_client_file_write_transfer_t* transfer) {
  return !iree_any_bit_set(
             transfer->flags,
             IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_WRITE_FAILED) &&
         iree_any_bit_set(
             transfer->flags,
             IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_SERVER_COMPLETE) &&
         transfer->pending_write_count == 0 &&
         iree_hal_remote_bulk_transfer_tracker_is_complete(
             &transfer->receive_tracker);
}

static void iree_hal_remote_client_file_write_chunk_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags);

static iree_status_t iree_hal_remote_client_file_write_chunk_submit_next(
    iree_hal_remote_client_file_write_chunk_t* chunk_context) {
  const iree_host_size_t remaining_length =
      chunk_context->chunk_length - chunk_context->file_progress;
  const iree_const_byte_span_t payload =
      iree_net_bulk_chunk_payload(chunk_context->chunk);
  iree_async_operation_zero(&chunk_context->write_op.base,
                            sizeof(chunk_context->write_op));
  iree_async_operation_initialize(
      &chunk_context->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_remote_client_file_write_chunk_complete, chunk_context);
  chunk_context->write_op.file = chunk_context->async_file;
  chunk_context->write_op.offset =
      chunk_context->file_offset + chunk_context->file_progress;
  chunk_context->write_op.buffer = iree_async_span_from_ptr(
      (void*)(payload.data + chunk_context->file_progress), remaining_length);
  return iree_async_proactor_submit_one(chunk_context->device->proactor,
                                        &chunk_context->write_op.base);
}

static iree_status_t iree_hal_remote_client_bulk_ack_file_write(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id) {
  if (!channel) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bulk channel is not available");
  }
  iree_status_t status = iree_net_bulk_channel_send_complete(
      channel, transfer_id, /*operation_user_data=*/0);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            transfer_id);
    if (table_transfer) {
      iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_client_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
        iree_hal_remote_client_file_write_notify_locked(
            &bulk_transfer->file_write, iree_ok_status());
      }
    }
    iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
    iree_hal_remote_client_bulk_download_receiver_cancel_transfer(device,
                                                                  transfer_id);
  } else {
    iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            transfer_id);
    if (table_transfer) {
      iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_client_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
        iree_hal_remote_client_file_write_notify_locked(
            &bulk_transfer->file_write, iree_status_clone(status));
      }
    }
    iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  }
  return status;
}

static void iree_hal_remote_client_file_write_chunk_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_remote_client_file_write_chunk_t* chunk_context =
      (iree_hal_remote_client_file_write_chunk_t*)user_data;
  iree_hal_remote_client_device_t* device = chunk_context->device;

  if (iree_status_is_ok(status) && chunk_context->write_op.bytes_written > 0) {
    chunk_context->file_progress += chunk_context->write_op.bytes_written;
    if (chunk_context->file_progress < chunk_context->chunk_length) {
      status =
          iree_hal_remote_client_file_write_chunk_submit_next(chunk_context);
      if (iree_status_is_ok(status)) return;
    }
  } else if (iree_status_is_ok(status) &&
             chunk_context->file_progress < chunk_context->chunk_length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short async file write: requested %" PRIhsz " bytes, wrote %" PRIhsz,
        chunk_context->chunk_length, chunk_context->file_progress);
  }

  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  bool send_complete = false;
  bool send_abort = false;
  bool transfer_terminal = false;
  uint64_t transfer_id = chunk_context->transfer_id;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                          transfer_id);
  if (!table_transfer) {
    iree_status_free(status);
    status = iree_ok_status();
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      status = iree_status_join(
          status,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "async file write completed for non-download "
                           "transfer_id=%" PRIu64,
                           transfer_id));
    } else {
      iree_hal_remote_client_file_write_transfer_t* file_transfer =
          &bulk_transfer->file_write;
      if (file_transfer->pending_write_count > 0) {
        --file_transfer->pending_write_count;
      }
      transfer_terminal = iree_any_bit_set(
          file_transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL);
      if (transfer_terminal) {
        iree_status_free(status);
        status = iree_ok_status();
        if (file_transfer->pending_write_count == 0) {
          iree_hal_remote_client_bulk_download_receiver_release_transfer(
              device->bulk_session.transfers, table_transfer);
          table_transfer = NULL;
        }
      } else if (iree_status_is_ok(status)) {
        send_complete =
            iree_hal_remote_client_file_write_is_ready_locked(file_transfer);
      } else {
        file_transfer->flags |=
            IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_WRITE_FAILED;
        send_abort = true;
      }
    }
    if (table_transfer && (!iree_status_is_ok(status) || send_abort)) {
      iree_hal_remote_client_bulk_download_receiver_release_transfer(
          device->bulk_session.transfers, table_transfer);
      table_transfer = NULL;
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  iree_hal_remote_client_file_write_chunk_release(chunk_context);

  iree_status_t transport_status = iree_ok_status();
  if (transfer_terminal || !table_transfer) {
    // Terminal failure already closed channel admission and notified any local
    // waiter. Admitted writes retire without generating more transport work.
  } else if (bulk_channel) {
    transport_status = iree_net_bulk_channel_send_credit(
        bulk_channel, /*credit_delta=*/1, /*operation_user_data=*/0);
  } else {
    transport_status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                        "remote bulk channel is not available");
  }
  if (iree_status_is_ok(transport_status) && send_complete) {
    transport_status = iree_hal_remote_client_bulk_ack_file_write(
        device, bulk_channel, transfer_id);
  }
  if (iree_status_is_ok(transport_status) && send_abort) {
    transport_status = iree_net_bulk_channel_send_abort(
        bulk_channel, transfer_id, iree_async_span_list_empty(),
        /*operation_user_data=*/0);
  }
  iree_status_free(status);
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_fail(device, transport_status);
  }
}

static iree_status_t iree_hal_remote_client_bulk_submit_async_file_write_locked(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_file_write_transfer_t* file_transfer,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease) {
  iree_net_bulk_chunk_t* chunk = NULL;
  iree_status_t status = iree_net_bulk_chunk_pool_acquire(
      device->bulk_session.receive_chunks, transfer_id, chunk_offset, sequence,
      flags, chunk_data, lease, /*user_value=*/0, &chunk);
  if (!iree_status_is_ok(status)) return status;

  iree_hal_remote_client_file_write_chunk_t* chunk_context =
      iree_hal_remote_client_file_write_chunk_storage(chunk);
  memset(chunk_context, 0, sizeof(*chunk_context));
  chunk_context->device = device;
  iree_hal_device_retain((iree_hal_device_t*)device);
  chunk_context->chunk = chunk;
  chunk_context->transfer_id = transfer_id;
  chunk_context->file_offset = file_transfer->target_offset + chunk_offset;
  chunk_context->chunk_length = chunk_data.data_length;
  chunk_context->async_file = file_transfer->target_file_view.async_file;
  iree_async_file_retain(chunk_context->async_file);
  ++file_transfer->pending_write_count;

  status = iree_hal_remote_client_file_write_chunk_submit_next(chunk_context);
  if (!iree_status_is_ok(status)) {
    --file_transfer->pending_write_count;
    iree_hal_remote_client_file_write_chunk_release(chunk_context);
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_download_receiver_on_data(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease, bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_status_t status = iree_ok_status();
  bool send_credit = false;
  bool send_abort = false;
  uint64_t abort_transfer_id = 0;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
    if (out_handled) *out_handled = true;
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return iree_ok_status();
    }
    if (out_handled) *out_handled = true;

    iree_hal_remote_client_file_write_transfer_t* file_transfer =
        &bulk_transfer->file_write;
    if (iree_any_bit_set(
            file_transfer->flags,
            IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL)) {
      status = iree_hal_remote_client_bulk_session_check_active_locked(
          &device->bulk_session);
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return status;
    }
    const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
    const bool chunk_range_overflow = chunk_offset > UINT64_MAX - chunk_length;
    const uint64_t chunk_end =
        chunk_range_overflow ? UINT64_MAX : chunk_offset + chunk_length;
    const bool final_chunk =
        iree_all_bits_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
    const bool expected_final_chunk =
        !chunk_range_overflow && chunk_end == file_transfer->total_length;
    if (chunk_range_overflow || chunk_end > file_transfer->total_length) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "remote client bulk DATA range [%" PRIu64 ", %" PRIu64
          ") exceeds transfer length %" PRIu64,
          chunk_offset, chunk_end, file_transfer->total_length);
    } else if (final_chunk != expected_final_chunk) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "remote client bulk DATA final flag mismatch for "
                           "transfer_id=%" PRIu64,
                           transfer_id);
    } else {
      status = iree_hal_remote_bulk_transfer_tracker_record_chunk(
          &file_transfer->receive_tracker, chunk_offset,
          chunk_data.data_length);
      if (iree_status_is_ok(status) &&
          file_transfer->target_file_view.kind ==
              IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION) {
        const uint64_t target_offset =
            file_transfer->target_offset + chunk_offset;
        memcpy(file_transfer->target_file_view.host_allocation.data +
                   (iree_host_size_t)target_offset,
               chunk_data.data, chunk_data.data_length);
        send_credit = true;
      } else if (iree_status_is_ok(status) &&
                 file_transfer->target_file_view.kind ==
                     IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE) {
        status = iree_hal_remote_client_bulk_submit_async_file_write_locked(
            device, file_transfer, transfer_id, chunk_offset, sequence, flags,
            chunk_data, lease);
        if (!iree_status_is_ok(status) &&
            iree_status_code(status) != IREE_STATUS_RESOURCE_EXHAUSTED) {
          iree_hal_remote_client_file_write_notify_locked(
              file_transfer, iree_status_clone(status));
          iree_status_free(status);
          status = iree_ok_status();
          send_credit = true;
          send_abort = true;
          abort_transfer_id = transfer_id;
          iree_hal_remote_client_bulk_download_receiver_release_transfer(
              device->bulk_session.transfers, transfer);
          transfer = NULL;
        }
      } else if (iree_status_is_ok(status)) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "remote queue_write bulk DATA requires a host allocation or "
            "async target file");
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_remote_client_file_write_notify_locked(
          file_transfer, iree_status_clone(status));
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && send_credit) {
    if (channel) {
      status = iree_net_bulk_channel_send_credit(channel, /*credit_delta=*/1,
                                                 /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  if (iree_status_is_ok(status) && send_abort) {
    if (channel) {
      status = iree_net_bulk_channel_send_abort(channel, abort_transfer_id,
                                                iree_async_span_list_empty(),
                                                /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_download_receiver_on_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_status_t status = iree_ok_status();
  bool send_complete = false;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
    if (out_handled) *out_handled = true;
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return iree_ok_status();
    }
    if (out_handled) *out_handled = true;

    iree_hal_remote_client_file_write_transfer_t* file_transfer =
        &bulk_transfer->file_write;
    if (iree_any_bit_set(
            file_transfer->flags,
            IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL)) {
      status = iree_hal_remote_client_bulk_session_check_active_locked(
          &device->bulk_session);
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return status;
    }
    file_transfer->flags |=
        IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_SERVER_COMPLETE;
    if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
            &file_transfer->receive_tracker)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote client bulk COMPLETE before all DATA "
                                "for transfer_id=%" PRIu64,
                                transfer_id);
      iree_hal_remote_client_file_write_notify_locked(
          file_transfer, iree_status_clone(status));
    } else {
      send_complete =
          iree_hal_remote_client_file_write_is_ready_locked(file_transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && send_complete) {
    status = iree_hal_remote_client_bulk_ack_file_write(device, channel,
                                                        transfer_id);
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_download_receiver_on_abort(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      if (out_handled) *out_handled = true;
      iree_hal_remote_client_file_write_notify_locked(
          &bulk_transfer->file_write,
          iree_make_status(IREE_STATUS_ABORTED,
                           "remote peer aborted bulk transfer"));
      iree_hal_remote_client_bulk_download_receiver_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return iree_ok_status();
}
