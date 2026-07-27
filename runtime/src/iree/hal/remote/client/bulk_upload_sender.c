// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_upload_sender.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/channel/bulk/chunk_pool.h"

typedef struct iree_hal_remote_client_file_read_chunk_t {
  // Device retained while the async file read or bulk DATA send is active.
  iree_hal_remote_client_device_t* device;

  // Owning chunk pool descriptor for this read/send stage.
  iree_net_bulk_chunk_t* chunk;

  // Bulk transfer ID this chunk belongs to.
  uint64_t transfer_id;

  // Transfer byte offset for this chunk.
  uint64_t transfer_offset;

  // Source file byte offset for this chunk.
  uint64_t file_offset;

  // Number of bytes expected for this chunk.
  iree_host_size_t chunk_length;

  // Number of bytes read across partial async file read completions.
  iree_host_size_t file_progress;

  // Staging buffer lease holding file bytes until bulk DATA send completion.
  iree_async_buffer_lease_t lease;

  // Async file retained while partial reads may be resubmitted.
  iree_async_file_t* async_file;

  // Async file read operation.
  iree_async_file_read_operation_t read_op;
} iree_hal_remote_client_file_read_chunk_t;

iree_host_size_t iree_hal_remote_client_bulk_upload_sender_chunk_storage_size(
    void) {
  return sizeof(iree_hal_remote_client_file_read_chunk_t);
}

iree_host_size_t
iree_hal_remote_client_bulk_upload_sender_chunk_storage_alignment(void) {
  return iree_alignof(iree_hal_remote_client_file_read_chunk_t);
}

static iree_hal_remote_client_file_read_chunk_t*
iree_hal_remote_client_file_read_chunk_storage(iree_net_bulk_chunk_t* chunk) {
  return (iree_hal_remote_client_file_read_chunk_t*)
      iree_net_bulk_chunk_user_storage(chunk)
          .data;
}

static void iree_hal_remote_client_file_read_chunk_release(
    iree_hal_remote_client_file_read_chunk_t* chunk_context) {
  if (!chunk_context) return;
  iree_hal_remote_client_device_t* device = chunk_context->device;
  iree_net_bulk_chunk_t* chunk = chunk_context->chunk;
  iree_async_buffer_lease_release(&chunk_context->lease);
  iree_async_file_release(chunk_context->async_file);
  memset(chunk_context, 0, sizeof(*chunk_context));
  if (chunk) {
    iree_net_bulk_chunk_release(device->bulk_session.send_chunks, chunk);
  }
  iree_hal_device_release((iree_hal_device_t*)device);
}

void iree_hal_remote_client_bulk_upload_sender_deinitialize_transfer(
    iree_hal_remote_client_file_read_transfer_t* transfer) {
  if (transfer->pending_chunk) {
    iree_hal_remote_client_file_read_chunk_release(
        iree_hal_remote_client_file_read_chunk_storage(
            transfer->pending_chunk));
    transfer->pending_chunk = NULL;
  }
  iree_hal_file_release(transfer->source_file);
}

bool iree_hal_remote_client_bulk_upload_sender_mark_terminal_locked(
    iree_hal_remote_client_file_read_transfer_t* transfer) {
  transfer->flags |= IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL;
  if (transfer->pending_chunk) {
    iree_hal_remote_client_file_read_chunk_release(
        iree_hal_remote_client_file_read_chunk_storage(
            transfer->pending_chunk));
    transfer->pending_chunk = NULL;
    transfer->flags &=
        ~IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT;
  }
  return transfer->pending_send_count == 0 &&
         !iree_any_bit_set(
             transfer->flags,
             IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT);
}

static void iree_hal_remote_client_bulk_upload_sender_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  const uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(transfer);
  iree_hal_remote_client_bulk_upload_sender_deinitialize_transfer(
      &bulk_transfer->file_read);
  memset(bulk_transfer, 0, sizeof(*bulk_transfer));
  iree_net_bulk_transfer_table_remove(table, transfer_id);
}

static void iree_hal_remote_client_file_read_chunk_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags);

static iree_status_t iree_hal_remote_client_file_read_chunk_submit_next(
    iree_hal_remote_client_file_read_chunk_t* chunk_context) {
  const iree_host_size_t remaining_length =
      chunk_context->chunk_length - chunk_context->file_progress;
  iree_async_operation_zero(&chunk_context->read_op.base,
                            sizeof(chunk_context->read_op));
  iree_async_operation_initialize(
      &chunk_context->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_remote_client_file_read_chunk_complete, chunk_context);
  chunk_context->read_op.offset =
      chunk_context->file_offset + chunk_context->file_progress;
  chunk_context->read_op.buffer = iree_async_span_make(
      chunk_context->lease.span.region,
      chunk_context->lease.span.offset + chunk_context->file_progress,
      remaining_length);
  chunk_context->read_op.file = chunk_context->async_file;
  return iree_async_proactor_submit_one(chunk_context->device->proactor,
                                        &chunk_context->read_op.base);
}

static iree_status_t iree_hal_remote_client_bulk_try_send_pending_upload_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  iree_hal_remote_client_file_read_transfer_t* transfer =
      &bulk_transfer->file_read;
  iree_net_bulk_chunk_t* chunk = transfer->pending_chunk;
  if (!chunk) return iree_ok_status();

  iree_hal_remote_client_file_read_chunk_t* chunk_context =
      iree_hal_remote_client_file_read_chunk_storage(chunk);
  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  const uint64_t chunk_end =
      chunk_context->transfer_offset + chunk_context->chunk_length;
  iree_net_bulk_frame_flags_t flags = chunk_end == total_length
                                          ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                          : IREE_NET_BULK_FRAME_FLAG_NONE;
  iree_async_span_t chunk_span =
      iree_async_span_from_ptr(iree_async_span_ptr(chunk_context->lease.span),
                               chunk_context->chunk_length);
  iree_async_span_list_t chunk_payload =
      iree_async_span_list_make(&chunk_span, 1);
  iree_status_t failure_status = iree_ok_status();
  iree_hal_remote_bulk_channel_send_result_t send_result =
      iree_hal_remote_bulk_channel_send_data(
          channel, transfer_id, chunk_context->transfer_offset,
          transfer->next_sequence, flags, chunk_payload,
          (uint64_t)(uintptr_t)chunk, &failure_status);
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
    return iree_ok_status();
  }
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
    return failure_status;
  }
  transfer->pending_chunk = NULL;
  transfer->send_offset = chunk_end;
  ++transfer->next_sequence;
  ++transfer->pending_send_count;
  return iree_ok_status();
}

static void iree_hal_remote_client_file_read_chunk_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_remote_client_file_read_chunk_t* chunk_context =
      (iree_hal_remote_client_file_read_chunk_t*)user_data;
  iree_hal_remote_client_device_t* device = chunk_context->device;

  if (iree_status_is_ok(status) && chunk_context->read_op.bytes_read > 0) {
    chunk_context->file_progress += chunk_context->read_op.bytes_read;
    if (chunk_context->file_progress < chunk_context->chunk_length) {
      status =
          iree_hal_remote_client_file_read_chunk_submit_next(chunk_context);
      if (iree_status_is_ok(status)) return;
    }
  } else if (iree_status_is_ok(status) &&
             chunk_context->file_progress < chunk_context->chunk_length) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "short async file read: requested %" PRIhsz " bytes, got %" PRIhsz,
        chunk_context->chunk_length, chunk_context->file_progress);
  }

  iree_net_bulk_channel_t* channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  const bool file_failure = !iree_status_is_ok(status);
  bool release_chunk = false;
  bool chunk_attached_to_transfer = false;
  bool send_abort = false;
  uint64_t abort_transfer_id = 0;
  iree_status_t transport_status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                          chunk_context->transfer_id);
  iree_hal_remote_client_file_read_transfer_t* file_transfer = NULL;
  if (table_transfer) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
      file_transfer = &bulk_transfer->file_read;
    }
  }
  if (file_transfer &&
      iree_any_bit_set(
          file_transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL)) {
    file_transfer->flags &=
        ~IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT;
    if (file_transfer->pending_send_count == 0) {
      iree_hal_remote_client_bulk_upload_sender_release_transfer(
          device->bulk_session.transfers, table_transfer);
      table_transfer = NULL;
    }
    iree_status_free(status);
    status = iree_ok_status();
    release_chunk = true;
  } else if (iree_status_is_ok(status) && file_transfer && channel) {
    file_transfer->pending_chunk = chunk_context->chunk;
    chunk_attached_to_transfer = true;
    status = iree_hal_remote_client_bulk_try_send_pending_upload_locked(
        device, channel, table_transfer);
  } else if (iree_status_is_ok(status) && !channel) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "remote bulk channel is not available");
  }

  if (!iree_status_is_ok(status)) {
    if (file_failure && table_transfer && channel) {
      send_abort = true;
      abort_transfer_id = chunk_context->transfer_id;
      iree_status_free(status);
      status = iree_ok_status();
    } else if (file_failure && !table_transfer) {
      iree_status_free(status);
      status = iree_ok_status();
    } else {
      transport_status = status;
      status = iree_ok_status();
    }
    if (table_transfer) {
      iree_hal_remote_client_bulk_upload_sender_release_transfer(
          device->bulk_session.transfers, table_transfer);
      table_transfer = NULL;
    }
    release_chunk = !chunk_attached_to_transfer;
  } else if (!table_transfer) {
    release_chunk = true;
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (release_chunk) {
    iree_hal_remote_client_file_read_chunk_release(chunk_context);
  }
  if (send_abort) {
    transport_status = iree_net_bulk_channel_send_abort(
        channel, abort_transfer_id, iree_async_span_list_empty(),
        /*operation_user_data=*/0);
  }
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_fail(device, transport_status);
  }
}

iree_status_t iree_hal_remote_client_bulk_begin_file_read(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_device_size_t length,
    uint64_t* out_transfer_id) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(source_file_view);
  IREE_ASSERT_ARGUMENT(out_transfer_id);
  *out_transfer_id = 0;

  const uint64_t source_length = (uint64_t)length;
  uint64_t source_end = source_offset;
  const bool source_range_overflow = source_offset > UINT64_MAX - source_length;
  if (!source_range_overflow) {
    source_end = source_offset + source_length;
  }
  if (source_range_overflow || source_offset > source_file_view->length ||
      source_length > source_file_view->length - source_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "remote queue_read source range [%" PRIu64 ", %" PRIu64
        ") exceeds client file length %" PRIu64,
        source_offset, source_end, source_file_view->length);
  }
  if (source_file_view->kind !=
          IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION &&
      source_file_view->kind != IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote queue_read bulk upload currently requires a host allocation "
        "or async source file");
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
    bulk_transfer->kind = IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ;
    bulk_transfer->file_read.source_file = source_file;
    iree_hal_file_retain(bulk_transfer->file_read.source_file);
    bulk_transfer->file_read.source_file_view = *source_file_view;
    bulk_transfer->file_read.source_offset = source_offset;
    bulk_transfer->file_read.total_length = (uint64_t)length;
    *out_transfer_id = iree_net_bulk_transfer_id(transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

iree_status_t iree_hal_remote_client_bulk_begin_buffer_unmap_write(
    iree_hal_remote_client_device_t* device,
    iree_const_byte_span_t source_bytes, uint64_t* out_transfer_id) {
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
        device->bulk_session.transfers, (uint64_t)source_bytes.data_length,
        /*user_value=*/0, &transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind = IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ;
    bulk_transfer->file_read.source_file = NULL;
    bulk_transfer->file_read.source_file_view =
        (iree_hal_remote_client_file_view_t){
            .kind = IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION,
            .access = IREE_HAL_MEMORY_ACCESS_READ,
            .length = (uint64_t)source_bytes.data_length,
            .host_allocation = iree_make_byte_span((void*)source_bytes.data,
                                                   source_bytes.data_length),
        };
    bulk_transfer->file_read.source_offset = 0;
    bulk_transfer->file_read.total_length = (uint64_t)source_bytes.data_length;
    *out_transfer_id = iree_net_bulk_transfer_id(transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

static iree_status_t iree_hal_remote_client_bulk_submit_async_file_read_locked(
    iree_hal_remote_client_device_t* device,
    iree_net_bulk_transfer_t* table_transfer, uint64_t chunk_offset,
    iree_host_size_t chunk_length) {
  iree_net_bulk_chunk_t* chunk = NULL;
  iree_status_t status = iree_net_bulk_chunk_pool_acquire(
      device->bulk_session.send_chunks,
      iree_net_bulk_transfer_id(table_transfer), chunk_offset, /*sequence=*/0,
      IREE_NET_BULK_FRAME_FLAG_NONE, iree_const_byte_span_empty(),
      /*lease=*/NULL, /*user_value=*/0, &chunk);
  if (!iree_status_is_ok(status)) return status;

  iree_hal_remote_client_file_read_chunk_t* chunk_context =
      iree_hal_remote_client_file_read_chunk_storage(chunk);
  memset(chunk_context, 0, sizeof(*chunk_context));
  chunk_context->device = device;
  iree_hal_device_retain((iree_hal_device_t*)device);
  chunk_context->chunk = chunk;
  chunk_context->transfer_id = iree_net_bulk_transfer_id(table_transfer);
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  chunk_context->transfer_offset = chunk_offset;
  chunk_context->file_offset =
      bulk_transfer->file_read.source_offset + chunk_offset;
  chunk_context->chunk_length = chunk_length;
  chunk_context->async_file =
      bulk_transfer->file_read.source_file_view.async_file;
  iree_async_file_retain(chunk_context->async_file);

  status = iree_async_buffer_pool_acquire(
      iree_hal_remote_recv_pool_buffer_pool(device->recv_pool),
      &chunk_context->lease);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_file_read_chunk_submit_next(chunk_context);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_file_read_chunk_release(chunk_context);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_bulk_try_send_upload_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
    return iree_ok_status();
  }
  iree_hal_remote_client_file_read_transfer_t* transfer =
      &bulk_transfer->file_read;
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL)) {
    if (transfer->pending_send_count == 0 &&
        !iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT)) {
      iree_hal_remote_client_bulk_upload_sender_release_transfer(
          device->bulk_session.transfers, table_transfer);
    }
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT)) {
    if (transfer->pending_send_count == 0) {
      iree_hal_remote_client_bulk_upload_sender_release_transfer(
          device->bulk_session.transfers, table_transfer);
    }
    return iree_ok_status();
  }
  if (!channel) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bulk channel is not available");
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_START_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_start(
            channel, transfer_id, total_length, IREE_NET_BULK_FRAME_FLAG_NONE,
            transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_START_SENT;
    ++transfer->pending_send_count;
  }

  if (transfer->source_file_view.kind ==
      IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE) {
    IREE_RETURN_IF_ERROR(
        iree_hal_remote_client_bulk_try_send_pending_upload_locked(
            device, channel, table_transfer));
    if (!iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT) &&
        transfer->send_offset < total_length &&
        iree_net_bulk_channel_remote_chunk_credit_count(channel) > 0) {
      const uint64_t remaining_length = total_length - transfer->send_offset;
      const iree_host_size_t chunk_length = (iree_host_size_t)iree_min(
          remaining_length, (uint64_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
      iree_status_t status =
          iree_hal_remote_client_bulk_submit_async_file_read_locked(
              device, table_transfer, transfer->send_offset, chunk_length);
      if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
        iree_status_free(status);
        return iree_ok_status();
      }
      if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        iree_hal_remote_client_bulk_upload_sender_release_transfer(
            device->bulk_session.transfers, table_transfer);
        return iree_net_bulk_channel_send_abort(channel, transfer_id,
                                                iree_async_span_list_empty(),
                                                /*operation_user_data=*/0);
      }
      transfer->flags |=
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT;
    }
  } else {
    while (transfer->send_offset < total_length) {
      if (iree_net_bulk_channel_remote_chunk_credit_count(channel) == 0) {
        return iree_ok_status();
      }
      const uint64_t remaining_length = total_length - transfer->send_offset;
      const iree_host_size_t chunk_length = (iree_host_size_t)iree_min(
          remaining_length, (uint64_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
      const uint64_t chunk_end = transfer->send_offset + chunk_length;
      iree_net_bulk_frame_flags_t flags =
          chunk_end == total_length ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                    : IREE_NET_BULK_FRAME_FLAG_NONE;
      iree_async_span_t chunk_span = iree_async_span_from_ptr(
          transfer->source_file_view.host_allocation.data +
              (iree_host_size_t)(transfer->source_offset +
                                 transfer->send_offset),
          chunk_length);
      iree_async_span_list_t chunk_payload =
          iree_async_span_list_make(&chunk_span, 1);
      iree_status_t failure_status = iree_ok_status();
      iree_hal_remote_bulk_channel_send_result_t send_result =
          iree_hal_remote_bulk_channel_send_data(
              channel, transfer_id, transfer->send_offset,
              transfer->next_sequence, flags, chunk_payload, transfer_id,
              &failure_status);
      if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
        return iree_ok_status();
      }
      if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
        return failure_status;
      }
      transfer->send_offset = chunk_end;
      ++transfer->next_sequence;
      ++transfer->pending_send_count;
    }
  }

  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT) &&
      !transfer->pending_chunk &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT) &&
      transfer->send_offset == total_length) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_complete(
            channel, transfer_id, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT;
    ++transfer->pending_send_count;
  }

  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT) &&
      transfer->pending_send_count == 0) {
    iree_hal_remote_client_bulk_upload_sender_release_transfer(
        device->bulk_session.transfers, table_transfer);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_bulk_upload_sender_upload(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  IREE_ASSERT_ARGUMENT(device);
  if (!transfer_id) return iree_ok_status();

  iree_net_bulk_channel_t* channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  if (!channel) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bulk channel is not available");
  }

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  iree_status_t status = iree_ok_status();
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk upload unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    status = iree_hal_remote_client_bulk_try_send_upload_locked(device, channel,
                                                                transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

static void iree_hal_remote_client_bulk_collect_ready_upload(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_transfer_id_list_t* id_list =
      (iree_hal_remote_client_bulk_transfer_id_list_t*)user_data;
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
    return;
  }
  if (iree_any_bit_set(
          bulk_transfer->file_read.flags,
          IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT)) {
    return;
  }
  if (id_list->transfer_count >= IREE_ARRAYSIZE(id_list->transfer_ids)) return;
  id_list->transfer_ids[id_list->transfer_count++] =
      iree_net_bulk_transfer_id(table_transfer);
}

iree_status_t iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel) {
  iree_hal_remote_client_bulk_transfer_id_list_t id_list;
  memset(&id_list, 0, sizeof(id_list));
  iree_net_bulk_transfer_table_visit(
      device->bulk_session.transfers,
      iree_hal_remote_client_bulk_collect_ready_upload, &id_list);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < id_list.transfer_count && iree_status_is_ok(status); ++i) {
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            id_list.transfer_ids[i]);
    if (table_transfer) {
      status = iree_hal_remote_client_bulk_try_send_upload_locked(
          device, channel, table_transfer);
    }
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_upload_sender_send_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    return status;
  }

  // Client transfer IDs are odd and DATA chunk pointers are aligned, so the
  // low bit classifies which completion path owns |operation_user_data|.
  if ((operation_user_data & 1u) == 0) {
    iree_net_bulk_chunk_t* chunk =
        (iree_net_bulk_chunk_t*)(uintptr_t)operation_user_data;
    iree_hal_remote_client_file_read_chunk_t* chunk_context =
        iree_hal_remote_client_file_read_chunk_storage(chunk);
    const uint64_t transfer_id = chunk_context->transfer_id;
    iree_status_t transport_status = iree_ok_status();
    iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            transfer_id);
    if (!table_transfer) {
      if (!iree_status_is_ok(status)) {
        transport_status = status;
        status = iree_ok_status();
      }
    } else {
      iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_client_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
        iree_hal_remote_client_file_read_transfer_t* transfer =
            &bulk_transfer->file_read;
        if (transfer->pending_send_count > 0) {
          --transfer->pending_send_count;
        }
        transfer->flags &=
            ~IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT;
        if (iree_any_bit_set(
                transfer->flags,
                IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL)) {
          iree_status_free(status);
          status = iree_ok_status();
          if (transfer->pending_send_count == 0) {
            iree_hal_remote_client_bulk_upload_sender_release_transfer(
                device->bulk_session.transfers, table_transfer);
            table_transfer = NULL;
          }
        } else if (!iree_status_is_ok(status)) {
          transport_status = status;
          status = iree_ok_status();
          iree_hal_remote_client_bulk_upload_sender_release_transfer(
              device->bulk_session.transfers, table_transfer);
          table_transfer = NULL;
        }
      } else if (!iree_status_is_ok(status)) {
        transport_status = status;
        status = iree_ok_status();
      }
    }
    iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

    iree_hal_remote_client_file_read_chunk_release(chunk_context);

    if (iree_status_is_ok(transport_status)) {
      iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
      transport_status =
          iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
              device, channel);
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
    }
    return transport_status;
  }

  const uint64_t transfer_id = operation_user_data;
  iree_status_t transport_status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                          transfer_id);
  if (!table_transfer) {
    if (!iree_status_is_ok(status)) {
      transport_status = status;
      status = iree_ok_status();
    }
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
      iree_hal_remote_client_file_read_transfer_t* transfer =
          &bulk_transfer->file_read;
      if (transfer->pending_send_count > 0) {
        --transfer->pending_send_count;
      }
      if (iree_any_bit_set(
              transfer->flags,
              IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL)) {
        iree_status_free(status);
        status = iree_ok_status();
        if (transfer->pending_send_count == 0 &&
            !iree_any_bit_set(
                transfer->flags,
                IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT)) {
          iree_hal_remote_client_bulk_upload_sender_release_transfer(
              device->bulk_session.transfers, table_transfer);
        }
      } else if (iree_status_is_ok(status)) {
        transport_status = iree_hal_remote_client_bulk_try_send_upload_locked(
            device, channel, table_transfer);
      } else {
        transport_status = status;
        status = iree_ok_status();
        iree_hal_remote_client_bulk_upload_sender_release_transfer(
            device->bulk_session.transfers, table_transfer);
      }
    } else if (!iree_status_is_ok(status)) {
      transport_status = status;
      status = iree_ok_status();
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  return transport_status;
}
