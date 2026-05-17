// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk.h"

#include "iree/async/operations/file.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
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
#define IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH (64 * 1024)

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status);

typedef enum iree_hal_remote_client_bulk_transfer_kind_e {
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_EMPTY = 0u,
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ = 1u,
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE = 2u,
} iree_hal_remote_client_bulk_transfer_kind_e;
typedef uint8_t iree_hal_remote_client_bulk_transfer_kind_t;

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

typedef struct iree_hal_remote_client_file_read_transfer_t {
  // Source HAL file retained for the transfer lifetime.
  iree_hal_file_t* source_file;

  // Resolved source file capabilities captured at queue_read submission.
  iree_hal_remote_client_file_view_t source_file_view;

  // Source file byte offset for the first transfer byte.
  uint64_t source_offset;

  // Total number of bytes sent by this transfer.
  uint64_t total_length;

  // Number of host bytes submitted as DATA frames.
  uint64_t send_offset;

  // Async file read chunk ready to send or awaiting send retry.
  iree_net_bulk_chunk_t* pending_chunk;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // Number of bulk START/DATA/COMPLETE frames still awaiting send completion.
  uint32_t pending_send_count;

  // True while an async file read chunk is reading or awaiting send completion.
  bool chunk_in_flight;

  // True once the START frame has been submitted.
  bool start_sent;

  // True once the COMPLETE frame has been submitted.
  bool complete_sent;
} iree_hal_remote_client_file_read_transfer_t;

typedef struct iree_hal_remote_client_file_write_transfer_t {
  // Target HAL file retained for the transfer lifetime.
  iree_hal_file_t* target_file;

  // Resolved target file capabilities captured at queue_write submission.
  iree_hal_remote_client_file_view_t target_file_view;

  // Target file byte offset for the first transfer byte.
  uint64_t target_offset;

  // Total number of bytes expected from the server.
  uint64_t total_length;

  // Tracks fixed-grid DATA chunks received from the server.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // Number of async file writes still in flight.
  uint32_t pending_write_count;

  // True once the server has sent its COMPLETE marker.
  bool server_complete;

  // True once an async file write failed and ABORT was sent or queued.
  bool write_failed;
} iree_hal_remote_client_file_write_transfer_t;

typedef struct iree_hal_remote_client_bulk_transfer_t {
  // Active transfer kind stored in the union below.
  iree_hal_remote_client_bulk_transfer_kind_t kind;

  union {
    // Client-local queue_read upload state.
    iree_hal_remote_client_file_read_transfer_t file_read;

    // Client-local queue_write download state.
    iree_hal_remote_client_file_write_transfer_t file_write;
  };
} iree_hal_remote_client_bulk_transfer_t;

static iree_hal_remote_client_file_read_chunk_t*
iree_hal_remote_client_file_read_chunk_storage(iree_net_bulk_chunk_t* chunk);
static void iree_hal_remote_client_file_read_chunk_release(
    iree_hal_remote_client_file_read_chunk_t* chunk_context);
static void iree_hal_remote_client_file_write_chunk_release(
    iree_hal_remote_client_file_write_chunk_t* chunk_context);

static void iree_hal_remote_client_bulk_transfer_deinitialize(
    iree_hal_remote_client_bulk_transfer_t* transfer) {
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ:
      if (transfer->file_read.pending_chunk) {
        iree_hal_remote_client_file_read_chunk_release(
            iree_hal_remote_client_file_read_chunk_storage(
                transfer->file_read.pending_chunk));
        transfer->file_read.pending_chunk = NULL;
      }
      iree_hal_file_release(transfer->file_read.source_file);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE:
      iree_hal_remote_bulk_transfer_tracker_deinitialize(
          &transfer->file_write.receive_tracker);
      iree_hal_file_release(transfer->file_write.target_file);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static iree_hal_remote_client_bulk_transfer_t*
iree_hal_remote_client_bulk_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_client_bulk_transfer_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
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
  iree_net_bulk_transfer_table_options_t options =
      iree_net_bulk_transfer_table_options_default();
  options.user_storage_size = sizeof(iree_hal_remote_client_bulk_transfer_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_client_bulk_transfer_t);
  options.initial_transfer_id = 1;
  options.transfer_id_stride = 2;
  iree_status_t status = iree_net_bulk_transfer_table_allocate(
      &options, device->host_allocator, &device->bulk_transfers);

  iree_net_bulk_chunk_pool_options_t chunk_options =
      iree_net_bulk_chunk_pool_options_default();
  chunk_options.user_storage_size =
      iree_max(sizeof(iree_hal_remote_client_file_read_chunk_t),
               sizeof(iree_hal_remote_client_file_write_chunk_t));
  chunk_options.user_storage_alignment =
      iree_max(iree_alignof(iree_hal_remote_client_file_read_chunk_t),
               iree_alignof(iree_hal_remote_client_file_write_chunk_t));
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_chunk_pool_allocate(
        &chunk_options, device->host_allocator, &device->bulk_send_chunks);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_chunk_pool_allocate(
        &chunk_options, device->host_allocator, &device->bulk_receive_chunks);
  }

  if (!iree_status_is_ok(status)) {
    iree_net_bulk_chunk_pool_free(device->bulk_receive_chunks);
    device->bulk_receive_chunks = NULL;
    iree_net_bulk_chunk_pool_free(device->bulk_send_chunks);
    device->bulk_send_chunks = NULL;
    iree_net_bulk_transfer_table_free(device->bulk_transfers);
    device->bulk_transfers = NULL;
  }
  return status;
}

void iree_hal_remote_client_device_deinitialize_bulk_transfers(
    iree_hal_remote_client_device_t* device) {
  if (!device->bulk_transfers && !device->bulk_send_chunks &&
      !device->bulk_receive_chunks) {
    return;
  }
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  if (device->bulk_transfers) {
    iree_net_bulk_transfer_table_visit(
        device->bulk_transfers,
        iree_hal_remote_client_bulk_deinitialize_transfer, NULL);
    iree_net_bulk_transfer_table_clear(device->bulk_transfers);
  }
  iree_net_bulk_transfer_table_t* table = device->bulk_transfers;
  device->bulk_transfers = NULL;
  iree_net_bulk_chunk_pool_t* send_chunks = device->bulk_send_chunks;
  device->bulk_send_chunks = NULL;
  iree_net_bulk_chunk_pool_t* receive_chunks = device->bulk_receive_chunks;
  device->bulk_receive_chunks = NULL;
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_table_free(table);
  iree_net_bulk_chunk_pool_free(send_chunks);
  iree_net_bulk_chunk_pool_free(receive_chunks);
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

  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer = NULL;
  iree_status_t status = iree_net_bulk_transfer_table_allocate_transfer(
      device->bulk_transfers, (uint64_t)length, /*user_value=*/0, &transfer);
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
      iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                   transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
  return status;
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

  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer = NULL;
  iree_status_t status = iree_net_bulk_transfer_table_allocate_transfer(
      device->bulk_transfers, (uint64_t)length, /*user_value=*/0, &transfer);
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
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
  return status;
}

void iree_hal_remote_client_bulk_cancel_transfer(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  if (!transfer_id) return;
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                 transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
}

static iree_hal_remote_client_file_read_chunk_t*
iree_hal_remote_client_file_read_chunk_storage(iree_net_bulk_chunk_t* chunk) {
  return (iree_hal_remote_client_file_read_chunk_t*)
      iree_net_bulk_chunk_user_storage(chunk)
          .data;
}

static iree_hal_remote_client_file_write_chunk_t*
iree_hal_remote_client_file_write_chunk_storage(iree_net_bulk_chunk_t* chunk) {
  return (iree_hal_remote_client_file_write_chunk_t*)
      iree_net_bulk_chunk_user_storage(chunk)
          .data;
}

static iree_status_t iree_hal_remote_client_bulk_try_send_all_file_reads_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel);

static void iree_hal_remote_client_file_read_chunk_release(
    iree_hal_remote_client_file_read_chunk_t* chunk_context) {
  if (!chunk_context) return;
  iree_hal_remote_client_device_t* device = chunk_context->device;
  iree_net_bulk_chunk_t* chunk = chunk_context->chunk;
  iree_async_buffer_lease_release(&chunk_context->lease);
  iree_async_file_release(chunk_context->async_file);
  memset(chunk_context, 0, sizeof(*chunk_context));
  iree_net_bulk_chunk_release(device->bulk_send_chunks, chunk);
  iree_hal_device_release((iree_hal_device_t*)device);
}

static void iree_hal_remote_client_file_write_chunk_release(
    iree_hal_remote_client_file_write_chunk_t* chunk_context) {
  if (!chunk_context) return;
  iree_hal_remote_client_device_t* device = chunk_context->device;
  iree_net_bulk_chunk_t* chunk = chunk_context->chunk;
  iree_async_file_release(chunk_context->async_file);
  memset(chunk_context, 0, sizeof(*chunk_context));
  iree_net_bulk_chunk_release(device->bulk_receive_chunks, chunk);
  iree_hal_device_release((iree_hal_device_t*)device);
}

static void iree_hal_remote_client_file_read_chunk_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags);
static void iree_hal_remote_client_file_write_chunk_complete(
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

static bool iree_hal_remote_client_file_write_is_ready_locked(
    const iree_hal_remote_client_file_write_transfer_t* transfer) {
  return !transfer->write_failed && transfer->server_complete &&
         transfer->pending_write_count == 0 &&
         iree_hal_remote_bulk_transfer_tracker_is_complete(
             &transfer->receive_tracker);
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
    iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_bulk_try_send_pending_read_locked(
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
  iree_status_t status = iree_net_bulk_channel_send_data(
      channel, transfer_id, chunk_context->transfer_offset,
      transfer->next_sequence, flags, chunk_payload,
      (uint64_t)(uintptr_t)chunk);
  if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    transfer->pending_chunk = NULL;
    transfer->send_offset = chunk_end;
    ++transfer->next_sequence;
    ++transfer->pending_send_count;
  }
  return status;
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

  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)iree_atomic_load(
      &device->bulk_channel, iree_memory_order_acquire);
  const bool file_failure = !iree_status_is_ok(status);
  bool release_chunk = false;
  bool chunk_attached_to_transfer = false;
  bool send_abort = false;
  uint64_t abort_transfer_id = 0;
  iree_status_t transport_status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers,
                                          chunk_context->transfer_id);
  if (iree_status_is_ok(status) && table_transfer && channel) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(table_transfer);
    bulk_transfer->file_read.pending_chunk = chunk_context->chunk;
    chunk_attached_to_transfer = true;
    status = iree_hal_remote_client_bulk_try_send_pending_read_locked(
        device, channel, table_transfer);
  } else if (iree_status_is_ok(status) && !channel) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "remote bulk channel is not available");
  }

  if (!iree_status_is_ok(status)) {
    if (file_failure && table_transfer && channel) {
      send_abort = true;
      abort_transfer_id = chunk_context->transfer_id;
      iree_status_ignore(status);
      status = iree_ok_status();
    } else if (file_failure && !table_transfer) {
      iree_status_ignore(status);
      status = iree_ok_status();
    } else {
      transport_status = status;
      status = iree_ok_status();
    }
    if (table_transfer) {
      iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                   table_transfer);
      table_transfer = NULL;
    }
    release_chunk = !chunk_attached_to_transfer;
  } else if (!table_transfer) {
    release_chunk = true;
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

  if (release_chunk) {
    iree_hal_remote_client_file_read_chunk_release(chunk_context);
  }
  if (send_abort) {
    transport_status = iree_net_bulk_channel_send_abort(
        channel, abort_transfer_id, iree_async_span_list_empty(),
        /*operation_user_data=*/0);
  }
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(device,
                                                          transport_status);
  }
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
      (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                 iree_memory_order_acquire);
  bool send_complete = false;
  bool send_abort = false;
  uint64_t transfer_id = chunk_context->transfer_id;
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  if (!table_transfer) {
    iree_status_ignore(status);
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
      if (iree_status_is_ok(status)) {
        send_complete =
            iree_hal_remote_client_file_write_is_ready_locked(file_transfer);
      } else {
        file_transfer->write_failed = true;
        send_abort = true;
      }
    }
    if (!iree_status_is_ok(status) || send_abort) {
      iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                   table_transfer);
      table_transfer = NULL;
    }
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

  iree_hal_remote_client_file_write_chunk_release(chunk_context);

  iree_status_t transport_status = iree_ok_status();
  if (bulk_channel) {
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
  iree_status_ignore(status);
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(device,
                                                          transport_status);
  }
}

static iree_status_t iree_hal_remote_client_bulk_submit_async_file_read_locked(
    iree_hal_remote_client_device_t* device,
    iree_net_bulk_transfer_t* table_transfer, uint64_t chunk_offset,
    iree_host_size_t chunk_length) {
  iree_net_bulk_chunk_t* chunk = NULL;
  iree_status_t status = iree_net_bulk_chunk_pool_acquire(
      device->bulk_send_chunks, iree_net_bulk_transfer_id(table_transfer),
      chunk_offset, /*sequence=*/0, IREE_NET_BULK_FRAME_FLAG_NONE,
      iree_const_byte_span_empty(), /*lease=*/NULL, /*user_value=*/0, &chunk);
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

static iree_status_t iree_hal_remote_client_bulk_submit_async_file_write_locked(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_file_write_transfer_t* file_transfer,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease) {
  iree_net_bulk_chunk_t* chunk = NULL;
  iree_status_t status = iree_net_bulk_chunk_pool_acquire(
      device->bulk_receive_chunks, transfer_id, chunk_offset, sequence, flags,
      chunk_data, lease, /*user_value=*/0, &chunk);
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

static iree_status_t iree_hal_remote_client_bulk_try_send_file_read_locked(
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
  if (transfer->complete_sent) return iree_ok_status();
  if (!channel) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bulk channel is not available");
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!transfer->start_sent) {
    iree_status_t status = iree_net_bulk_channel_send_start(
        channel, transfer_id, total_length, IREE_NET_BULK_FRAME_FLAG_NONE,
        transfer_id);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_ignore(status);
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) {
      return status;
    }
    transfer->start_sent = true;
    ++transfer->pending_send_count;
  }

  if (transfer->source_file_view.kind ==
      IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE) {
    IREE_RETURN_IF_ERROR(
        iree_hal_remote_client_bulk_try_send_pending_read_locked(
            device, channel, table_transfer));
    if (!transfer->chunk_in_flight && transfer->send_offset < total_length &&
        iree_net_bulk_channel_remote_chunk_credit_count(channel) > 0) {
      const uint64_t remaining_length = total_length - transfer->send_offset;
      const iree_host_size_t chunk_length = (iree_host_size_t)iree_min(
          remaining_length, (uint64_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
      iree_status_t status =
          iree_hal_remote_client_bulk_submit_async_file_read_locked(
              device, table_transfer, transfer->send_offset, chunk_length);
      if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
        iree_status_ignore(status);
        return iree_ok_status();
      }
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                     table_transfer);
        return iree_net_bulk_channel_send_abort(channel, transfer_id,
                                                iree_async_span_list_empty(),
                                                /*operation_user_data=*/0);
      }
      transfer->chunk_in_flight = true;
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
      iree_status_t status = iree_net_bulk_channel_send_data(
          channel, transfer_id, transfer->send_offset, transfer->next_sequence,
          flags, chunk_payload, transfer_id);
      if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
        iree_status_ignore(status);
        return iree_ok_status();
      }
      if (!iree_status_is_ok(status)) {
        return status;
      }
      transfer->send_offset = chunk_end;
      ++transfer->next_sequence;
      ++transfer->pending_send_count;
    }
  }

  if (!transfer->chunk_in_flight && !transfer->pending_chunk &&
      !transfer->complete_sent && transfer->send_offset == total_length) {
    iree_status_t status =
        iree_net_bulk_channel_send_complete(channel, transfer_id, transfer_id);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_ignore(status);
      return iree_ok_status();
    }
    if (!iree_status_is_ok(status)) {
      return status;
    }
    transfer->complete_sent = true;
    ++transfer->pending_send_count;
  }

  if (transfer->complete_sent && transfer->pending_send_count == 0) {
    iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                 table_transfer);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_bulk_upload_file_read(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  IREE_ASSERT_ARGUMENT(device);
  if (!transfer_id) return iree_ok_status();

  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)iree_atomic_load(
      &device->bulk_channel, iree_memory_order_acquire);
  if (!channel) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote bulk channel is not available");
  }

  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  iree_status_t status = iree_ok_status();
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk upload unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    status = iree_hal_remote_client_bulk_try_send_file_read_locked(
        device, channel, transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(
        device, iree_status_clone(status));
  }
  return status;
}

typedef struct iree_hal_remote_client_bulk_transfer_id_list_t {
  // Active transfer IDs collected for retry.
  uint64_t transfer_ids[IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY];

  // Number of populated entries in |transfer_ids|.
  iree_host_size_t transfer_count;
} iree_hal_remote_client_bulk_transfer_id_list_t;

static void iree_hal_remote_client_bulk_collect_ready_file_read(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_transfer_id_list_t* id_list =
      (iree_hal_remote_client_bulk_transfer_id_list_t*)user_data;
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ) {
    return;
  }
  if (bulk_transfer->file_read.complete_sent) return;
  if (id_list->transfer_count >= IREE_ARRAYSIZE(id_list->transfer_ids)) return;
  id_list->transfer_ids[id_list->transfer_count++] =
      iree_net_bulk_transfer_id(table_transfer);
}

static iree_status_t iree_hal_remote_client_bulk_try_send_all_file_reads_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel) {
  iree_hal_remote_client_bulk_transfer_id_list_t id_list;
  memset(&id_list, 0, sizeof(id_list));
  iree_net_bulk_transfer_table_visit(
      device->bulk_transfers,
      iree_hal_remote_client_bulk_collect_ready_file_read, &id_list);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < id_list.transfer_count; ++i) {
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_transfers,
                                            id_list.transfer_ids[i]);
    if (table_transfer) {
      status = iree_hal_remote_client_bulk_try_send_file_read_locked(
          device, channel, table_transfer);
      if (!iree_status_is_ok(status)) break;
    }
  }
  return status;
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

  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  iree_status_t status = iree_ok_status();
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk START unknown transfer_id=%" PRIu64, transfer_id);
  } else if (iree_net_bulk_transfer_total_size(transfer) != total_size) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote client bulk START size mismatch for transfer_id=%" PRIu64,
        transfer_id);
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_t* bulk_channel =
        (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                   iree_memory_order_acquire);
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
      (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                 iree_memory_order_acquire);
  iree_status_t status = iree_ok_status();
  bool send_credit = false;
  bool send_abort = false;
  uint64_t abort_transfer_id = 0;
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote client bulk DATA received for "
                                "non-download transfer_id=%" PRIu64,
                                transfer_id);
    } else {
      iree_hal_remote_client_file_write_transfer_t* file_transfer =
          &bulk_transfer->file_write;
      const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
      const bool chunk_range_overflow =
          chunk_offset > UINT64_MAX - chunk_length;
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
            iree_status_ignore(status);
            status = iree_ok_status();
            send_credit = true;
            send_abort = true;
            abort_transfer_id = transfer_id;
            iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                         transfer);
            transfer = NULL;
          }
        } else if (iree_status_is_ok(status)) {
          status = iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "remote queue_write bulk DATA requires a host allocation or "
              "async target file");
        }
      }
    }
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

  if (iree_status_is_ok(status) && send_credit) {
    if (bulk_channel) {
      status = iree_net_bulk_channel_send_credit(
          bulk_channel, /*credit_delta=*/1, /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  if (iree_status_is_ok(status) && send_abort) {
    if (bulk_channel) {
      status = iree_net_bulk_channel_send_abort(bulk_channel, abort_transfer_id,
                                                iree_async_span_list_empty(),
                                                /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_complete(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                 iree_memory_order_acquire);

  iree_status_t status = iree_ok_status();
  bool send_complete = false;
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "remote client bulk COMPLETE received for non-download "
          "transfer_id=%" PRIu64,
          transfer_id);
    } else {
      iree_hal_remote_client_file_write_transfer_t* file_transfer =
          &bulk_transfer->file_write;
      file_transfer->server_complete = true;
      if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
              &file_transfer->receive_tracker)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "remote client bulk COMPLETE before all DATA "
                                  "for transfer_id=%" PRIu64,
                                  transfer_id);
      } else {
        send_complete =
            iree_hal_remote_client_file_write_is_ready_locked(file_transfer);
      }
    }
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

  if (iree_status_is_ok(status) && send_complete) {
    status = iree_hal_remote_client_bulk_ack_file_write(device, bulk_channel,
                                                        transfer_id);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  (void)abort_data;
  (void)lease;
  iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
  return iree_ok_status();
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
  if (operation_user_data == 0) {
    iree_status_ignore(status);
    return;
  }

  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                 iree_memory_order_acquire);

  if ((operation_user_data & 1u) == 0) {
    iree_net_bulk_chunk_t* chunk =
        (iree_net_bulk_chunk_t*)(uintptr_t)operation_user_data;
    iree_hal_remote_client_file_read_chunk_t* chunk_context =
        iree_hal_remote_client_file_read_chunk_storage(chunk);
    const uint64_t transfer_id = chunk_context->transfer_id;
    iree_status_t transport_status = iree_ok_status();
    iree_slim_mutex_lock(&device->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_transfers,
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
        transfer->chunk_in_flight = false;
        if (!iree_status_is_ok(status)) {
          transport_status = status;
          status = iree_ok_status();
          iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                       table_transfer);
          table_transfer = NULL;
        }
      } else if (!iree_status_is_ok(status)) {
        transport_status = status;
        status = iree_ok_status();
      }
    }
    iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

    iree_hal_remote_client_file_read_chunk_release(chunk_context);

    if (iree_status_is_ok(transport_status)) {
      iree_slim_mutex_lock(&device->bulk_transfer_mutex);
      transport_status =
          iree_hal_remote_client_bulk_try_send_all_file_reads_locked(
              device, bulk_channel);
      iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
    }
    iree_status_ignore(status);
    if (!iree_status_is_ok(transport_status)) {
      iree_hal_remote_client_device_on_bulk_transport_error(user_data,
                                                            transport_status);
    }
    return;
  }

  const uint64_t transfer_id = operation_user_data;
  iree_status_t transport_status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(device->bulk_transfers, transfer_id);
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
      if (iree_status_is_ok(status)) {
        transport_status =
            iree_hal_remote_client_bulk_try_send_file_read_locked(
                device, bulk_channel, table_transfer);
      } else {
        transport_status = status;
        status = iree_ok_status();
        iree_hal_remote_client_bulk_release_transfer(device->bulk_transfers,
                                                     table_transfer);
      }
    } else if (!iree_status_is_ok(status)) {
      transport_status = status;
      status = iree_ok_status();
    }
  }
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);

  iree_status_ignore(status);
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
      (iree_net_bulk_channel_t*)iree_atomic_load(&device->bulk_channel,
                                                 iree_memory_order_acquire);
  iree_slim_mutex_lock(&device->bulk_transfer_mutex);
  iree_status_t status =
      iree_hal_remote_client_bulk_try_send_all_file_reads_locked(device,
                                                                 bulk_channel);
  iree_slim_mutex_unlock(&device->bulk_transfer_mutex);
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
      const uint32_t initial_chunk_credit =
          (uint32_t)iree_net_bulk_chunk_pool_capacity(
              device->bulk_receive_chunks);
      status = iree_net_bulk_channel_send_credit(
          bulk_channel, initial_chunk_credit, /*operation_user_data=*/0);
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
