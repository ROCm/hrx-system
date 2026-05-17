// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk.h"

#include "iree/async/semaphore.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/io/file_handle.h"
#include "iree/net/channel/bulk/transfer_table.h"

#define IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH (64 * 1024)

// Host allocation wrapped as a HAL memory file for local queue_write staging.
typedef struct iree_hal_remote_server_bulk_host_allocation_t {
  // Host allocator used to free this allocation.
  iree_allocator_t host_allocator;

  // Byte-addressable file contents.
  uint8_t data[];
} iree_hal_remote_server_bulk_host_allocation_t;

typedef struct iree_hal_remote_server_client_file_read_transfer_t {
  // Server retained while transfer state may outlive the session lock.
  iree_hal_remote_server_t* server;

  // Local HAL device borrowed from |server|.
  iree_hal_device_t* local_device;

  // Target buffer retained until the local queue_read is submitted.
  iree_hal_buffer_t* target_buffer;

  // Target buffer byte offset for the first uploaded byte.
  iree_device_size_t target_offset;

  // Wait semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // File handle owning the host allocation until imported as a HAL file.
  iree_io_file_handle_t* host_file_handle;

  // HAL memory file wrapping |host_contents| after command attachment.
  iree_hal_file_t* host_file;

  // Host allocation receiving DATA chunks from the client.
  iree_byte_span_t host_contents;

  // Tracks fixed-grid DATA chunks received from the client.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // True once the client START frame has been received.
  bool start_received;

  // True once the client COMPLETE frame has been received.
  bool peer_complete;

  // True once the queue command has attached local HAL resources.
  bool command_ready;

  // True once the cloned semaphore lists have been freed or moved away.
  bool semaphore_lists_consumed;
} iree_hal_remote_server_client_file_read_transfer_t;

typedef struct iree_hal_remote_server_client_file_read_update_t {
  // Server retained until the local queue_read has captured its inputs.
  iree_hal_remote_server_t* server;

  // Local HAL device borrowed from |server|.
  iree_hal_device_t* local_device;

  // Target buffer retained until the local queue_read has captured it.
  iree_hal_buffer_t* target_buffer;

  // Target buffer byte offset for the first uploaded byte.
  iree_device_size_t target_offset;

  // Wait semaphore list consumed by the local queue_read.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphore list consumed by the local queue_read.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // HAL memory file containing the uploaded bytes.
  iree_hal_file_t* host_file;

  // Total number of uploaded bytes.
  iree_device_size_t total_length;
} iree_hal_remote_server_client_file_read_update_t;

typedef struct iree_hal_remote_server_client_file_write_transfer_t {
  // Server retained while callbacks may reference the session array.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer table entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Server-side memory file receiving the local queue_write.
  iree_hal_file_t* host_file;

  // Host allocation contents streamed to the client after local queue_write.
  iree_byte_span_t host_contents;

  // Final local signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Number of host bytes that have been submitted as DATA frames.
  uint64_t send_offset;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // Number of bulk START/DATA/COMPLETE frames still awaiting send completion.
  uint32_t pending_send_count;

  // True once the local queue_write into |host_file| has completed.
  bool local_write_complete;

  // True once the START frame has been submitted.
  bool start_sent;

  // True once the server COMPLETE frame has been submitted.
  bool complete_sent;

  // True once the client has acknowledged completion with its COMPLETE frame.
  bool peer_complete;

  // True once |signal_semaphore_list| has been signaled or failed and freed.
  bool signal_consumed;
} iree_hal_remote_server_client_file_write_transfer_t;

typedef enum iree_hal_remote_server_bulk_transfer_kind_e {
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_EMPTY = 0u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ = 1u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE = 2u,
} iree_hal_remote_server_bulk_transfer_kind_e;
typedef uint8_t iree_hal_remote_server_bulk_transfer_kind_t;

typedef struct iree_hal_remote_server_bulk_transfer_t {
  // Active transfer kind stored in the union below.
  iree_hal_remote_server_bulk_transfer_kind_t kind;

  union {
    // Client-file queue_read upload state.
    iree_hal_remote_server_client_file_read_transfer_t client_file_read;

    // Client-file queue_write download state.
    iree_hal_remote_server_client_file_write_transfer_t client_file_write;
  };
} iree_hal_remote_server_bulk_transfer_t;

typedef struct iree_hal_remote_server_client_file_write_ready_t {
  // Timepoint registered on the local queue_write signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;

  // Server retained while the timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the timepoint fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after local queue_write completion.
  uint64_t transfer_id;

  // Local queue_write signal semaphore retained for the timepoint.
  iree_hal_semaphore_t* local_semaphore;

  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
} iree_hal_remote_server_client_file_write_ready_t;

static iree_hal_remote_server_bulk_transfer_t*
iree_hal_remote_server_bulk_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
}

static iree_hal_remote_server_client_file_read_transfer_t*
iree_hal_remote_server_client_file_read_storage(
    iree_net_bulk_transfer_t* transfer) {
  return &iree_hal_remote_server_bulk_transfer_storage(transfer)
              ->client_file_read;
}

static iree_hal_remote_server_client_file_write_transfer_t*
iree_hal_remote_server_client_file_write_storage(
    iree_net_bulk_transfer_t* transfer) {
  return &iree_hal_remote_server_bulk_transfer_storage(transfer)
              ->client_file_write;
}

static void iree_hal_remote_server_bulk_host_allocation_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  (void)user_data;
  IREE_ASSERT(handle_primitive.type ==
              IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
  iree_byte_span_t host_allocation = handle_primitive.value.host_allocation;
  if (!host_allocation.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_allocation.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static void iree_hal_remote_server_bulk_host_contents_free(
    iree_byte_span_t host_contents) {
  if (!host_contents.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_contents.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static iree_status_t iree_hal_remote_server_bulk_host_contents_allocate(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents) {
  *out_host_contents = iree_byte_span_empty();

  iree_host_size_t total_size = 0;
  iree_host_size_t data_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_bulk_host_allocation_t), &total_size,
      IREE_STRUCT_FIELD(allocation_size, uint8_t, &data_offset)));

  iree_hal_remote_server_bulk_host_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocation));
  allocation->host_allocator = host_allocator;

  iree_byte_span_t host_contents =
      iree_make_byte_span((uint8_t*)allocation + data_offset, allocation_size);
  *out_host_contents = host_contents;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_bulk_host_allocation_create(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents,
    iree_io_file_handle_t** out_file_handle) {
  *out_file_handle = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_bulk_host_contents_allocate(
      allocation_size, host_allocator, out_host_contents));

  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_remote_server_bulk_host_allocation_release,
      .user_data = NULL,
  };
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, *out_host_contents,
      release_callback, host_allocator, out_file_handle);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_host_contents_free(*out_host_contents);
    *out_host_contents = iree_byte_span_empty();
  }
  return status;
}

static void iree_hal_remote_server_client_file_read_free_semaphore_lists(
    iree_hal_remote_server_client_file_read_transfer_t* transfer) {
  if (transfer->semaphore_lists_consumed) return;
  iree_hal_semaphore_list_free(transfer->wait_semaphore_list,
                               transfer->server->host_allocator);
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->server->host_allocator);
  transfer->wait_semaphore_list = iree_hal_semaphore_list_empty();
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
  transfer->semaphore_lists_consumed = true;
}

static void iree_hal_remote_server_client_file_read_signal_failure(
    iree_hal_remote_server_client_file_read_transfer_t* transfer,
    iree_status_t status) {
  if (!transfer->command_ready || transfer->semaphore_lists_consumed) {
    iree_status_ignore(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
  iree_hal_remote_server_client_file_read_free_semaphore_lists(transfer);
}

static void iree_hal_remote_server_client_file_read_deinitialize(
    iree_hal_remote_server_client_file_read_transfer_t* transfer) {
  if (transfer->command_ready && !transfer->semaphore_lists_consumed) {
    iree_hal_remote_server_client_file_read_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  iree_hal_file_release(transfer->host_file);
  iree_io_file_handle_release(transfer->host_file_handle);
  iree_hal_buffer_release(transfer->target_buffer);
  iree_hal_remote_bulk_transfer_tracker_deinitialize(
      &transfer->receive_tracker);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_client_file_read_update_deinitialize(
    iree_hal_remote_server_client_file_read_update_t* update) {
  if (!update->server) return;
  iree_allocator_t host_allocator = update->server->host_allocator;
  iree_hal_semaphore_list_free(update->wait_semaphore_list, host_allocator);
  iree_hal_semaphore_list_free(update->signal_semaphore_list, host_allocator);
  iree_hal_file_release(update->host_file);
  iree_hal_buffer_release(update->target_buffer);
  iree_hal_remote_server_release(update->server);
  memset(update, 0, sizeof(*update));
}

static void iree_hal_remote_server_client_file_read_update_submit(
    iree_hal_remote_server_client_file_read_update_t* update) {
  if (!update->server) return;
  iree_status_t status = iree_hal_device_queue_read(
      update->local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      update->wait_semaphore_list, update->signal_semaphore_list,
      update->host_file, /*source_offset=*/0, update->target_buffer,
      update->target_offset, update->total_length, IREE_HAL_READ_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(update->signal_semaphore_list, status);
  }
  iree_hal_remote_server_client_file_read_update_deinitialize(update);
}

static void iree_hal_remote_server_client_file_write_signal_failure(
    iree_hal_remote_server_client_file_write_transfer_t* transfer,
    iree_status_t status) {
  if (transfer->signal_consumed) {
    iree_status_ignore(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->server->host_allocator);
  memset(&transfer->signal_semaphore_list, 0,
         sizeof(transfer->signal_semaphore_list));
  transfer->signal_consumed = true;
}

static void iree_hal_remote_server_client_file_write_deinitialize(
    iree_hal_remote_server_client_file_write_transfer_t* transfer) {
  if (!transfer->signal_consumed) {
    iree_hal_remote_server_client_file_write_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  iree_hal_file_release(transfer->host_file);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_bulk_transfer_deinitialize(
    iree_hal_remote_server_bulk_transfer_t* transfer) {
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
      iree_hal_remote_server_client_file_read_deinitialize(
          &transfer->client_file_read);
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
      iree_hal_remote_server_client_file_write_deinitialize(
          &transfer->client_file_write);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_bulk_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_server_bulk_transfer_deinitialize(
      iree_hal_remote_server_bulk_transfer_storage(transfer));
  iree_net_bulk_transfer_table_remove(table, transfer_id);
}

static void iree_hal_remote_server_client_file_write_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_server_bulk_release_transfer(table, transfer);
}

static void iree_hal_remote_server_client_file_write_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  if (transfer->peer_complete && !transfer->signal_consumed) {
    iree_status_t status = iree_hal_semaphore_list_signal(
        transfer->signal_semaphore_list, /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
    }
    iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                                 transfer->server->host_allocator);
    memset(&transfer->signal_semaphore_list, 0,
           sizeof(transfer->signal_semaphore_list));
    transfer->signal_consumed = true;
  }
  if (transfer->signal_consumed && transfer->pending_send_count == 0) {
    iree_hal_remote_server_client_file_write_release_transfer(
        session_slot->bulk_transfers, table_transfer);
  }
}

static void iree_hal_remote_server_client_file_write_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  iree_hal_remote_server_client_file_write_signal_failure(transfer, status);
  if (transfer->pending_send_count == 0) {
    iree_hal_remote_server_client_file_write_release_transfer(
        session_slot->bulk_transfers, table_transfer);
  }
}

static void iree_hal_remote_server_client_file_write_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  if (transfer->signal_consumed || !transfer->local_write_complete) return;
  if (!bulk_channel) {
    iree_hal_remote_server_client_file_write_fail_locked(
        session_slot, table_transfer,
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "remote bulk channel is not available"));
    return;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!transfer->start_sent) {
    iree_status_t status = iree_net_bulk_channel_send_start(
        bulk_channel, transfer_id, total_length, IREE_NET_BULK_FRAME_FLAG_NONE,
        transfer_id);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_ignore(status);
      return;
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, status);
      return;
    }
    transfer->start_sent = true;
    ++transfer->pending_send_count;
  }

  while (transfer->send_offset < total_length) {
    if (iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel) == 0) {
      return;
    }
    const uint64_t remaining_length = total_length - transfer->send_offset;
    const iree_host_size_t chunk_length = (iree_host_size_t)iree_min(
        remaining_length, (uint64_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
    const uint64_t chunk_end = transfer->send_offset + chunk_length;
    iree_net_bulk_frame_flags_t flags =
        chunk_end == total_length ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                  : IREE_NET_BULK_FRAME_FLAG_NONE;
    iree_async_span_t chunk_span = iree_async_span_from_ptr(
        transfer->host_contents.data + (iree_host_size_t)transfer->send_offset,
        chunk_length);
    iree_async_span_list_t chunk_payload =
        iree_async_span_list_make(&chunk_span, 1);
    iree_status_t status = iree_net_bulk_channel_send_data(
        bulk_channel, transfer_id, transfer->send_offset,
        transfer->next_sequence, flags, chunk_payload, transfer_id);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_ignore(status);
      return;
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, status);
      return;
    }
    transfer->send_offset = chunk_end;
    ++transfer->next_sequence;
    ++transfer->pending_send_count;
  }

  if (!transfer->complete_sent) {
    iree_status_t status = iree_net_bulk_channel_send_complete(
        bulk_channel, transfer_id, transfer_id);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_ignore(status);
      return;
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, status);
      return;
    }
    transfer->complete_sent = true;
    ++transfer->pending_send_count;
  }

  iree_hal_remote_server_client_file_write_try_finish_locked(session_slot,
                                                             table_transfer);
}

typedef struct iree_hal_remote_server_bulk_transfer_id_list_t {
  // Active transfer IDs collected for retry.
  uint64_t transfer_ids[IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY];

  // Number of populated entries in |transfer_ids|.
  iree_host_size_t transfer_count;
} iree_hal_remote_server_bulk_transfer_id_list_t;

static void iree_hal_remote_server_collect_ready_client_file_write(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_transfer_id_list_t* id_list =
      (iree_hal_remote_server_bulk_transfer_id_list_t*)user_data;
  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    return;
  }
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      &bulk_transfer->client_file_write;
  if (!transfer->local_write_complete || transfer->signal_consumed) return;
  if (id_list->transfer_count >= IREE_ARRAYSIZE(id_list->transfer_ids)) return;
  id_list->transfer_ids[id_list->transfer_count++] =
      iree_net_bulk_transfer_id(table_transfer);
}

static void iree_hal_remote_server_client_file_write_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel) {
  iree_hal_remote_server_bulk_transfer_id_list_t id_list;
  memset(&id_list, 0, sizeof(id_list));
  iree_net_bulk_transfer_table_visit(
      session_slot->bulk_transfers,
      iree_hal_remote_server_collect_ready_client_file_write, &id_list);
  for (iree_host_size_t i = 0; i < id_list.transfer_count; ++i) {
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                            id_list.transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_server_client_file_write_try_send_locked(
          session_slot, bulk_channel, table_transfer);
    }
  }
}

static iree_status_t
iree_hal_remote_server_client_file_read_get_or_insert_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_length, iree_net_bulk_transfer_t** out_table_transfer) {
  *out_table_transfer = NULL;
  if (total_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CLIENT_FILE_READ length %" PRIu64
                            " exceeds host size max %" PRIhsz,
                            total_length, IREE_HOST_SIZE_MAX);
  }

  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                          transfer_id);
  if (table_transfer) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bulk transfer_id=%" PRIu64
                              " is already used by another transfer kind",
                              transfer_id);
    }
    if (iree_net_bulk_transfer_total_size(table_transfer) != total_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "CLIENT_FILE_READ size mismatch for transfer_id=%" PRIu64,
          transfer_id);
    }
    *out_table_transfer = table_transfer;
    return iree_ok_status();
  }

  iree_status_t status = iree_net_bulk_transfer_table_insert(
      session_slot->bulk_transfers, transfer_id, total_length,
      /*user_value=*/0, &table_transfer);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind =
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ;
    iree_hal_remote_server_client_file_read_transfer_t* transfer =
        &bulk_transfer->client_file_read;
    transfer->server = session_slot->server;
    iree_hal_remote_server_retain(transfer->server);
    status = iree_hal_remote_server_bulk_host_allocation_create(
        (iree_host_size_t)total_length, session_slot->server->host_allocator,
        &transfer->host_contents, &transfer->host_file_handle);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_bulk_transfer_tracker_initialize(
          total_length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
          session_slot->server->host_allocator, &transfer->receive_tracker);
    }
  }
  if (!iree_status_is_ok(status) && table_transfer) {
    iree_hal_remote_server_bulk_release_transfer(session_slot->bulk_transfers,
                                                 table_transfer);
    table_transfer = NULL;
  }

  *out_table_transfer = table_transfer;
  return status;
}

static iree_status_t
iree_hal_remote_server_client_file_read_try_take_update_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer,
    iree_hal_remote_server_client_file_read_update_t* out_update) {
  memset(out_update, 0, sizeof(*out_update));
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      iree_hal_remote_server_client_file_read_storage(table_transfer);
  if (!transfer->command_ready || !transfer->peer_complete) {
    return iree_ok_status();
  }

  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
          &transfer->receive_tracker)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CLIENT_FILE_READ COMPLETE before all DATA for transfer_id=%" PRIu64,
        transfer_id);
  }

  out_update->server = transfer->server;
  transfer->server = NULL;
  out_update->local_device = transfer->local_device;
  out_update->target_buffer = transfer->target_buffer;
  transfer->target_buffer = NULL;
  out_update->target_offset = transfer->target_offset;
  out_update->wait_semaphore_list = transfer->wait_semaphore_list;
  transfer->wait_semaphore_list = iree_hal_semaphore_list_empty();
  out_update->signal_semaphore_list = transfer->signal_semaphore_list;
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
  out_update->host_file = transfer->host_file;
  transfer->host_file = NULL;
  out_update->total_length = (iree_device_size_t)total_length;
  transfer->semaphore_lists_consumed = true;

  iree_hal_remote_server_bulk_release_transfer(session_slot->bulk_transfers,
                                               table_transfer);
  return iree_ok_status();
}

static void iree_hal_remote_server_client_file_write_ready_context_release(
    iree_hal_remote_server_client_file_write_ready_t* context) {
  if (!context) return;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_semaphore_release(context->local_semaphore);
  iree_hal_remote_server_release(context->server);
  iree_allocator_free(host_allocator, context);
}

static void iree_hal_remote_server_client_file_write_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_client_file_write_ready_t* context =
      (iree_hal_remote_server_client_file_write_ready_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == context->session_id &&
                   session_slot->session != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (!session_active) {
    iree_status_ignore(status);
  } else {
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                            context->transfer_id);
    if (!table_transfer) {
      iree_status_ignore(status);
    } else if (iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_write_transfer_t* transfer =
          iree_hal_remote_server_client_file_write_storage(table_transfer);
      transfer->local_write_complete = true;
      iree_hal_remote_server_client_file_write_try_send_locked(
          session_slot, bulk_channel, table_transfer);
    } else {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, status);
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_channel_release(bulk_channel);
  }

  iree_hal_remote_server_client_file_write_ready_context_release(context);
}

static void iree_hal_remote_server_client_file_write_deinitialize_visit(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_transfer_deinitialize(
      iree_hal_remote_server_bulk_transfer_storage(table_transfer));
}

iree_status_t iree_hal_remote_server_session_initialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  iree_net_bulk_transfer_table_options_t options =
      iree_net_bulk_transfer_table_options_default();
  options.user_storage_size = sizeof(iree_hal_remote_server_bulk_transfer_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_server_bulk_transfer_t);
  options.initial_transfer_id = 2;
  options.transfer_id_stride = 2;
  return iree_net_bulk_transfer_table_allocate(&options, host_allocator,
                                               &session_slot->bulk_transfers);
}

void iree_hal_remote_server_session_deinitialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_transfers) return;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_table_visit(
      session_slot->bulk_transfers,
      iree_hal_remote_server_client_file_write_deinitialize_visit, NULL);
  iree_net_bulk_transfer_table_clear(session_slot->bulk_transfers);
  iree_net_bulk_transfer_table_t* table = session_slot->bulk_transfers;
  session_slot->bulk_transfers = NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_table_free(table);
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_read(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_status_t status =
      iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  if (iree_status_is_ok(status) && flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported CLIENT_FILE_READ flags: 0x%" PRIx64,
                              flags);
  }

  iree_hal_remote_server_client_file_read_update_t update;
  memset(&update, 0, sizeof(update));
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfers) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }

    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_client_file_read_get_or_insert_locked(
          session_slot, transfer_id, (uint64_t)length, &table_transfer);
    }

    iree_hal_semaphore_list_t cloned_wait_list =
        iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t cloned_signal_list =
        iree_hal_semaphore_list_empty();
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          iree_hal_remote_server_client_file_read_storage(table_transfer);
      if (transfer->command_ready) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "CLIENT_FILE_READ command already attached to transfer_id=%" PRIu64,
            transfer_id);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_semaphore_list_clone(
            &wait_list, session_slot->server->host_allocator,
            &cloned_wait_list);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_semaphore_list_clone(
            &signal_list, session_slot->server->host_allocator,
            &cloned_signal_list);
      }
      if (iree_status_is_ok(status)) {
        transfer->local_device = local_device;
        if (!transfer->host_file) {
          status = iree_hal_file_import(
              local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
              IREE_HAL_MEMORY_ACCESS_READ, transfer->host_file_handle,
              IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &transfer->host_file);
          if (iree_status_is_ok(status)) {
            iree_io_file_handle_release(transfer->host_file_handle);
            transfer->host_file_handle = NULL;
          }
        }
      }
      if (iree_status_is_ok(status)) {
        transfer->target_buffer = target_buffer;
        iree_hal_buffer_retain(transfer->target_buffer);
        transfer->target_offset = target_offset;
        transfer->wait_semaphore_list = cloned_wait_list;
        transfer->signal_semaphore_list = cloned_signal_list;
        transfer->command_ready = true;
        status = iree_hal_remote_server_client_file_read_try_take_update_locked(
            session_slot, table_transfer, &update);
      } else {
        iree_hal_semaphore_list_free(cloned_wait_list,
                                     session_slot->server->host_allocator);
        iree_hal_semaphore_list_free(cloned_signal_list,
                                     session_slot->server->host_allocator);
      }
    }
    if (!iree_status_is_ok(status) && table_transfer) {
      iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_server_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
        iree_hal_remote_server_client_file_read_transfer_t* transfer =
            &bulk_transfer->client_file_read;
        if (transfer->command_ready) {
          iree_hal_remote_server_client_file_read_free_semaphore_lists(
              transfer);
        }
      }
      iree_hal_remote_server_bulk_release_transfer(session_slot->bulk_transfers,
                                                   table_transfer);
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_update_submit(&update);
  } else {
    iree_hal_remote_server_client_file_read_update_deinitialize(&update);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_write(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_status_t status =
      iree_hal_buffer_validate_range(source_buffer, source_offset, length);

  iree_allocator_t host_allocator = session_slot->server->host_allocator;
  iree_byte_span_t host_contents = iree_byte_span_empty();
  iree_io_file_handle_t* host_file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_host_allocation_create(
        (iree_host_size_t)length, host_allocator, &host_contents,
        &host_file_handle);
  }

  iree_hal_file_t* host_file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        host_file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &host_file);
  }
  iree_io_file_handle_release(host_file_handle);

  iree_hal_semaphore_t* local_write_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &local_write_semaphore);
  }

  iree_hal_remote_server_t* server = session_slot->server;
  uint64_t session_id = 0;
  iree_net_bulk_transfer_t* table_transfer = NULL;
  bool transfer_inserted = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    session_id = session_slot->session_id;
    bool session_active = session_slot->session != NULL;
    iree_slim_mutex_unlock(&server->session_mutex);

    if (!session_active) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfers) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    } else {
      status = iree_net_bulk_transfer_table_insert(
          session_slot->bulk_transfers, transfer_id, length,
          /*user_value=*/0, &table_transfer);
      transfer_inserted = iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
            iree_hal_remote_server_bulk_transfer_storage(table_transfer);
        memset(bulk_transfer, 0, sizeof(*bulk_transfer));
        bulk_transfer->kind =
            IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE;
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            iree_hal_remote_server_client_file_write_storage(table_transfer);
        transfer->server = server;
        iree_hal_remote_server_retain(transfer->server);
        transfer->session_slot = session_slot;
        transfer->session_id = session_id;
        transfer->host_file = host_file;
        iree_hal_file_retain(transfer->host_file);
        transfer->host_contents = host_contents;
        status = iree_hal_semaphore_list_clone(
            &signal_list, host_allocator, &transfer->signal_semaphore_list);
      }
      if (!iree_status_is_ok(status) && transfer_inserted) {
        iree_hal_remote_server_client_file_write_release_transfer(
            session_slot->bulk_transfers, table_transfer);
        transfer_inserted = false;
        table_transfer = NULL;
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  iree_hal_remote_server_client_file_write_ready_t* ready_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*ready_context),
                                   (void**)&ready_context);
  }
  if (iree_status_is_ok(status)) {
    memset(ready_context, 0, sizeof(*ready_context));
    ready_context->server = server;
    iree_hal_remote_server_retain(ready_context->server);
    ready_context->session_slot = session_slot;
    ready_context->session_id = session_id;
    ready_context->transfer_id = transfer_id;
    ready_context->local_semaphore = local_write_semaphore;
    iree_hal_semaphore_retain(ready_context->local_semaphore);
    ready_context->host_allocator = host_allocator;
    ready_context->timepoint.callback =
        iree_hal_remote_server_client_file_write_ready_callback;
    ready_context->timepoint.user_data = ready_context;
  }

  uint64_t local_write_value = 1;
  iree_hal_semaphore_list_t local_write_signal_list = {
      .count = 1,
      .semaphores = &local_write_semaphore,
      .payload_values = &local_write_value,
  };
  bool local_write_submitted = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
        local_write_signal_list, source_buffer, source_offset, host_file,
        /*target_offset=*/0, length, flags);
    local_write_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)local_write_semaphore,
        /*minimum_value=*/1, &ready_context->timepoint);
    if (iree_status_is_ok(status)) {
      ready_context = NULL;
    }
  }

  if (!iree_status_is_ok(status) && transfer_inserted &&
      !local_write_submitted) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    table_transfer = iree_net_bulk_transfer_table_lookup(
        session_slot->bulk_transfers, transfer_id);
    if (table_transfer) {
      iree_hal_remote_server_client_file_write_release_transfer(
          session_slot->bulk_transfers, table_transfer);
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  iree_hal_remote_server_client_file_write_ready_context_release(ready_context);
  iree_hal_semaphore_release(local_write_semaphore);
  iree_hal_file_release(host_file);
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_start(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_size, iree_net_bulk_frame_flags_t flags) {
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote server bulk START flags: 0x%02x", flags);
  }

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_ok_status();
  if (!session_slot->bulk_transfers) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_client_file_read_get_or_insert_locked(
        session_slot, transfer_id, total_size, &table_transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_transfer_t* transfer =
        iree_hal_remote_server_client_file_read_storage(table_transfer);
    transfer->start_received = true;
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status) && session_slot->bulk_channel) {
    status = iree_net_bulk_channel_refresh_credit(session_slot->bulk_channel,
                                                  /*operation_user_data=*/0);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_data(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  (void)sequence;
  (void)lease;
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote server bulk DATA flags: 0x%02x",
                            flags);
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                          transfer_id);
  if (!table_transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote server bulk DATA received for "
                                "non-upload transfer_id=%" PRIu64,
                                transfer_id);
    } else {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          &bulk_transfer->client_file_read;
      const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
      const bool chunk_range_overflow =
          chunk_offset > UINT64_MAX - chunk_length;
      const uint64_t chunk_end =
          chunk_range_overflow ? UINT64_MAX : chunk_offset + chunk_length;
      const uint64_t total_length =
          iree_net_bulk_transfer_total_size(table_transfer);
      const bool final_chunk =
          iree_all_bits_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
      const bool expected_final_chunk =
          !chunk_range_overflow && chunk_end == total_length;
      if (!transfer->start_received) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "remote server bulk DATA before START for transfer_id=%" PRIu64,
            transfer_id);
      } else if (chunk_range_overflow || chunk_end > total_length) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "remote server bulk DATA range [%" PRIu64
                             ", %" PRIu64 ") exceeds transfer length %" PRIu64,
                             chunk_offset, chunk_end, total_length);
      } else if (final_chunk != expected_final_chunk) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "remote server bulk DATA final flag mismatch for "
                             "transfer_id=%" PRIu64,
                             transfer_id);
      } else {
        status = iree_hal_remote_bulk_transfer_tracker_record_chunk(
            &transfer->receive_tracker, chunk_offset, chunk_data.data_length);
      }
      if (iree_status_is_ok(status)) {
        memcpy(transfer->host_contents.data + (iree_host_size_t)chunk_offset,
               chunk_data.data, chunk_data.data_length);
      }
    }
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_channel_send_credit(session_slot->bulk_channel,
                                               /*credit_delta=*/1,
                                               /*operation_user_data=*/0);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  iree_hal_remote_server_client_file_read_update_t update;
  memset(&update, 0, sizeof(update));

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                          transfer_id);
  iree_status_t status = iree_ok_status();
  if (!table_transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
  } else {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    switch (bulk_transfer->kind) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ: {
        iree_hal_remote_server_client_file_read_transfer_t* transfer =
            &bulk_transfer->client_file_read;
        transfer->peer_complete = true;
        if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
                &transfer->receive_tracker)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "CLIENT_FILE_READ COMPLETE before all DATA "
                                    "for transfer_id=%" PRIu64,
                                    transfer_id);
        } else {
          status =
              iree_hal_remote_server_client_file_read_try_take_update_locked(
                  session_slot, table_transfer, &update);
        }
        break;
      }
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE: {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            &bulk_transfer->client_file_write;
        transfer->peer_complete = true;
        iree_hal_remote_server_client_file_write_try_finish_locked(
            session_slot, table_transfer);
        break;
      }
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "remote server bulk COMPLETE received for "
                                  "empty transfer_id=%" PRIu64,
                                  transfer_id);
        break;
    }
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_update_submit(&update);
  } else {
    iree_hal_remote_server_client_file_read_update_deinitialize(&update);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_abort(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                          transfer_id);
  if (table_transfer) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    switch (bulk_transfer->kind) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
        iree_hal_remote_server_client_file_read_signal_failure(
            &bulk_transfer->client_file_read,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        iree_hal_remote_server_bulk_release_transfer(
            session_slot->bulk_transfers, table_transfer);
        break;
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
        iree_hal_remote_server_client_file_write_fail_locked(
            session_slot, table_transfer,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        break;
      default:
        break;
    }
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  return iree_ok_status();
}

void iree_hal_remote_server_bulk_on_send_complete(
    iree_hal_remote_server_session_t* session_slot,
    uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    iree_status_ignore(status);
    return;
  }
  const uint64_t transfer_id = operation_user_data;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer =
      iree_net_bulk_transfer_table_lookup(session_slot->bulk_transfers,
                                          transfer_id);
  if (!table_transfer) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    return;
  }

  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    return;
  }

  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      &bulk_transfer->client_file_write;
  if (transfer->pending_send_count > 0) {
    --transfer->pending_send_count;
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_write_try_send_locked(
        session_slot, session_slot->bulk_channel, table_transfer);
    table_transfer = iree_net_bulk_transfer_table_lookup(
        session_slot->bulk_transfers, transfer_id);
    if (table_transfer) {
      iree_hal_remote_server_client_file_write_try_finish_locked(
          session_slot, table_transfer);
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    return;
  }

  iree_hal_remote_server_client_file_write_fail_locked(
      session_slot, table_transfer, iree_status_clone(status));
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_ignore(status);
}

void iree_hal_remote_server_bulk_on_credit(
    iree_hal_remote_server_session_t* session_slot) {
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_client_file_write_try_send_all_locked(
      session_slot, session_slot->bulk_channel);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
}
