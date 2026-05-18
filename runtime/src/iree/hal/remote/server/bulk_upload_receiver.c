// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_upload_receiver.h"

#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/channel/bulk/chunk_pool.h"

typedef struct iree_hal_remote_server_bulk_upload_chunk_list_t {
  // Retained DATA chunks collected for release or processing.
  iree_net_bulk_chunk_t* chunks[IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT];

  // Number of populated entries in |chunks|.
  iree_host_size_t chunk_count;

  // Transfer ID selected for collection.
  uint64_t transfer_id;
} iree_hal_remote_server_bulk_upload_chunk_list_t;

typedef struct iree_hal_remote_server_bulk_upload_ready_chunk_query_t {
  // Session slot containing the transfer scheduler used for chunk readiness.
  iree_hal_remote_server_session_t* session_slot;

  // Scheduler user value identifying upload transfers.
  uint64_t transfer_kind;

  // First chunk whose transfer is ready to process.
  iree_net_bulk_chunk_t* chunk;
} iree_hal_remote_server_bulk_upload_ready_chunk_query_t;

static void iree_hal_remote_server_bulk_upload_collect_chunks_for_transfer(
    void* user_data, iree_net_bulk_chunk_t* chunk) {
  iree_hal_remote_server_bulk_upload_chunk_list_t* chunk_list =
      (iree_hal_remote_server_bulk_upload_chunk_list_t*)user_data;
  if (iree_net_bulk_chunk_transfer_id(chunk) != chunk_list->transfer_id) return;
  if (chunk_list->chunk_count >= IREE_ARRAYSIZE(chunk_list->chunks)) return;
  chunk_list->chunks[chunk_list->chunk_count++] = chunk;
}

static void iree_hal_remote_server_bulk_upload_release_chunks_for_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  if (!session_slot || !session_slot->bulk_receive_window) return;
  iree_hal_remote_server_bulk_upload_chunk_list_t chunk_list;
  memset(&chunk_list, 0, sizeof(chunk_list));
  chunk_list.transfer_id = transfer_id;
  iree_net_bulk_receive_window_visit_chunks(
      session_slot->bulk_receive_window,
      iree_hal_remote_server_bulk_upload_collect_chunks_for_transfer,
      &chunk_list);
  for (iree_host_size_t i = 0; i < chunk_list.chunk_count; ++i) {
    iree_net_bulk_receive_window_release_chunk(
        session_slot->bulk_receive_window, chunk_list.chunks[i]);
  }
}

static void iree_hal_remote_server_bulk_upload_free_signal_list(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer) {
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->server->host_allocator);
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
}

static void iree_hal_remote_server_bulk_upload_signal_failure(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    iree_status_t status) {
  if (!iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY) ||
      iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_free(status);
    return;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_CONTROL_RESPONSE)) {
    iree_status_t send_status =
        iree_hal_remote_server_session_send_error_response(
            transfer->session_slot, &transfer->response_envelope, status);
    iree_status_free(send_status);
  } else {
    iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
    iree_hal_remote_server_bulk_upload_free_signal_list(transfer);
  }
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
}

iree_status_t iree_hal_remote_server_bulk_upload_transfer_initialize(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t transfer_id, uint64_t total_length,
    iree_host_size_t data_chunk_length, iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_upload_transfer_t* transfer) {
  memset(transfer, 0, sizeof(*transfer));
  transfer->server = server;
  iree_hal_remote_server_retain(transfer->server);
  transfer->session_slot = session_slot;
  transfer->session_id = session_id;
  transfer->transfer_id = transfer_id;
  iree_status_t status = iree_hal_remote_bulk_transfer_tracker_initialize(
      total_length, data_chunk_length, host_allocator,
      &transfer->receive_tracker);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_release(transfer->server);
    memset(transfer, 0, sizeof(*transfer));
  }
  return status;
}

void iree_hal_remote_server_bulk_upload_transfer_deinitialize(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer) {
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_hal_remote_server_bulk_upload_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  iree_hal_remote_server_bulk_upload_release_chunks_for_transfer(
      transfer->session_slot, transfer->transfer_id);
  iree_hal_remote_server_bulk_upload_ready_release(transfer->ready_context);
  iree_hal_semaphore_release(transfer->ready_semaphore);
  iree_hal_buffer_release(transfer->target_buffer);
  iree_hal_remote_bulk_transfer_tracker_deinitialize(
      &transfer->receive_tracker);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

iree_status_t iree_hal_remote_server_bulk_upload_transfer_mark_start(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    iree_net_bulk_frame_flags_t flags) {
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote server bulk START flags: 0x%02x", flags);
  }
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_START_RECEIVED;
  return iree_ok_status();
}

iree_status_t iree_hal_remote_server_bulk_upload_transfer_record_data(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    uint64_t transfer_id, uint64_t total_length, uint64_t chunk_offset,
    iree_host_size_t chunk_length, iree_net_bulk_frame_flags_t flags) {
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote server bulk DATA flags: 0x%02x",
                            flags);
  }
  const bool chunk_range_overflow =
      chunk_offset > UINT64_MAX - (uint64_t)chunk_length;
  const uint64_t chunk_end =
      chunk_range_overflow ? UINT64_MAX : chunk_offset + (uint64_t)chunk_length;
  const bool final_chunk =
      iree_all_bits_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
  const bool expected_final_chunk =
      !chunk_range_overflow && chunk_end == total_length;
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_START_RECEIVED)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote server bulk DATA before START for transfer_id=%" PRIu64,
        transfer_id);
  } else if (chunk_range_overflow || chunk_end > total_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remote server bulk DATA range [%" PRIu64
                            ", %" PRIu64 ") exceeds transfer length %" PRIu64,
                            chunk_offset, chunk_end, total_length);
  } else if (final_chunk != expected_final_chunk) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote server bulk DATA final flag mismatch for transfer_id=%" PRIu64,
        transfer_id);
  }
  return iree_hal_remote_bulk_transfer_tracker_record_chunk(
      &transfer->receive_tracker, chunk_offset, chunk_length);
}

iree_status_t iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    uint64_t transfer_id) {
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_PEER_COMPLETE;
  if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
          &transfer->receive_tracker)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CLIENT_FILE_READ COMPLETE before all DATA "
                            "for transfer_id=%" PRIu64,
                            transfer_id);
  }
  return iree_ok_status();
}

void iree_hal_remote_server_bulk_upload_ready_retain(
    iree_hal_remote_server_bulk_upload_ready_t* context) {
  if (!context) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

void iree_hal_remote_server_bulk_upload_ready_release(
    iree_hal_remote_server_bulk_upload_ready_t* context) {
  if (!context) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) != 1) return;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_semaphore_release(context->local_semaphore);
  iree_hal_remote_server_release(context->server);
  iree_allocator_free(host_allocator, context);
}

iree_hal_remote_server_bulk_upload_transfer_t*
iree_hal_remote_server_bulk_upload_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer) {
  return (iree_hal_remote_server_bulk_upload_transfer_t*)
      iree_net_bulk_transfer_user_storage(table_transfer)
          .data;
}

iree_net_bulk_transfer_t* iree_hal_remote_server_bulk_upload_lookup_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id) {
  if (!session_slot->bulk_transfer_scheduler) return NULL;
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer ||
      iree_net_bulk_transfer_user_value(table_transfer) != transfer_kind) {
    return NULL;
  }
  return table_transfer;
}

iree_status_t iree_hal_remote_server_bulk_upload_get_or_insert_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t total_length,
    iree_host_size_t data_chunk_length,
    iree_net_bulk_transfer_t** out_table_transfer) {
  *out_table_transfer = NULL;
  if (total_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CLIENT_FILE_READ length %" PRIu64
                            " exceeds host size max %" PRIhsz,
                            total_length, IREE_HOST_SIZE_MAX);
  }

  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (table_transfer) {
    if (iree_net_bulk_transfer_user_value(table_transfer) != transfer_kind) {
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

  iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      session_slot->bulk_transfer_scheduler, transfer_id, total_length,
      transfer_kind, &table_transfer);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_transfer_t* transfer =
        iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
    status = iree_hal_remote_server_bulk_upload_transfer_initialize(
        session_slot->server, session_slot, session_slot->session_id,
        transfer_id, total_length, data_chunk_length,
        session_slot->server->host_allocator, transfer);
  }
  if (!iree_status_is_ok(status) && table_transfer) {
    iree_hal_remote_bulk_transfer_scheduler_release(
        session_slot->bulk_transfer_scheduler, table_transfer);
    table_transfer = NULL;
  }

  *out_table_transfer = table_transfer;
  return status;
}

void iree_hal_remote_server_bulk_upload_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY |
              IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_PEER_COMPLETE |
              IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0 &&
      iree_hal_remote_bulk_transfer_tracker_is_complete(
          &transfer->receive_tracker)) {
    if (iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_CONTROL_RESPONSE)) {
      iree_status_t status = iree_hal_remote_server_session_send_response(
          transfer->session_slot, &transfer->response_envelope, IREE_STATUS_OK,
          NULL, 0);
      iree_status_free(status);
    } else {
      iree_status_t status = iree_hal_semaphore_list_signal(
          transfer->signal_semaphore_list, /*frontier=*/NULL);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
      }
      iree_hal_remote_server_bulk_upload_free_signal_list(transfer);
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0) {
    iree_hal_remote_bulk_transfer_scheduler_release(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

void iree_hal_remote_server_bulk_upload_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  iree_hal_remote_server_bulk_upload_signal_failure(transfer, status);
  iree_hal_remote_server_bulk_upload_release_chunks_for_transfer(
      session_slot, transfer->transfer_id);
  if (transfer->pending_operation_count == 0) {
    iree_hal_remote_bulk_transfer_scheduler_release(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

static void iree_hal_remote_server_bulk_upload_collect_ready_chunk(
    void* user_data, iree_net_bulk_chunk_t* chunk) {
  iree_hal_remote_server_bulk_upload_ready_chunk_query_t* query =
      (iree_hal_remote_server_bulk_upload_ready_chunk_query_t*)user_data;
  if (query->chunk) return;

  iree_hal_remote_server_session_t* session_slot = query->session_slot;
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler,
          iree_net_bulk_chunk_transfer_id(chunk));
  if (!table_transfer || iree_net_bulk_transfer_user_value(table_transfer) !=
                             query->transfer_kind) {
    return;
  }

  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  if (!iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY) ||
      iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    return;
  }

  query->chunk = chunk;
}

static iree_net_bulk_chunk_t*
iree_hal_remote_server_bulk_upload_take_ready_chunk_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind) {
  iree_hal_remote_server_bulk_upload_ready_chunk_query_t query;
  memset(&query, 0, sizeof(query));
  query.session_slot = session_slot;
  query.transfer_kind = transfer_kind;
  iree_net_bulk_receive_window_visit_chunks(
      session_slot->bulk_receive_window,
      iree_hal_remote_server_bulk_upload_collect_ready_chunk, &query);
  return query.chunk;
}

static bool iree_hal_remote_server_bulk_upload_submit_chunk_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    iree_net_bulk_chunk_t* chunk,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback) {
  const uint64_t transfer_id = iree_net_bulk_chunk_transfer_id(chunk);
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_server_bulk_upload_lookup_locked(
          session_slot, transfer_kind, transfer_id);
  if (!table_transfer) return false;
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);

  iree_hal_remote_server_bulk_staging_slot_t* staging_slot = NULL;
  iree_status_t status = iree_hal_remote_server_bulk_staging_pool_try_acquire(
      session_slot->bulk_staging_pool, transfer->local_device, &staging_slot);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_fail_locked(session_slot, table_transfer,
                                                   status);
    return false;
  }
  if (!staging_slot) return false;

  const uint64_t chunk_offset = iree_net_bulk_chunk_offset(chunk);
  const iree_const_byte_span_t chunk_payload =
      iree_net_bulk_chunk_payload(chunk);
  iree_byte_span_t staging_contents =
      iree_hal_remote_server_bulk_staging_slot_contents(staging_slot);
  memcpy(staging_contents.data, chunk_payload.data, chunk_payload.data_length);
  iree_net_bulk_receive_window_release_chunk(session_slot->bulk_receive_window,
                                             chunk);

  iree_hal_remote_server_t* server = transfer->server;
  iree_hal_remote_server_retain(server);
  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_buffer_t* target_buffer = transfer->target_buffer;
  iree_hal_buffer_retain(target_buffer);
  iree_hal_file_t* staging_file =
      iree_hal_remote_server_bulk_staging_slot_file(staging_slot);
  iree_hal_file_retain(staging_file);
  iree_hal_semaphore_t* staging_semaphore =
      iree_hal_remote_server_bulk_staging_slot_semaphore(staging_slot);
  iree_hal_semaphore_retain(staging_semaphore);
  iree_hal_remote_server_bulk_upload_staging_callback_t* callback_state =
      (iree_hal_remote_server_bulk_upload_staging_callback_t*)
          iree_hal_remote_server_bulk_staging_slot_user_storage(staging_slot)
              .data;
  callback_state->server = server;
  callback_state->session_slot = session_slot;
  callback_state->session_id = transfer->session_id;
  callback_state->transfer_id = transfer_id;

  uint64_t ready_wait_value = 1;
  iree_hal_semaphore_t* ready_semaphore = transfer->ready_semaphore;
  iree_hal_semaphore_list_t wait_list = {
      .count = 1,
      .semaphores = &ready_semaphore,
      .payload_values = &ready_wait_value,
  };
  uint64_t staging_signal_value =
      iree_hal_remote_server_bulk_staging_slot_next_signal_value(staging_slot);
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &staging_semaphore,
      .payload_values = &staging_signal_value,
  };

  ++transfer->pending_operation_count;

  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  status = iree_hal_device_queue_read(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      staging_file, /*source_offset=*/0, target_buffer,
      transfer->target_offset + (iree_device_size_t)chunk_offset,
      (iree_device_size_t)chunk_payload.data_length, IREE_HAL_READ_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_staging_slot_acquire_timepoint(
        staging_slot, staging_signal_value, staging_callback, callback_state);
  }
  iree_hal_semaphore_release(staging_semaphore);
  iree_hal_file_release(staging_file);
  iree_hal_buffer_release(target_buffer);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) return true;

  iree_hal_remote_server_release(callback_state->server);
  memset(callback_state, 0, sizeof(*callback_state));
  iree_hal_remote_server_bulk_staging_slot_release(
      staging_slot,
      iree_hal_remote_server_bulk_staging_slot_last_signal_value(staging_slot));
  table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
      session_slot, transfer_kind, transfer_id);
  if (table_transfer) {
    transfer =
        iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
    if (transfer->pending_operation_count > 0) {
      --transfer->pending_operation_count;
    }
    iree_hal_remote_server_bulk_upload_fail_locked(session_slot, table_transfer,
                                                   status);
  } else {
    iree_status_free(status);
  }
  return false;
}

void iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback) {
  if (!session_slot->bulk_transfer_scheduler ||
      !session_slot->bulk_receive_window) {
    return;
  }
  while (true) {
    iree_net_bulk_chunk_t* chunk =
        iree_hal_remote_server_bulk_upload_take_ready_chunk_locked(
            session_slot, transfer_kind);
    if (!chunk) return;
    if (!iree_hal_remote_server_bulk_upload_submit_chunk_locked(
            session_slot, transfer_kind, chunk, staging_callback)) {
      return;
    }
  }
}

void iree_hal_remote_server_bulk_upload_submit_ready_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_kind,
    iree_hal_semaphore_list_t wait_list,
    iree_async_semaphore_timepoint_fn_t ready_callback) {
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  const iree_hal_remote_server_bulk_upload_transfer_flags_t
      ready_done_or_pending_flags =
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_PENDING |
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_COMPLETE |
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
  if (iree_any_bit_set(transfer->flags, ready_done_or_pending_flags)) {
    return;
  }

  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_semaphore_t* ready_semaphore = transfer->ready_semaphore;
  iree_hal_semaphore_retain(ready_semaphore);
  iree_hal_remote_server_bulk_upload_ready_t* ready_context =
      transfer->ready_context;
  iree_hal_remote_server_bulk_upload_ready_retain(ready_context);

  uint64_t ready_signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &ready_semaphore,
      .payload_values = &ready_signal_value,
  };

  ready_context->timepoint.callback = ready_callback;
  ready_context->timepoint.user_data = ready_context;
  iree_hal_remote_server_bulk_upload_ready_retain(ready_context);
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_PENDING;
  ++transfer->pending_operation_count;

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_hal_device_queue_barrier(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)ready_semaphore, ready_signal_value,
        &ready_context->timepoint);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_ready_release(ready_context);
  }
  iree_hal_remote_server_bulk_upload_ready_release(ready_context);
  iree_hal_semaphore_release(ready_semaphore);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) return;
  table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
      session_slot, transfer_kind, transfer_id);
  if (table_transfer) {
    transfer =
        iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
    if (transfer->pending_operation_count > 0) {
      --transfer->pending_operation_count;
    }
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_PENDING;
    iree_hal_remote_server_bulk_upload_fail_locked(session_slot, table_transfer,
                                                   status);
  } else {
    iree_status_free(status);
  }
}

iree_status_t iree_hal_remote_server_bulk_upload_on_start_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags, iree_host_size_t data_chunk_length) {
  if (!session_slot->bulk_transfer_scheduler) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote server bulk START flags: 0x%02x", flags);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  iree_status_t status =
      iree_hal_remote_server_bulk_upload_get_or_insert_locked(
          session_slot, transfer_kind, transfer_id, total_size,
          data_chunk_length, &table_transfer);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_transfer_t* transfer =
        iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
    status =
        iree_hal_remote_server_bulk_upload_transfer_mark_start(transfer, flags);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_upload_on_data_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback) {
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote server bulk DATA flags: 0x%02x",
                            flags);
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (!table_transfer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
  } else if (iree_net_bulk_transfer_user_value(table_transfer) !=
             transfer_kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote server bulk DATA received for "
                            "non-upload transfer_id=%" PRIu64,
                            transfer_id);
  }

  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  iree_status_t status =
      iree_hal_remote_server_bulk_upload_transfer_record_data(
          transfer, transfer_id,
          iree_net_bulk_transfer_total_size(table_transfer), chunk_offset,
          chunk_data.data_length, flags);
  if (iree_status_is_ok(status)) {
    iree_net_bulk_chunk_t* chunk = NULL;
    status = iree_net_bulk_receive_window_acquire_chunk(
        session_slot->bulk_receive_window, transfer_id, chunk_offset, sequence,
        flags, chunk_data, lease, /*user_value=*/0, &chunk);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
        session_slot, transfer_kind, staging_callback);
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_upload_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_id) {
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  iree_status_t status =
      iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
          transfer, transfer_id);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                         table_transfer);
  }
  return status;
}

void iree_hal_remote_server_bulk_upload_on_ready_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active, iree_status_t status,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback) {
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_server_bulk_upload_lookup_locked(
          session_slot, transfer_kind, transfer_id);
  if (!table_transfer) {
    iree_status_free(status);
    return;
  }

  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  if (transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_PENDING;
  if (!session_active) {
    iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                         table_transfer);
    iree_status_free(status);
  } else if (iree_status_is_ok(status)) {
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_COMPLETE;
    iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
        session_slot, transfer_kind, staging_callback);
    table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
        session_slot, transfer_kind, transfer_id);
    if (table_transfer) {
      iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                           table_transfer);
    }
  } else {
    iree_hal_remote_server_bulk_upload_fail_locked(session_slot, table_transfer,
                                                   status);
  }
}

void iree_hal_remote_server_bulk_upload_on_staging_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active,
    iree_hal_remote_server_bulk_staging_slot_t* staging_slot,
    uint64_t signal_value, iree_status_t status,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback) {
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_server_bulk_upload_lookup_locked(
          session_slot, transfer_kind, transfer_id);
  iree_hal_remote_server_bulk_staging_slot_release(staging_slot, signal_value);
  if (!table_transfer) {
    iree_status_free(status);
    return;
  }

  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  if (transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  if (!session_active) {
    iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                         table_transfer);
    iree_status_free(status);
  } else if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
        session_slot, transfer_kind, staging_callback);
    table_transfer = iree_hal_remote_server_bulk_upload_lookup_locked(
        session_slot, transfer_kind, transfer_id);
    if (table_transfer) {
      iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                           table_transfer);
    }
  } else {
    iree_hal_remote_server_bulk_upload_fail_locked(session_slot, table_transfer,
                                                   status);
  }
}

iree_status_t iree_hal_remote_server_bulk_upload_attach_command_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t signal_list, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_semaphore_t** ready_semaphore,
    iree_hal_remote_server_bulk_upload_ready_t** ready_context,
    const iree_hal_remote_control_envelope_t* response_envelope) {
  iree_hal_remote_server_bulk_upload_transfer_t* transfer =
      iree_hal_remote_server_bulk_upload_transfer_storage(table_transfer);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "CLIENT_FILE_READ command already attached to transfer_id=%" PRIu64,
        iree_net_bulk_transfer_id(table_transfer));
  }

  iree_status_t status = iree_ok_status();
  if (response_envelope) {
    transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
  } else {
    status = iree_hal_semaphore_list_clone(&signal_list,
                                           session_slot->server->host_allocator,
                                           &transfer->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    transfer->local_device = local_device;
    transfer->target_buffer = target_buffer;
    iree_hal_buffer_retain(transfer->target_buffer);
    transfer->target_offset = target_offset;
    transfer->ready_semaphore = *ready_semaphore;
    *ready_semaphore = NULL;
    transfer->ready_context = *ready_context;
    *ready_context = NULL;
    if (response_envelope) {
      transfer->response_envelope = *response_envelope;
      transfer->flags |=
          IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_CONTROL_RESPONSE;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY;
  }
  return status;
}
