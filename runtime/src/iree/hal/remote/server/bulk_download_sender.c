// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_download_sender.h"

#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/net/channel/bulk/transfer_table.h"

static void iree_hal_remote_server_bulk_download_free_signal_list(
    iree_hal_remote_server_bulk_download_transfer_t* transfer) {
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->host_allocator);
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
}

static void iree_hal_remote_server_bulk_download_signal_failure(
    iree_hal_remote_server_bulk_download_transfer_t* transfer,
    iree_status_t status) {
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
  iree_hal_remote_server_bulk_download_free_signal_list(transfer);
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
}

static void iree_hal_remote_server_bulk_download_free_initial_wait_list(
    iree_hal_remote_server_bulk_download_transfer_t* transfer) {
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED)) {
    return;
  }
  iree_hal_semaphore_list_free(transfer->initial_wait_semaphore_list,
                               transfer->host_allocator);
  transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED;
}

void iree_hal_remote_server_bulk_download_ready_retain(
    iree_hal_remote_server_bulk_download_ready_t* context) {
  if (!context) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

void iree_hal_remote_server_bulk_download_ready_release(
    iree_hal_remote_server_bulk_download_ready_t* context) {
  if (!context) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) != 1) return;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_semaphore_release(context->local_semaphore);
  iree_hal_remote_server_release(context->server);
  iree_allocator_free(host_allocator, context);
}

iree_hal_remote_server_bulk_download_transfer_t*
iree_hal_remote_server_bulk_download_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer) {
  return (iree_hal_remote_server_bulk_download_transfer_t*)
      iree_net_bulk_transfer_user_storage(table_transfer)
          .data;
}

iree_status_t iree_hal_remote_server_bulk_download_transfer_initialize(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t transfer_id, iree_hal_device_t* local_device,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_write_flags_t write_flags,
    iree_async_semaphore_timepoint_fn_t ready_callback,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_download_transfer_t* transfer) {
  memset(transfer, 0, sizeof(*transfer));
  transfer->server = server;
  iree_hal_remote_server_retain(transfer->server);
  transfer->host_allocator = host_allocator;
  transfer->session_slot = session_slot;
  transfer->session_id = session_id;
  transfer->local_device = local_device;
  transfer->source_buffer = source_buffer;
  iree_hal_buffer_retain(transfer->source_buffer);
  transfer->source_offset = source_offset;
  transfer->write_flags = write_flags;
  transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();

  iree_hal_remote_server_bulk_download_ready_t* ready_context = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*ready_context), (void**)&ready_context);
  if (iree_status_is_ok(status)) {
    memset(ready_context, 0, sizeof(*ready_context));
    iree_atomic_ref_count_init(&ready_context->ref_count);
    ready_context->server = server;
    iree_hal_remote_server_retain(ready_context->server);
    ready_context->session_slot = session_slot;
    ready_context->session_id = session_id;
    ready_context->transfer_id = transfer_id;
    ready_context->host_allocator = host_allocator;
    ready_context->timepoint.callback = ready_callback;
    ready_context->timepoint.user_data = ready_context;
    transfer->ready_context = ready_context;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(transfer->source_buffer);
    iree_hal_remote_server_release(transfer->server);
    memset(transfer, 0, sizeof(*transfer));
  }
  return status;
}

void iree_hal_remote_server_bulk_download_transfer_deinitialize(
    iree_hal_remote_server_bulk_download_transfer_t* transfer) {
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_hal_remote_server_bulk_download_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  iree_hal_remote_server_bulk_download_free_initial_wait_list(transfer);
  iree_hal_remote_server_bulk_staging_slot_release(
      transfer->staging_slot, transfer->last_staging_signal_value);
  iree_hal_remote_server_bulk_download_ready_release(transfer->ready_context);
  iree_hal_buffer_release(transfer->source_buffer);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_bulk_download_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_bulk_transfer_scheduler_release(scheduler, table_transfer);
}

void iree_hal_remote_server_bulk_download_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_PEER_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_t status = iree_hal_semaphore_list_signal(
        transfer->signal_semaphore_list, /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
    }
    iree_hal_remote_server_bulk_download_free_signal_list(transfer);
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0) {
    iree_hal_remote_server_bulk_download_release_transfer(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        table_transfer);
  }
}

void iree_hal_remote_server_bulk_download_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  iree_hal_remote_server_bulk_download_signal_failure(transfer, status);
  if (transfer->pending_operation_count == 0) {
    iree_hal_remote_server_bulk_download_release_transfer(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        table_transfer);
  }
}

static void
iree_hal_remote_server_bulk_download_submit_next_staging_write_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  const iree_hal_remote_server_bulk_download_transfer_flags_t staging_busy_flags =
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_WRITE_PENDING |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, staging_busy_flags)) {
    return;
  }

  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (transfer->next_staging_offset >= total_length) {
    return;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t remaining_length =
      total_length - transfer->next_staging_offset;
  const iree_device_size_t staging_length = (iree_device_size_t)iree_min(
      remaining_length, (uint64_t)transfer->staging_contents.data_length);
  const iree_device_size_t source_offset =
      transfer->source_offset + transfer->next_staging_offset;
  uint64_t staging_signal_value = transfer->last_staging_signal_value + 1;

  iree_allocator_t host_allocator = transfer->host_allocator;
  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_buffer_t* source_buffer = transfer->source_buffer;
  iree_hal_buffer_retain(source_buffer);
  iree_hal_file_t* staging_file = transfer->staging_file;
  iree_hal_file_retain(staging_file);
  iree_hal_semaphore_t* staging_semaphore = transfer->staging_semaphore;
  iree_hal_semaphore_retain(staging_semaphore);
  iree_hal_write_flags_t write_flags = transfer->write_flags;
  iree_hal_remote_server_bulk_download_ready_t* ready_context =
      transfer->ready_context;
  iree_hal_remote_server_bulk_download_ready_retain(ready_context);

  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  uint64_t staging_wait_value = 0;
  const bool uses_initial_wait_list = transfer->next_staging_offset == 0;
  if (uses_initial_wait_list) {
    wait_list = transfer->initial_wait_semaphore_list;
    transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED;
  } else {
    staging_wait_value = transfer->last_staging_signal_value;
    wait_list = (iree_hal_semaphore_list_t){
        .count = 1,
        .semaphores = &staging_semaphore,
        .payload_values = &staging_wait_value,
    };
  }
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &staging_semaphore,
      .payload_values = &staging_signal_value,
  };

  transfer->staging_offset = transfer->next_staging_offset;
  transfer->staging_length = staging_length;
  transfer->next_staging_offset += staging_length;
  transfer->last_staging_signal_value = staging_signal_value;
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_WRITE_PENDING;
  ++transfer->pending_operation_count;

  iree_hal_remote_server_bulk_download_ready_retain(ready_context);

  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_status_t status = iree_hal_device_queue_write(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_buffer, source_offset, staging_file,
      /*target_offset=*/0, staging_length, write_flags);
  if (uses_initial_wait_list) {
    iree_hal_semaphore_list_free(wait_list, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)staging_semaphore, staging_signal_value,
        &ready_context->timepoint);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_ready_release(ready_context);
  }
  iree_hal_remote_server_bulk_download_ready_release(ready_context);
  iree_hal_semaphore_release(staging_semaphore);
  iree_hal_file_release(staging_file);
  iree_hal_buffer_release(source_buffer);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));

  if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    iree_status_free(status);
    return;
  }
  table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
      iree_hal_remote_server_bulk_session_scheduler(session_slot), transfer_id);
  if (!table_transfer) {
    iree_status_free(status);
    return;
  }
  transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  if (!iree_status_is_ok(status) && transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SUBMIT_PENDING;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_try_send_locked(
        session_slot, iree_hal_remote_server_bulk_session_channel(session_slot),
        table_transfer);
    if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
      iree_status_free(status);
      return;
    }
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        transfer_id);
    if (!table_transfer) {
      iree_status_free(status);
      return;
    }
    transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    if (iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
        transfer->pending_operation_count == 0) {
      iree_hal_remote_server_bulk_download_release_transfer(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          table_transfer);
    }
    iree_status_free(status);
  } else {
    iree_hal_remote_server_bulk_download_fail_locked(session_slot,
                                                     table_transfer, status);
  }
}

void iree_hal_remote_server_bulk_download_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  const iree_hal_remote_server_bulk_download_transfer_flags_t
      terminal_or_send_pending_flags =
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED |
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, terminal_or_send_pending_flags)) {
    return;
  }
  if (!bulk_channel) {
    iree_hal_remote_server_bulk_download_fail_locked(
        session_slot, table_transfer,
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "remote bulk channel is not available"));
    return;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_START_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_start(
            bulk_channel, transfer_id, total_length,
            IREE_NET_BULK_FRAME_FLAG_NONE, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_bulk_download_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_START_SENT |
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->pending_operation_count;
    return;
  }

  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY)) {
    if (iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel) == 0) {
      return;
    }
    const uint64_t chunk_end =
        transfer->staging_offset + transfer->staging_length;
    iree_net_bulk_frame_flags_t flags =
        chunk_end == total_length ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                  : IREE_NET_BULK_FRAME_FLAG_NONE;
    iree_async_span_t chunk_span = iree_async_span_from_ptr(
        transfer->staging_contents.data, transfer->staging_length);
    iree_async_span_list_t chunk_payload =
        iree_async_span_list_make(&chunk_span, 1);
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_data(
            bulk_channel, transfer_id, transfer->staging_offset,
            transfer->next_sequence, flags, chunk_payload, transfer_id,
            &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_bulk_download_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY;
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING |
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->next_sequence;
    ++transfer->pending_operation_count;
    return;
  }

  const iree_hal_remote_server_bulk_download_transfer_flags_t staging_pending_flags =
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_WRITE_PENDING |
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, staging_pending_flags)) {
    return;
  }

  if (transfer->next_staging_offset < total_length) {
    iree_hal_remote_server_bulk_download_submit_next_staging_write_locked(
        session_slot, table_transfer);
    return;
  }

  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_COMPLETE_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_complete(
            bulk_channel, transfer_id, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_bulk_download_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_COMPLETE_SENT |
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->pending_operation_count;
    return;
  }

  iree_hal_remote_server_bulk_download_try_finish_locked(session_slot,
                                                         table_transfer);
}

static bool iree_hal_remote_server_bulk_download_select_ready(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  const uint64_t transfer_kind = *(const uint64_t*)user_data;
  if (iree_net_bulk_transfer_user_value(table_transfer) != transfer_kind) {
    return false;
  }
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    return false;
  }
  return true;
}

void iree_hal_remote_server_bulk_download_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind) {
  if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) return;
  uint64_t transfer_ids[IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY];
  iree_host_size_t transfer_count = 0;
  const uint64_t selected_transfer_kind = transfer_kind;
  bool all_ids_collected =
      iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          iree_hal_remote_server_bulk_download_select_ready,
          (void*)&selected_transfer_kind, transfer_ids,
          IREE_ARRAYSIZE(transfer_ids), &transfer_count);
  IREE_ASSERT(all_ids_collected);
  (void)all_ids_collected;
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) return;
    iree_net_bulk_transfer_t* table_transfer =
        iree_hal_remote_bulk_transfer_scheduler_lookup(
            iree_hal_remote_server_bulk_session_scheduler(session_slot),
            transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_server_bulk_download_try_send_locked(
          session_slot, bulk_channel, table_transfer);
    }
  }
}

iree_status_t iree_hal_remote_server_bulk_download_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_id) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_PEER_COMPLETE;
  iree_hal_remote_server_bulk_download_try_finish_locked(session_slot,
                                                         table_transfer);
  (void)transfer_id;
  return iree_ok_status();
}

void iree_hal_remote_server_bulk_download_on_send_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  if (transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING;
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING)) {
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING;
  }
  if (iree_status_is_ok(status)) {
    const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
    iree_hal_remote_server_bulk_download_try_send_locked(
        session_slot, bulk_channel, table_transfer);
    table_transfer = NULL;
    if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          transfer_id);
    }
    if (table_transfer) {
      iree_hal_remote_server_bulk_download_try_finish_locked(session_slot,
                                                             table_transfer);
    }
    iree_status_free(status);
    return;
  }

  iree_hal_remote_server_bulk_download_fail_locked(session_slot, table_transfer,
                                                   status);
}

void iree_hal_remote_server_bulk_download_on_ready_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active, iree_status_t status) {
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        transfer_id);
  }
  if (!table_transfer ||
      iree_net_bulk_transfer_user_value(table_transfer) != transfer_kind) {
    iree_status_free(status);
    return;
  }

  iree_hal_remote_server_bulk_download_transfer_t* transfer =
      iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
  if (transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_WRITE_PENDING;

  if (!session_active) {
    iree_status_free(status);
    iree_hal_remote_server_bulk_download_try_finish_locked(session_slot,
                                                           table_transfer);
  } else if (iree_status_is_ok(status)) {
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY;
    iree_hal_remote_server_bulk_download_try_send_locked(
        session_slot, bulk_channel, table_transfer);
    table_transfer = NULL;
    if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          transfer_id);
    }
    if (table_transfer) {
      iree_hal_remote_server_bulk_download_try_finish_locked(session_slot,
                                                             table_transfer);
    }
    iree_status_free(status);
  } else {
    iree_hal_remote_server_bulk_download_fail_locked(session_slot,
                                                     table_transfer, status);
  }
}

iree_status_t iree_hal_remote_server_bulk_download_submit_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind,
    uint64_t session_id, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    uint64_t transfer_id, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_device_size_t length,
    iree_hal_write_flags_t flags,
    iree_async_semaphore_timepoint_fn_t ready_callback) {
  iree_status_t status =
      iree_hal_buffer_validate_range(source_buffer, source_offset, length);

  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_net_bulk_transfer_t* table_transfer = NULL;
  bool transfer_inserted = false;
  if (iree_status_is_ok(status) &&
      !iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        transfer_id, length, transfer_kind, &table_transfer);
    transfer_inserted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_transfer_t* transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    status = iree_hal_remote_server_bulk_download_transfer_initialize(
        server, session_slot, session_id, transfer_id, local_device,
        source_buffer, source_offset, flags, ready_callback, host_allocator,
        transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_transfer_t* transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    status = iree_hal_remote_server_bulk_staging_pool_acquire(
        iree_hal_remote_server_bulk_session_staging_pool(session_slot),
        local_device, &transfer->staging_slot);
    if (iree_status_is_ok(status)) {
      transfer->staging_file =
          iree_hal_remote_server_bulk_staging_slot_file(transfer->staging_slot);
      transfer->staging_contents =
          iree_hal_remote_server_bulk_staging_slot_contents(
              transfer->staging_slot);
      transfer->staging_semaphore =
          iree_hal_remote_server_bulk_staging_slot_semaphore(
              transfer->staging_slot);
      transfer->last_staging_signal_value =
          iree_hal_remote_server_bulk_staging_slot_last_signal_value(
              transfer->staging_slot);
      transfer->ready_context->local_semaphore = transfer->staging_semaphore;
      iree_hal_semaphore_retain(transfer->ready_context->local_semaphore);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_transfer_t* transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    status = iree_hal_semaphore_list_clone(
        &wait_list, host_allocator, &transfer->initial_wait_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_transfer_t* transfer =
        iree_hal_remote_server_bulk_download_transfer_storage(table_transfer);
    status = iree_hal_semaphore_list_clone(&signal_list, host_allocator,
                                           &transfer->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_download_try_send_locked(
        session_slot, bulk_channel, table_transfer);
  }
  if (!iree_status_is_ok(status) && transfer_inserted) {
    iree_hal_remote_server_bulk_download_release_transfer(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        table_transfer);
  }
  return status;
}
