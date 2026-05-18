// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk.h"

#include "iree/async/semaphore.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/bulk_download_sender.h"
#include "iree/hal/remote/server/bulk_profile_sender.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/hal/remote/server/bulk_upload_receiver.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"

static iree_hal_remote_server_bulk_transfer_kind_t
iree_hal_remote_server_bulk_transfer_kind(iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_kind_t)
      iree_net_bulk_transfer_user_value(transfer);
}

static void iree_hal_remote_server_client_file_read_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_client_file_read_chunk_callback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* staging_slot,
    uint64_t signal_value, iree_status_t status);

static void iree_hal_remote_server_bulk_shutdown_session_on_error(
    iree_hal_remote_server_session_t* session_slot, iree_status_t status,
    iree_string_view_t reason) {
  if (iree_status_is_ok(status)) return;
  iree_net_session_t* session = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  session = session_slot->session;
  if (session) iree_net_session_retain(session);
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  if (session) {
    status = iree_status_join(
        status, iree_net_session_shutdown(session, /*reason_code=*/0, reason));
    iree_net_session_release(session);
  }
  iree_status_free(status);
}

static void iree_hal_remote_server_bulk_download_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_bulk_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_scheduler_release(scheduler, transfer);
}

static void iree_hal_remote_server_client_file_read_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_bulk_upload_ready_t* context =
      (iree_hal_remote_server_bulk_upload_ready_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active =
      session_slot->session_id == context->session_id &&
      session_slot->session != NULL &&
      iree_hal_remote_server_bulk_session_channel(session_slot) != NULL;
  if (session_active) {
    bulk_channel = iree_hal_remote_server_bulk_session_channel(session_slot);
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_hal_remote_server_bulk_upload_on_ready_timepoint_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      context->transfer_id, session_active, status,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_session_flush_receive_window(session_slot);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, flush_status,
        iree_make_cstring_view("bulk receive credit flush failed"));
  }
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_net_bulk_channel_release(bulk_channel);
  iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
  iree_hal_remote_server_bulk_upload_ready_release(context);
}

static void iree_hal_remote_server_client_file_read_chunk_callback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* staging_slot,
    uint64_t signal_value, iree_status_t status) {
  iree_hal_remote_server_bulk_upload_staging_callback_t* callback_state =
      (iree_hal_remote_server_bulk_upload_staging_callback_t*)user_data;
  iree_hal_remote_server_t* server = callback_state->server;
  iree_hal_remote_server_session_t* session_slot = callback_state->session_slot;
  const uint64_t session_id = callback_state->session_id;
  const uint64_t transfer_id = callback_state->transfer_id;
  memset(callback_state, 0, sizeof(*callback_state));

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active =
      session_slot->session_id == session_id && session_slot->session != NULL &&
      iree_hal_remote_server_bulk_session_channel(session_slot) != NULL;
  if (session_active) {
    bulk_channel = iree_hal_remote_server_bulk_session_channel(session_slot);
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_hal_remote_server_bulk_upload_on_staging_timepoint_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, session_active, staging_slot, signal_value, status,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_session_flush_receive_window(session_slot);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, flush_status,
        iree_make_cstring_view("bulk receive credit flush failed"));
  }
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_net_bulk_channel_release(bulk_channel);
  iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
  iree_hal_remote_server_release(server);
}

static void iree_hal_remote_server_bulk_download_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_bulk_download_ready_t* context =
      (iree_hal_remote_server_bulk_download_ready_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active =
      session_slot->session_id == context->session_id &&
      session_slot->session != NULL &&
      iree_hal_remote_server_bulk_session_channel(session_slot) != NULL;
  if (session_active) {
    bulk_channel = iree_hal_remote_server_bulk_session_channel(session_slot);
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_hal_remote_server_bulk_download_on_ready_timepoint_locked(
      session_slot, bulk_channel,
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE,
      context->transfer_id, session_active, status);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_net_bulk_channel_release(bulk_channel);

  iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
  iree_hal_remote_server_bulk_download_ready_release(context);
}

static iree_status_t iree_hal_remote_server_bulk_submit_client_file_read_impl(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags,
    const iree_hal_remote_control_envelope_t* response_envelope) {
  iree_status_t status =
      iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  if (iree_status_is_ok(status) && flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported CLIENT_FILE_READ flags: 0x%" PRIx64,
                              flags);
  }

  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_hal_semaphore_t* ready_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &ready_semaphore);
  }

  iree_hal_remote_server_bulk_upload_ready_t* ready_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*ready_context),
                                   (void**)&ready_context);
  }
  if (iree_status_is_ok(status)) {
    memset(ready_context, 0, sizeof(*ready_context));
    iree_atomic_ref_count_init(&ready_context->ref_count);
    ready_context->server = server;
    iree_hal_remote_server_retain(ready_context->server);
    ready_context->session_slot = session_slot;
    ready_context->session_id = session_slot->session_id;
    ready_context->transfer_id = transfer_id;
    ready_context->local_semaphore = ready_semaphore;
    iree_hal_semaphore_retain(ready_context->local_semaphore);
    ready_context->host_allocator = host_allocator;
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  bool transfer_inserted_or_found = false;
  bool drain_profile_pending = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    if (!iree_hal_remote_server_bulk_session_scheduler(session_slot) ||
        !iree_hal_remote_server_bulk_session_receive_window(session_slot)) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_get_or_insert_locked(
          session_slot,
          IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
          transfer_id, (uint64_t)length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
          &table_transfer);
      transfer_inserted_or_found = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_attach_command_locked(
          session_slot, table_transfer, local_device, signal_list,
          target_buffer, target_offset, &ready_semaphore, &ready_context,
          response_envelope);
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_bulk_upload_submit_ready_locked(
            session_slot, table_transfer,
            IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
            wait_list, iree_hal_remote_server_client_file_read_ready_callback);
        table_transfer = NULL;
        if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              iree_hal_remote_server_bulk_session_scheduler(session_slot),
              transfer_id);
        }
      }
      if (iree_status_is_ok(status) && table_transfer) {
        iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
            session_slot,
            IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
            iree_hal_remote_server_client_file_read_chunk_callback);
        table_transfer = NULL;
        if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              iree_hal_remote_server_bulk_session_scheduler(session_slot),
              transfer_id);
        }
      }
      if (iree_status_is_ok(status) && table_transfer) {
        iree_hal_remote_server_bulk_upload_try_finish_locked(session_slot,
                                                             table_transfer);
      }
    }
    if (!iree_status_is_ok(status) && transfer_inserted_or_found &&
        table_transfer) {
      iree_hal_remote_server_bulk_release_transfer(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          table_transfer);
    }
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_server_bulk_session_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot,
                    iree_hal_remote_server_bulk_session_channel(session_slot),
                    /*returned_sequence=*/0));
  }
  iree_hal_remote_server_bulk_upload_ready_release(ready_context);
  iree_hal_semaphore_release(ready_semaphore);
  return status;
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_read(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_hal_remote_server_bulk_submit_client_file_read_impl(
      session_slot, local_device, wait_list, signal_list, transfer_id,
      target_buffer, target_offset, length, flags, /*response_envelope=*/NULL);
}

iree_status_t iree_hal_remote_server_bulk_submit_buffer_unmap(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device,
    const iree_hal_remote_control_envelope_t* response_envelope,
    uint64_t transfer_id, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_hal_remote_server_bulk_submit_client_file_read_impl(
      session_slot, local_device, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), transfer_id, target_buffer,
      target_offset, length, IREE_HAL_READ_FLAG_NONE, response_envelope);
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_write(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_status_t status = iree_ok_status();
  uint64_t session_id = 0;
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_id = session_slot->session_id;
  bool session_active = session_slot->session != NULL;
  iree_slim_mutex_unlock(&server->session_mutex);
  if (!session_active) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    status = iree_hal_remote_server_bulk_download_submit_locked(
        session_slot, iree_hal_remote_server_bulk_session_channel(session_slot),
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE, session_id,
        local_device, wait_list, signal_list, transfer_id, source_buffer,
        source_offset, length, flags,
        iree_hal_remote_server_bulk_download_ready_callback);
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
  }

  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot,
                    iree_hal_remote_server_bulk_session_channel(session_slot),
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_start(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_size, iree_net_bulk_frame_flags_t flags) {
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_status_t status = iree_hal_remote_server_bulk_upload_on_start_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, total_size, flags, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (iree_status_is_ok(status) &&
      iree_hal_remote_server_bulk_session_channel(session_slot)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_refresh_credit(
            iree_hal_remote_server_bulk_session_channel(session_slot),
            /*operation_user_data=*/0, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      status = failure_status;
    }
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_data(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_status_t status = iree_hal_remote_server_bulk_upload_on_data_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, chunk_offset, sequence, flags, chunk_data, lease,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_server_bulk_session_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot,
                    iree_hal_remote_server_bulk_session_channel(session_slot),
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  iree_status_t status = iree_ok_status();
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        transfer_id);
  }
  if (!table_transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
  } else {
    switch (iree_hal_remote_server_bulk_transfer_kind(table_transfer)) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ: {
        status = iree_hal_remote_server_bulk_upload_on_complete_locked(
            session_slot, table_transfer, transfer_id);
        break;
      }
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE: {
        status = iree_hal_remote_server_bulk_download_on_complete_locked(
            session_slot, table_transfer, transfer_id);
        break;
      }
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND: {
        profile_sequence = iree_hal_remote_server_profile_on_complete_locked(
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
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (iree_status_is_ok(status) && profile_sequence != 0) {
    status = iree_hal_remote_server_profile_relay_observe_transfer(
        session_slot, profile_sequence, iree_ok_status());
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot,
                    iree_hal_remote_server_bulk_session_channel(session_slot),
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_abort(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        iree_hal_remote_server_bulk_session_scheduler(session_slot),
        transfer_id);
  }
  if (table_transfer) {
    switch (iree_hal_remote_server_bulk_transfer_kind(table_transfer)) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
        iree_hal_remote_server_bulk_upload_fail_locked(
            session_slot, table_transfer,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        break;
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
        iree_hal_remote_server_bulk_download_fail_locked(
            session_slot, table_transfer,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        break;
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
        profile_sequence = iree_hal_remote_server_profile_on_abort_locked(
            session_slot, table_transfer);
        break;
      default:
        break;
    }
  }
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_status_t status = iree_ok_status();
  if (profile_sequence != 0) {
    status = iree_hal_remote_server_profile_relay_observe_transfer(
        session_slot, profile_sequence,
        iree_make_status(IREE_STATUS_ABORTED,
                         "remote client aborted profile transfer"));
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot,
                    iree_hal_remote_server_bulk_session_channel(session_slot),
                    /*returned_sequence=*/0));
  }
  return status;
}

void iree_hal_remote_server_bulk_on_send_complete(
    iree_hal_remote_server_session_t* session_slot,
    uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    iree_status_free(status);
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }
  const uint64_t transfer_id = operation_user_data;

  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (!iree_hal_remote_server_bulk_session_scheduler(session_slot)) {
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    iree_status_free(status);
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          iree_hal_remote_server_bulk_session_scheduler(session_slot),
          transfer_id);
  if (!table_transfer) {
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    iree_status_free(status);
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }

  if (iree_hal_remote_server_bulk_transfer_kind(table_transfer) ==
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND) {
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    iree_status_t profile_status =
        iree_hal_remote_server_profile_on_send_complete(session_slot,
                                                        transfer_id, status);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, profile_status,
        iree_make_cstring_view("profile transfer send completion failed"));
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }

  if (iree_hal_remote_server_bulk_transfer_kind(table_transfer) !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    iree_slim_mutex_unlock(
        iree_hal_remote_server_bulk_session_mutex(session_slot));
    iree_status_free(status);
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }

  bool drain_profile_pending = false;
  iree_hal_remote_server_bulk_download_on_send_complete_locked(
      session_slot, iree_hal_remote_server_bulk_session_channel(session_slot),
      table_transfer, status);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot,
            iree_hal_remote_server_bulk_session_channel(session_slot),
            /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
}

void iree_hal_remote_server_bulk_on_credit(
    iree_hal_remote_server_session_t* session_slot) {
  uint64_t failed_profile_sequence = 0;
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_hal_remote_server_bulk_download_try_send_all_locked(
      session_slot, iree_hal_remote_server_bulk_session_channel(session_slot),
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE);
  iree_status_t status = iree_hal_remote_server_profile_try_send_all_locked(
      session_slot, iree_hal_remote_server_bulk_session_channel(session_slot),
      &failed_profile_sequence);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  if (!iree_status_is_ok(status) && failed_profile_sequence != 0) {
    status = iree_hal_remote_server_profile_relay_observe_transfer(
        session_slot, failed_profile_sequence, status);
  } else {
    iree_status_free(status);
  }
  iree_hal_remote_server_bulk_shutdown_session_on_error(
      session_slot, status,
      iree_make_cstring_view("profile transfer send failed"));
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot,
            iree_hal_remote_server_bulk_session_channel(session_slot),
            /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
}
