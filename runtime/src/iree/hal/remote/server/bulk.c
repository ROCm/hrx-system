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
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/hal/remote/server/bulk_upload_receiver.h"
#include "iree/hal/remote/server/profile.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/net/channel/bulk/receive_window.h"

typedef union iree_hal_remote_server_bulk_transfer_storage_t {
  // Client-file queue_read upload state.
  iree_hal_remote_server_bulk_upload_transfer_t client_file_read;

  // Client-file queue_write download state.
  iree_hal_remote_server_bulk_download_transfer_t client_file_write;

  // Server-originated profile callback transfer state.
  iree_hal_remote_server_profile_transfer_t profile_send;
} iree_hal_remote_server_bulk_transfer_storage_t;

static iree_hal_remote_server_bulk_transfer_kind_t
iree_hal_remote_server_bulk_transfer_kind(iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_kind_t)
      iree_net_bulk_transfer_user_value(transfer);
}

static iree_hal_remote_server_bulk_transfer_storage_t*
iree_hal_remote_server_bulk_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_storage_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
}

static void iree_hal_remote_server_client_file_read_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_client_file_read_chunk_callback(
    void* user_data, iree_hal_remote_server_bulk_staging_slot_t* staging_slot,
    uint64_t signal_value, iree_status_t status);

static void iree_hal_remote_server_session_try_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot);

static iree_status_t iree_hal_remote_server_bulk_receive_window_send_credit(
    void* user_data, uint32_t credit_delta) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  if (!session_slot->bulk_channel) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_status_t failure_status = iree_ok_status();
  iree_hal_remote_bulk_channel_send_result_t send_result =
      iree_hal_remote_bulk_channel_send_credit(
          session_slot->bulk_channel, credit_delta, /*operation_user_data=*/0,
          &failure_status);
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED) {
    return iree_ok_status();
  }
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  return failure_status;
}

static iree_status_t iree_hal_remote_server_bulk_flush_receive_window_locked(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_receive_window) return iree_ok_status();
  iree_status_t status = iree_net_bulk_receive_window_flush_credit(
      session_slot->bulk_receive_window);
  if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_flush_receive_window(
    iree_hal_remote_server_session_t* session_slot) {
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status =
      iree_hal_remote_server_bulk_flush_receive_window_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  return status;
}

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

static void iree_hal_remote_server_bulk_transfer_deinitialize(
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_transfer_storage_t* transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  switch (iree_hal_remote_server_bulk_transfer_kind(table_transfer)) {
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
      iree_hal_remote_server_bulk_upload_transfer_deinitialize(
          &transfer->client_file_read);
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
      iree_hal_remote_server_bulk_download_transfer_deinitialize(
          &transfer->client_file_write);
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
      iree_hal_remote_server_profile_transfer_deinitialize(
          &transfer->profile_send);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_bulk_transfer_deinitialize_callback(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_transfer_deinitialize(transfer);
}

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
  session_active = session_slot->session_id == context->session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_bulk_upload_on_ready_timepoint_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      context->transfer_id, session_active, status,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_flush_receive_window(session_slot);
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
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
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
  session_active = session_slot->session_id == session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_bulk_upload_on_staging_timepoint_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, session_active, staging_slot, signal_value, status,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_flush_receive_window(session_slot);
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
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
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
  session_active = session_slot->session_id == context->session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_on_ready_timepoint_locked(
      session_slot, bulk_channel,
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE,
      context->transfer_id, session_active, status);
  drain_profile_pending =
      session_active &&
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_net_bulk_channel_release(bulk_channel);

  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
  iree_hal_remote_server_bulk_download_ready_release(context);
}

static void iree_hal_remote_server_bulk_fail_transfer_for_drain_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  switch (iree_hal_remote_server_bulk_transfer_kind(table_transfer)) {
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
      iree_hal_remote_server_bulk_upload_fail_locked(
          session_slot, table_transfer,
          iree_make_status(IREE_STATUS_CANCELLED,
                           "remote bulk transfer cancelled"));
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
      iree_hal_remote_server_bulk_download_fail_locked(
          session_slot, table_transfer,
          iree_make_status(IREE_STATUS_CANCELLED,
                           "remote bulk transfer cancelled"));
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      break;
    default:
      iree_hal_remote_server_bulk_release_transfer(
          session_slot->bulk_transfer_scheduler, table_transfer);
      break;
  }
}

iree_status_t iree_hal_remote_server_session_initialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  options.capacity = IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY;
  options.user_storage_size =
      sizeof(iree_hal_remote_server_bulk_transfer_storage_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_server_bulk_transfer_storage_t);
  options.initial_transfer_id = 2;
  options.transfer_id_stride = 2;
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {
      .deinitialize =
          iree_hal_remote_server_bulk_transfer_deinitialize_callback,
      .user_data = NULL,
  };
  iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_allocate(
      &options, callbacks, host_allocator,
      &session_slot->bulk_transfer_scheduler);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_staging_pool_options_t staging_options =
        iree_hal_remote_server_bulk_staging_pool_options_default();
    staging_options.slot_count = IREE_HAL_REMOTE_BULK_STAGING_SLOT_COUNT;
    staging_options.slot_length = IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH;
    staging_options.user_storage_size =
        sizeof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    staging_options.user_storage_alignment =
        iree_alignof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    status = iree_hal_remote_server_bulk_staging_pool_create(
        &staging_options, host_allocator, &session_slot->bulk_staging_pool);
  }
  iree_net_bulk_receive_window_options_t receive_window_options =
      iree_net_bulk_receive_window_options_default();
  receive_window_options.chunk_pool.capacity =
      IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT;
  iree_net_bulk_receive_window_callbacks_t receive_window_callbacks = {
      .send_credit = iree_hal_remote_server_bulk_receive_window_send_credit,
      .user_data = session_slot,
  };
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_receive_window_allocate(
        &receive_window_options, receive_window_callbacks, host_allocator,
        &session_slot->bulk_receive_window);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_bulk_receive_window_free(session_slot->bulk_receive_window);
    session_slot->bulk_receive_window = NULL;
    iree_hal_remote_server_bulk_staging_pool_release(
        session_slot->bulk_staging_pool);
    session_slot->bulk_staging_pool = NULL;
    iree_hal_remote_bulk_transfer_scheduler_free(
        session_slot->bulk_transfer_scheduler);
    session_slot->bulk_transfer_scheduler = NULL;
  }
  return status;
}

static bool iree_hal_remote_server_session_take_drained_bulk_state_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler,
    iree_hal_remote_server_bulk_staging_pool_t** out_staging_pool,
    iree_net_bulk_receive_window_t** out_receive_window) {
  *out_scheduler = NULL;
  *out_staging_pool = NULL;
  *out_receive_window = NULL;

  if (!iree_any_bit_set(
          session_slot->flags,
          IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING)) {
    return false;
  }
  if (session_slot->bulk_transfer_scheduler &&
      iree_hal_remote_bulk_transfer_scheduler_count(
          session_slot->bulk_transfer_scheduler) != 0) {
    return false;
  }
  if (iree_hal_remote_server_profile_has_pending_transfers_locked(
          session_slot)) {
    return false;
  }
  if (session_slot->bulk_channel &&
      iree_net_bulk_channel_has_pending_sends(session_slot->bulk_channel)) {
    return false;
  }

  *out_scheduler = session_slot->bulk_transfer_scheduler;
  session_slot->bulk_transfer_scheduler = NULL;
  *out_staging_pool = session_slot->bulk_staging_pool;
  session_slot->bulk_staging_pool = NULL;
  *out_receive_window = session_slot->bulk_receive_window;
  session_slot->bulk_receive_window = NULL;
  return true;
}

static void iree_hal_remote_server_session_try_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool = NULL;
  iree_net_bulk_receive_window_t* receive_window = NULL;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  bool drained = iree_hal_remote_server_session_take_drained_bulk_state_locked(
      session_slot, &scheduler, &staging_pool, &receive_window);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (drained) {
    iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
    iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
    iree_net_bulk_receive_window_free(receive_window);
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
}

void iree_hal_remote_server_session_deinitialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_transfer_scheduler &&
      !session_slot->bulk_staging_pool && !session_slot->bulk_receive_window &&
      !iree_hal_remote_server_profile_has_pending_transfers_locked(
          session_slot)) {
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  const bool drain_pending =
      iree_any_bit_set(session_slot->flags,
                       IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING);
  bool bulk_drained = false;
  iree_hal_remote_server_profile_pending_transfer_t* pending_profile_transfers =
      NULL;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  pending_profile_transfers =
      iree_hal_remote_server_profile_take_pending_transfers_locked(
          session_slot);
  if (!drain_pending && session_slot->bulk_transfer_scheduler) {
    iree_hal_remote_bulk_transfer_scheduler_clear(
        session_slot->bulk_transfer_scheduler);
  } else if (drain_pending && session_slot->bulk_transfer_scheduler) {
    uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
    iree_host_size_t transfer_count = 0;
    bool all_ids_collected =
        iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
            session_slot->bulk_transfer_scheduler, /*select=*/NULL, NULL,
            transfer_ids, IREE_ARRAYSIZE(transfer_ids), &transfer_count);
    IREE_ASSERT(all_ids_collected);
    (void)all_ids_collected;
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      if (!session_slot->bulk_transfer_scheduler) break;
      iree_net_bulk_transfer_t* table_transfer =
          iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_ids[i]);
      if (table_transfer) {
        iree_hal_remote_server_bulk_fail_transfer_for_drain_locked(
            session_slot, table_transfer);
      }
    }
  }
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler =
      session_slot->bulk_transfer_scheduler;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool =
      session_slot->bulk_staging_pool;
  iree_net_bulk_receive_window_t* receive_window =
      session_slot->bulk_receive_window;
  if (!drain_pending) {
    session_slot->bulk_transfer_scheduler = NULL;
    session_slot->bulk_staging_pool = NULL;
    session_slot->bulk_receive_window = NULL;
  } else {
    bulk_drained =
        iree_hal_remote_server_session_take_drained_bulk_state_locked(
            session_slot, &scheduler, &staging_pool, &receive_window);
    if (!bulk_drained) {
      scheduler = NULL;
      staging_pool = NULL;
      receive_window = NULL;
    }
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_profile_pending_transfer_free_list(
      session_slot->server, pending_profile_transfers);
  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
  iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
  iree_net_bulk_receive_window_free(receive_window);
  if (drain_pending && bulk_drained) {
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
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
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfer_scheduler ||
        !session_slot->bulk_receive_window) {
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
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_id);
        }
      }
      if (iree_status_is_ok(status) && table_transfer) {
        iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
            session_slot,
            IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
            iree_hal_remote_server_client_file_read_chunk_callback);
        table_transfer = NULL;
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_id);
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
          session_slot->bulk_transfer_scheduler, table_transfer);
    }
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
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
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    status = iree_hal_remote_server_bulk_download_submit_locked(
        session_slot, session_slot->bulk_channel,
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE, session_id,
        local_device, wait_list, signal_list, transfer_id, source_buffer,
        source_offset, length, flags,
        iree_hal_remote_server_bulk_download_ready_callback);
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_start(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_size, iree_net_bulk_frame_flags_t flags) {
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_hal_remote_server_bulk_upload_on_start_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, total_size, flags, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status) && session_slot->bulk_channel) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_refresh_credit(session_slot->bulk_channel,
                                                    /*operation_user_data=*/0,
                                                    &failure_status);
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
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_hal_remote_server_bulk_upload_on_data_locked(
      session_slot, IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ,
      transfer_id, chunk_offset, sequence, flags, chunk_data, lease,
      iree_hal_remote_server_client_file_read_chunk_callback);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  iree_status_t status = iree_ok_status();
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  if (!session_slot->bulk_transfer_scheduler) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
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
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status) && profile_sequence != 0) {
    status = iree_hal_remote_server_profile_observe_transfer(
        session_slot, profile_sequence, iree_ok_status());
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_abort(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
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
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_ok_status();
  if (profile_sequence != 0) {
    status = iree_hal_remote_server_profile_observe_transfer(
        session_slot, profile_sequence,
        iree_make_status(IREE_STATUS_ABORTED,
                         "remote client aborted profile transfer"));
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

void iree_hal_remote_server_bulk_on_send_complete(
    iree_hal_remote_server_session_t* session_slot,
    uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    iree_status_free(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }
  const uint64_t transfer_id = operation_user_data;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  if (!session_slot->bulk_transfer_scheduler) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  if (iree_hal_remote_server_bulk_transfer_kind(table_transfer) ==
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_t profile_status =
        iree_hal_remote_server_profile_on_send_complete(session_slot,
                                                        transfer_id, status);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, profile_status,
        iree_make_cstring_view("profile transfer send completion failed"));
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  if (iree_hal_remote_server_bulk_transfer_kind(table_transfer) !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  bool drain_profile_pending = false;
  iree_hal_remote_server_bulk_download_on_send_complete_locked(
      session_slot, session_slot->bulk_channel, table_transfer, status);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, session_slot->bulk_channel,
            /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
}

void iree_hal_remote_server_bulk_on_credit(
    iree_hal_remote_server_session_t* session_slot) {
  uint64_t failed_profile_sequence = 0;
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_bulk_download_try_send_all_locked(
      session_slot, session_slot->bulk_channel,
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE);
  iree_status_t status = iree_hal_remote_server_profile_try_send_all_locked(
      session_slot, session_slot->bulk_channel, &failed_profile_sequence);
  drain_profile_pending =
      iree_hal_remote_server_profile_has_pending_transfers_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (!iree_status_is_ok(status) && failed_profile_sequence != 0) {
    status = iree_hal_remote_server_profile_observe_transfer(
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
            session_slot, session_slot->bulk_channel,
            /*returned_sequence=*/0);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, drain_status,
        iree_make_cstring_view("profile transfer drain failed"));
  }
}
