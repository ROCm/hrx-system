// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_session.h"

#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/bulk_download_sender.h"
#include "iree/hal/remote/server/bulk_profile_sender.h"
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/hal/remote/server/bulk_upload_receiver.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/hal/remote/util/bulk_router.h"
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

struct iree_hal_remote_server_bulk_session_t {
  // References held by the session slot owner and in-flight callbacks.
  iree_atomic_ref_count_t ref_count;

  // Session slot whose identity gates callbacks using this state.
  iree_hal_remote_server_session_t* session_slot;

  // Host allocator used to allocate and release this component.
  iree_allocator_t host_allocator;

  // Protects active bulk transfer state and receive-window state.
  iree_slim_mutex_t transfer_mutex;

  // Bulk channel callback router bound to |session_slot|.
  iree_hal_remote_bulk_router_t router;

  // Bulk channel for large payload transfers, or NULL before endpoint open.
  iree_net_bulk_channel_t* channel;

  // Active bulk transfer scheduler and lifecycle owner.
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler;

  // Reusable host staging slots for bulk queue file operations.
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool;

  // Receive window retaining client-to-server DATA chunks and CREDIT state.
  iree_net_bulk_receive_window_t* receive_window;

  // Server-originated profiling callback relay state.
  iree_hal_remote_server_profile_relay_t profile_relay;
};

static iree_hal_remote_server_bulk_transfer_kind_t
iree_hal_remote_server_bulk_session_transfer_kind(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_kind_t)
      iree_net_bulk_transfer_user_value(transfer);
}

static iree_hal_remote_server_bulk_transfer_storage_t*
iree_hal_remote_server_bulk_session_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_storage_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
}

static void iree_hal_remote_server_bulk_session_deinitialize_transfer(
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_transfer_storage_t* transfer =
      iree_hal_remote_server_bulk_session_transfer_storage(table_transfer);
  switch (iree_hal_remote_server_bulk_session_transfer_kind(table_transfer)) {
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

static void iree_hal_remote_server_bulk_session_deinitialize_transfer_callback(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_session_deinitialize_transfer(transfer);
}

iree_hal_remote_server_bulk_session_options_t
iree_hal_remote_server_bulk_session_options_default(void) {
  iree_hal_remote_server_bulk_session_options_t options;
  memset(&options, 0, sizeof(options));
  options.active_transfer_capacity =
      IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY;
  options.initial_transfer_id = 2;
  options.transfer_id_stride = 2;
  options.staging_slot_count = IREE_HAL_REMOTE_BULK_STAGING_SLOT_COUNT;
  options.staging_slot_length = IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH;
  options.receive_chunk_capacity = IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT;
  return options;
}

static iree_status_t iree_hal_remote_server_bulk_receive_window_send_credit(
    void* user_data, uint32_t credit_delta) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  if (!bulk_session->channel) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_status_t failure_status = iree_ok_status();
  iree_hal_remote_bulk_channel_send_result_t send_result =
      iree_hal_remote_bulk_channel_send_credit(
          bulk_session->channel, credit_delta, /*operation_user_data=*/0,
          &failure_status);
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED) {
    return iree_ok_status();
  }
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  return failure_status;
}

static void iree_hal_remote_server_bulk_session_retain(void* user_data) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  iree_atomic_ref_count_inc(&bulk_session->ref_count);
  iree_hal_remote_server_retain(bulk_session->session_slot->server);
}

static void iree_hal_remote_server_bulk_session_release(void* user_data) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  iree_hal_remote_server_t* server = bulk_session->session_slot->server;
  iree_hal_remote_server_bulk_session_free(bulk_session);
  iree_hal_remote_server_release(server);
}

static iree_status_t iree_hal_remote_server_bulk_session_start(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  return iree_hal_remote_server_bulk_on_start(bulk_session->session_slot,
                                              transfer_id, total_size, flags);
}

static iree_status_t iree_hal_remote_server_bulk_session_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  return iree_hal_remote_server_bulk_on_data(
      bulk_session->session_slot, transfer_id, chunk_offset, sequence, flags,
      chunk_data, lease);
}

static iree_status_t iree_hal_remote_server_bulk_session_complete(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  return iree_hal_remote_server_bulk_on_complete(bulk_session->session_slot,
                                                 transfer_id);
}

static iree_status_t iree_hal_remote_server_bulk_session_abort(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  return iree_hal_remote_server_bulk_on_abort(bulk_session->session_slot,
                                              transfer_id);
}

static void iree_hal_remote_server_bulk_session_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  iree_hal_remote_server_session_t* session_slot = bulk_session->session_slot;
  iree_net_session_t* session = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  session = session_slot->session;
  if (session) iree_net_session_retain(session);
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  if (session) {
    status = iree_status_join(
        status, iree_net_session_shutdown(
                    session, /*reason_code=*/0,
                    iree_make_cstring_view("bulk channel transport error")));
    iree_net_session_release(session);
  }
  iree_status_free(status);
}

static void iree_hal_remote_server_bulk_session_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  iree_hal_remote_server_bulk_on_send_complete(bulk_session->session_slot,
                                               operation_user_data, status);
}

static void iree_hal_remote_server_bulk_session_credit(void* user_data) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      (iree_hal_remote_server_bulk_session_t*)user_data;
  iree_hal_remote_server_bulk_on_credit(bulk_session->session_slot);
}

static iree_hal_remote_bulk_router_operations_t
iree_hal_remote_server_bulk_session_router_operations(void) {
  iree_hal_remote_bulk_router_operations_t operations = {
      .retain = iree_hal_remote_server_bulk_session_retain,
      .release = iree_hal_remote_server_bulk_session_release,
      .start = iree_hal_remote_server_bulk_session_start,
      .data = iree_hal_remote_server_bulk_session_data,
      .complete = iree_hal_remote_server_bulk_session_complete,
      .abort = iree_hal_remote_server_bulk_session_abort,
      .transport_error = iree_hal_remote_server_bulk_session_transport_error,
      .send_complete = iree_hal_remote_server_bulk_session_send_complete,
      .credit = iree_hal_remote_server_bulk_session_credit,
  };
  return operations;
}

iree_status_t iree_hal_remote_server_bulk_session_create(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_server_bulk_session_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_session_t** out_bulk_session) {
  IREE_ASSERT_ARGUMENT(session_slot);
  IREE_ASSERT_ARGUMENT(out_bulk_session);
  *out_bulk_session = NULL;
  iree_hal_remote_server_bulk_session_options_t default_options =
      iree_hal_remote_server_bulk_session_options_default();
  if (!options) options = &default_options;

  iree_hal_remote_server_bulk_session_t* bulk_session = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*bulk_session), (void**)&bulk_session);
  if (iree_status_is_ok(status)) {
    memset(bulk_session, 0, sizeof(*bulk_session));
    iree_atomic_ref_count_init(&bulk_session->ref_count);
    bulk_session->session_slot = session_slot;
    bulk_session->host_allocator = host_allocator;
    session_slot->bulk_session = bulk_session;
    iree_slim_mutex_initialize(&bulk_session->transfer_mutex);
    iree_hal_remote_bulk_router_initialize(
        iree_hal_remote_server_bulk_session_router_operations(), bulk_session,
        &bulk_session->router);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_relay_initialize(session_slot,
                                                             host_allocator);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_bulk_transfer_scheduler_options_t scheduler_options =
        iree_hal_remote_bulk_transfer_scheduler_options_default();
    scheduler_options.capacity = options->active_transfer_capacity;
    scheduler_options.user_storage_size =
        sizeof(iree_hal_remote_server_bulk_transfer_storage_t);
    scheduler_options.user_storage_alignment =
        iree_alignof(iree_hal_remote_server_bulk_transfer_storage_t);
    scheduler_options.initial_transfer_id = options->initial_transfer_id;
    scheduler_options.transfer_id_stride = options->transfer_id_stride;
    iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {
        .deinitialize =
            iree_hal_remote_server_bulk_session_deinitialize_transfer_callback,
        .user_data = NULL,
    };
    status = iree_hal_remote_bulk_transfer_scheduler_allocate(
        &scheduler_options, callbacks, host_allocator,
        &bulk_session->scheduler);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_staging_pool_options_t staging_options =
        iree_hal_remote_server_bulk_staging_pool_options_default();
    staging_options.slot_count = options->staging_slot_count;
    staging_options.slot_length = options->staging_slot_length;
    staging_options.user_storage_size =
        sizeof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    staging_options.user_storage_alignment =
        iree_alignof(iree_hal_remote_server_bulk_upload_staging_callback_t);
    status = iree_hal_remote_server_bulk_staging_pool_create(
        &staging_options, host_allocator, &bulk_session->staging_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_net_bulk_receive_window_options_t receive_window_options =
        iree_net_bulk_receive_window_options_default();
    receive_window_options.chunk_pool.capacity =
        options->receive_chunk_capacity;
    iree_net_bulk_receive_window_callbacks_t receive_window_callbacks = {
        .send_credit = iree_hal_remote_server_bulk_receive_window_send_credit,
        .user_data = bulk_session,
    };
    status = iree_net_bulk_receive_window_allocate(
        &receive_window_options, receive_window_callbacks, host_allocator,
        &bulk_session->receive_window);
  }

  if (iree_status_is_ok(status)) {
    *out_bulk_session = bulk_session;
  } else if (bulk_session) {
    iree_hal_remote_server_bulk_session_free(bulk_session);
    session_slot->bulk_session = NULL;
  }
  return status;
}

static void iree_hal_remote_server_bulk_session_destroy(
    iree_hal_remote_server_bulk_session_t* bulk_session) {
  iree_allocator_t host_allocator = bulk_session->host_allocator;
  iree_hal_remote_server_session_t* session_slot = bulk_session->session_slot;

  if (!iree_hal_remote_server_profile_relay_is_empty(
          &bulk_session->profile_relay)) {
    iree_hal_remote_server_profile_relay_cancel(session_slot);
    iree_hal_remote_server_profile_relay_deinitialize(session_slot);
  }

  iree_hal_remote_bulk_transfer_scheduler_free(bulk_session->scheduler);
  bulk_session->scheduler = NULL;
  iree_hal_remote_server_bulk_staging_pool_release(bulk_session->staging_pool);
  bulk_session->staging_pool = NULL;
  iree_net_bulk_receive_window_free(bulk_session->receive_window);
  bulk_session->receive_window = NULL;

  iree_net_bulk_channel_detach(bulk_session->channel);
  iree_net_bulk_channel_release(bulk_session->channel);
  bulk_session->channel = NULL;

  iree_hal_remote_bulk_router_deinitialize(&bulk_session->router);
  iree_slim_mutex_deinitialize(&bulk_session->transfer_mutex);
  iree_allocator_free(host_allocator, bulk_session);
}

void iree_hal_remote_server_bulk_session_free(
    iree_hal_remote_server_bulk_session_t* bulk_session) {
  if (bulk_session &&
      iree_atomic_ref_count_dec(&bulk_session->ref_count) == 1) {
    iree_hal_remote_server_bulk_session_destroy(bulk_session);
  }
}

bool iree_hal_remote_server_bulk_session_is_empty(
    const iree_hal_remote_server_bulk_session_t* bulk_session) {
  return !bulk_session;
}

iree_net_bulk_channel_callbacks_t
iree_hal_remote_server_bulk_session_channel_callbacks(
    iree_hal_remote_server_session_t* session_slot) {
  return iree_hal_remote_bulk_router_callbacks(
      &session_slot->bulk_session->router);
}

iree_status_t iree_hal_remote_server_bulk_session_attach_channel(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel) {
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (!session_slot->session || !bulk_session) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (bulk_session->channel) {
    status = iree_status_from_code(IREE_STATUS_ALREADY_EXISTS);
  } else {
    iree_net_bulk_channel_retain(bulk_channel);
    bulk_session->channel = bulk_channel;
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  return status;
}

void iree_hal_remote_server_bulk_session_detach_channel(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (bulk_session) {
    iree_net_bulk_channel_detach(bulk_session->channel);
  }
}

iree_net_bulk_channel_t* iree_hal_remote_server_bulk_session_take_channel(
    iree_hal_remote_server_session_t* session_slot) {
  iree_net_bulk_channel_t* bulk_channel = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (bulk_session) {
    bulk_channel = bulk_session->channel;
    bulk_session->channel = NULL;
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  return bulk_channel;
}

iree_net_bulk_channel_t* iree_hal_remote_server_bulk_session_channel(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  return bulk_session ? bulk_session->channel : NULL;
}

iree_net_bulk_channel_t*
iree_hal_remote_server_bulk_session_retain_channel_if_active(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id) {
  iree_net_bulk_channel_t* bulk_channel = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (session_slot->session_id == session_id && session_slot->session &&
      bulk_session && bulk_session->channel) {
    bulk_channel = bulk_session->channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  return bulk_channel;
}

iree_slim_mutex_t* iree_hal_remote_server_bulk_session_mutex(
    iree_hal_remote_server_session_t* session_slot) {
  return &session_slot->bulk_session->transfer_mutex;
}

iree_hal_remote_bulk_transfer_scheduler_t*
iree_hal_remote_server_bulk_session_scheduler(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  return bulk_session ? bulk_session->scheduler : NULL;
}

iree_hal_remote_server_bulk_staging_pool_t*
iree_hal_remote_server_bulk_session_staging_pool(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  return bulk_session ? bulk_session->staging_pool : NULL;
}

iree_net_bulk_receive_window_t*
iree_hal_remote_server_bulk_session_receive_window(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  return bulk_session ? bulk_session->receive_window : NULL;
}

iree_hal_remote_server_profile_relay_t*
iree_hal_remote_server_bulk_session_profile_relay(
    const iree_hal_remote_server_session_t* session_slot) {
  return &session_slot->bulk_session->profile_relay;
}

static iree_status_t
iree_hal_remote_server_bulk_session_flush_receive_window_locked(
    iree_hal_remote_server_session_t* session_slot) {
  iree_net_bulk_receive_window_t* receive_window =
      iree_hal_remote_server_bulk_session_receive_window(session_slot);
  if (!receive_window) return iree_ok_status();
  iree_status_t status =
      iree_net_bulk_receive_window_flush_credit(receive_window);
  if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_session_flush_receive_window(
    iree_hal_remote_server_session_t* session_slot) {
  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  iree_status_t status =
      iree_hal_remote_server_bulk_session_flush_receive_window_locked(
          session_slot);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));
  return status;
}

static void iree_hal_remote_server_bulk_session_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_scheduler_release(scheduler, transfer);
}

static void iree_hal_remote_server_bulk_session_fail_transfer_for_drain_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler =
      iree_hal_remote_server_bulk_session_scheduler(session_slot);
  switch (iree_hal_remote_server_bulk_session_transfer_kind(table_transfer)) {
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
      iree_hal_remote_server_bulk_session_release_transfer(scheduler,
                                                           table_transfer);
      break;
  }
}

static bool iree_hal_remote_server_bulk_session_take_drained_state_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler,
    iree_hal_remote_server_bulk_staging_pool_t** out_staging_pool,
    iree_net_bulk_receive_window_t** out_receive_window) {
  *out_scheduler = NULL;
  *out_staging_pool = NULL;
  *out_receive_window = NULL;

  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (!bulk_session ||
      !iree_any_bit_set(
          session_slot->flags,
          IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING)) {
    return false;
  }
  if (bulk_session->scheduler && iree_hal_remote_bulk_transfer_scheduler_count(
                                     bulk_session->scheduler) != 0) {
    return false;
  }
  if (iree_hal_remote_server_profile_has_pending_transfers_locked(
          session_slot)) {
    return false;
  }
  if (bulk_session->channel &&
      iree_net_bulk_channel_has_pending_sends(bulk_session->channel)) {
    return false;
  }

  *out_scheduler = bulk_session->scheduler;
  bulk_session->scheduler = NULL;
  *out_staging_pool = bulk_session->staging_pool;
  bulk_session->staging_pool = NULL;
  *out_receive_window = bulk_session->receive_window;
  bulk_session->receive_window = NULL;
  return true;
}

void iree_hal_remote_server_bulk_session_try_complete_drain(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool = NULL;
  iree_net_bulk_receive_window_t* receive_window = NULL;

  iree_slim_mutex_lock(iree_hal_remote_server_bulk_session_mutex(session_slot));
  bool drained = iree_hal_remote_server_bulk_session_take_drained_state_locked(
      session_slot, &scheduler, &staging_pool, &receive_window);
  iree_slim_mutex_unlock(
      iree_hal_remote_server_bulk_session_mutex(session_slot));

  if (drained) {
    iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
    iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
    iree_net_bulk_receive_window_free(receive_window);
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
}

void iree_hal_remote_server_bulk_session_deinitialize_transfers(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_bulk_session_t* bulk_session =
      session_slot->bulk_session;
  if (!bulk_session) return;

  if (!bulk_session->scheduler && !bulk_session->staging_pool &&
      !bulk_session->receive_window &&
      !iree_hal_remote_server_profile_has_pending_transfers_locked(
          session_slot)) {
    iree_hal_remote_server_bulk_session_try_complete_drain(session_slot);
    return;
  }

  const bool drain_pending =
      iree_any_bit_set(session_slot->flags,
                       IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING);
  bool bulk_drained = false;
  iree_hal_remote_server_profile_pending_transfer_t* pending_profile_transfers =
      NULL;
  iree_slim_mutex_lock(&bulk_session->transfer_mutex);
  pending_profile_transfers =
      iree_hal_remote_server_profile_take_pending_transfers_locked(
          session_slot);
  if (!drain_pending && bulk_session->scheduler) {
    iree_hal_remote_bulk_transfer_scheduler_clear(bulk_session->scheduler);
  } else if (drain_pending && bulk_session->scheduler) {
    uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
    iree_host_size_t transfer_count = 0;
    bool all_ids_collected =
        iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
            bulk_session->scheduler, /*select=*/NULL, NULL, transfer_ids,
            IREE_ARRAYSIZE(transfer_ids), &transfer_count);
    IREE_ASSERT(all_ids_collected);
    (void)all_ids_collected;
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      if (!bulk_session->scheduler) break;
      iree_net_bulk_transfer_t* table_transfer =
          iree_hal_remote_bulk_transfer_scheduler_lookup(
              bulk_session->scheduler, transfer_ids[i]);
      if (table_transfer) {
        iree_hal_remote_server_bulk_session_fail_transfer_for_drain_locked(
            session_slot, table_transfer);
      }
    }
  }
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler =
      bulk_session->scheduler;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool =
      bulk_session->staging_pool;
  iree_net_bulk_receive_window_t* receive_window = bulk_session->receive_window;
  if (!drain_pending) {
    bulk_session->scheduler = NULL;
    bulk_session->staging_pool = NULL;
    bulk_session->receive_window = NULL;
  } else {
    bulk_drained =
        iree_hal_remote_server_bulk_session_take_drained_state_locked(
            session_slot, &scheduler, &staging_pool, &receive_window);
    if (!bulk_drained) {
      scheduler = NULL;
      staging_pool = NULL;
      receive_window = NULL;
    }
  }
  iree_slim_mutex_unlock(&bulk_session->transfer_mutex);
  iree_hal_remote_server_profile_pending_transfer_free_list(
      session_slot->server, pending_profile_transfers);
  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
  iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
  iree_net_bulk_receive_window_free(receive_window);
  if (drain_pending && bulk_drained) {
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
}
