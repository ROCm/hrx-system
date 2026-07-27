// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_session.h"

#include <string.h>

#include "iree/hal/api.h"
#include "iree/net/channel/bulk/chunk_pool.h"
#include "iree/net/channel/bulk/transfer_table.h"

iree_status_t iree_hal_remote_client_bulk_session_initialize(
    iree_allocator_t host_allocator,
    iree_hal_remote_client_bulk_session_t* out_session) {
  IREE_ASSERT_ARGUMENT(out_session);
  memset(out_session, 0, sizeof(*out_session));
  iree_atomic_store(&out_session->channel, 0, iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&out_session->transfer_mutex);
  out_session->flags |= IREE_HAL_REMOTE_CLIENT_BULK_SESSION_FLAG_INITIALIZED;

  iree_status_t status = iree_net_sequence_window_initialize(
      /*initial_observed_sequence=*/0, /*initial_capacity=*/64, host_allocator,
      &out_session->profile_sequence_window);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_session_deinitialize(out_session);
  }
  return status;
}

void iree_hal_remote_client_bulk_session_deinitialize(
    iree_hal_remote_client_bulk_session_t* session) {
  if (!session || !iree_all_bits_set(
                      session->flags,
                      IREE_HAL_REMOTE_CLIENT_BULK_SESSION_FLAG_INITIALIZED)) {
    return;
  }
  IREE_ASSERT(!session->transfers && !session->send_chunks &&
              !session->receive_chunks);
  iree_status_free(session->terminal_status);
  iree_net_sequence_window_deinitialize(&session->profile_sequence_window);
  iree_hal_profile_sink_release(session->profile_sink);
  session->profile_sink = NULL;
  iree_slim_mutex_deinitialize(&session->transfer_mutex);
  memset(session, 0, sizeof(*session));
}

static void iree_hal_remote_client_bulk_session_clear_transfers(
    void* user_data, iree_net_bulk_transfer_table_t* transfers) {
  (void)user_data;
  iree_net_bulk_transfer_table_clear(transfers);
}

iree_status_t iree_hal_remote_client_bulk_session_initialize_transfers(
    iree_hal_remote_client_bulk_session_t* session,
    const iree_hal_remote_client_bulk_session_transfer_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT(iree_all_bits_set(
      session->flags, IREE_HAL_REMOTE_CLIENT_BULK_SESSION_FLAG_INITIALIZED));

  iree_status_t status = iree_ok_status();
  if (session->transfers || session->send_chunks || session->receive_chunks) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "client bulk transfer state is already active");
  }

  iree_net_bulk_transfer_table_options_t transfer_options =
      iree_net_bulk_transfer_table_options_default();
  transfer_options.user_storage_size = options->transfer_user_storage_size;
  transfer_options.user_storage_alignment =
      options->transfer_user_storage_alignment;
  transfer_options.initial_transfer_id = 1;
  transfer_options.transfer_id_stride = 2;
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_transfer_table_allocate(
        &transfer_options, host_allocator, &session->transfers);
  }

  iree_net_bulk_chunk_pool_options_t chunk_options =
      iree_net_bulk_chunk_pool_options_default();
  chunk_options.user_storage_size = options->chunk_user_storage_size;
  chunk_options.user_storage_alignment = options->chunk_user_storage_alignment;
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_chunk_pool_allocate(&chunk_options, host_allocator,
                                               &session->send_chunks);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_chunk_pool_allocate(&chunk_options, host_allocator,
                                               &session->receive_chunks);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_session_deinitialize_transfers(
        session, iree_hal_remote_client_bulk_session_clear_transfers,
        /*deinitialize_user_data=*/NULL);
  }
  return status;
}

void iree_hal_remote_client_bulk_session_deinitialize_transfers(
    iree_hal_remote_client_bulk_session_t* session,
    iree_hal_remote_client_bulk_session_deinitialize_transfers_fn_t
        deinitialize_transfers,
    void* deinitialize_user_data) {
  if (!session || (!session->transfers && !session->send_chunks &&
                   !session->receive_chunks)) {
    return;
  }

  iree_slim_mutex_lock(&session->transfer_mutex);
  if (session->transfers) {
    IREE_ASSERT_ARGUMENT(deinitialize_transfers);
    deinitialize_transfers(deinitialize_user_data, session->transfers);
  }
  iree_net_bulk_transfer_table_t* transfers = session->transfers;
  session->transfers = NULL;
  iree_net_bulk_chunk_pool_t* send_chunks = session->send_chunks;
  session->send_chunks = NULL;
  iree_net_bulk_chunk_pool_t* receive_chunks = session->receive_chunks;
  session->receive_chunks = NULL;
  iree_slim_mutex_unlock(&session->transfer_mutex);

  iree_net_bulk_transfer_table_free(transfers);
  iree_net_bulk_chunk_pool_free(send_chunks);
  iree_net_bulk_chunk_pool_free(receive_chunks);
}

iree_net_bulk_channel_t* iree_hal_remote_client_bulk_session_load_channel(
    iree_hal_remote_client_bulk_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return (iree_net_bulk_channel_t*)iree_atomic_load(&session->channel,
                                                    iree_memory_order_acquire);
}

iree_net_bulk_channel_t* iree_hal_remote_client_bulk_session_exchange_channel(
    iree_hal_remote_client_bulk_session_t* session,
    iree_net_bulk_channel_t* new_channel) {
  IREE_ASSERT_ARGUMENT(session);
  return (iree_net_bulk_channel_t*)iree_atomic_exchange(
      &session->channel, (intptr_t)new_channel, iree_memory_order_acq_rel);
}

iree_status_t iree_hal_remote_client_bulk_session_check_active_locked(
    iree_hal_remote_client_bulk_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_status_is_ok(session->terminal_status)
             ? iree_ok_status()
             : iree_status_clone(session->terminal_status);
}
