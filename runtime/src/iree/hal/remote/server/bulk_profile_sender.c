// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_profile_sender.h"

#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"

struct iree_hal_remote_server_profile_pending_transfer_t {
  // Next queued profile transfer in FIFO order.
  iree_hal_remote_server_profile_pending_transfer_t* next;

  // Session ID expected in |session_slot| when the transfer is activated.
  uint64_t session_id;

  // Profile callback sequence number carried by |payload|.
  uint64_t sequence;

  // Retained profile callback payload bytes.
  iree_byte_span_t payload;
};

iree_hal_remote_server_profile_transfer_t*
iree_hal_remote_server_profile_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer) {
  return (iree_hal_remote_server_profile_transfer_t*)
      iree_net_bulk_transfer_user_storage(table_transfer)
          .data;
}

void iree_hal_remote_server_profile_transfer_deinitialize(
    iree_hal_remote_server_profile_transfer_t* transfer) {
  if (transfer->payload.data) {
    iree_allocator_free(transfer->server->host_allocator,
                        transfer->payload.data);
  }
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_profile_pending_transfer_free(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer) {
  if (!pending_transfer) return;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_allocator_free(host_allocator, pending_transfer->payload.data);
  iree_allocator_free(host_allocator, pending_transfer);
}

void iree_hal_remote_server_profile_pending_transfer_free_list(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer) {
  while (pending_transfer) {
    iree_hal_remote_server_profile_pending_transfer_t* next =
        pending_transfer->next;
    iree_hal_remote_server_profile_pending_transfer_free(server,
                                                         pending_transfer);
    pending_transfer = next;
  }
}

bool iree_hal_remote_server_profile_has_pending_transfers_locked(
    const iree_hal_remote_server_session_t* session_slot) {
  return session_slot->profile_relay.pending_transfer_head != NULL ||
         session_slot->profile_relay.pending_transfer_tail != NULL;
}

iree_hal_remote_server_profile_pending_transfer_t*
iree_hal_remote_server_profile_take_pending_transfers_locked(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_profile_pending_transfer_t* pending_transfers =
      session_slot->profile_relay.pending_transfer_head;
  session_slot->profile_relay.pending_transfer_head = NULL;
  session_slot->profile_relay.pending_transfer_tail = NULL;
  return pending_transfers;
}

static void iree_hal_remote_server_profile_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_bulk_transfer_scheduler_release(scheduler, table_transfer);
}

static void iree_hal_remote_server_profile_transfer_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  if (iree_any_bit_set(transfer->flags,
                       IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_FAILED)) {
    if (!iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING)) {
      iree_hal_remote_server_profile_release_transfer(
          session_slot->bulk_transfer_scheduler, table_transfer);
    }
    return;
  }
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT |
              IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING)) {
    iree_hal_remote_server_profile_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

void iree_hal_remote_server_profile_transfer_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  transfer->flags |= IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_FAILED;
  iree_hal_remote_server_profile_transfer_try_finish_locked(session_slot,
                                                            table_transfer);
}

static iree_status_t iree_hal_remote_server_profile_transfer_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  const iree_hal_remote_server_profile_transfer_flags_t
      terminal_or_send_pending_flags =
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE |
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING |
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_FAILED;
  if (iree_any_bit_set(transfer->flags, terminal_or_send_pending_flags)) {
    return iree_ok_status();
  }
  if (!bulk_channel) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_UNAVAILABLE, "remote bulk channel is not available");
    iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                        table_transfer);
    return status;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_start(
            bulk_channel, transfer_id, total_length,
            IREE_NET_BULK_FRAME_FLAG_NONE, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT |
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  if (transfer->send_offset < total_length) {
    if (iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel) == 0) {
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
        transfer->payload.data + (iree_host_size_t)transfer->send_offset,
        chunk_length);
    iree_async_span_list_t chunk_payload =
        iree_async_span_list_make(&chunk_span, 1);
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_data(
            bulk_channel, transfer_id, transfer->send_offset,
            transfer->next_sequence, flags, chunk_payload, transfer_id,
            &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->send_offset = chunk_end;
    ++transfer->next_sequence;
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_complete(
            bulk_channel, transfer_id, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT |
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  iree_hal_remote_server_profile_transfer_try_finish_locked(session_slot,
                                                            table_transfer);
  return iree_ok_status();
}

static iree_status_t
iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t sequence, iree_byte_span_t* payload) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_hal_remote_server_profile_pending_transfer_t* pending_transfer = NULL;
  iree_status_t status =
      iree_allocator_malloc(server->host_allocator, sizeof(*pending_transfer),
                            (void**)&pending_transfer);
  if (iree_status_is_ok(status)) {
    memset(pending_transfer, 0, sizeof(*pending_transfer));
    pending_transfer->session_id = session_id;
    pending_transfer->sequence = sequence;
    pending_transfer->payload = *payload;
    *payload = iree_byte_span_empty();
    if (session_slot->profile_relay.pending_transfer_tail) {
      session_slot->profile_relay.pending_transfer_tail->next =
          pending_transfer;
    } else {
      session_slot->profile_relay.pending_transfer_head = pending_transfer;
    }
    session_slot->profile_relay.pending_transfer_tail = pending_transfer;
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_transfer_activate_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t session_id,
    uint64_t profile_sequence, iree_byte_span_t* payload,
    bool* out_table_full) {
  *out_table_full = false;
  iree_status_t status = iree_ok_status();
  if (!session_slot->bulk_transfer_scheduler) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (iree_hal_remote_bulk_transfer_scheduler_count(
                 session_slot->bulk_transfer_scheduler) >=
             iree_hal_remote_bulk_transfer_scheduler_capacity(
                 session_slot->bulk_transfer_scheduler)) {
    *out_table_full = true;
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status) && !*out_table_full) {
    status = iree_hal_remote_bulk_transfer_scheduler_allocate_local(
        session_slot->bulk_transfer_scheduler, payload->data_length,
        /*user_value=*/IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND,
        &table_transfer);
  }
  if (iree_status_is_ok(status) && table_transfer) {
    iree_hal_remote_server_profile_transfer_t* transfer =
        iree_hal_remote_server_profile_transfer_storage(table_transfer);
    memset(transfer, 0, sizeof(*transfer));
    transfer->server = session_slot->server;
    iree_hal_remote_server_retain(transfer->server);
    transfer->session_slot = session_slot;
    transfer->session_id = session_id;
    transfer->payload = *payload;
    transfer->sequence = profile_sequence;
    *payload = iree_byte_span_empty();
    status = iree_hal_remote_server_profile_transfer_try_send_locked(
        session_slot, bulk_channel, table_transfer);
  }
  return status;
}

static iree_status_t
iree_hal_remote_server_profile_pending_transfer_try_drain_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t* out_failed_sequence) {
  *out_failed_sequence = 0;
  iree_status_t status = iree_ok_status();
  while (session_slot->bulk_transfer_scheduler && iree_status_is_ok(status) &&
         session_slot->profile_relay.pending_transfer_head) {
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer =
        session_slot->profile_relay.pending_transfer_head;
    iree_byte_span_t payload = pending_transfer->payload;
    bool table_full = false;
    status = iree_hal_remote_server_profile_transfer_activate_locked(
        session_slot, bulk_channel, pending_transfer->session_id,
        pending_transfer->sequence, &payload, &table_full);
    if (table_full) break;

    session_slot->profile_relay.pending_transfer_head = pending_transfer->next;
    if (!session_slot->profile_relay.pending_transfer_head) {
      session_slot->profile_relay.pending_transfer_tail = NULL;
    }
    pending_transfer->next = NULL;
    pending_transfer->payload = payload;
    if (!iree_status_is_ok(status)) {
      *out_failed_sequence = pending_transfer->sequence;
    }
    iree_hal_remote_server_profile_pending_transfer_free(session_slot->server,
                                                         pending_transfer);
  }
  return status;
}

iree_status_t iree_hal_remote_server_profile_pending_transfer_drain(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t returned_sequence) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    uint64_t failed_sequence = 0;
    iree_status_t drain_status = iree_ok_status();
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    drain_status =
        iree_hal_remote_server_profile_pending_transfer_try_drain_locked(
            session_slot, bulk_channel, &failed_sequence);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    if (iree_status_is_ok(drain_status)) break;
    if (failed_sequence == returned_sequence) {
      status = drain_status;
    } else if (failed_sequence != 0) {
      status = iree_hal_remote_server_profile_relay_observe_transfer(
          session_slot, failed_sequence, drain_status);
    } else {
      status = drain_status;
    }
  }
  return status;
}

static bool iree_hal_remote_server_select_ready_profile_transfer(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  (void)user_data;
  if ((iree_hal_remote_server_bulk_transfer_kind_t)
          iree_net_bulk_transfer_user_value(table_transfer) !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND) {
    return false;
  }
  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE |
              IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING |
              IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_FAILED)) {
    return false;
  }
  return true;
}

iree_status_t iree_hal_remote_server_profile_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t* out_failed_sequence) {
  *out_failed_sequence = 0;
  if (!session_slot->bulk_transfer_scheduler) return iree_ok_status();
  uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
  iree_host_size_t transfer_count = 0;
  bool all_ids_collected =
      iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
          session_slot->bulk_transfer_scheduler,
          iree_hal_remote_server_select_ready_profile_transfer, NULL,
          transfer_ids, IREE_ARRAYSIZE(transfer_ids), &transfer_count);
  IREE_ASSERT(all_ids_collected);
  (void)all_ids_collected;
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (!session_slot->bulk_transfer_scheduler) return iree_ok_status();
    iree_net_bulk_transfer_t* table_transfer =
        iree_hal_remote_bulk_transfer_scheduler_lookup(
            session_slot->bulk_transfer_scheduler, transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_server_profile_transfer_t* transfer =
          iree_hal_remote_server_profile_transfer_storage(table_transfer);
      const uint64_t profile_sequence = transfer->sequence;
      iree_status_t status =
          iree_hal_remote_server_profile_transfer_try_send_locked(
              session_slot, bulk_channel, table_transfer);
      if (!iree_status_is_ok(status)) {
        *out_failed_sequence = profile_sequence;
        return status;
      }
    }
  }
  return iree_ok_status();
}

uint64_t iree_hal_remote_server_profile_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  const uint64_t profile_sequence = transfer->sequence;
  transfer->flags |= IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE;
  iree_hal_remote_server_profile_transfer_try_finish_locked(session_slot,
                                                            table_transfer);
  return profile_sequence;
}

uint64_t iree_hal_remote_server_profile_on_abort_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  const uint64_t profile_sequence =
      iree_hal_remote_server_profile_transfer_storage(table_transfer)->sequence;
  iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                      table_transfer);
  return profile_sequence;
}

iree_status_t iree_hal_remote_server_profile_on_send_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    iree_status_t status) {
  iree_status_t result_status = iree_ok_status();
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  if (!session_slot->bulk_transfer_scheduler) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    return iree_ok_status();
  }
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    return iree_ok_status();
  }

  iree_hal_remote_server_profile_transfer_t* transfer =
      iree_hal_remote_server_profile_transfer_storage(table_transfer);
  profile_sequence = transfer->sequence;
  transfer->flags &= ~IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
  if (iree_status_is_ok(status)) {
    iree_status_t send_status = iree_ok_status();
    if (iree_any_bit_set(transfer->flags,
                         IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_FAILED)) {
      iree_hal_remote_server_profile_transfer_try_finish_locked(session_slot,
                                                                table_transfer);
    } else {
      send_status = iree_hal_remote_server_profile_transfer_try_send_locked(
          session_slot, session_slot->bulk_channel, table_transfer);
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
      if (table_transfer) {
        iree_hal_remote_server_profile_transfer_try_finish_locked(
            session_slot, table_transfer);
      }
    }
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_free(status);
    if (!iree_status_is_ok(send_status)) {
      result_status = iree_hal_remote_server_profile_relay_observe_transfer(
          session_slot, profile_sequence, send_status);
    }
  } else {
    iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                        table_transfer);
    drain_profile_pending =
        iree_hal_remote_server_profile_has_pending_transfers_locked(
            session_slot);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    result_status = iree_hal_remote_server_profile_relay_observe_transfer(
        session_slot, profile_sequence, status);
  }

  if (iree_status_is_ok(result_status) && drain_profile_pending) {
    result_status = iree_hal_remote_server_profile_pending_transfer_drain(
        session_slot, session_slot->bulk_channel, /*returned_sequence=*/0);
  }
  return result_status;
}

iree_status_t iree_hal_remote_server_profile_submit_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_hal_profile_sink_t* profile_sink, iree_byte_span_t payload) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;

  iree_status_t status = iree_ok_status();
  uint64_t profile_sequence = 0;
  if (payload.data_length < sizeof(iree_hal_remote_profile_transfer_header_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "profile payload too small for header: %" PRIhsz
                              " bytes",
                              payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_profile_transfer_header_t* header =
        (const iree_hal_remote_profile_transfer_header_t*)payload.data;
    profile_sequence = header->sequence;
    if (profile_sequence == 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "profile payload must have a nonzero callback sequence");
    }
  }

  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_status_is_ok(status)) {
    bool session_active = false;
    iree_slim_mutex_lock(&server->session_mutex);
    session_active = session_slot->session_id == session_id &&
                     session_slot->session != NULL &&
                     session_slot->bulk_channel != NULL &&
                     session_slot->profile_relay.active_sink == profile_sink;
    if (session_active) {
      bulk_channel = session_slot->bulk_channel;
      iree_net_bulk_channel_retain(bulk_channel);
    }
    iree_slim_mutex_unlock(&server->session_mutex);
    if (!session_active) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
  }

  bool drain_profile_pending = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfer_scheduler) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    if (iree_status_is_ok(status)) {
      if (session_slot->profile_relay.pending_transfer_head) {
        status = iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
            session_slot, session_id, profile_sequence, &payload);
        drain_profile_pending = iree_status_is_ok(status);
      } else {
        bool table_full = false;
        status = iree_hal_remote_server_profile_transfer_activate_locked(
            session_slot, bulk_channel, session_id, profile_sequence, &payload,
            &table_full);
        if (iree_status_is_ok(status) && table_full) {
          status =
              iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
                  session_slot, session_id, profile_sequence, &payload);
          drain_profile_pending = iree_status_is_ok(status);
        }
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }
  if (iree_status_is_ok(status) && drain_profile_pending) {
    status = iree_hal_remote_server_profile_pending_transfer_drain(
        session_slot, bulk_channel, profile_sequence);
  }

  iree_net_bulk_channel_release(bulk_channel);
  iree_allocator_free(host_allocator, payload.data);
  return status;
}
