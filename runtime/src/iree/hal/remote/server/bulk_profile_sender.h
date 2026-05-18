// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server bulk profile sender.
//
// Owns server-originated profile callback transfer semantics: retained callback
// payloads, START/DATA/COMPLETE sequencing, peer credit backpressure, pending
// callback FIFO backpressure when active transfer capacity is exhausted, peer
// COMPLETE/ABORT, and callback transfer observation. Callers serialize all
// functions suffixed _locked with the session bulk-transfer mutex.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_PROFILE_SENDER_H_
#define IREE_HAL_REMOTE_SERVER_BULK_PROFILE_SENDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;
typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;
typedef struct iree_hal_remote_server_profile_pending_transfer_t
    iree_hal_remote_server_profile_pending_transfer_t;

typedef uint8_t iree_hal_remote_server_profile_transfer_flags_t;
enum iree_hal_remote_server_profile_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT = 1u << 0,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING = 1u << 1,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT = 1u << 2,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE = 1u << 3,
};

// Active server-originated profile callback transfer state.
typedef struct iree_hal_remote_server_profile_transfer_t {
  // Server retained while callbacks may reference the session array.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Retained profile callback payload bytes.
  iree_byte_span_t payload;

  // Profile callback sequence number carried by |payload|.
  uint64_t sequence;

  // Transfer-relative byte offset for the next DATA frame.
  uint64_t send_offset;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // State bits from iree_hal_remote_server_profile_transfer_flag_bits_e.
  iree_hal_remote_server_profile_transfer_flags_t flags;
} iree_hal_remote_server_profile_transfer_t;

// Returns the profile transfer storage attached to |table_transfer|.
iree_hal_remote_server_profile_transfer_t*
iree_hal_remote_server_profile_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer);

// Deinitializes in-place profile transfer state.
void iree_hal_remote_server_profile_transfer_deinitialize(
    iree_hal_remote_server_profile_transfer_t* transfer);

// Returns true if the session has queued profile callback transfers.
bool iree_hal_remote_server_profile_has_pending_transfers_locked(
    const iree_hal_remote_server_session_t* session_slot);

// Takes all queued profile callback transfers from |session_slot|.
iree_hal_remote_server_profile_pending_transfer_t*
iree_hal_remote_server_profile_take_pending_transfers_locked(
    iree_hal_remote_server_session_t* session_slot);

// Releases a queued profile callback transfer list.
void iree_hal_remote_server_profile_pending_transfer_free_list(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer);

// Fails a profile transfer and releases it.
void iree_hal_remote_server_profile_transfer_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer);

// Handles peer COMPLETE for a profile transfer and returns the callback
// sequence that should be observed after releasing the bulk-transfer mutex.
uint64_t iree_hal_remote_server_profile_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer);

// Handles peer ABORT for a profile transfer and returns the callback sequence
// that should be observed after releasing the bulk-transfer mutex.
uint64_t iree_hal_remote_server_profile_on_abort_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer);

// Handles a completed bulk send operation for a profile transfer. Consumes
// |status| and observes any transfer failure.
iree_status_t iree_hal_remote_server_profile_on_send_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    iree_status_t status);

// Attempts to send all profile transfers with pending progress after peer
// credit replenishment.
iree_status_t iree_hal_remote_server_profile_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t* out_failed_sequence);

// Attempts to drain queued profile callbacks into the active transfer table.
iree_status_t iree_hal_remote_server_profile_pending_transfer_drain(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t returned_sequence);

// Submits one server-to-client profile callback payload. If active transfer
// capacity is exhausted, the payload may be queued until a transfer retires.
// Consumes |payload| regardless of the result.
iree_status_t iree_hal_remote_server_profile_submit_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_hal_profile_sink_t* profile_sink, iree_byte_span_t payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_PROFILE_SENDER_H_
