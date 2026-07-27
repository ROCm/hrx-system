// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client bulk session state.
//
// Owns client-local bulk channel publication, active transfer tables, retained
// chunk pools, and remote profiling receive ordering. Higher-level client bulk
// operations own transfer-specific semantics.

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_SESSION_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_SESSION_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/channel/util/sequence_window.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_profile_sink_t iree_hal_profile_sink_t;
typedef struct iree_net_bulk_channel_t iree_net_bulk_channel_t;
typedef struct iree_net_bulk_chunk_pool_t iree_net_bulk_chunk_pool_t;
typedef struct iree_net_bulk_transfer_table_t iree_net_bulk_transfer_table_t;

typedef uint8_t iree_hal_remote_client_bulk_session_flags_t;
enum iree_hal_remote_client_bulk_session_flag_bits_e {
  IREE_HAL_REMOTE_CLIENT_BULK_SESSION_FLAG_INITIALIZED = 1u << 0,
};

typedef struct iree_hal_remote_client_bulk_session_transfer_options_t {
  // Byte length of owner-managed storage attached to each transfer.
  iree_host_size_t transfer_user_storage_size;

  // Alignment of owner-managed transfer storage.
  iree_host_size_t transfer_user_storage_alignment;

  // Byte length of owner-managed storage attached to each chunk.
  iree_host_size_t chunk_user_storage_size;

  // Alignment of owner-managed chunk storage.
  iree_host_size_t chunk_user_storage_alignment;
} iree_hal_remote_client_bulk_session_transfer_options_t;

// Called under the session mutex to release resources referenced by active
// transfers before the session clears and frees its transfer table.
typedef void (*iree_hal_remote_client_bulk_session_deinitialize_transfers_fn_t)(
    void* user_data, iree_net_bulk_transfer_table_t* transfers);

// Returns conservative defaults for client bulk transfer state.
static inline iree_hal_remote_client_bulk_session_transfer_options_t
iree_hal_remote_client_bulk_session_transfer_options_default(void) {
  iree_hal_remote_client_bulk_session_transfer_options_t options = {0};
  options.transfer_user_storage_alignment = iree_max_align_t;
  options.chunk_user_storage_alignment = iree_max_align_t;
  return options;
}

typedef struct iree_hal_remote_client_bulk_session_t {
  // Published bulk channel, or 0 before the bulk endpoint opens. Once
  // published, the object remains retained through terminal device
  // deactivation and is released during final device destruction.
  iree_atomic_intptr_t channel;

  // Protects client-local transfer, chunk, and profile state.
  iree_slim_mutex_t transfer_mutex;

  // Fixed-capacity table of client-local bulk transfers.
  iree_net_bulk_transfer_table_t* transfers;

  // Fixed-capacity pool of outgoing async file read staging chunks.
  iree_net_bulk_chunk_pool_t* send_chunks;

  // Fixed-capacity pool of retained incoming bulk DATA chunks.
  iree_net_bulk_chunk_pool_t* receive_chunks;

  // Client-owned sink retained while a remote profiling session is active.
  iree_hal_profile_sink_t* profile_sink;

  // Reconstructs in-order profile callback dispatch from completed transfers.
  iree_net_sequence_window_t profile_sequence_window;

  // First terminal device failure. Protected by transfer_mutex and retained
  // until deinitialization so racing transfer admission observes the same
  // failure as the rest of the device.
  iree_status_t terminal_status;

  // Lifecycle bits from iree_hal_remote_client_bulk_session_flag_bits_e.
  iree_hal_remote_client_bulk_session_flags_t flags;
} iree_hal_remote_client_bulk_session_t;

// Initializes an embedded client bulk session.
iree_status_t iree_hal_remote_client_bulk_session_initialize(
    iree_allocator_t host_allocator,
    iree_hal_remote_client_bulk_session_t* out_session);

// Deinitializes an embedded client bulk session.
//
// Transfer state must have been deinitialized with
// iree_hal_remote_client_bulk_session_deinitialize_transfers before calling.
void iree_hal_remote_client_bulk_session_deinitialize(
    iree_hal_remote_client_bulk_session_t* session);

// Initializes bounded transfer and chunk state owned by |session|.
iree_status_t iree_hal_remote_client_bulk_session_initialize_transfers(
    iree_hal_remote_client_bulk_session_t* session,
    const iree_hal_remote_client_bulk_session_transfer_options_t* options,
    iree_allocator_t host_allocator);

// Deinitializes bounded transfer and chunk state owned by |session|.
//
// |deinitialize_transfers| is required when the transfer table has been
// initialized. It must release all owner-managed resources referenced by active
// transfers before returning and may mutate |transfers|.
void iree_hal_remote_client_bulk_session_deinitialize_transfers(
    iree_hal_remote_client_bulk_session_t* session,
    iree_hal_remote_client_bulk_session_deinitialize_transfers_fn_t
        deinitialize_transfers,
    void* deinitialize_user_data);

// Loads the published bulk channel with acquire ordering.
iree_net_bulk_channel_t* iree_hal_remote_client_bulk_session_load_channel(
    iree_hal_remote_client_bulk_session_t* session);

// Publishes |new_channel| with acq_rel ordering and returns the old channel.
iree_net_bulk_channel_t* iree_hal_remote_client_bulk_session_exchange_channel(
    iree_hal_remote_client_bulk_session_t* session,
    iree_net_bulk_channel_t* new_channel);

// Returns OK when transfer admission is active or clones the terminal status.
// The caller must hold |transfer_mutex|.
iree_status_t iree_hal_remote_client_bulk_session_check_active_locked(
    iree_hal_remote_client_bulk_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_SESSION_H_
