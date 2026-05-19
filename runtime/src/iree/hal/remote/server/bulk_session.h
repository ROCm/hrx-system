// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server per-session bulk transfer state.
//
// Owns the mutable representation shared by server bulk components: active
// transfer table, staging slots, receive DATA window, profile callback relay,
// bulk channel router, and the mutex serializing those structures. The session
// slot owns identity and lifetime; this component owns the bulk transfer
// machinery attached to that identity.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_SESSION_H_
#define IREE_HAL_REMOTE_SERVER_BULK_SESSION_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_bulk_session_t
    iree_hal_remote_server_bulk_session_t;
typedef struct iree_hal_remote_server_bulk_staging_pool_t
    iree_hal_remote_server_bulk_staging_pool_t;
typedef struct iree_hal_remote_bulk_transfer_scheduler_t
    iree_hal_remote_bulk_transfer_scheduler_t;
typedef struct iree_hal_remote_server_profile_relay_t
    iree_hal_remote_server_profile_relay_t;
typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;
typedef struct iree_net_bulk_receive_window_t iree_net_bulk_receive_window_t;

// Creation parameters for per-session bulk transfer state.
typedef struct iree_hal_remote_server_bulk_session_options_t {
  // Maximum number of concurrent bulk transfers in the active transfer table.
  iree_host_size_t active_transfer_capacity;

  // First locally allocated transfer ID for server-originated transfers.
  uint64_t initial_transfer_id;

  // Increment between locally allocated transfer IDs.
  uint64_t transfer_id_stride;

  // Number of reusable host staging slots for queue file operations.
  iree_host_size_t staging_slot_count;

  // Byte length of each reusable staging slot.
  iree_host_size_t staging_slot_length;

  // Number of client-to-server DATA chunks retained by the receive window.
  iree_host_size_t receive_chunk_capacity;
} iree_hal_remote_server_bulk_session_options_t;

// Returns default production bulk session options.
iree_hal_remote_server_bulk_session_options_t
iree_hal_remote_server_bulk_session_options_default(void);

// Creates and initializes bulk transfer state for |session_slot|.
iree_status_t iree_hal_remote_server_bulk_session_create(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_server_bulk_session_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_session_t** out_bulk_session);

// Frees bulk transfer state and releases the attached bulk channel, if any.
// Normal session removal must use the drain path first so zero-copy sends have
// retired before the component representation disappears.
void iree_hal_remote_server_bulk_session_free(
    iree_hal_remote_server_bulk_session_t* bulk_session);

// Returns true when |bulk_session| has no reusable slot state.
bool iree_hal_remote_server_bulk_session_is_empty(
    const iree_hal_remote_server_bulk_session_t* bulk_session);

// Returns callbacks suitable for iree_net_bulk_channel_create.
iree_net_bulk_channel_callbacks_t
iree_hal_remote_server_bulk_session_channel_callbacks(
    iree_hal_remote_server_session_t* session_slot);

// Attaches |bulk_channel| to |session_slot| if the session is still active.
// Retains |bulk_channel| on success.
iree_status_t iree_hal_remote_server_bulk_session_attach_channel(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel);

// Detaches the attached bulk channel from its endpoint, if present.
void iree_hal_remote_server_bulk_session_detach_channel(
    iree_hal_remote_server_session_t* session_slot);

// Takes the attached bulk channel reference from |session_slot|. The caller
// owns the returned reference.
iree_net_bulk_channel_t* iree_hal_remote_server_bulk_session_take_channel(
    iree_hal_remote_server_session_t* session_slot);

// Returns the attached bulk channel, or NULL when no channel is active.
iree_net_bulk_channel_t* iree_hal_remote_server_bulk_session_channel(
    iree_hal_remote_server_session_t* session_slot);

// Retains and returns the attached bulk channel when the session is active.
iree_net_bulk_channel_t*
iree_hal_remote_server_bulk_session_retain_channel_if_active(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id);

// Returns the mutex protecting bulk transfer component state.
iree_slim_mutex_t* iree_hal_remote_server_bulk_session_mutex(
    iree_hal_remote_server_session_t* session_slot);

// Returns the active transfer scheduler, or NULL during teardown.
iree_hal_remote_bulk_transfer_scheduler_t*
iree_hal_remote_server_bulk_session_scheduler(
    iree_hal_remote_server_session_t* session_slot);

// Returns the reusable staging pool, or NULL during teardown.
iree_hal_remote_server_bulk_staging_pool_t*
iree_hal_remote_server_bulk_session_staging_pool(
    iree_hal_remote_server_session_t* session_slot);

// Returns the receive DATA window, or NULL during teardown.
iree_net_bulk_receive_window_t*
iree_hal_remote_server_bulk_session_receive_window(
    iree_hal_remote_server_session_t* session_slot);

// Returns the profile relay state owned by the bulk session.
iree_hal_remote_server_profile_relay_t*
iree_hal_remote_server_bulk_session_profile_relay(
    const iree_hal_remote_server_session_t* session_slot);

// Flushes any unadvertised local receive capacity to the peer.
iree_status_t iree_hal_remote_server_bulk_session_flush_receive_window(
    iree_hal_remote_server_session_t* session_slot);

// Returns true if a drain completion was deferred to the active CREDIT flush.
bool iree_hal_remote_server_bulk_session_defer_drain_if_flushing(
    iree_hal_remote_server_session_t* session_slot);

// Deinitializes or drains active bulk transfers for |session_slot|.
void iree_hal_remote_server_bulk_session_deinitialize_transfers(
    iree_hal_remote_server_session_t* session_slot);

// Attempts to finish asynchronous drain after a callback retired work.
void iree_hal_remote_server_bulk_session_try_complete_drain(
    iree_hal_remote_server_session_t* session_slot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_SESSION_H_
