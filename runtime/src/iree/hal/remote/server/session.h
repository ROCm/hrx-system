// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_SESSION_H_
#define IREE_HAL_REMOTE_SERVER_SESSION_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/resource_table.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/sequence_window.h"
#include "iree/net/session.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;
typedef struct iree_net_bulk_channel_t iree_net_bulk_channel_t;
typedef struct iree_net_bulk_transfer_table_t iree_net_bulk_transfer_table_t;

typedef uint8_t iree_hal_remote_server_epoch_slot_state_t;

enum iree_hal_remote_server_epoch_slot_state_e {
  IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_EMPTY = 0,
  IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED = 1,
  IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_TOMBSTONE = 2,
};

// Per-client session tracking entry.
// Stored in the server's sessions array (indexed by slot).
typedef struct iree_hal_remote_server_session_t {
  // Back-pointer to the owning server. Used by application channel callbacks
  // to access server->devices without a global search.
  iree_hal_remote_server_t* server;

  // The net-layer session handling bootstrap and control channel.
  // NULL when the slot is free.
  iree_net_session_t* session;

  // Server-assigned session ID (unique, monotonically increasing).
  uint64_t session_id;

  // Queue channel for HAL command dispatch (NULL until queue endpoint opens).
  // The channel owns the header pool for its frame_sender (freed on channel
  // destroy). This ensures the pool remains valid as long as any reference
  // to the channel exists (e.g., command completion contexts).
  iree_net_queue_channel_t* queue_channel;

  // Bulk channel for large payload transfers (NULL until bulk endpoint opens).
  iree_net_bulk_channel_t* bulk_channel;

  // Protects active bulk transfer state.
  iree_slim_mutex_t bulk_transfer_mutex;

  // Fixed-capacity table of active bulk transfers.
  iree_net_bulk_transfer_table_t* bulk_transfers;

  // Resource table mapping resource_ids to retained HAL resources (buffers,
  // semaphores, etc.). Initialized when the session is accepted, deinitialized
  // when the session is removed.
  iree_hal_remote_resource_table_t resource_table;

  // (axis, epoch)→local semaphore mapping for wait frontier resolution. Each
  // COMMAND creates a local semaphore for completion tracking; subsequent
  // commands with wait frontiers look up earlier frontier entries to build
  // local wait semaphore lists. The mapping retains each semaphore until its
  // command completes or the session is removed.
  struct {
    // Occupancy state for each open-addressed slot.
    iree_hal_remote_server_epoch_slot_state_t* states;
    // Signal axis for each occupied slot.
    iree_async_axis_t* axes;
    // Signal epoch for each occupied slot.
    uint64_t* epochs;
    // Retained local completion semaphore for each occupied slot.
    iree_hal_semaphore_t** semaphores;
    // Number of occupied slots.
    iree_host_size_t count;
    // Number of occupied or tombstone slots.
    iree_host_size_t used_count;
    // Power-of-two capacity of all slot arrays.
    iree_host_size_t capacity;
  } epoch_semaphore_map;

  // COMMAND submission epochs processed by the server after referenced
  // resources have either been retained by the local HAL or rejected.
  iree_net_sequence_window_t observed_submission_window;

  // COMMAND signal epochs completed by the wrapped local HAL. Exact
  // observations let later wait frontier resolution skip retired local
  // semaphores, while the contiguous prefix gates ordered ADVANCE frames.
  iree_net_sequence_window_t completed_signal_window;

  // Provisional→resolved resource ID mapping. Populated during BUFFER_ALLOCA
  // processing (the server assigns a canonical ID and records the mapping).
  // Queried during subsequent commands that reference the provisional ID
  // (fill, copy, dealloca, etc.). Entries persist for the session lifetime
  // to handle out-of-order or late-arriving commands.
  struct {
    iree_hal_remote_resource_id_t* provisional_ids;
    iree_hal_remote_resource_id_t* resolved_ids;
    iree_host_size_t count;
    iree_host_size_t capacity;
  } provisional_map;
} iree_hal_remote_server_session_t;

// Called when session bootstrap completes and the session is ready for use.
void iree_hal_remote_server_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology);

// Called when the peer initiates a graceful disconnect.
void iree_hal_remote_server_on_session_goaway(void* user_data,
                                              iree_net_session_t* session,
                                              uint32_t reason_code,
                                              iree_string_view_t message);

// Called when the session encounters an unrecoverable error. Consumes |status|.
void iree_hal_remote_server_on_session_error(void* user_data,
                                             iree_net_session_t* session,
                                             iree_status_t status);

// Called when a control-channel send completes.
void iree_hal_remote_server_on_session_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Dispatches an incoming control channel frame to the appropriate handler
// (buffer alloc, query heaps, resource release, etc.).
iree_status_t iree_hal_remote_server_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Removes a session from the server's tracking. Called when a session reaches
// a terminal state (CLOSED or ERROR). Safe to call multiple times for the
// same session (second call is a no-op).
void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session);

// Deinitializes sequence windows and releases any owner-managed pending nodes.
void iree_hal_remote_server_session_deinitialize_windows(
    iree_hal_remote_server_session_t* session_slot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_SESSION_H_
