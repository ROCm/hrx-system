// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_SESSION_H_
#define IREE_HAL_REMOTE_SERVER_SESSION_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/epoch_semaphore_map.h"
#include "iree/hal/remote/server/resource_table.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/sequence_window.h"
#include "iree/net/session.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;
typedef struct iree_hal_remote_server_bulk_session_t
    iree_hal_remote_server_bulk_session_t;
typedef struct iree_hal_remote_control_envelope_t
    iree_hal_remote_control_envelope_t;
typedef struct iree_hal_remote_server_pending_queue_command_t
    iree_hal_remote_server_pending_queue_command_t;

typedef uint8_t iree_hal_remote_server_provisional_state_t;

enum iree_hal_remote_server_provisional_state_e {
  IREE_HAL_REMOTE_SERVER_PROVISIONAL_PENDING = 0,
  IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED = 1,
  IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED = 2,
};

typedef uint8_t iree_hal_remote_server_session_flags_t;

enum iree_hal_remote_server_session_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING = 1u << 0,
};

typedef uint8_t iree_hal_remote_server_queue_flags_t;

enum iree_hal_remote_server_queue_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL = 1u << 0,
  IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE = 1u << 1,
  IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED = 1u << 2,
  IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_READY_PENDING = 1u << 3,
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

  // Borrowed carrier backing the session connection, or NULL if unavailable.
  iree_net_carrier_t* carrier;

  // FILE_REGISTER capabilities derived from |carrier|.
  iree_hal_remote_file_registration_capabilities_t
      file_registration_capabilities;

  // Server-assigned session ID (unique, monotonically increasing).
  uint64_t session_id;

  // State bits from iree_hal_remote_server_session_flag_bits_e.
  iree_hal_remote_server_session_flags_t flags;

  // Queue state bits protected by server->session_mutex.
  iree_hal_remote_server_queue_flags_t queue_flags;

  // Queue channel for HAL command dispatch (NULL until queue endpoint opens).
  // The channel owns the header pool for its frame_sender (freed on channel
  // destroy). This ensures the pool remains valid as long as any reference
  // to the channel exists (e.g., command completion contexts).
  iree_net_queue_channel_t* queue_channel;

  // Owned bulk transfer state, or NULL when the slot has no active session.
  iree_hal_remote_server_bulk_session_t* bulk_session;

  // Resource table mapping resource_ids to retained HAL resources (buffers,
  // semaphores, etc.). Initialized when the session is accepted, deinitialized
  // when the session is removed.
  iree_hal_remote_resource_table_t resource_table;

  // (axis, epoch)->local semaphore mapping for wait frontier resolution. Each
  // COMMAND creates a local semaphore for completion tracking; subsequent
  // commands with wait frontiers look up earlier frontier entries to build
  // local wait semaphore lists. The mapping retains each semaphore until its
  // command completes or the session is removed.
  iree_hal_remote_server_epoch_semaphore_map_t epoch_semaphore_map;

  // COMMAND submission epochs processed by the server after referenced
  // resources have either been retained by the local HAL or rejected.
  iree_net_sequence_window_t observed_submission_window;

  // COMMAND signal epochs completed by the wrapped local HAL. Exact
  // observations let later wait frontier resolution skip retired local
  // semaphores, while the contiguous prefix gates ordered ADVANCE frames.
  iree_net_sequence_window_t completed_signal_window;

  // Ordered ADVANCE records waiting for transport admission. Records leave
  // this queue only after send admission transfers them to the queue channel.
  struct {
    // First pending ADVANCE record.
    iree_net_sequence_node_t* head;
    // Last pending ADVANCE record.
    iree_net_sequence_node_t* tail;
    // Number of pending ADVANCE records.
    iree_host_size_t count;
  } pending_advances;

  // Provisional→resolved resource ID mapping for resources whose client-visible
  // ID can appear on the queue channel before the control operation that
  // resolves it. Queue commands parked on unresolved provisionals are replayed
  // when resolution succeeds or failed with an ordered ADVANCE when resolution
  // fails or the provisional is released.
  struct {
    // Client-assigned provisional resource IDs.
    iree_hal_remote_resource_id_t* provisional_ids;
    // Server-assigned resolved resource IDs for RESOLVED entries.
    iree_hal_remote_resource_id_t* resolved_ids;
    // Resolution state for each provisional ID.
    iree_hal_remote_server_provisional_state_t* states;
    // Failure status codes for FAILED entries.
    iree_status_code_t* status_codes;
    // FIFO head of queue commands parked on unresolved provisional IDs.
    iree_hal_remote_server_pending_queue_command_t** pending_heads;
    // FIFO tail of queue commands parked on unresolved provisional IDs.
    iree_hal_remote_server_pending_queue_command_t** pending_tails;
    // Number of initialized provisional entries.
    iree_host_size_t count;
    // Capacity of each provisional map array.
    iree_host_size_t capacity;
  } provisional_map;

  // Buffer resource IDs that represent virtual address reservations. These
  // must be released through iree_hal_allocator_virtual_memory_release instead
  // of the generic buffer destructor.
  struct {
    // Resource IDs in |resource_table| that are virtual reservations.
    iree_hal_remote_resource_id_t* resource_ids;
    // Number of virtual reservation IDs currently tracked.
    iree_host_size_t count;
    // Capacity of |resource_ids|.
    iree_host_size_t capacity;
  } virtual_buffer_map;
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

// Dispatches an incoming queue channel COMMAND frame to the wrapped HAL.
iree_status_t iree_hal_remote_server_on_queue_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease);

// Called when a queue ADVANCE send completes.
void iree_hal_remote_server_on_queue_send_complete(void* user_data,
                                                   uint64_t operation_user_data,
                                                   iree_status_t status);

// Called when a backpressured queue ADVANCE send may make progress.
void iree_hal_remote_server_on_queue_send_ready(void* user_data);

// Removes a session from the server's tracking. Called when a session reaches
// a terminal state (CLOSED or ERROR). Safe to call multiple times for the
// same session (second call is a no-op).
void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session);

// Completes asynchronous bulk transfer teardown for a removed session.
void iree_hal_remote_server_session_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot);

// Sends a control-channel response from asynchronous session-owned work.
iree_status_t iree_hal_remote_server_session_send_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body,
    iree_host_size_t body_length);

// Sends a control-channel error response from asynchronous session-owned work.
// Consumes |status|.
iree_status_t iree_hal_remote_server_session_send_error_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_t status);

// Deinitializes sequence windows and releases any owner-managed pending nodes.
void iree_hal_remote_server_session_deinitialize_windows(
    iree_hal_remote_server_session_t* session_slot);

// Deinitializes provisional mappings and releases any parked queue commands.
void iree_hal_remote_server_session_deinitialize_provisionals(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator);

// Deinitializes the resource table and VM-reservation metadata.
void iree_hal_remote_server_session_deinitialize_resource_table(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator);

// Resolves a session-local resource ID that may still use its provisional
// client-assigned form. Unresolved IDs are returned unchanged so the owning
// resource lookup fails with its ordinary not-found path.
iree_hal_remote_resource_id_t iree_hal_remote_server_resolve_resource_id(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id);

// Resolves a direct or binding-table buffer reference for command recording.
// Direct resources are borrowed from the session resource table; indirect
// references remain unresolved until queue execution binds the slot.
iree_status_t iree_hal_remote_server_resolve_command_buffer_ref(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t buffer_id, uint32_t buffer_slot,
    uint64_t offset, uint64_t length, const char* command_name,
    iree_hal_buffer_ref_t* out_ref);

// Resolves a direct buffer reference used by a queue operation. The resource is
// borrowed from the session table and |buffer_id| must not be zero.
iree_status_t iree_hal_remote_server_resolve_queue_buffer_ref(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t buffer_id, uint64_t offset, uint64_t length,
    const char* command_name, iree_hal_buffer_ref_t* out_ref);

// Derives the local mode used to replay an uploaded reusable command buffer.
// Peer-side validation and retention hints are ignored because the server owns
// both contracts. Metadata retention hints are preserved for local profiling.
iree_status_t iree_hal_remote_server_derive_uploaded_command_buffer_mode(
    uint32_t wire_mode, iree_hal_command_buffer_mode_t* out_local_mode);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_SESSION_H_
