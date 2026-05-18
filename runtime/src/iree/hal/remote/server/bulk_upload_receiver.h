// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server bulk upload receiver.
//
// Owns CLIENT_FILE_READ and BUFFER_UNMAP client-to-server transfer semantics:
// peer START/DATA/COMPLETE validation, retained DATA chunks, readiness waits,
// staged HAL queue reads into the target buffer, and terminal signal/response
// completion. Callers serialize all functions suffixed _locked with the
// session bulk-transfer mutex.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_RECEIVER_H_
#define IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_RECEIVER_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/net/channel/bulk/receive_window.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;
typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;

typedef uint8_t iree_hal_remote_server_bulk_upload_transfer_flags_t;
enum iree_hal_remote_server_bulk_upload_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_START_RECEIVED = 1u << 0,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_PEER_COMPLETE = 1u << 1,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_COMMAND_READY = 1u << 2,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_PENDING = 1u << 3,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_READY_COMPLETE = 1u << 4,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED = 1u << 5,
  IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_TRANSFER_FLAG_CONTROL_RESPONSE = 1u << 6,
};

// Timepoint context used by the readiness barrier for an upload transfer.
typedef struct iree_hal_remote_server_bulk_upload_ready_t {
  // Reference count covering the transfer state and active timepoint callback.
  iree_atomic_ref_count_t ref_count;

  // Timepoint registered on the local readiness barrier signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;

  // Server retained while the timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the timepoint fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after barrier completion.
  uint64_t transfer_id;

  // Internal local semaphore retained for the timepoint.
  iree_hal_semaphore_t* local_semaphore;

  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
} iree_hal_remote_server_bulk_upload_ready_t;

// Active CLIENT_FILE_READ/BUFFER_UNMAP upload transfer state.
typedef struct iree_hal_remote_server_bulk_upload_transfer_t {
  // Server retained while transfer state may outlive the session lock.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Bulk transfer ID used for retained receive chunk cleanup.
  uint64_t transfer_id;

  // Local HAL device borrowed from |server|.
  iree_hal_device_t* local_device;

  // Target buffer retained until all local queue reads finish.
  iree_hal_buffer_t* target_buffer;

  // Target buffer byte offset for the first uploaded byte.
  iree_device_size_t target_offset;

  // Signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Control response envelope used by BUFFER_UNMAP bulk uploads.
  iree_hal_remote_control_envelope_t response_envelope;

  // Internal semaphore signaled after the original command waits resolve.
  iree_hal_semaphore_t* ready_semaphore;

  // Reusable timepoint context for the readiness barrier completion.
  iree_hal_remote_server_bulk_upload_ready_t* ready_context;

  // Tracks fixed-grid DATA chunks received from the client.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // Number of active local barrier/read callbacks referencing this state.
  uint32_t pending_operation_count;

  // State bits from iree_hal_remote_server_bulk_upload_transfer_flag_bits_e.
  iree_hal_remote_server_bulk_upload_transfer_flags_t flags;
} iree_hal_remote_server_bulk_upload_transfer_t;

// Timepoint context stored in a staging slot during a queued file read.
typedef struct iree_hal_remote_server_bulk_upload_staging_callback_t {
  // Server retained while the staging slot timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the callback transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the callback fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after local queue read completion.
  uint64_t transfer_id;
} iree_hal_remote_server_bulk_upload_staging_callback_t;

// Initializes in-place upload transfer state.
iree_status_t iree_hal_remote_server_bulk_upload_transfer_initialize(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t transfer_id, uint64_t total_length,
    iree_host_size_t data_chunk_length, iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_upload_transfer_t* transfer);

// Deinitializes in-place upload transfer state.
void iree_hal_remote_server_bulk_upload_transfer_deinitialize(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer);

// Marks the peer START frame as received.
iree_status_t iree_hal_remote_server_bulk_upload_transfer_mark_start(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    iree_net_bulk_frame_flags_t flags);

// Validates and records one peer DATA frame.
iree_status_t iree_hal_remote_server_bulk_upload_transfer_record_data(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    uint64_t transfer_id, uint64_t total_length, uint64_t chunk_offset,
    iree_host_size_t chunk_length, iree_net_bulk_frame_flags_t flags);

// Marks the peer COMPLETE frame as received.
iree_status_t iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
    iree_hal_remote_server_bulk_upload_transfer_t* transfer,
    uint64_t transfer_id);

// Retains |context| for a pending callback.
void iree_hal_remote_server_bulk_upload_ready_retain(
    iree_hal_remote_server_bulk_upload_ready_t* context);

// Releases |context| from a pending callback.
void iree_hal_remote_server_bulk_upload_ready_release(
    iree_hal_remote_server_bulk_upload_ready_t* context);

// Returns the upload transfer storage attached to |table_transfer|.
iree_hal_remote_server_bulk_upload_transfer_t*
iree_hal_remote_server_bulk_upload_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer);

// Looks up an upload transfer by ID.
iree_net_bulk_transfer_t* iree_hal_remote_server_bulk_upload_lookup_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id);

// Looks up or inserts an upload transfer with a peer-assigned ID.
iree_status_t iree_hal_remote_server_bulk_upload_get_or_insert_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t total_length,
    iree_host_size_t data_chunk_length,
    iree_net_bulk_transfer_t** out_table_transfer);

// Fails an upload transfer and releases it if no callbacks are active. Consumes
// |status|.
void iree_hal_remote_server_bulk_upload_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status);

// Attempts terminal upload completion.
void iree_hal_remote_server_bulk_upload_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer);

// Submits the command wait barrier for an attached upload transfer.
void iree_hal_remote_server_bulk_upload_submit_ready_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_kind,
    iree_hal_semaphore_list_t wait_list,
    iree_async_semaphore_timepoint_fn_t ready_callback);

// Attempts to stage and queue all currently ready DATA chunks.
void iree_hal_remote_server_bulk_upload_try_process_chunks_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback);

// Handles a peer START frame while holding the session bulk-transfer mutex.
iree_status_t iree_hal_remote_server_bulk_upload_on_start_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags, iree_host_size_t data_chunk_length);

// Handles a peer DATA frame while holding the session bulk-transfer mutex.
iree_status_t iree_hal_remote_server_bulk_upload_on_data_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback);

// Handles a peer COMPLETE frame for an upload transfer.
iree_status_t iree_hal_remote_server_bulk_upload_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_id);

// Handles readiness barrier timepoint completion. Consumes |status|.
void iree_hal_remote_server_bulk_upload_on_ready_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active, iree_status_t status,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback);

// Handles staging queue-read timepoint completion. Consumes |status|.
void iree_hal_remote_server_bulk_upload_on_staging_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active,
    iree_hal_remote_server_bulk_staging_slot_t* staging_slot,
    uint64_t signal_value, iree_status_t status,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t staging_callback);

// Attaches a CLIENT_FILE_READ or BUFFER_UNMAP command to an upload transfer.
iree_status_t iree_hal_remote_server_bulk_upload_attach_command_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t signal_list, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_hal_semaphore_t** ready_semaphore,
    iree_hal_remote_server_bulk_upload_ready_t** ready_context,
    const iree_hal_remote_control_envelope_t* response_envelope);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_UPLOAD_RECEIVER_H_
