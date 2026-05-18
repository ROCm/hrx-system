// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Server bulk download sender.
//
// Owns CLIENT_FILE_WRITE server-to-client transfer semantics: local HAL
// queue_write staging from source buffers, START/DATA/COMPLETE sequencing,
// peer credit backpressure, send-completion accounting, peer COMPLETE/ABORT,
// and terminal local signal behavior. Callers serialize all functions suffixed
// _locked with the session bulk-transfer mutex.

#ifndef IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_SENDER_H_
#define IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_SENDER_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/bulk_staging_pool.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;
typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;

typedef uint16_t iree_hal_remote_server_bulk_download_transfer_flags_t;
enum iree_hal_remote_server_bulk_download_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_START_SENT = 1u << 0,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SEND_PENDING = 1u << 1,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_WRITE_PENDING =
      1u << 2,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SUBMIT_PENDING =
      1u << 3,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_DATA_READY = 1u
                                                                          << 4,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_STAGING_SEND_PENDING =
      1u << 5,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_COMPLETE_SENT = 1u << 6,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_PEER_COMPLETE = 1u << 7,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_SIGNAL_CONSUMED = 1u << 8,
  IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED =
      1u << 9,
};

// Timepoint context used by the local staging queue_write completion.
typedef struct iree_hal_remote_server_bulk_download_ready_t {
  // Reference count covering the transfer state and active timepoint callback.
  iree_atomic_ref_count_t ref_count;

  // Timepoint registered on the local queue_write signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;

  // Server retained while the timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the timepoint fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after local queue_write completion.
  uint64_t transfer_id;

  // Local queue_write signal semaphore retained for the timepoint.
  iree_hal_semaphore_t* local_semaphore;

  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
} iree_hal_remote_server_bulk_download_ready_t;

// Active CLIENT_FILE_WRITE download transfer state.
typedef struct iree_hal_remote_server_bulk_download_transfer_t {
  // Server retained while callbacks may reference the session array.
  iree_hal_remote_server_t* server;

  // Host allocator used for transfer-owned cloned lists.
  iree_allocator_t host_allocator;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Local HAL device borrowed from |server| for staging queue writes.
  iree_hal_device_t* local_device;

  // Source buffer retained until all chunks have been staged.
  iree_hal_buffer_t* source_buffer;

  // Source buffer byte offset for the first streamed byte.
  iree_device_size_t source_offset;

  // Queue write flags provided by the remote command.
  iree_hal_write_flags_t write_flags;

  // Acquired staging slot returned at transfer teardown.
  iree_hal_remote_server_bulk_staging_slot_t* staging_slot;

  // Server-side memory file borrowed from |staging_slot|.
  iree_hal_file_t* staging_file;

  // Host allocation contents borrowed from |staging_slot|.
  iree_byte_span_t staging_contents;

  // Local semaphore borrowed from |staging_slot|.
  iree_hal_semaphore_t* staging_semaphore;

  // Reusable timepoint context for staging queue_write completion.
  iree_hal_remote_server_bulk_download_ready_t* ready_context;

  // Initial local wait semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t initial_wait_semaphore_list;

  // Final local signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Transfer-relative byte offset currently resident in |staging_contents|.
  uint64_t staging_offset;

  // Number of bytes currently resident in |staging_contents|.
  iree_device_size_t staging_length;

  // Transfer-relative byte offset for the next staging queue_write.
  uint64_t next_staging_offset;

  // Last payload value signaled on |staging_semaphore|.
  uint64_t last_staging_signal_value;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // Number of active callbacks or unlocked submissions referencing this state.
  uint32_t pending_operation_count;

  // State bits from iree_hal_remote_server_bulk_download_transfer_flag_bits_e.
  iree_hal_remote_server_bulk_download_transfer_flags_t flags;
} iree_hal_remote_server_bulk_download_transfer_t;

// Retains |context| for a pending callback.
void iree_hal_remote_server_bulk_download_ready_retain(
    iree_hal_remote_server_bulk_download_ready_t* context);

// Releases |context| from a pending callback.
void iree_hal_remote_server_bulk_download_ready_release(
    iree_hal_remote_server_bulk_download_ready_t* context);

// Returns the download transfer storage attached to |table_transfer|.
iree_hal_remote_server_bulk_download_transfer_t*
iree_hal_remote_server_bulk_download_transfer_storage(
    iree_net_bulk_transfer_t* table_transfer);

// Initializes in-place download transfer state.
iree_status_t iree_hal_remote_server_bulk_download_transfer_initialize(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t transfer_id, iree_hal_device_t* local_device,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_write_flags_t write_flags,
    iree_async_semaphore_timepoint_fn_t ready_callback,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_download_transfer_t* transfer);

// Deinitializes in-place download transfer state.
void iree_hal_remote_server_bulk_download_transfer_deinitialize(
    iree_hal_remote_server_bulk_download_transfer_t* transfer);

// Fails a download transfer and releases it if no callbacks are active.
// Consumes |status|.
void iree_hal_remote_server_bulk_download_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status);

// Attempts terminal download completion.
void iree_hal_remote_server_bulk_download_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer);

// Attempts to send the next download frame or submit the next staging write.
void iree_hal_remote_server_bulk_download_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer);

// Attempts to send all ready download transfers.
void iree_hal_remote_server_bulk_download_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind);

// Handles peer COMPLETE for a download transfer.
iree_status_t iree_hal_remote_server_bulk_download_on_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, uint64_t transfer_id);

// Handles bulk send completion for a download transfer. Consumes |status|.
void iree_hal_remote_server_bulk_download_on_send_complete_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status);

// Handles staging queue_write timepoint completion. Consumes |status|.
void iree_hal_remote_server_bulk_download_on_ready_timepoint_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind,
    uint64_t transfer_id, bool session_active, iree_status_t status);

// Attaches and starts a CLIENT_FILE_WRITE command.
iree_status_t iree_hal_remote_server_bulk_download_submit_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t transfer_kind,
    uint64_t session_id, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    uint64_t transfer_id, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_device_size_t length,
    iree_hal_write_flags_t flags,
    iree_async_semaphore_timepoint_fn_t ready_callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_DOWNLOAD_SENDER_H_
