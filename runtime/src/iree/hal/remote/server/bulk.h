// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_BULK_H_
#define IREE_HAL_REMOTE_SERVER_BULK_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_session_t
    iree_hal_remote_server_session_t;
typedef struct iree_hal_remote_control_envelope_t
    iree_hal_remote_control_envelope_t;

// Header buffers used by the bulk channel frame sender. Bulk DATA payloads are
// not copied into this pool; only the 40-byte frame headers are retained until
// send completion.
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT 128
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE 128
#define IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT 64

// DATA payloads share the frame with a bulk header; keep chunks comfortably
// below the default queue frame size instead of filling the whole frame.
#define IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH (32 * 1024)
#define IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY \
  IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY
#define IREE_HAL_REMOTE_BULK_STAGING_SLOT_COUNT \
  IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY

typedef enum iree_hal_remote_server_bulk_transfer_kind_e {
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_EMPTY = 0u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ = 1u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE = 2u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND = 3u,
} iree_hal_remote_server_bulk_transfer_kind_e;
typedef uint8_t iree_hal_remote_server_bulk_transfer_kind_t;

// Initializes bounded bulk transfer state for a session slot.
iree_status_t iree_hal_remote_server_session_initialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator);

// Deinitializes bounded bulk transfer state for a session slot.
void iree_hal_remote_server_session_deinitialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot);

// Submits a CLIENT_FILE_READ command once bulk DATA has uploaded the file.
iree_status_t iree_hal_remote_server_bulk_submit_client_file_read(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Submits a BUFFER_UNMAP bulk upload and sends |response_envelope| once the
// uploaded bytes are visible in |target_buffer|.
iree_status_t iree_hal_remote_server_bulk_submit_buffer_unmap(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device,
    const iree_hal_remote_control_envelope_t* response_envelope,
    uint64_t transfer_id, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

// Submits a CLIENT_FILE_WRITE command using the wrapped local HAL queue_write.
iree_status_t iree_hal_remote_server_bulk_submit_client_file_write(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// Flushes any unadvertised local receive capacity to the peer.
iree_status_t iree_hal_remote_server_bulk_flush_receive_window(
    iree_hal_remote_server_session_t* session_slot);

// Handles a peer START frame.
iree_status_t iree_hal_remote_server_bulk_on_start(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_size, iree_net_bulk_frame_flags_t flags);

// Handles a peer DATA frame.
iree_status_t iree_hal_remote_server_bulk_on_data(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease);

// Handles a peer COMPLETE frame.
iree_status_t iree_hal_remote_server_bulk_on_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id);

// Handles a peer ABORT frame.
iree_status_t iree_hal_remote_server_bulk_on_abort(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id);

// Handles a completed bulk send operation. Consumes |status|.
void iree_hal_remote_server_bulk_on_send_complete(
    iree_hal_remote_server_session_t* session_slot,
    uint64_t operation_user_data, iree_status_t status);

// Handles peer receive credit replenishment.
void iree_hal_remote_server_bulk_on_credit(
    iree_hal_remote_server_session_t* session_slot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_BULK_H_
