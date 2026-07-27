// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client bulk download receiver.
//
// Owns server-to-client DATA receive handling for queue_write and buffer-map
// transfers, including host copies, async file writes, transfer ACK/ABORT
// policy, and receive chunk lifetime.

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_DOWNLOAD_RECEIVER_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_DOWNLOAD_RECEIVER_H_

#include "iree/base/api.h"
#include "iree/hal/remote/client/bulk_transfer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/file.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the byte length of receiver-owned chunk storage.
iree_host_size_t
iree_hal_remote_client_bulk_download_receiver_chunk_storage_size(void);

// Returns the alignment of receiver-owned chunk storage.
iree_host_size_t
iree_hal_remote_client_bulk_download_receiver_chunk_storage_alignment(void);

// Deinitializes download transfer state.
void iree_hal_remote_client_bulk_download_receiver_deinitialize_transfer(
    iree_hal_remote_client_file_write_transfer_t* transfer);

// Begins a client-local queue_write bulk transfer.
iree_status_t iree_hal_remote_client_bulk_begin_file_write(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* target_file,
    const iree_hal_remote_client_file_view_t* target_file_view,
    uint64_t target_offset, iree_device_size_t length,
    uint64_t* out_transfer_id);

// Begins a server-to-client buffer map bulk transfer into |target_bytes|.
iree_status_t iree_hal_remote_client_bulk_begin_buffer_map_read(
    iree_hal_remote_client_device_t* device, iree_byte_span_t target_bytes,
    iree_hal_remote_client_bulk_completion_callback_t callback,
    uint64_t* out_transfer_id);

// Completes a download transfer's local waiter with |status| while locked.
void iree_hal_remote_client_bulk_download_receiver_fail_locked(
    iree_hal_remote_client_file_write_transfer_t* transfer,
    iree_status_t status);

// Handles a DATA frame if it belongs to a download transfer.
iree_status_t iree_hal_remote_client_bulk_download_receiver_on_data(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, uint64_t chunk_offset, uint32_t sequence,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    iree_async_buffer_lease_t* lease, bool* out_handled);

// Handles a COMPLETE frame if it belongs to a download transfer.
iree_status_t iree_hal_remote_client_bulk_download_receiver_on_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, bool* out_handled);

// Handles an ABORT frame if it belongs to a download transfer.
iree_status_t iree_hal_remote_client_bulk_download_receiver_on_abort(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_DOWNLOAD_RECEIVER_H_
