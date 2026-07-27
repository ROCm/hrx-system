// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client bulk upload sender.
//
// Owns client-to-server START/DATA/COMPLETE sequencing for queue_read and
// buffer-unmap transfers. The sender runs under the client bulk session mutex
// while mutating transfer state and uses bulk channel credit as backpressure.

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_UPLOAD_SENDER_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_UPLOAD_SENDER_H_

#include "iree/base/api.h"
#include "iree/hal/remote/client/bulk_transfer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/file.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the byte length of sender-owned chunk storage.
iree_host_size_t iree_hal_remote_client_bulk_upload_sender_chunk_storage_size(
    void);

// Returns the alignment of sender-owned chunk storage.
iree_host_size_t
iree_hal_remote_client_bulk_upload_sender_chunk_storage_alignment(void);

// Deinitializes upload transfer state.
void iree_hal_remote_client_bulk_upload_sender_deinitialize_transfer(
    iree_hal_remote_client_file_read_transfer_t* transfer);

// Marks |transfer| terminal while the transfer mutex is held. Releases any
// locally pending chunk with no async or transport owner. Returns true when no
// admitted callback still requires the transfer descriptor.
bool iree_hal_remote_client_bulk_upload_sender_mark_terminal_locked(
    iree_hal_remote_client_file_read_transfer_t* transfer);

// Begins a client-local queue_read bulk transfer.
iree_status_t iree_hal_remote_client_bulk_begin_file_read(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_device_size_t length,
    uint64_t* out_transfer_id);

// Begins a client-to-server buffer unmap bulk transfer from |source_bytes|.
iree_status_t iree_hal_remote_client_bulk_begin_buffer_unmap_write(
    iree_hal_remote_client_device_t* device,
    iree_const_byte_span_t source_bytes, uint64_t* out_transfer_id);

// Starts or resumes uploading a client-local transfer.
iree_status_t iree_hal_remote_client_bulk_upload_sender_upload(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id);

// Tries to send all upload transfers currently able to make progress.
iree_status_t iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel);

// Handles a bulk channel send completion for an upload START/DATA/COMPLETE.
//
// Consumes |status| and returns a transport status to report to the device.
iree_status_t iree_hal_remote_client_bulk_upload_sender_send_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t operation_user_data, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_UPLOAD_SENDER_H_
