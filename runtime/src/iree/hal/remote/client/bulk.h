// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_H_

#include "iree/hal/remote/client/bulk_transfer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/file.h"
#include "iree/net/channel/queue/queue_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opens the bulk endpoint after the queue channel has been created.
//
// On success this consumes |queue_channel| and publishes it only after the bulk
// channel is also ready. On failure the caller still owns |queue_channel|.
iree_status_t iree_hal_remote_client_device_open_bulk_endpoint(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* queue_channel);

// Initializes bounded client-local bulk transfer state on |device|.
iree_status_t iree_hal_remote_client_device_initialize_bulk_transfers(
    iree_hal_remote_client_device_t* device);

// Releases client-local bulk transfer state on |device|.
void iree_hal_remote_client_device_deinitialize_bulk_transfers(
    iree_hal_remote_client_device_t* device);

// Terminalizes active bulk transfers with |status| without releasing storage
// still retained by admitted sends or async file operations.
void iree_hal_remote_client_bulk_fail_transfers(
    iree_hal_remote_client_device_t* device, iree_status_t status);

// Begins a client-local queue_write bulk transfer.
iree_status_t iree_hal_remote_client_bulk_begin_file_write(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* target_file,
    const iree_hal_remote_client_file_view_t* target_file_view,
    uint64_t target_offset, iree_device_size_t length,
    uint64_t* out_transfer_id);

// Begins a client-local queue_read bulk transfer.
iree_status_t iree_hal_remote_client_bulk_begin_file_read(
    iree_hal_remote_client_device_t* device, iree_hal_file_t* source_file,
    const iree_hal_remote_client_file_view_t* source_file_view,
    uint64_t source_offset, iree_device_size_t length,
    uint64_t* out_transfer_id);

// Starts or resumes uploading a client-local queue_read transfer.
iree_status_t iree_hal_remote_client_bulk_upload_file_read(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id);

// Cancels a client-local file transfer before the server observes it.
void iree_hal_remote_client_bulk_cancel_transfer(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id);

// Begins a server-to-client buffer map bulk transfer into |target_bytes|.
iree_status_t iree_hal_remote_client_bulk_begin_buffer_map_read(
    iree_hal_remote_client_device_t* device, iree_byte_span_t target_bytes,
    iree_hal_remote_client_bulk_completion_callback_t callback,
    uint64_t* out_transfer_id);

// Begins a client-to-server buffer unmap bulk transfer from |source_bytes|.
iree_status_t iree_hal_remote_client_bulk_begin_buffer_unmap_write(
    iree_hal_remote_client_device_t* device,
    iree_const_byte_span_t source_bytes, uint64_t* out_transfer_id);

// Starts or resumes uploading a client-to-server buffer unmap transfer.
iree_status_t iree_hal_remote_client_bulk_upload_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id);

// Releases client-side bookkeeping after the server has consumed the upload.
void iree_hal_remote_client_bulk_end_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id);

// Begins a remote profiling session and prepares to receive profile bulk
// transfers. |sink| is retained until end_profile_session.
iree_status_t iree_hal_remote_client_bulk_begin_profile_session(
    iree_hal_remote_client_device_t* device, iree_hal_profile_sink_t* sink);

// Returns true when a remote profiling session is active locally.
bool iree_hal_remote_client_bulk_has_profile_session(
    iree_hal_remote_client_device_t* device);

// Ends any active remote profiling session and drops pending profile transfers.
void iree_hal_remote_client_bulk_end_profile_session(
    iree_hal_remote_client_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_H_
