// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_H_

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_H_
