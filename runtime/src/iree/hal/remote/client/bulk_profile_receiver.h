// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client bulk profile receiver.
//
// Owns remote profiling session state, profile payload reassembly, sequence
// ordering, sink dispatch, and profile transfer ACK/ABORT policy.

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_PROFILE_RECEIVER_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_PROFILE_RECEIVER_H_

#include "iree/base/api.h"
#include "iree/hal/remote/client/bulk_transfer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Deinitializes an owned profile transfer.
void iree_hal_remote_client_bulk_profile_receiver_free_transfer(
    iree_hal_remote_client_profile_transfer_t* transfer);

// Releases all active and pending profile transfers while locked.
void iree_hal_remote_client_bulk_profile_receiver_release_all_locked(
    iree_hal_remote_client_device_t* device);

// Begins a remote profiling session and retains |sink| until the session ends.
iree_status_t iree_hal_remote_client_bulk_begin_profile_session(
    iree_hal_remote_client_device_t* device, iree_hal_profile_sink_t* sink);

// Returns true when a remote profiling session is active locally.
bool iree_hal_remote_client_bulk_has_profile_session(
    iree_hal_remote_client_device_t* device);

// Ends any active remote profiling session and drops pending profile transfers.
void iree_hal_remote_client_bulk_end_profile_session(
    iree_hal_remote_client_device_t* device);

// Begins a profile receive transfer while locked.
iree_status_t iree_hal_remote_client_bulk_profile_receiver_begin_locked(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    uint64_t total_size);

// Handles a DATA frame if it belongs to a profile transfer.
iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_data(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, uint64_t chunk_offset,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    bool* out_handled);

// Handles a COMPLETE frame if it belongs to a profile transfer.
iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, bool* out_handled);

// Handles an ABORT frame if it belongs to a profile transfer.
iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_abort(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_PROFILE_RECEIVER_H_
