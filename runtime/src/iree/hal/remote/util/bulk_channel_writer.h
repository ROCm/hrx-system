// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk channel write helpers.
//
// These helpers convert the net bulk channel's status-returning send API into
// the contract transfer engines actually need: accepted, temporarily blocked,
// or failed. RESOURCE_EXHAUSTED from the channel means a bounded send resource
// was unavailable and the frame was not queued; it is not a protocol failure.

#ifndef IREE_HAL_REMOTE_UTIL_BULK_CHANNEL_WRITER_H_
#define IREE_HAL_REMOTE_UTIL_BULK_CHANNEL_WRITER_H_

#include "iree/base/api.h"
#include "iree/net/channel/bulk/bulk_channel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Result from attempting to queue a bulk channel frame.
typedef uint8_t iree_hal_remote_bulk_channel_send_result_t;
enum iree_hal_remote_bulk_channel_send_result_e {
  // The frame was accepted by the channel and will complete asynchronously.
  IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED = 0,
  // The channel could not accept the frame now. The caller must not advance
  // transfer state and should retry when send capacity may have changed.
  IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED = 1,
  // The send failed permanently. |out_failure_status| owns the diagnostic.
  IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED = 2,
};

// Classifies a raw channel send |status|.
//
// OK becomes ACCEPTED. RESOURCE_EXHAUSTED is consumed and becomes BLOCKED.
// Any other failure becomes FAILED and is transferred to |out_failure_status|.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_result_from_status(
    iree_status_t status, iree_status_t* out_failure_status);

// Attempts to send a START frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_start(iree_net_bulk_channel_t* channel,
                                        uint64_t transfer_id,
                                        uint64_t total_size,
                                        iree_net_bulk_frame_flags_t flags,
                                        uint64_t operation_user_data,
                                        iree_status_t* out_failure_status);

// Attempts to send a DATA frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_data(iree_net_bulk_channel_t* channel,
                                       uint64_t transfer_id,
                                       uint64_t chunk_offset, uint32_t sequence,
                                       iree_net_bulk_frame_flags_t flags,
                                       iree_async_span_list_t chunk_payload,
                                       uint64_t operation_user_data,
                                       iree_status_t* out_failure_status);

// Attempts to send a COMPLETE frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_complete(iree_net_bulk_channel_t* channel,
                                           uint64_t transfer_id,
                                           uint64_t operation_user_data,
                                           iree_status_t* out_failure_status);

// Attempts to send a CREDIT frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_credit(iree_net_bulk_channel_t* channel,
                                         uint32_t credit_delta,
                                         uint64_t operation_user_data,
                                         iree_status_t* out_failure_status);

// Attempts to refresh the last cumulative CREDIT frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_refresh_credit(iree_net_bulk_channel_t* channel,
                                            uint64_t operation_user_data,
                                            iree_status_t* out_failure_status);

// Attempts to send an ABORT frame.
iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_abort(iree_net_bulk_channel_t* channel,
                                        uint64_t transfer_id,
                                        iree_async_span_list_t abort_payload,
                                        uint64_t operation_user_data,
                                        iree_status_t* out_failure_status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_UTIL_BULK_CHANNEL_WRITER_H_
