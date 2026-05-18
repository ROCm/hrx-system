// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_channel_writer.h"

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_result_from_status(
    iree_status_t status, iree_status_t* out_failure_status) {
  IREE_ASSERT_ARGUMENT(out_failure_status);
  *out_failure_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    return IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED;
  }
  if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    return IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED;
  }
  *out_failure_status = status;
  return IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED;
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_start(iree_net_bulk_channel_t* channel,
                                        uint64_t transfer_id,
                                        uint64_t total_size,
                                        iree_net_bulk_frame_flags_t flags,
                                        uint64_t operation_user_data,
                                        iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_send_start(channel, transfer_id, total_size, flags,
                                       operation_user_data),
      out_failure_status);
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_data(iree_net_bulk_channel_t* channel,
                                       uint64_t transfer_id,
                                       uint64_t chunk_offset, uint32_t sequence,
                                       iree_net_bulk_frame_flags_t flags,
                                       iree_async_span_list_t chunk_payload,
                                       uint64_t operation_user_data,
                                       iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_send_data(channel, transfer_id, chunk_offset,
                                      sequence, flags, chunk_payload,
                                      operation_user_data),
      out_failure_status);
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_complete(iree_net_bulk_channel_t* channel,
                                           uint64_t transfer_id,
                                           uint64_t operation_user_data,
                                           iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_send_complete(channel, transfer_id,
                                          operation_user_data),
      out_failure_status);
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_credit(iree_net_bulk_channel_t* channel,
                                         uint32_t credit_delta,
                                         uint64_t operation_user_data,
                                         iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_send_credit(channel, credit_delta,
                                        operation_user_data),
      out_failure_status);
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_refresh_credit(iree_net_bulk_channel_t* channel,
                                            uint64_t operation_user_data,
                                            iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_refresh_credit(channel, operation_user_data),
      out_failure_status);
}

iree_hal_remote_bulk_channel_send_result_t
iree_hal_remote_bulk_channel_send_abort(iree_net_bulk_channel_t* channel,
                                        uint64_t transfer_id,
                                        iree_async_span_list_t abort_payload,
                                        uint64_t operation_user_data,
                                        iree_status_t* out_failure_status) {
  return iree_hal_remote_bulk_channel_send_result_from_status(
      iree_net_bulk_channel_send_abort(channel, transfer_id, abort_payload,
                                       operation_user_data),
      out_failure_status);
}
