// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/timestamp.h"

#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/session.h"

static_assert(IREE_HAL_TIMESTAMP_FLAG_NONE == 0,
              "remote timestamp flags must match the HAL");

iree_status_t iree_hal_remote_server_queue_timestamp(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list,
    iree_const_byte_span_t command_data) {
  if (command_data.data_length !=
      sizeof(iree_hal_remote_queue_timestamp_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "QUEUE_TIMESTAMP command length %" PRIhsz
                            " does not match canonical length %" PRIhsz,
                            command_data.data_length,
                            sizeof(iree_hal_remote_queue_timestamp_op_t));
  }

  iree_hal_remote_queue_timestamp_op_t op;
  memcpy(&op, command_data.data, sizeof(op));
  if (op.header.type != IREE_HAL_REMOTE_QUEUE_OP_QUEUE_TIMESTAMP) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "QUEUE_TIMESTAMP command has unexpected type 0x%04x", op.header.type);
  }
  if (op.header.flags != 0 || op.header.reserved != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "QUEUE_TIMESTAMP command header reserved fields are nonzero");
  }
  if (op.flags != IREE_HAL_TIMESTAMP_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported timestamp flags: 0x%016" PRIx64,
                            op.flags);
  }

  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_queue_buffer_ref(
      session_slot, op.target.buffer_id, op.target.offset, sizeof(uint64_t),
      "QUEUE_TIMESTAMP", &target_ref));
  return iree_hal_device_queue_timestamp(
      local_device, (iree_hal_queue_affinity_t)op.target.queue_affinity,
      wait_list, signal_list, target_ref.buffer, target_ref.offset,
      (iree_hal_timestamp_flags_t)op.flags);
}
