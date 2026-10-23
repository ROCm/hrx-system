// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch_upload.h"

#include "loom/serve/runtime/program/transfer_plan.h"

iree_status_t loom_serve_text_epoch_upload(
    loom_serve_execution_t* execution, iree_const_byte_span_t plan,
    const iree_const_byte_span_t packets[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT],
    iree_hal_buffer_t* const targets[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT]) {
  uint64_t source_lengths[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT];
  uint64_t target_lengths[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT];
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT; ++i) {
    source_lengths[i] = packets[i].data_length;
    target_lengths[i] = iree_hal_buffer_byte_length(targets[i]);
  }
  loom_serve_transfer_plan_t placements;
  IREE_RETURN_IF_ERROR(loom_serve_transfer_plan_initialize(
      plan, IREE_ARRAYSIZE(source_lengths), source_lengths,
      IREE_ARRAYSIZE(target_lengths), target_lengths, &placements));
  // Batching bounds stack use, not the number of source ranges. HAL captures
  // the descriptors during submission; only packet bytes escape this call.
  iree_hal_transfer_operation_t operations[64];
  iree_host_size_t operation_count = 0;
  uint64_t completion = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < placements.count && iree_status_is_ok(status); ++i) {
    const loom_serve_transfer_range_t range =
        loom_serve_transfer_plan_range(placements, i);
    if (!range.length) {
      continue;
    }
    operations[operation_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source =
                       packets[range.source_binding].data + range.source_offset,
                   .target_buffer = targets[range.target_binding],
                   .target_offset = range.target_offset,
                   .length = range.length},
    };
    if (operation_count == IREE_ARRAYSIZE(operations)) {
      status = loom_serve_execution_transfer(execution, operation_count,
                                             operations, &completion);
      operation_count = 0;
    }
  }
  if (iree_status_is_ok(status) && operation_count) {
    status = loom_serve_execution_transfer(execution, operation_count,
                                           operations, &completion);
  }
  return status;
}
