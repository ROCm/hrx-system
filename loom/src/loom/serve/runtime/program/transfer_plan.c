// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/transfer_plan.h"

loom_serve_transfer_range_t loom_serve_transfer_plan_range(
    loom_serve_transfer_plan_t plan, iree_host_size_t index) {
  const uint8_t* record =
      plan.records + index * LOOM_SERVE_TRANSFER_RANGE_BYTES;
  return (loom_serve_transfer_range_t){
      .source_binding = iree_unaligned_load_le_u64(record),
      .source_offset = iree_unaligned_load_le_u64(record + 8),
      .target_binding = iree_unaligned_load_le_u64(record + 16),
      .target_offset = iree_unaligned_load_le_u64(record + 24),
      .length = iree_unaligned_load_le_u64(record + 32),
  };
}

iree_status_t loom_serve_transfer_plan_initialize(
    iree_const_byte_span_t bytes, iree_host_size_t source_count,
    const uint64_t* source_lengths, iree_host_size_t target_count,
    const uint64_t* target_lengths, loom_serve_transfer_plan_t* out_plan) {
  *out_plan = (loom_serve_transfer_plan_t){0};
  if (bytes.data_length % LOOM_SERVE_TRANSFER_RANGE_BYTES) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transfer plan requires complete byte ranges");
  }
  const loom_serve_transfer_plan_t plan = {
      .records = bytes.data,
      .count = bytes.data_length / LOOM_SERVE_TRANSFER_RANGE_BYTES,
  };
  uint64_t previous_target = 0;
  uint64_t previous_end = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < plan.count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_transfer_range_t range =
        loom_serve_transfer_plan_range(plan, i);
    if (range.source_binding >= source_count ||
        range.target_binding >= target_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid binding in transfer range %zu", i);
    } else if (range.source_offset > source_lengths[range.source_binding] ||
               range.length >
                   source_lengths[range.source_binding] - range.source_offset ||
               range.target_offset > target_lengths[range.target_binding] ||
               range.length >
                   target_lengths[range.target_binding] - range.target_offset) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "transfer range %zu exceeds bound storage", i);
    } else if (range.length) {
      if (range.target_binding < previous_target ||
          (range.target_binding == previous_target &&
           range.target_offset < previous_end)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "transfer range %zu must follow disjoint target ranges", i);
      } else {
        previous_target = range.target_binding;
        previous_end = range.target_offset + range.length;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  }
  return status;
}
