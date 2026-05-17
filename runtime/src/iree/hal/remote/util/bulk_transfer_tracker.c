// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_transfer_tracker.h"

#include <string.h>

iree_status_t iree_hal_remote_bulk_transfer_tracker_initialize(
    uint64_t total_length, iree_host_size_t chunk_length,
    iree_allocator_t host_allocator,
    iree_hal_remote_bulk_transfer_tracker_t* out_tracker) {
  IREE_ASSERT_ARGUMENT(out_tracker);
  memset(out_tracker, 0, sizeof(*out_tracker));

  if (chunk_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk transfer chunk length must be non-zero");
  }
  uint64_t expected_chunk_count64 =
      total_length == 0 ? 0 : 1 + (total_length - 1) / (uint64_t)chunk_length;
  if (expected_chunk_count64 > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bulk transfer chunk count %" PRIu64
                            " exceeds host size max %" PRIhsz,
                            expected_chunk_count64, IREE_HOST_SIZE_MAX);
  }

  iree_host_size_t expected_chunk_count =
      (iree_host_size_t)expected_chunk_count64;
  iree_host_size_t word_count =
      iree_bitmap_calculate_words(expected_chunk_count);
  uint64_t* words = out_tracker->inline_words;
  if (word_count > IREE_ARRAYSIZE(out_tracker->inline_words)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, word_count, sizeof(uint64_t),
        (void**)&out_tracker->allocated_words));
    words = out_tracker->allocated_words;
  }
  memset(words, 0, word_count * sizeof(uint64_t));

  out_tracker->host_allocator = host_allocator;
  out_tracker->total_length = total_length;
  out_tracker->chunk_length = chunk_length;
  out_tracker->expected_chunk_count = expected_chunk_count;
  out_tracker->received_chunk_count = 0;
  out_tracker->received_chunks = (iree_bitmap_t){
      .bit_count = expected_chunk_count,
      .words = words,
  };
  return iree_ok_status();
}

void iree_hal_remote_bulk_transfer_tracker_deinitialize(
    iree_hal_remote_bulk_transfer_tracker_t* tracker) {
  if (!tracker) return;
  iree_allocator_t host_allocator = tracker->host_allocator;
  iree_allocator_free(host_allocator, tracker->allocated_words);
  memset(tracker, 0, sizeof(*tracker));
}

iree_status_t iree_hal_remote_bulk_transfer_tracker_record_chunk(
    iree_hal_remote_bulk_transfer_tracker_t* tracker, uint64_t chunk_offset,
    iree_host_size_t data_length) {
  IREE_ASSERT_ARGUMENT(tracker);

  const uint64_t chunk_length = (uint64_t)tracker->chunk_length;
  if (data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk DATA chunks must be non-empty");
  }
  if (chunk_offset % chunk_length != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk DATA offset %" PRIu64
                            " is not aligned to chunk length %" PRIu64,
                            chunk_offset, chunk_length);
  }

  const uint64_t data_length64 = (uint64_t)data_length;
  const bool chunk_range_overflow = chunk_offset > UINT64_MAX - data_length64;
  const uint64_t chunk_end =
      chunk_range_overflow ? UINT64_MAX : chunk_offset + data_length64;
  if (chunk_range_overflow || chunk_end > tracker->total_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk DATA range [%" PRIu64 ", %" PRIu64
                            ") exceeds transfer length %" PRIu64,
                            chunk_offset, chunk_end, tracker->total_length);
  }

  const iree_host_size_t chunk_index =
      (iree_host_size_t)(chunk_offset / chunk_length);
  const uint64_t expected_length =
      iree_min(chunk_length, tracker->total_length - chunk_offset);
  if (data_length64 != expected_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk DATA chunk %" PRIhsz " length %" PRIu64
                            " does not match expected length %" PRIu64,
                            chunk_index, data_length64, expected_length);
  }
  if (iree_bitmap_test(tracker->received_chunks, chunk_index)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "duplicate bulk DATA chunk index %" PRIhsz,
                            chunk_index);
  }

  iree_bitmap_set(tracker->received_chunks, chunk_index);
  ++tracker->received_chunk_count;
  return iree_ok_status();
}

bool iree_hal_remote_bulk_transfer_tracker_is_complete(
    const iree_hal_remote_bulk_transfer_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  return tracker->received_chunk_count == tracker->expected_chunk_count;
}
