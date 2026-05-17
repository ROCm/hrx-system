// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk transfer receive tracker.
//
// Tracks fixed-grid DATA chunks for a single transfer. HAL remote currently
// sends queue file I/O DATA in fixed-size chunks, with only the final chunk
// allowed to be short. This tracker makes receiver accounting independent of
// delivery order and rejects duplicate, overlapping, missing, and malformed
// chunks before COMPLETE can finish a transfer.

#ifndef IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_TRACKER_H_
#define IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_TRACKER_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Inline bitmap words stored directly in the tracker.
#define IREE_HAL_REMOTE_BULK_TRANSFER_TRACKER_INLINE_WORD_COUNT 4

typedef struct iree_hal_remote_bulk_transfer_tracker_t {
  // Host allocator used for overflow bitmap storage.
  iree_allocator_t host_allocator;

  // Total transfer byte length.
  uint64_t total_length;

  // Nominal DATA chunk byte length.
  iree_host_size_t chunk_length;

  // Number of DATA chunks expected for the transfer.
  iree_host_size_t expected_chunk_count;

  // Number of unique DATA chunks received so far.
  iree_host_size_t received_chunk_count;

  // Bitmap of received DATA chunk indexes.
  iree_bitmap_t received_chunks;

  // Heap-allocated bitmap words when the inline storage is not large enough.
  uint64_t* allocated_words;

  // Inline bitmap words for small transfers.
  uint64_t
      inline_words[IREE_HAL_REMOTE_BULK_TRANSFER_TRACKER_INLINE_WORD_COUNT];
} iree_hal_remote_bulk_transfer_tracker_t;

// Initializes |out_tracker| for a transfer split into fixed-size DATA chunks.
iree_status_t iree_hal_remote_bulk_transfer_tracker_initialize(
    uint64_t total_length, iree_host_size_t chunk_length,
    iree_allocator_t host_allocator,
    iree_hal_remote_bulk_transfer_tracker_t* out_tracker);

// Deinitializes |tracker| and releases any overflow bitmap storage.
void iree_hal_remote_bulk_transfer_tracker_deinitialize(
    iree_hal_remote_bulk_transfer_tracker_t* tracker);

// Records a received DATA chunk.
//
// Chunks must begin on the configured fixed chunk grid. Non-final chunks must
// have exactly |chunk_length| bytes, and the final chunk must have the
// remaining transfer byte count. Duplicate and overlapping chunks fail loudly.
iree_status_t iree_hal_remote_bulk_transfer_tracker_record_chunk(
    iree_hal_remote_bulk_transfer_tracker_t* tracker, uint64_t chunk_offset,
    iree_host_size_t data_length);

// Returns true once every expected DATA chunk has been recorded.
bool iree_hal_remote_bulk_transfer_tracker_is_complete(
    const iree_hal_remote_bulk_transfer_tracker_t* tracker);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_UTIL_BULK_TRANSFER_TRACKER_H_
