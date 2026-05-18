// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_upload_receiver.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH (32 * 1024)
#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT 64
#define IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH \
  (IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH *    \
   IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT)

static iree_status_t BM_RecordCompleteTransfer(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  (void)benchmark_def;
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  while (
      iree_status_is_ok(status) &&
      iree_benchmark_keep_running(
          benchmark_state, IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT)) {
    iree_hal_remote_server_bulk_upload_transfer_t transfer;
    bool transfer_initialized = false;
    status = iree_hal_remote_server_bulk_upload_transfer_initialize(
        /*server=*/NULL, /*session_slot=*/NULL, /*session_id=*/0,
        /*transfer_id=*/1, IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH,
        IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH, host_allocator,
        &transfer);
    transfer_initialized = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_transfer_mark_start(
          &transfer, IREE_NET_BULK_FRAME_FLAG_NONE);
    }
    for (iree_host_size_t i = 0;
         i < IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT &&
         iree_status_is_ok(status);
         ++i) {
      const bool final_chunk =
          i + 1 == IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_COUNT;
      status = iree_hal_remote_server_bulk_upload_transfer_record_data(
          &transfer, /*transfer_id=*/1,
          IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_TOTAL_LENGTH,
          (uint64_t)i * IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH,
          IREE_HAL_REMOTE_BULK_UPLOAD_BENCHMARK_CHUNK_LENGTH,
          final_chunk ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                      : IREE_NET_BULK_FRAME_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_bulk_upload_transfer_mark_peer_complete(
          &transfer, /*transfer_id=*/1);
    }
    if (transfer_initialized) {
      iree_hal_remote_server_bulk_upload_transfer_deinitialize(&transfer);
    }
  }
  return status;
}

IREE_BENCHMARK_REGISTER(BM_RecordCompleteTransfer);
