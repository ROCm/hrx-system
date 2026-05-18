// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY 64

typedef struct benchmark_transfer_storage_t {
  // Benchmark readiness flags.
  uint32_t flags;
} benchmark_transfer_storage_t;

static void BenchmarkTransferDeinitialize(void* user_data,
                                          iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  benchmark_transfer_storage_t* storage =
      (benchmark_transfer_storage_t*)iree_net_bulk_transfer_user_storage(
          transfer)
          .data;
  storage->flags = 0;
}

static iree_status_t AllocateBenchmarkScheduler(
    iree_allocator_t host_allocator,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  options.capacity = IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY;
  options.user_storage_size = sizeof(benchmark_transfer_storage_t);
  options.user_storage_alignment = iree_alignof(benchmark_transfer_storage_t);
  options.initial_transfer_id = 2;
  options.transfer_id_stride = 2;
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {};
  callbacks.deinitialize = BenchmarkTransferDeinitialize;
  callbacks.user_data = NULL;
  return iree_hal_remote_bulk_transfer_scheduler_allocate(
      &options, callbacks, host_allocator, out_scheduler);
}

static bool SelectReadyTransfers(void* user_data,
                                 iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  benchmark_transfer_storage_t* storage =
      (benchmark_transfer_storage_t*)iree_net_bulk_transfer_user_storage(
          transfer)
          .data;
  return storage->flags != 0;
}

static iree_status_t BM_AllocateLookupRelease(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  iree_status_t status = AllocateBenchmarkScheduler(host_allocator, &scheduler);

  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(
             benchmark_state,
             IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY)) {
    iree_net_bulk_transfer_t*
        transfers[IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY];
    memset(transfers, 0, sizeof(transfers));
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(transfers) && iree_status_is_ok(status); ++i) {
      status = iree_hal_remote_bulk_transfer_scheduler_allocate_local(
          scheduler, /*total_size=*/4096, /*user_value=*/0, &transfers[i]);
      if (iree_status_is_ok(status)) {
        iree_net_bulk_transfer_t* lookup =
            iree_hal_remote_bulk_transfer_scheduler_lookup(
                scheduler, iree_net_bulk_transfer_id(transfers[i]));
        iree_benchmark_use_ptr((char const volatile*)lookup);
      }
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(transfers); ++i) {
      iree_hal_remote_bulk_transfer_scheduler_release(scheduler, transfers[i]);
    }
  }

  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
  return status;
}

static iree_status_t BM_CollectReadyTransferIds(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  iree_status_t status = AllocateBenchmarkScheduler(host_allocator, &scheduler);

  for (iree_host_size_t i = 0;
       i < IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY &&
       iree_status_is_ok(status);
       ++i) {
    iree_net_bulk_transfer_t* transfer = NULL;
    status = iree_hal_remote_bulk_transfer_scheduler_allocate_local(
        scheduler, /*total_size=*/4096, /*user_value=*/0, &transfer);
    if (iree_status_is_ok(status)) {
      benchmark_transfer_storage_t* storage =
          (benchmark_transfer_storage_t*)iree_net_bulk_transfer_user_storage(
              transfer)
              .data;
      storage->flags = (i & 1) == 0 ? 1 : 0;
    }
  }

  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    uint64_t transfer_ids
        [IREE_HAL_REMOTE_BULK_TRANSFER_SCHEDULER_BENCHMARK_CAPACITY];
    iree_host_size_t transfer_count = 0;
    bool all_ids_collected =
        iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
            scheduler, SelectReadyTransfers, NULL, transfer_ids,
            IREE_ARRAYSIZE(transfer_ids), &transfer_count);
    iree_benchmark_use_ptr((char const volatile*)transfer_ids);
    iree_benchmark_use_ptr((char const volatile*)&transfer_count);
    iree_benchmark_use_ptr((char const volatile*)&all_ids_collected);
  }

  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
  return status;
}

IREE_BENCHMARK_REGISTER(BM_AllocateLookupRelease);
IREE_BENCHMARK_REGISTER(BM_CollectReadyTransferIds);
