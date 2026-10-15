// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/publisher.h"

namespace {

using Entry = iree_hal_amd_gpu_publication_entry_t;

// Measures only metadata handoff. Native encoding, doorbells, notifications,
// device execution and retirement belong to end-to-end queue measurements.
void BM_IdleCaller(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  Entry entry;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_enqueue(&publisher, &entry));
    benchmark::DoNotOptimize(iree_hal_amd_gpu_publisher_next(&publisher));
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr));
  }
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IdleCaller);

// Captures a burst behind an active caller, detaches each entry, and returns
// the owner to idle. Work execution and reclamation are excluded.
void BM_ReadyBurst(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  std::array<Entry, 129> entries;
  const size_t count = state.range(0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[0]));
    benchmark::DoNotOptimize(iree_hal_amd_gpu_publisher_next(&publisher));
    for (size_t i = 1; i <= count; ++i) {
      benchmark::DoNotOptimize(
          iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[i]));
    }
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr));
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_acquire_service(&publisher));
    for (size_t i = 0; i < count; ++i) {
      benchmark::DoNotOptimize(iree_hal_amd_gpu_publisher_next(&publisher));
    }
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr));
  }
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations() * (count + 1));
}
BENCHMARK(BM_ReadyBurst)->Arg(1)->Arg(8)->Arg(32)->Arg(128);

// Repeatedly defers one detached entry ahead of the unconsumed FIFO. Capacity
// pressure adds no traversal, regardless of the number of queued entries.
void BM_DeferredEntry(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  std::array<Entry, 129> entries;
  const size_t count = state.range(0);
  iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[0]);
  iree_hal_amd_gpu_publisher_next(&publisher);
  for (size_t i = 1; i <= count; ++i) {
    iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[i]);
  }
  iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr);
  iree_hal_amd_gpu_publisher_acquire_service(&publisher);
  Entry* work = iree_hal_amd_gpu_publisher_next(&publisher);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, work));
    iree_hal_amd_gpu_publisher_acquire_service(&publisher);
    work = iree_hal_amd_gpu_publisher_next(&publisher);
    benchmark::DoNotOptimize(work);
  }
  while (iree_hal_amd_gpu_publisher_next(&publisher)) {
  }
  iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr);
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DeferredEntry)->Arg(1)->Arg(8)->Arg(32)->Arg(128);

}  // namespace
