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
using List = iree_hal_amd_gpu_publication_list_t;

// Measures only metadata handoff. Native encoding, doorbells, notifications,
// device execution and retirement belong to end-to-end queue measurements.
void BM_IdleCaller(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  Entry entry;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_enqueue(&publisher, &entry));
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, {}));
  }
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IdleCaller);

// Captures a ready burst behind an active caller, detaches it in O(1), and
// returns the owner to idle. Work execution and reclamation are excluded.
void BM_ReadyBurst(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  std::array<Entry, 129> entries;
  const size_t count = state.range(0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[0]));
    for (size_t i = 1; i <= count; ++i) {
      benchmark::DoNotOptimize(
          iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[i]));
    }
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, {}));
    benchmark::DoNotOptimize(iree_hal_amd_gpu_publisher_take(&publisher));
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, {}));
  }
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations() * (count + 1));
}
BENCHMARK(BM_ReadyBurst)->Arg(1)->Arg(8)->Arg(32)->Arg(128);

// Repeatedly hands an unpublished suffix back to the service. Length does not
// change the two pointer updates; capacity pressure adds no list traversal.
void BM_DeferredSuffix(benchmark::State& state) {
  iree_hal_amd_gpu_publisher_t publisher;
  iree_hal_amd_gpu_publisher_initialize(&publisher);
  std::array<Entry, 129> entries;
  const size_t count = state.range(0);
  for (size_t i = 0; i <= count; ++i) {
    iree_hal_amd_gpu_publisher_enqueue(&publisher, &entries[i]);
  }
  iree_hal_amd_gpu_publisher_handoff(&publisher, {});
  List work = iree_hal_amd_gpu_publisher_take(&publisher);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_publisher_handoff(&publisher, work));
    work = iree_hal_amd_gpu_publisher_take(&publisher);
    benchmark::DoNotOptimize(work);
  }
  iree_hal_amd_gpu_publisher_handoff(&publisher, {});
  iree_hal_amd_gpu_publisher_deinitialize(&publisher);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DeferredSuffix)->Arg(1)->Arg(8)->Arg(32)->Arg(128);

}  // namespace
