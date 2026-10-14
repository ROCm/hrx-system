// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

namespace {

struct Storage {
  Storage() {
    iree_hal_amd_gpu_pm4_signal_pool_initialize(
        values.data(), 0x1000, signals.data(), signals.size(), {}, &pool);
  }
  ~Storage() { iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool); }

  // CPU-only stand-in for native control storage; no MMIO is timed here.
  std::array<uint64_t, 64> values;
  // Metadata shared by benchmark producer threads.
  std::array<iree_hal_amd_gpu_pm4_signal_t, 64> signals;
  // Shared finite free list.
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
};

void BM_Pool(benchmark::State& state) {
  static Storage storage;
  for (auto _ : state) {
    auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    benchmark::DoNotOptimize(signal);
    iree_hal_amd_gpu_pm4_signal_release(signal);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["host_bytes_per_slot"] = benchmark::Counter(
      sizeof(storage.signals[0]), benchmark::Counter::kAvgThreads);
  state.counters["native_bytes_per_slot"] = benchmark::Counter(
      sizeof(storage.values[0]), benchmark::Counter::kAvgThreads);
}
BENCHMARK(BM_Pool)->Threads(1)->Threads(2)->Threads(4)->Threads(8);

void BM_SharedUse(benchmark::State& state) {
  Storage storage;
  auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  for (auto _ : state) {
    benchmark::DoNotOptimize(signal);
    iree_hal_amd_gpu_pm4_signal_retain(signal);
    iree_hal_amd_gpu_pm4_signal_release(signal);
  }
  iree_hal_amd_gpu_pm4_signal_release(signal);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SharedUse);

}  // namespace
