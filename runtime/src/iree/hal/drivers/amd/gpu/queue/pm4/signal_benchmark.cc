// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

namespace {

struct Storage {
  Storage() {
    iree_hal_amd_gpu_pm4_signal_pool_initialize(this, values.data(), 0x1000,
                                                signals.data(), signals.size(),
                                                1, frontiers.data(), {}, &pool);
  }
  ~Storage() { iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool); }

  // CPU-only stand-in for native control storage; no MMIO is timed here.
  std::array<uint64_t, 64> values;
  // Metadata shared by benchmark producer threads.
  std::array<iree_hal_amd_gpu_pm4_signal_t, 64> signals;
  // Shared finite free list.
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
  // One-queue causal vector per native signal.
  std::array<uint64_t, 64> frontiers;
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

// Snapshot and transitive join over the actual provisioned queue count. Pool
// construction and signal acquisition stay outside the timed region.
void BM_Frontier(benchmark::State& state) {
  const uint32_t queue_count = static_cast<uint32_t>(state.range(0));
  std::vector<uint64_t> source(queue_count, 1);
  std::vector<uint64_t> target(queue_count, 0);
  std::vector<uint64_t> snapshot(queue_count);
  uint64_t value = 0;
  iree_hal_amd_gpu_pm4_signal_t metadata;
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
  iree_hal_amd_gpu_pm4_signal_pool_initialize(&pool, &value, 0x1000, &metadata,
                                              1, queue_count, snapshot.data(),
                                              {}, &pool);
  auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&pool);
  for (auto _ : state) {
    ++source[0];
    iree_hal_amd_gpu_pm4_signal_publish_frontier(signal, source.data());
    iree_hal_amd_gpu_pm4_signal_join_frontier(signal, target.data());
    benchmark::DoNotOptimize(target.data());
    benchmark::ClobberMemory();
  }
  iree_hal_amd_gpu_pm4_signal_release(signal);
  iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool);
  state.SetItemsProcessed(state.iterations());
  state.counters["snapshot_bytes"] = queue_count * sizeof(uint64_t);
}
BENCHMARK(BM_Frontier)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(32)->Arg(256);

}  // namespace
