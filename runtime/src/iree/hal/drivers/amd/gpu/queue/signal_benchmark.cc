// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"

namespace {

struct Storage {
  Storage() {
    iree_hal_amd_gpu_signal_pool_initialize({}, &pool);
    iree_hal_amd_gpu_signal_block_initialize(
        &pool, this, values.data(), 0x1000, 8, 0, signals.data(),
        signals.size(), 1, frontiers.data(), &block);
  }
  ~Storage() {
    iree_hal_amd_gpu_signal_block_deinitialize(&block);
    iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
  }

  // CPU-only stand-in for native control storage; no MMIO is timed here.
  std::array<uint64_t, 64> values;
  // Metadata owned by this queue's exclusive publisher.
  std::array<iree_hal_amd_gpu_signal_t, 64> signals = {};
  // Finite pool with one acquiring publisher and concurrent final returns.
  iree_hal_amd_gpu_signal_pool_t pool;
  // Attached native extent and its signal metadata.
  iree_hal_amd_gpu_signal_block_t block;
  // One-queue causal vector per native signal.
  std::array<uint64_t, 64> frontiers;
};

void BM_Pool(benchmark::State& state) {
  // Each benchmark thread models a distinct queue's exclusive publisher.
  Storage storage;
  for (auto _ : state) {
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    benchmark::DoNotOptimize(signal);
    iree_hal_amd_gpu_signal_release(signal);
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
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(signal);
    iree_hal_amd_gpu_signal_retain(signal);
    iree_hal_amd_gpu_signal_release(signal);
  }
  iree_hal_amd_gpu_signal_release(signal);
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
  iree_hal_amd_gpu_signal_t metadata = {};
  iree_hal_amd_gpu_signal_pool_t pool;
  // Attached native extent and its signal metadata.
  iree_hal_amd_gpu_signal_block_t block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &pool);
  iree_hal_amd_gpu_signal_block_initialize(&pool, &pool, &value, 0x1000, 8, 0,
                                           &metadata, 1, queue_count,
                                           snapshot.data(), &block);
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0);
  for (auto _ : state) {
    ++source[0];
    iree_hal_amd_gpu_signal_publish_frontier(signal, source.data());
    iree_hal_amd_gpu_signal_join_frontier(signal, target.data());
    benchmark::DoNotOptimize(target.data());
    benchmark::ClobberMemory();
  }
  iree_hal_amd_gpu_signal_release(signal);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
  state.SetItemsProcessed(state.iterations());
  state.counters["snapshot_bytes"] = queue_count * sizeof(uint64_t);
}
BENCHMARK(BM_Frontier)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(32)->Arg(256);

}  // namespace
