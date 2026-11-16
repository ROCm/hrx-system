// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/queue/capture_pool.h"

namespace {

// Isolates block acquisition/return and whole-list transfer. The public queue
// benchmark measures concurrent callers, native execution, and final
// retirement.
void BM_RecycleBurst(benchmark::State& state) {
  iree_hal_amd_gpu_capture_pool_t pool;
  iree_hal_amd_gpu_capture_pool_initialize(1024, iree_allocator_system(),
                                           &pool);
  std::vector<iree_byte_span_t> blocks(state.range(0));
  IREE_CHECK_OK(
      iree_hal_amd_gpu_capture_pool_preallocate(&pool, blocks.size()));
  for (auto& block : blocks) {
    IREE_CHECK_OK(
        iree_hal_amd_gpu_capture_pool_acquire(&pool, state.range(1), &block));
  }
  for (auto block : blocks) {
    iree_hal_amd_gpu_capture_pool_release(&pool, block);
  }
  for (auto _ : state) {
    for (auto& block : blocks) {
      IREE_CHECK_OK(
          iree_hal_amd_gpu_capture_pool_acquire(&pool, state.range(1), &block));
      benchmark::DoNotOptimize(block.data);
    }
    for (auto block : blocks) {
      iree_hal_amd_gpu_capture_pool_release(&pool, block);
    }
  }
  iree_hal_amd_gpu_capture_pool_deinitialize(&pool);
  state.SetItemsProcessed(state.iterations() * blocks.size());
}
BENCHMARK(BM_RecycleBurst)
    ->Args({1, 208})
    ->Args({1, 1024})
    ->Args({1, 1025})
    ->Args({1, 1256})
    ->Args({1, 4096})
    ->Args({1, 65537})
    ->Args({8, 208})
    ->Args({64, 208})
    ->Args({1024, 208});

}  // namespace
