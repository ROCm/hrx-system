// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/fabric/transient_buffer_pool.h"

namespace {

void Release(const iree_hal_amd_fabric_transient_buffer_slots_t& slots) {
  for (iree_host_size_t i = 0; i < slots.count; ++i) {
    iree_hal_amd_fabric_transient_buffer_pool_release(
        slots.chunk, slots.data + i * slots.stride);
  }
}

// Measures metadata reuse independently of allocation and native queue work.
// Small slots return into a partially available chunk; the one-slot class
// also exercises publication of a formerly full chunk on each iteration.
void BM_Recycle(benchmark::State& state) {
  iree_hal_amd_fabric_transient_buffer_pool_t pool;
  iree_hal_amd_fabric_transient_buffer_pool_initialize(iree_allocator_system(),
                                                       &pool);
  iree_hal_amd_fabric_transient_buffer_slots_t slots;
  IREE_CHECK_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
      &pool, state.range(0), state.range(1), &slots));
  Release(slots);
  for (auto _ : state) {
    IREE_CHECK_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
        &pool, state.range(0), state.range(1), &slots));
    benchmark::DoNotOptimize(slots.data);
    Release(slots);
  }
  state.SetItemsProcessed(state.iterations() * slots.count);
  iree_hal_amd_fabric_transient_buffer_pool_deinitialize(&pool);
}
BENCHMARK(BM_Recycle)
    ->Args({512, 1})
    ->Args({512, 8})
    ->Args({512, 64})
    ->Args({32768, 64})
    ->Args({1 << 20, 1});

}  // namespace
