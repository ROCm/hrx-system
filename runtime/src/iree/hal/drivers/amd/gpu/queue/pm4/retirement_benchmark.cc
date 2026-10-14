// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"

namespace {

// CPU encoding and observation only; native execution, MMIO, worker wakeup and
// host action processing are separate costs.
void BM_Emit(benchmark::State& state) {
  std::array<iree_hal_amd_gpu_pm4_retirement_record_t, 32> records;
  iree_hal_amd_gpu_pm4_retirement_ring_t ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000, 32,
                                                  &ring);
  std::array<uint32_t, 14> words;
  uint64_t position = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ring);
    benchmark::DoNotOptimize(iree_hal_amd_gpu_pm4_retirement_emit(
        words.data(), &ring, position++, &state, 0x30e));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["record_bytes"] = sizeof(records[0]);
}
BENCHMARK(BM_Emit);

void BM_Poll(benchmark::State& state) {
  std::array<iree_hal_amd_gpu_pm4_retirement_record_t, 32> records;
  iree_hal_amd_gpu_pm4_retirement_ring_t ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000, 32,
                                                  &ring);
  for (auto& record : records) {
    record.action = reinterpret_cast<uintptr_t>(&state);
    iree_atomic_store(&record.generation, state.range(0),
                      iree_memory_order_release);
  }
  // Rotate across an initialized generation so indexing and generation
  // selection remain in the measured path. This measures observation only;
  // the benchmark does not model native production or capacity return.
  uint64_t position = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ring);
    benchmark::DoNotOptimize(position);
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_pm4_retirement_poll(&ring, position));
    position = (position + 1) & (records.size() - 1);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Poll)->ArgName("ready")->Arg(0)->Arg(1);

}  // namespace
