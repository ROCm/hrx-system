// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/ring.h"

namespace {

// Reservation and padding only. Native stores, doorbells, execution and the
// independent completion consumer are outside this component measurement.
void BM_Reserve(benchmark::State& state) {
  std::array<uint32_t, 1024> words;
  iree_hal_amd_gpu_pm4_ring_t ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  uint64_t consumed = 0;
  const uint32_t body = state.range(0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(ring);
    iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
    const bool reserved = iree_hal_amd_gpu_pm4_ring_try_reserve(
        &ring, consumed, body, &reservation);
    benchmark::DoNotOptimize(reserved);
    benchmark::DoNotOptimize(reservation.words);
    benchmark::ClobberMemory();
    consumed = reservation.position + body - 8;
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["metadata_bytes"] = sizeof(ring);
}
BENCHMARK(BM_Reserve)->Arg(32)->Arg(76)->Arg(252);

void BM_CapacityBlocked(benchmark::State& state) {
  std::array<uint32_t, 1024> words;
  iree_hal_amd_gpu_pm4_ring_t ring;
  iree_hal_amd_gpu_pm4_ring_initialize(words.data(), words.size(), &ring);
  iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
  for (int i = 0; i < 3; ++i) {
    iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 256, &reservation);
  }
  for (auto _ : state) {
    // The completed prefix may change between attempts in the real queue.
    // Keep the capacity decision observable instead of folding it to false.
    benchmark::DoNotOptimize(ring);
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_pm4_ring_try_reserve(&ring, 0, 256, &reservation));
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CapacityBlocked);

}  // namespace
