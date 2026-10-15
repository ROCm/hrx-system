// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"

namespace {

struct Storage {
  Storage() {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_create(
        proactor, 0, 16, iree_allocator_system(), &semaphore));
    iree_hal_amd_gpu_pm4_signal_pool_initialize(
        this, values.data(), 0x1000, signals.data(), signals.size(), {}, &pool);
  }
  ~Storage() {
    iree_hal_semaphore_release(semaphore);
    iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool);
    iree_async_proactor_release(proactor);
  }
  // Borrowed platform facility; no polling or native execution is measured.
  iree_async_proactor_t* proactor = nullptr;
  // Owned semaphore with completed and submitted state kept separate.
  iree_hal_semaphore_t* semaphore = nullptr;
  // CPU RAM standing in for native control cells; no MMIO is timed.
  std::array<uint64_t, 4> values;
  // Native signal metadata, allocated once for the entire benchmark.
  std::array<iree_hal_amd_gpu_pm4_signal_t, 4> signals;
  // Finite backing shared by publication and independent readers.
  iree_hal_amd_gpu_pm4_signal_pool_t pool;
};

void BM_CompletedWait(benchmark::State& state) {
  Storage storage;
  IREE_CHECK_OK(iree_hal_semaphore_signal(storage.semaphore, 1, nullptr));
  for (auto _ : state) {
    iree_hal_amd_gpu_pm4_semaphore_wait_t wait;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_acquire_wait(storage.semaphore,
                                                              1, &wait));
    benchmark::DoNotOptimize(wait);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CompletedWait);

void BM_NativeWait(benchmark::State& state) {
  Storage storage;
  auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_signal(
      storage.semaphore, 1, {1, 1}, signal));
  for (auto _ : state) {
    iree_hal_amd_gpu_pm4_semaphore_wait_t wait;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_acquire_wait(storage.semaphore,
                                                              1, &wait));
    benchmark::DoNotOptimize(wait);
    iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  }
  iree_hal_amd_gpu_pm4_signal_release(signal);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NativeWait);

// Metadata lifecycle, including producer allocation, publication, an
// independent native reader, observed completion, and last reader return. The
// second case includes an already-created availability timeline with no pending
// timepoints.
void BM_Lifecycle(benchmark::State& state) {
  Storage storage;
  if (state.range(0)) {
    iree_hal_semaphore_t* submission = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
        storage.semaphore, &submission));
    iree_hal_semaphore_release(submission);
  }
  uint64_t value = 0;
  for (auto _ : state) {
    ++value;
    auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_signal(
        storage.semaphore, value, {1, value}, signal));
    iree_hal_amd_gpu_pm4_semaphore_wait_t wait;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_acquire_wait(storage.semaphore,
                                                              value, &wait));
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_completed(
        storage.semaphore, value, nullptr));
    iree_hal_amd_gpu_pm4_signal_release(signal);
    iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Lifecycle)->Arg(0)->Arg(1);

}  // namespace
