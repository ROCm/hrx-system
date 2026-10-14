// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <semaphore>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/utils/host_semaphore.h"

namespace {

// Measures cold host admission only. Ready/native dependencies never enter this
// path. Each iteration uses actual host semaphores and the actual service
// owner; the producer satisfies the captured waits only after registration
// returns.
void HostWait(benchmark::State& state, bool include_completion) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_CHECK_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_memory_maintenance_t* service = nullptr;
  IREE_CHECK_OK(iree_hal_memory_maintenance_thread_create(
      {}, iree_allocator_system(), &service));
  iree_async_semaphore_wait_context_t context;
  iree_async_semaphore_wait_context_initialize(&context);
  const size_t count = state.range(0);
  std::vector<iree_hal_semaphore_t*> semaphores(count);
  std::vector<uint64_t> values(count, 0);
  for (auto& semaphore : semaphores) {
    IREE_CHECK_OK(iree_hal_host_semaphore_create(
        proactor, 0, iree_allocator_system(), &semaphore));
  }
  iree_hal_amd_gpu_host_wait_t wait;
  std::binary_semaphore complete{0};
  for (auto _ : state) {
    for (auto& value : values) {
      ++value;
    }
    IREE_CHECK_OK(iree_hal_amd_gpu_host_wait_begin(
        &context, service, {count, semaphores.data(), values.data()},
        [](void* data, iree_status_t status) {
          IREE_CHECK_OK(status);
          static_cast<std::binary_semaphore*>(data)->release();
        },
        &complete, iree_allocator_system(), &wait));
    if (!include_completion) {
      state.PauseTiming();
    }
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(
          iree_hal_semaphore_signal(semaphores[i], values[i], nullptr));
    }
    complete.acquire();
    if (!include_completion) {
      state.ResumeTiming();
    }
  }
  iree_hal_memory_maintenance_release(service);
  iree_async_semaphore_wait_context_deinitialize(&context);
  for (auto* semaphore : semaphores) {
    iree_hal_semaphore_release(semaphore);
  }
  iree_async_proactor_release(proactor);
  state.SetItemsProcessed(state.iterations());
  state.counters["sidecar_bytes"] = sizeof(wait);
}

// Includes tracker allocation, semaphore retention and pending registration.
// Producer signaling, service wake, finalization and completion are excluded.
void BM_HostWaitBegin(benchmark::State& state) { HostWait(state, false); }
BENCHMARK(BM_HostWaitBegin)->Arg(1)->Arg(4)->Arg(16);

// Includes registration, producer signaling, service finalization and the
// user-visible completion join. Wall time includes both executor threads.
void BM_HostWaitEndToEnd(benchmark::State& state) { HostWait(state, true); }
BENCHMARK(BM_HostWaitEndToEnd)->Arg(1)->Arg(4)->Arg(16)->UseRealTime();

}  // namespace
