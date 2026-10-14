// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <semaphore>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/continuation.h"
#include "iree/hal/memory/maintenance_thread.h"

namespace {

struct Gate : iree_hal_memory_maintenance_entry_t {
  // Indicates that the service is occupied before notifications are measured.
  std::binary_semaphore entered{0};
  // Controller-owned release after the timed region.
  std::binary_semaphore resume{0};
};

// Measures notifications to an already-scheduled queue, as when several
// completions return capacity during a service turn or a cold native call.
// This is metadata-only timing; it excludes worker wake and native execution.
void BM_CoalescedNotification(benchmark::State& state) {
  iree_hal_memory_maintenance_t* maintenance = nullptr;
  IREE_CHECK_OK(iree_hal_memory_maintenance_thread_create(
      {}, iree_allocator_system(), &maintenance));
  Gate gate;
  gate.fn = [](iree_hal_memory_maintenance_entry_t* entry) {
    auto* gate = static_cast<Gate*>(entry);
    gate->entered.release();
    gate->resume.acquire();
  };
  iree_hal_memory_maintenance_enqueue(maintenance, &gate);
  gate.entered.acquire();
  iree_hal_amd_gpu_continuation_t continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance, [](void*) { return false; }, nullptr, &continuation);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  for (auto _ : state) {
    iree_hal_amd_gpu_continuation_notify(&continuation);
  }
  gate.resume.release();
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  iree_hal_memory_maintenance_release(maintenance);
  state.SetItemsProcessed(state.iterations());
  state.counters["metadata_bytes"] = sizeof(continuation);
}
BENCHMARK(BM_CoalescedNotification);

}  // namespace
