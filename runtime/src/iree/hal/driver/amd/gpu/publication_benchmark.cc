// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <semaphore>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/publication.h"

namespace {

// Includes the handoff's completion boundary. One changed identity is embedded
// in a table spanning one, two, or all five readiness words; no GPU is opened.
void BM_SourceReplacement(benchmark::State& state) {
  std::array<iree_hal_amd_gpu_publication_source_t,
             IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES>
      first = {},
      second = {};
  const uint32_t count = state.range(0);
  first[count - 1] = {
      .publish = [](void*) { return false; },
      .user_data = first.data(),
  };
  second[count - 1] = first[count - 1];
  second[count - 1].user_data = second.data();
  iree_hal_amd_gpu_publication_t publication;
  IREE_CHECK_OK(iree_hal_amd_gpu_publication_initialize(
      count, first.data(), {}, iree_allocator_system(), &publication));
  for (auto _ : state) {
    iree_hal_amd_gpu_publication_set_sources(&publication, count,
                                             second.data());
    iree_hal_amd_gpu_publication_set_sources(&publication, count, first.data());
  }
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  state.SetItemsProcessed(2 * state.iterations());
  state.counters["replacements"] = 2;
  state.counters["publication_bytes"] = sizeof(publication);
}
BENCHMARK(BM_SourceReplacement)
    ->Arg(1)
    ->Arg(65)
    ->Arg(IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Measures the added removal handoff separately from native queue shutdown.
// Both rows remove and restore one source; the split row first disables the
// callback, allowing its exact join to happen outside membership exclusion.
void BM_SourceRemoval(benchmark::State& state, bool split) {
  iree_hal_amd_gpu_publication_source_t source = {
      .publish = [](void*) { return false; },
  };
  const auto binding = source;
  iree_hal_amd_gpu_publication_source_t vacant = {};
  iree_hal_amd_gpu_publication_t publication;
  IREE_CHECK_OK(iree_hal_amd_gpu_publication_initialize(
      1, &source, {}, iree_allocator_system(), &publication));
  for (auto _ : state) {
    if (split) {
      iree_hal_amd_gpu_publication_remove_source(&publication, 0);
    }
    iree_hal_amd_gpu_publication_set_sources(&publication, 1, &vacant);
    source = binding;
    iree_hal_amd_gpu_publication_set_sources(&publication, 1, &source);
  }
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_CAPTURE(BM_SourceRemoval, Replace, false)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_SourceRemoval, DisableThenReplace, true)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Includes notification and the final worker completion. The large ready
// suffix amortizes that boundary to expose the cost of one deferred turn.
// A turn is one callback, which may publish many operations in a native queue.
void BM_ReadyTurns(benchmark::State& state) {
  struct Source {
    // Worker-owned turn count, reset only after the completed batch joins.
    uint32_t remaining = 0;
    // Publishes the final count access before the controller resets it.
    std::binary_semaphore completed{0};
  } source;
  iree_hal_amd_gpu_publication_source_t binding = {
      .publish =
          [](void* user_data) {
            auto* source = static_cast<Source*>(user_data);
            if (--source->remaining) {
              return true;
            }
            source->completed.release();
            return false;
          },
      .user_data = &source,
  };
  iree_hal_amd_gpu_publication_t publication;
  IREE_CHECK_OK(iree_hal_amd_gpu_publication_initialize(
      1, &binding, {}, iree_allocator_system(), &publication));
  const uint32_t turns = state.range(0);
  for (auto _ : state) {
    source.remaining = turns;
    iree_hal_amd_gpu_publication_notify(&publication, 0);
    source.completed.acquire();
  }
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  state.SetItemsProcessed(turns * state.iterations());
  state.counters["turns"] = turns;
  state.counters["publication_bytes"] = sizeof(publication);
}
BENCHMARK(BM_ReadyTurns)
    ->Arg(1)
    ->Arg(4096)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

}  // namespace
