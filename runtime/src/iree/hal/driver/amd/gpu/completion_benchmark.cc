// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/completion.h"

namespace {

// Isolates cold callback-table bookkeeping before worker startup, the same
// state used by completion initialization. Native queue shutdown and worker
// scheduling are deliberately outside this component measurement. Both rows
// remove and reinstall the last source; the split row first disables it as
// queue membership does before joining outside its own mutex.
void BM_SourceRemoval(benchmark::State& state, bool split) {
  const size_t count = state.range(0);
  std::vector<iree_hal_amd_gpu_completion_source_t> first(count);
  std::vector<iree_hal_amd_gpu_completion_source_t> second(count);
  for (size_t i = 0; i < count; ++i) {
    first[i] = {
        .fn = [](void*, amdf_status_t, iree_hal_amd_gpu_completion_scope_t*,
                 iree_hal_amd_gpu_completion_wait_t*,
                 iree_hal_amd_gpu_completion_copy_t*) { return false; },
        .user_data = &first[i],
    };
  }
  second = first;
  const auto source = first.back();
  iree_hal_amd_gpu_completion_t completion = {};
  iree_slim_mutex_initialize(&completion.drain_mutex);
  iree_hal_amd_gpu_completion_set_sources(&completion, count, first.data());
  for (auto _ : state) {
    if (split) {
      iree_hal_amd_gpu_completion_remove_source(&completion, source.fn,
                                                source.user_data);
    }
    iree_hal_amd_gpu_completion_set_sources(&completion, count - 1,
                                            second.data());
    first.back() = source;
    iree_hal_amd_gpu_completion_set_sources(&completion, count, first.data());
  }
  iree_hal_amd_gpu_completion_set_sources(&completion, 0, nullptr);
  iree_slim_mutex_deinitialize(&completion.drain_mutex);
  state.SetItemsProcessed(state.iterations());
  state.counters["completion_bytes"] = sizeof(completion);
}
BENCHMARK_CAPTURE(BM_SourceRemoval, Replace, false)
    ->Arg(1)
    ->Arg(65)
    ->Arg(259)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_SourceRemoval, DisableThenReplace, true)
    ->Arg(1)
    ->Arg(65)
    ->Arg(259)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

}  // namespace
