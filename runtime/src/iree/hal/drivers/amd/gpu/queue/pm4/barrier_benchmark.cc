// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

namespace iree::hal::amd::gpu {
namespace {

static void BM_Prepare(benchmark::State& state) {
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  IREE_CHECK_OK(
      iree_hal_amd_gpu_pm4_barrier_profile_initialize({12, 0, 1}, &profile));
  std::array<iree_hal_barrier_t, 16> values;
  const iree_hal_memory_barrier_t access = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ};
  values.fill({
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
      .memory_barrier_count = 1,
      .memory_barriers = &access,
  });
  const iree_hal_barrier_list_t list = {
      static_cast<iree_host_size_t>(state.range(0) > 0 ? state.range(0) : 0),
      values.data()};
  const auto* barriers = state.range(0) < 0 ? nullptr : &list;
  benchmark::DoNotOptimize(profile);
  benchmark::DoNotOptimize(barriers);
  for (auto _ : state) {
    const auto result =
        iree_hal_amd_gpu_pm4_barrier_prepare(&profile, barriers);
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Prepare)->Arg(-1)->Arg(0)->Arg(1)->Arg(4)->Arg(16);

// CPU encoding only. No publication, driver, native execution, or completion.
static void BM_Emit(benchmark::State& state) {
  iree_hal_amd_gpu_pm4_barrier_profile_t profile;
  IREE_CHECK_OK(
      iree_hal_amd_gpu_pm4_barrier_profile_initialize({12, 0, 1}, &profile));
  const iree_hal_barrier_list_t empty = {};
  auto barrier = iree_hal_amd_gpu_pm4_barrier_prepare(
      &profile, state.range(0) ? nullptr : &empty);
  benchmark::DoNotOptimize(barrier);
  std::array<uint32_t, 17> words;
  for (auto _ : state) {
    if (state.range(1)) {
      uint32_t cache_control;
      auto* end = iree_hal_amd_gpu_pm4_barrier_emit_tail(words.data(), barrier,
                                                         &cache_control);
      benchmark::DoNotOptimize(cache_control);
      benchmark::DoNotOptimize(end);
    } else {
      auto* end = iree_hal_amd_gpu_pm4_barrier_emit(words.data(), barrier);
      benchmark::DoNotOptimize(end);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Emit)->Args({0, 0})->Args({1, 0})->Args({0, 1})->Args({1, 1});

}  // namespace
}  // namespace iree::hal::amd::gpu
