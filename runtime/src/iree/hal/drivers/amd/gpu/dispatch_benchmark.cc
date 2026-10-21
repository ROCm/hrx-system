// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/dispatch.h"

namespace {

// Immutable image/native facts are warm, as in repeated libhrx launch-size
// queries. This measures residency arithmetic, not executable lookup, public
// queue validation, submission, or device execution.
static void BM_QueryConcurrency(benchmark::State& state) {
  const uint32_t major = state.range(0);
  amdf_gpu_endpoint_info_t target = {
      .gfx_ip = {major, major == 9 ? 4u : 0u, major == 9 ? 2u : 0u},
      .compute =
          {
              .wavefront_size = major == 9 ? 64u : 32u,
              .compute_unit_count = 96,
              .maximum_wave_count_per_compute_unit = 32,
              .local_data_share_byte_length = 65536,
          },
  };
  iree_hal_amd_gpu_image_kernel_t kernel = {
      .compute_pgm_rsrc1 = 19u | (7u << 6),
      .code_properties = static_cast<uint16_t>(major == 9 ? 0 : 1u << 10),
  };
  iree_hal_amd_gpu_metadata_kernel_t metadata = {
      .group_segment_fixed_size = 1024,
      .max_flat_workgroup_size = 1024,
  };
  iree_hal_queue_dispatch_concurrency_params_t params = {
      .workgroup_size = {256, 1, 1},
      .dynamic_workgroup_local_memory = static_cast<uint32_t>(state.range(1)),
  };
  benchmark::DoNotOptimize(target);
  benchmark::DoNotOptimize(kernel);
  benchmark::DoNotOptimize(metadata);
  benchmark::DoNotOptimize(params);
  for (auto _ : state) {
    iree_hal_queue_dispatch_concurrency_t concurrency;
    IREE_CHECK_OK(iree_hal_amd_gpu_dispatch_query_concurrency(
        &target, &kernel, &metadata, 16, params, &concurrency));
    benchmark::DoNotOptimize(concurrency);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueryConcurrency)
    ->ArgNames({"gfx_major", "dynamic_lds"})
    ->Args({9, 0})
    ->Args({9, 32768})
    ->Args({11, 0})
    ->Args({11, 32768})
    ->Args({12, 0})
    ->Args({12, 32768});

}  // namespace
