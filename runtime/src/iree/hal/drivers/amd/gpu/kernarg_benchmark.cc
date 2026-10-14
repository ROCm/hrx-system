// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"

namespace {

// CPU packing only: native storage, resource retention, publication, and
// completion are outside these rows. Each pair of explicit arguments is one
// pointer and four bytes, with a native alignment hole before the next pair.
static void BM_Emplace(benchmark::State& state) {
  const uint32_t count = state.range(0);
  const bool runtime = state.range(1) != 0;
  const bool native = state.range(2) != 0;
  std::vector<iree_hal_amd_gpu_metadata_argument_t> arguments;
  for (uint32_t i = 0; i < count; ++i) {
    arguments.push_back({
        .offset = i * 16,
        .size = 8,
        .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER,
    });
    arguments.push_back({
        .offset = i * 16 + 8,
        .size = 4,
        .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE,
    });
  }
  if (runtime) {
    for (uint32_t i = 0; i < 3; ++i) {
      arguments.push_back({
          .offset = count * 16 + i * 4,
          .size = 4,
          .kind = static_cast<iree_hal_amd_gpu_metadata_argument_kind_t>(
              IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X + i),
      });
    }
  }
  const iree_hal_amd_gpu_metadata_kernel_t kernel = {
      .kernarg_segment_size = count * 16 + (runtime ? 16u : 0u),
      .kernarg_segment_alignment = 8,
      .has_argument_metadata = true,
      .argument_count = arguments.size(),
      .arguments = arguments.data(),
  };
  iree_hal_amd_gpu_kernarg_layout_t* layout = nullptr;
  IREE_CHECK_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
      &kernel, iree_allocator_system(), &layout));
  std::vector<uint8_t> constants(
      native ? layout->native_byte_length : layout->constant_byte_length, 0x5a);
  std::vector<uint64_t> bindings(count, UINT64_C(0x12340000));
  const iree_hal_amd_gpu_kernarg_inputs_t inputs = {.block_count = {7, 3, 2}};
  alignas(64) std::array<uint8_t, 1024> target;
  benchmark::DoNotOptimize(layout);
  benchmark::DoNotOptimize(constants.data());
  benchmark::DoNotOptimize(bindings.data());
  benchmark::DoNotOptimize(inputs);
  if (native) {
    const auto source =
        iree_make_const_byte_span(constants.data(), constants.size());
    for (auto _ : state) {
      iree_hal_amd_gpu_kernarg_layout_emplace_native(layout, source, &inputs,
                                                     target.data());
      benchmark::ClobberMemory();
    }
  } else {
    for (auto _ : state) {
      iree_hal_amd_gpu_kernarg_layout_emplace(
          layout, constants.data(), bindings.data(), &inputs, target.data());
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * layout->byte_length);
  iree_hal_amd_gpu_kernarg_layout_free(layout);
}
BENCHMARK(BM_Emplace)
    ->ArgNames({"pairs", "runtime", "native"})
    ->Args({1, 0, 0})
    ->Args({4, 0, 0})
    ->Args({16, 0, 0})
    ->Args({4, 1, 0})
    ->Args({4, 1, 1});

}  // namespace
