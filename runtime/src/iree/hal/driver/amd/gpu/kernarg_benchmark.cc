// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/kernarg.h"

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
  iree_hal_amd_gpu_kernarg_inputs_t inputs = {.block_count = {7, 3, 2}};
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

// V5 compiler layout for a pointer, two scalars, and a by-value payload. The
// 256-byte payload occupies a 528-byte ABI with thirteen hidden geometry
// fields. Both emplacement paths measure CPU packing into cacheable host memory
// only.
static void BM_EmplaceLaunch(benchmark::State& state) {
  const uint32_t payload_length = state.range(0);
  const bool native = state.range(1) != 0;
  using Kind = iree_hal_amd_gpu_metadata_argument_kind_t;
  const uint32_t explicit_length = 16 + payload_length;
  std::vector<iree_hal_amd_gpu_metadata_argument_t> arguments = {
      {
          .offset = 0,
          .size = 8,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER,
      },
      {
          .offset = 8,
          .size = 4,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE,
      },
      {
          .offset = 12,
          .size = 4,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE,
      },
      {
          .offset = 16,
          .size = payload_length,
          .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE,
      },
  };
  for (uint32_t i = 0; i < 9; ++i) {
    arguments.push_back({
        .offset = explicit_length + (i < 3 ? i * 4 : 12 + (i - 3) * 2),
        .size = i < 3 ? 4u : 2u,
        .kind = static_cast<Kind>(
            IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X + i),
    });
  }
  for (uint32_t i = 0; i < 3; ++i) {
    arguments.push_back({
        .offset = explicit_length + 40 + i * 8,
        .size = 8,
        .kind = static_cast<Kind>(
            IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GLOBAL_OFFSET_X + i),
    });
  }
  arguments.push_back({
      .offset = explicit_length + 64,
      .size = 2,
      .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GRID_DIMS,
  });
  const iree_hal_amd_gpu_metadata_kernel_t kernel = {
      .kernarg_segment_size = explicit_length + 256,
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
  const uint64_t binding = UINT64_C(0x12340000);
  const iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .block_count = {8192, 1, 1},
      .group_size = {256, 1, 1},
      .grid_dimensions = 1,
  };
  alignas(64) std::array<uint8_t, 1024> target;
  benchmark::DoNotOptimize(layout);
  benchmark::DoNotOptimize(constants.data());
  benchmark::DoNotOptimize(binding);
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
      iree_hal_amd_gpu_kernarg_layout_emplace(layout, constants.data(),
                                              &binding, &inputs, target.data());
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * layout->byte_length);
  iree_hal_amd_gpu_kernarg_layout_free(layout);
}
BENCHMARK(BM_EmplaceLaunch)
    ->ArgNames({"payload", "native"})
    ->Args({64, 0})
    ->Args({64, 1})
    ->Args({256, 0})
    ->Args({256, 1});

}  // namespace
