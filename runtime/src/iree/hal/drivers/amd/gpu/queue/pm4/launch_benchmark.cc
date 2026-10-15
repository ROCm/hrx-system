// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

namespace iree::hal::amd::gpu {
namespace {

static iree_hal_amd_gpu_image_kernel_t Kernel(uint16_t properties,
                                              uint32_t register_count) {
  return {
      .entry_offset = 0x200,
      .compute_pgm_rsrc1 = 0xe0af0000,
      .compute_pgm_rsrc2 = 0x80u | (register_count << 1),
      .compute_pgm_rsrc3 = 0x20,
      .code_properties = properties,
  };
}

static void BM_PrepareLaunch(benchmark::State& state) {
  auto kernel = Kernel(0x408, state.range(0));
  benchmark::DoNotOptimize(kernel);
  for (auto _ : state) {
    iree_hal_amd_gpu_pm4_launch_t launch;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        {11, 0, 0}, &kernel, UINT64_C(0x1234000000), &launch));
    benchmark::DoNotOptimize(launch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_PrepareLaunch)->Arg(2)->Arg(15);

// CPU emission only. Native capture, publication, and completion are excluded.
static void BM_EmitLaunch(benchmark::State& state) {
  auto kernel = Kernel(state.range(0), state.range(1));
  alignas(64) iree_hal_amd_gpu_pm4_launch_t launch;
  IREE_CHECK_OK(iree_hal_amd_gpu_pm4_launch_initialize(
      {11, 0, 0}, &kernel, UINT64_C(0x1234000000), &launch));
  uint32_t workgroup_size[] = {128, 1, 1};
  uint32_t partial_size[] = {0, 0, 0};
  uint32_t dimensions[] = {256, 1, 1};
  uint32_t group_byte_length = 1024;
  iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .dispatch_pointer = UINT64_C(0x2000000040),
      .queue_pointer = UINT64_C(0x2000000080),
      .kernarg_pointer = UINT64_C(0x2000000100),
      .dispatch_id = 17,
      .private_segment_size = 0,
  };
  benchmark::DoNotOptimize(launch);
  benchmark::DoNotOptimize(workgroup_size);
  benchmark::DoNotOptimize(partial_size);
  benchmark::DoNotOptimize(dimensions);
  benchmark::DoNotOptimize(group_byte_length);
  benchmark::DoNotOptimize(inputs);
  alignas(64) std::array<uint32_t, 64> words;
  for (auto _ : state) {
    auto* end = iree_hal_amd_gpu_pm4_launch_emit(words.data(), &launch,
                                                 workgroup_size, partial_size,
                                                 group_byte_length, &inputs);
    end = iree_hal_amd_gpu_pm4_emit_dispatch_direct(end, dimensions,
                                                    launch.dispatch_initiator);
    benchmark::DoNotOptimize(end);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(
      state.iterations() *
      (launch.user_data_count ? 29 + launch.user_data_count : 27) *
      sizeof(uint32_t));
}
BENCHMARK(BM_EmitLaunch)
    ->Args({0x400, 0})
    ->Args({0x408, 2})
    ->Args({0x408, 15})
    ->Args({0x45e, 16});

// Private argument/packet preparation on the caller or recorder. GPU-side
// indirect count conversion and native publication are measured separately.
static void BM_EmplaceDispatchArguments(benchmark::State& state) {
  const iree_hal_amd_gpu_metadata_kernel_t metadata = {};
  iree_hal_amd_gpu_kernarg_layout_t layout = {
      .byte_length = 32,
      .alignment = 8,
      .native_byte_length = 32,
  };
  const iree_hal_amd_gpu_pm4_dispatch_packet_info_t packet = {
      .kernel_object = UINT64_C(0x1234000040),
  };
  const iree_hal_amd_gpu_pm4_function_t function = {
      .metadata = &metadata,
      .kernarg = &layout,
      .packet = state.range(0) ? &packet : nullptr,
  };
  const iree_hal_amd_gpu_pm4_dispatch_t dispatch = {
      .workgroup_size = {32, 1, 1},
      .workgroup_count = {64, 1, 1},
  };
  const iree_hal_dispatch_flags_t flags =
      (state.range(1) ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS
                      : 0) |
      (state.range(2) ? IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS
                      : IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  const std::array<uint64_t, 4> constants = {0x100000000, 17, 23, 29};
  alignas(64) std::array<uint8_t, 192> bytes;
  benchmark::DoNotOptimize(function);
  benchmark::DoNotOptimize(dispatch);
  benchmark::DoNotOptimize(flags);
  for (auto _ : state) {
    const auto storage =
        iree_hal_amd_gpu_pm4_dispatch_storage(&function, flags);
    if (storage.length) {
      iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
          &function, &dispatch,
          iree_make_const_byte_span(constants.data(), sizeof(constants)),
          nullptr, flags, bytes.data());
    }
    benchmark::DoNotOptimize(storage);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_EmplaceDispatchArguments)
    ->Args({0, 0, 0})
    ->Args({0, 1, 0})
    ->Args({0, 0, 1})
    ->Args({1, 0, 0})
    ->Args({1, 1, 0})
    ->Args({1, 0, 1})
    ->Args({1, 1, 1});

}  // namespace
}  // namespace iree::hal::amd::gpu
