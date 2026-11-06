// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/kernels/private_roundtrip.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

TEST_F(Pm4DispatchTest, PrivateStorageReplacementAndReuseAcrossQueues) {
  if (!(family_.format_features &
        AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH)) {
    GTEST_SKIP() << "command-owned scratch is unavailable";
  }
  const auto* kernel =
      kernels::private_roundtrip::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  ASSERT_NE(kernel->program.resource2 & 1u, 0u);
  ASSERT_GT(kernel->private_segment_byte_length, 0u);
  const uint32_t engine_count =
      gpu_endpoint_info_.topology.shader_engine_count_per_xcc;
  const uint32_t waves_per_engine =
      ((gpu_endpoint_info_.compute.compute_unit_count + engine_count - 1) /
       engine_count) *
      gpu_endpoint_info_.compute.maximum_scratch_wave_count_per_compute_unit;
  ASSERT_GT(waves_per_engine, 0u);
  ASSERT_LE(waves_per_engine, 0xfffu);
  const uint32_t wave_byte_length =
      (kernel->private_segment_byte_length * kernel->wavefront_size + 255u) &
      ~255u;
  const uint64_t scratch_byte_length =
      uint64_t{engine_count} * waves_per_engine * wave_byte_length;
  RecordProperty("scratch_bytes_per_queue",
                 std::to_string(scratch_byte_length));
  RecordProperty("scratch_bytes_per_wave", wave_byte_length);
  RecordProperty("scratch_waves_per_engine", waves_per_engine);
  RecordProperty("scratch_kernel_target", kernel->target);

  constexpr uint32_t kEpochs = 3;
  constexpr uint32_t kWorkitems = 4096;
  constexpr uint32_t kWords = kWorkitems * 9;
  constexpr uint32_t kGuardWords = 64;
  constexpr uint32_t kGuard = 0x759bf13du;
  struct Workset {
    // Independent native publisher borrowing only this workset's resources.
    GpuCommandQueue* queue;
    // Two backing generations replaced after a GPU execution join.
    std::array<GpuMemory*, 2> scratch;
    // Output records with independent prefix and suffix guards.
    GpuMemory* output;
    // Immutable argument records, one per dispatch.
    GpuMemory* arguments;
    // Monotonic completion value written after payload release.
    GpuMemory* completion;
  };
  std::array<Workset, 2> worksets = {};
  Pm4ComputeProgram program = {0,
                               kernel->program.resource1,
                               kernel->program.resource2,
                               kernel->program.resource3,
                               kernel->group_segment_byte_length,
                               kernel->wavefront_size,
                               {kernel->workgroup_size(), 1, 1}};
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel->executable, kernel->entry_byte_offset, &program, "private"));
  for (uint32_t index = 0; index < worksets.size(); ++index) {
    auto& workset = worksets[index];
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&workset.queue));
    ASSERT_GE(workset.queue->words().size(), 3 * 96u);
    for (auto& scratch : workset.scratch) {
      ASSERT_NO_FATAL_FAILURE(
          CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                       scratch_byte_length + 2 * kGuardWords * sizeof(uint32_t),
                       AMDF_MEMORY_FLAG_HOST_COHERENT, &scratch));
      ASSERT_EQ(scratch->device_address % 256, 0u);
      auto* words = static_cast<uint32_t*>(scratch->host.pointer);
      std::fill_n(words, scratch->info.byte_length / sizeof(uint32_t), kGuard);
    }
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     (kEpochs * kWords + 2 * kGuardWords) * sizeof(uint32_t),
                     &workset.output));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &workset.arguments));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
                     &workset.completion));
    std::memset(workset.completion->host.pointer, 0, 4096);
    auto* output = static_cast<uint32_t*>(workset.output->host.pointer);
    std::fill_n(output, kEpochs * kWords + 2 * kGuardWords, kGuard);
    for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
      const kernels::private_roundtrip::Arguments arguments = {
          workset.output->device_address +
              (kGuardWords + epoch * kWords) * sizeof(uint32_t),
          0x13579bdfu + index * 0x24681357u + epoch, epoch + 1};
      std::memcpy(
          static_cast<uint8_t*>(workset.arguments->host.pointer) + epoch * 256,
          &arguments, sizeof(arguments));
    }
  }

  // Publish every generation and reuse before observing either queue. Only
  // the GPU execution join orders replacement; no host wait hides that edge.
  for (auto& workset : worksets) {
    Pm4CommandWriter commands(workset.queue->words().data(), *pm4_profile_);
    commands.SystemBarrier();
    for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
      if (epoch < 2) {
        if (epoch) {
          commands.WaitCompute();
        }
        commands.BindScratch(workset.scratch[epoch]->device_address +
                                 kGuardWords * sizeof(uint32_t),
                             waves_per_engine, wave_byte_length);
      }
      commands.BindCompute(program,
                           workset.arguments->device_address + epoch * 256,
                           family_.format_features);
      commands.Dispatch(program, kWorkitems, 1, 1);
    }
    commands.ReleaseSystem32(workset.completion->device_address, 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        workset.queue->Publish(api_, gpu_api_, commands.word_count()));
  }
  for (uint32_t index = 0; index < worksets.size(); ++index) {
    auto& workset = worksets[index];
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(workset.completion->host.pointer), 1));
    const auto* output =
        static_cast<const uint32_t*>(workset.output->host.pointer);
    for (uint32_t epoch = 0; epoch < kEpochs; ++epoch) {
      for (uint32_t workitem = 0; workitem < kWorkitems; ++workitem) {
        const auto expected = kernels::private_roundtrip::ExpectedRecord(
            workitem, 0x13579bdfu + index * 0x24681357u + epoch, epoch + 1);
        for (uint32_t word = 0; word < expected.size(); ++word) {
          EXPECT_EQ(output[kGuardWords + epoch * kWords + workitem * 9 + word],
                    expected[word])
              << "queue=" << index << " epoch=" << epoch
              << " workitem=" << workitem;
        }
      }
    }
    for (uint32_t word = 0; word < kGuardWords; ++word) {
      EXPECT_EQ(output[word], kGuard);
      EXPECT_EQ(output[kGuardWords + kEpochs * kWords + word], kGuard);
      for (const auto* scratch : workset.scratch) {
        const auto* words = static_cast<const uint32_t*>(scratch->host.pointer);
        EXPECT_EQ(words[word], kGuard);
        EXPECT_EQ(words[kGuardWords + scratch_byte_length / 4 + word], kGuard);
      }
    }
    ASSERT_NO_FATAL_FAILURE(workset.queue->WaitRetired(api_));
  }
}

}  // namespace
