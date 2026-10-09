// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/lds_exchange.h"
#include "libamdf/cts/gpu/kernels/lds_exchange_kernels.h"
#include "libamdf/cts/gpu/kernels/lds_exchange_wave64_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"

namespace {

enum class LaunchKind { kDirect, kIndirect };

class Pm4WaveModeTest : public Pm4DispatchTest {
 protected:
  void Execute(LaunchKind launch_kind) {
    if (!pm4_profile_->supports_wave64) {
      GTEST_SKIP() << "physical PM4 target supports wave32 only";
    }
    using Arguments = kernels::lds_exchange::Arguments;
    constexpr std::array<uint32_t, 3> kGroups = {4, 3, 5};
    constexpr std::array<uint32_t, 3> kSeeds = {0x13579bdfu, 0xa5c31f27u,
                                                0x729de081u};
    constexpr uint32_t kWorkgroupSize = 128;
    constexpr uint32_t kStageOutputWords = 2048;
    constexpr uint32_t kGuardWords = 16;
    constexpr uint32_t kPageBytes = 4096;
    constexpr uint32_t kArgumentStride = 256;
    constexpr uint32_t kTupleOffset = 1024;
    constexpr uint32_t kTupleStride = 16;
    constexpr uint32_t kCompletionOffset = 256;
    constexpr uint32_t kCompletionWord = kCompletionOffset / sizeof(uint32_t);
    constexpr uint32_t kTailGuard = 0x7c42a695u;
    constexpr uint32_t kPrefixGuard = 0x619b30d5u;
    constexpr uint32_t kSuffixGuard = 0xe270c84bu;
    constexpr uint32_t kControlGuard = 0x68d329b7u;

    amdf_gpu_endpoint_info_t endpoint_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint_info)};
    ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    const std::array<const kernels::Kernel*, 2> selected_kernels = {
        kernels::lds_exchange::kKernels.Find(endpoint_info),
        kernels::lds_exchange_wave64::kKernels.Find(endpoint_info)};
    std::array<Pm4ComputeProgram, 2> programs = {};
    std::array<GpuMemory*, 2> code = {};
    std::array<std::vector<uint8_t>, 2> expected_code;
    std::array<std::vector<uint8_t>, 2> observed_code;
    for (size_t mode = 0; mode < selected_kernels.size(); ++mode) {
      ASSERT_NE(selected_kernels[mode], nullptr)
          << "missing compiled LDS product for legal wave mode " << mode;
      const auto& kernel = *selected_kernels[mode];
      ASSERT_EQ(kernel.wavefront_size, mode == 0 ? 32u : 64u);
      ASSERT_EQ(kernel.workgroup_size(), kWorkgroupSize);
      ASSERT_LE(kernel.group_segment_byte_length,
                endpoint_info.compute.local_data_share_byte_length);
      programs[mode] = {0,
                        kernel.program.resource1,
                        kernel.program.resource2,
                        kernel.program.resource3,
                        kernel.group_segment_byte_length,
                        kernel.wavefront_size,
                        {kWorkgroupSize, 1, 1}};
      const std::string prefix = "wave" + std::to_string(kernel.wavefront_size);
      ASSERT_NO_FATAL_FAILURE(
          PrepareProgram(kernel.executable, kernel.entry_byte_offset,
                         &programs[mode], prefix.c_str(), &code[mode]));
      RecordProperty(prefix + "_target", kernel.target);
      expected_code[mode].resize(code[mode]->info.byte_length, 0);
      observed_code[mode].resize(code[mode]->info.byte_length);
      std::memcpy(expected_code[mode].data(), kernel.executable.words,
                  kernel.executable.byte_length);
    }

    GpuMemory* output = nullptr;
    GpuMemory* inputs = nullptr;
    GpuMemory* completion = nullptr;
    std::array<uint32_t, kStageOutputWords * kGroups.size()> expected_output;
    std::array<uint32_t, kStageOutputWords * kGroups.size()> observed_output;
    std::array<uint8_t, kPageBytes> expected_inputs;
    std::array<uint8_t, kPageBytes> observed_inputs;
    std::array<uint32_t, kPageBytes / sizeof(uint32_t)> expected_control;
    std::array<uint32_t, kPageBytes / sizeof(uint32_t)> observed_control;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     sizeof(expected_output), &output));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageBytes, &inputs));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kPageBytes, &completion));
    for (const auto* kernel : selected_kernels) {
      ASSERT_EQ(inputs->device_address % kernel->arguments.alignment, 0u);
      ASSERT_EQ(kArgumentStride % kernel->arguments.alignment, 0u);
    }
    ASSERT_EQ((inputs->device_address + kTupleOffset) % sizeof(uint32_t), 0u);
    ASSERT_EQ(
        (completion->device_address + kCompletionOffset) % sizeof(uint32_t),
        0u);
    expected_output.fill(kTailGuard);
    expected_inputs.fill(0xab);
    expected_control.fill(kControlGuard);
    expected_control[kCompletionWord] = 1;
    observed_control = expected_control;
    observed_control[kCompletionWord] = 0;
    std::memcpy(completion->host.pointer, observed_control.data(),
                sizeof(observed_control));

    for (size_t stage = 0; stage < kGroups.size(); ++stage) {
      const uint32_t first_word = stage * kStageOutputWords;
      const uint32_t workitem_count = kGroups[stage] * kWorkgroupSize;
      for (uint32_t guard = 0; guard < kGuardWords; ++guard) {
        expected_output[first_word + guard] = kPrefixGuard;
        expected_output[first_word + kGuardWords + workitem_count * 2 + guard] =
            kSuffixGuard;
      }
      for (uint32_t workitem = 0; workitem < workitem_count; ++workitem) {
        const auto record =
            kernels::lds_exchange::ExpectedRecord(workitem, kSeeds[stage]);
        const uint32_t word = first_word + kGuardWords + workitem * 2;
        expected_output[word] = record[0];
        expected_output[word + 1] = record[1];
      }
      const Arguments arguments = {
          output->device_address +
              (first_word + kGuardWords) * sizeof(uint32_t),
          kSeeds[stage]};
      std::memset(expected_inputs.data() + stage * kArgumentStride, 0,
                  sizeof(Arguments));
      std::memcpy(expected_inputs.data() + stage * kArgumentStride, &arguments,
                  kernels::lds_exchange::kArgumentByteLength);
      const std::array<uint32_t, 3> tuple = {kGroups[stage], 1, 1};
      std::memcpy(expected_inputs.data() + kTupleOffset + stage * kTupleStride,
                  tuple.data(), sizeof(tuple));
    }
    observed_output = expected_output;
    for (size_t stage = 0; stage < kGroups.size(); ++stage) {
      for (uint32_t word = 0; word < kGroups[stage] * kWorkgroupSize * 2;
           ++word) {
        observed_output[stage * kStageOutputWords + kGuardWords + word] ^=
            UINT32_MAX;
      }
    }
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    std::memcpy(inputs->host.pointer, expected_inputs.data(),
                sizeof(expected_inputs));

    GpuCommandQueue* queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
    std::array<uint32_t, 256> expected_commands = {};
    std::array<uint32_t, 256> observed_commands = {};
    ASSERT_GT(queue->words().size(), expected_commands.size());
    Pm4CommandWriter commands(expected_commands.data(), *pm4_profile_);
    commands.SystemBarrier();
    for (size_t stage = 0; stage < kGroups.size(); ++stage) {
      const auto& program = programs[stage == 1 ? 1 : 0];
      commands.BindCompute(program,
                           inputs->device_address + stage * kArgumentStride);
      if (launch_kind == LaunchKind::kDirect) {
        commands.Dispatch(program, kGroups[stage] * kWorkgroupSize, 1, 1);
      } else {
        commands.DispatchIndirect(
            program,
            inputs->device_address + kTupleOffset + stage * kTupleStride);
      }
      // Each complete dispatch drains before the next binding replaces its
      // resources. No CPU publication or wait advances the intermediate stages.
      commands.SystemBarrier();
      const std::string prefix = "stage_" + std::to_string(stage);
      RecordProperty(prefix + "_wavefront_size", program.wavefront_size);
      RecordProperty(prefix + "_workgroup_count", kGroups[stage]);
      RecordProperty(prefix + "_seed", std::to_string(kSeeds[stage]));
    }
    commands.WriteData32(completion->device_address + kCompletionOffset, 1);
    commands.PadToEightWords();
    ASSERT_LE(commands.word_count(), expected_commands.size());
    const size_t command_bytes = commands.word_count() * sizeof(uint32_t);
    std::memcpy(queue->words().data(), expected_commands.data(), command_bytes);
    RecordProperty("wave_mode_launch",
                   launch_kind == LaunchKind::kDirect ? "direct" : "indirect");
    RecordProperty("wave_mode_command_words",
                   std::to_string(commands.word_count()));
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer) +
            kCompletionOffset,
        1);

    // Snapshot before retirement or diagnostics can add another visibility
    // edge. Immutable inputs and commands share the same final-use boundary.
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_inputs.data(), inputs->host.pointer,
                sizeof(observed_inputs));
    std::memcpy(observed_control.data(), completion->host.pointer,
                sizeof(observed_control));
    std::memcpy(observed_commands.data(), queue->words().data(), command_bytes);
    for (size_t mode = 0; mode < code.size(); ++mode) {
      std::memcpy(observed_code[mode].data(), code[mode]->host.pointer,
                  observed_code[mode].size());
    }
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    EXPECT_EQ(observed_output, expected_output);
    EXPECT_EQ(observed_inputs, expected_inputs);
    EXPECT_EQ(observed_control, expected_control);
    EXPECT_EQ(observed_commands, expected_commands);
    EXPECT_EQ(observed_code, expected_code);
    if (!HasFailure()) {
      RecordProperty("wave_mode_completed_dispatches", kGroups.size());
    }
  }
};

TEST_F(Pm4WaveModeTest, DirectLdsDispatchRestoresWave32AfterWave64) {
  Execute(LaunchKind::kDirect);
}

TEST_F(Pm4WaveModeTest, IndirectLdsDispatchRestoresWave32AfterWave64) {
  Execute(LaunchKind::kIndirect);
}

}  // namespace
