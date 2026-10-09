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
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"

namespace {

using Arguments = kernels::lds_exchange::Arguments;

using Pm4LdsTest = Pm4DispatchTest;

TEST_F(Pm4LdsTest, StaticGroupMemoryExchangesAcrossWaves) {
  constexpr std::array<uint32_t, 2> kSeeds = {0x13579bdfu, 0xa5c31f27u};
  constexpr uint32_t kGridSize = 512;
  constexpr uint32_t kPayloadWordCount = kGridSize * 2;
  constexpr uint32_t kOutputWordCount = 2048;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kControlWordCount = kPageByteLength / sizeof(uint32_t);
  constexpr uint32_t kCompletionByteOffset = 256;
  constexpr uint32_t kCompletionWordIndex =
      kCompletionByteOffset / sizeof(uint32_t);
  constexpr uint32_t kCommandWordCountPerEpoch = 64;
  constexpr uint32_t kPrefixGuard = 0x619b30d5u;
  constexpr uint32_t kSuffixGuard = 0xe270c84bu;
  constexpr uint32_t kTailGuard = 0x7c42a695u;
  constexpr uint32_t kControlGuard = 0x68d329b7u;

  amdf_gpu_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info)};
  ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  const auto* selected = kernels::lds_exchange::kKernels.Find(endpoint_info);
  ASSERT_NE(selected, nullptr) << "missing compiled LDS kernel for endpoint";
  const auto& kernel = *selected;
  RecordProperty("lds_kernel_target", kernel.target);

  ASSERT_LE(kernel.group_segment_byte_length,
            endpoint_info.compute.local_data_share_byte_length);

  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  GpuMemory* code = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   uint64_t{kOutputWordCount} * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(completion->device_address % sizeof(uint32_t), 0u);
  Pm4ComputeProgram program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {kernel.workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel.executable, kernel.entry_byte_offset, &program, "pm4", &code));

  std::array<uint32_t, kOutputWordCount> expected_output;
  std::array<uint32_t, kOutputWordCount> observed_output;
  std::array<uint8_t, kPageByteLength> expected_arguments;
  std::array<uint8_t, kPageByteLength> observed_arguments;
  std::array<uint32_t, kControlWordCount> expected_control;
  std::array<uint32_t, kControlWordCount> observed_control;
  std::vector<uint8_t> expected_code(code->info.byte_length, 0);
  std::vector<uint8_t> observed_code(code->info.byte_length);
  std::memcpy(expected_code.data(), kernel.executable.words,
              kernel.executable.byte_length);
  expected_control.fill(kControlGuard);
  expected_control[kCompletionWordIndex] = 0;
  // Initialize the whole page once. Only GPU writes advance its epoch word.
  std::memcpy(completion->host.pointer, expected_control.data(),
              sizeof(expected_control));
  const uintptr_t completion_address =
      reinterpret_cast<uintptr_t>(completion->host.pointer) +
      kCompletionByteOffset;

  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  const uint64_t command_capacity = queue->words().size();
  // Each 56-word sequence has a complete eight-word NOP. All batches occupy
  // distinct resident command ranges.
  ASSERT_GE(command_capacity, kCommandWordCountPerEpoch * kSeeds.size());
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  RecordProperty(
      "pm4_lds_capacity_per_compute_unit",
      std::to_string(endpoint_info.compute.local_data_share_byte_length));
  RecordProperty("pm4_lds_fixed_group_byte_length",
                 kernel.group_segment_byte_length);
  // PAL/Mesa's four-wave SIMD destination policy remains fixed across rows.
  RecordProperty("pm4_lds_compute_resource_limits", 0x00400000);
  RecordProperty("pm4_lds_kernarg_semantic_byte_length",
                 kernels::lds_exchange::kArgumentByteLength);
  RecordProperty("pm4_lds_kernarg_slot_byte_length", sizeof(Arguments));
  RecordProperty("pm4_lds_workgroup_size", kernel.workgroup_size());
  RecordProperty("pm4_lds_wavefront_size", kernel.wavefront_size);
  RecordProperty("pm4_lds_reported_wavefront_size",
                 endpoint_info.compute.wavefront_size);
  RecordProperty("pm4_lds_waves_per_workgroup",
                 kernel.workgroup_size() / kernel.wavefront_size);
  RecordProperty("pm4_lds_grid_size", kGridSize);
  RecordProperty("pm4_lds_output_words_per_epoch", kPayloadWordCount);
  RecordProperty("pm4_lds_checked_output_byte_length", sizeof(expected_output));
  RecordProperty("pm4_lds_checked_argument_byte_length",
                 sizeof(expected_arguments));
  RecordProperty("pm4_lds_checked_control_byte_length",
                 sizeof(expected_control));
  RecordProperty("pm4_lds_checked_code_byte_length",
                 std::to_string(expected_code.size()));
  RecordProperty("pm4_lds_completion_byte_offset", kCompletionByteOffset);
  RecordProperty("pm4_lds_command_word_count_per_epoch",
                 kCommandWordCountPerEpoch);
  RecordProperty("pm4_lds_first_published_word_count",
                 std::to_string(commands.word_count()));
  RecordProperty("pm4_lds_command_capacity_dwords",
                 std::to_string(command_capacity));

  for (uint32_t epoch = 0; epoch < kSeeds.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    expected_output.fill(kTailGuard);
    for (uint32_t word = 0; word < kGuardWordCount; ++word) {
      expected_output[word] = kPrefixGuard;
      expected_output[kGuardWordCount + kPayloadWordCount + word] =
          kSuffixGuard;
    }
    for (uint32_t workitem = 0; workitem < kGridSize; ++workitem) {
      // Division/remainder, +/-64 and wider arithmetic independently derive
      // the other wave's token and the global workitem's position stamp.
      const auto record =
          kernels::lds_exchange::ExpectedRecord(workitem, kSeeds[epoch]);
      const uint32_t position = kGuardWordCount + workitem * 2;
      expected_output[position] = record[0];
      expected_output[position + 1] = record[1];
    }
    observed_output = expected_output;
    for (uint32_t word = 0; word < kPayloadWordCount; ++word) {
      observed_output[kGuardWordCount + word] =
          ~expected_output[kGuardWordCount + word];
    }
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    const Arguments payload = {
        output->device_address + kGuardWordCount * sizeof(uint32_t),
        kSeeds[epoch],
    };
    // The full argument page is initialized; only semantic bytes come from
    // typed fields, and all bytes beyond the argument record remain zero.
    expected_arguments.fill(0);
    std::memcpy(expected_arguments.data(), &payload,
                kernels::lds_exchange::kArgumentByteLength);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                sizeof(expected_arguments));
    expected_control[kCompletionWordIndex] = epoch + 1;

    commands.SystemBarrier();
    commands.BindCompute(program, arguments->device_address);
    commands.Dispatch(program, kGridSize, 1, 1);
    commands.SystemBarrier();
    commands.WriteData32(completion->device_address + kCompletionByteOffset,
                         epoch + 1);
    commands.PadToEightWords();
    ASSERT_EQ(commands.word_count(), (epoch + 1) * kCommandWordCountPerEpoch);
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    GpuWaitEqual<uint32_t>(completion_address, epoch + 1);

    // Capture every initialized extent before diagnostics or command storage
    // retirement can add synchronization to these completion-visible
    // observations.
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_control.data(), completion->host.pointer,
                sizeof(observed_control));
    std::memcpy(observed_code.data(), code->host.pointer, observed_code.size());
    for (uint32_t word = 0; word < kOutputWordCount; ++word) {
      EXPECT_EQ(observed_output[word], expected_output[word])
          << "output word=" << word;
    }
    for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
      EXPECT_EQ(observed_arguments[byte], expected_arguments[byte])
          << "argument byte=" << byte;
    }
    for (size_t byte = 0; byte < expected_code.size(); ++byte) {
      EXPECT_EQ(observed_code[byte], expected_code[byte])
          << "code byte=" << byte;
    }
    for (uint32_t word = 0; word < kControlWordCount; ++word) {
      EXPECT_EQ(observed_control[word], expected_control[word])
          << "control word=" << word;
    }
    // An oracle failure still retires the whole stream before stopping. The
    // immutable image and all other backing remain owned through teardown.
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "pm4_lds_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_seed", std::to_string(kSeeds[epoch]));
    RecordProperty(prefix + "_group_byte_length",
                   program.group_segment_byte_length);
    RecordProperty(prefix + "_bound_compute_pgm_rsrc2",
                   kernel.program.resource2 |
                       ((program.group_segment_byte_length / 512u) << 15));
    RecordProperty(prefix + "_published_word_count",
                   std::to_string(commands.word_count()));
  }
  RecordProperty("pm4_lds_completed_epochs", kSeeds.size());
  RecordProperty("pm4_lds_command_word_count",
                 std::to_string(commands.word_count()));
}

}  // namespace
