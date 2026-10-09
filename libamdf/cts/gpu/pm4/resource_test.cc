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
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"

namespace {

static_assert(sizeof(kernels::lds_exchange::Arguments) <= 32);

TEST_F(Pm4DispatchTest, SwitchesBetweenTransformAndLdsKernels) {
  const auto* transform_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(transform_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& transform = *transform_product;
  RecordProperty("transform_kernel_target", transform.target);

  constexpr uint32_t kTransformGridSize = 1024;
  constexpr uint32_t kLdsGridSize = 512;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kControlWordCount = kPageByteLength / sizeof(uint32_t);
  constexpr uint32_t kCompletionByteOffset = 256;
  constexpr uint32_t kCompletionWordIndex =
      kCompletionByteOffset / sizeof(uint32_t);
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint32_t kCommandWordsPerEpoch = 144;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kFirstAddends = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kSeeds = {0x13579bdfu, 0xa5c31f27u};
  constexpr std::array<uint32_t, 2> kLastAddends = {11, 0x10203045u};
  static_assert(kLdsGridSize * 2 == kTransformGridSize);

  amdf_gpu_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info)};
  ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  const auto* selected = kernels::lds_exchange::kKernels.Find(endpoint_info);
  ASSERT_NE(selected, nullptr) << "missing compiled LDS kernel for endpoint";
  const auto& lds = *selected;
  RecordProperty("lds_kernel_target", lds.target);

  ASSERT_GE(endpoint_info.compute.local_data_share_byte_length,
            lds.group_segment_byte_length);

  GpuMemory* input = nullptr;
  GpuMemory* transform_output = nullptr;
  GpuMemory* lds_output = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  GpuMemory* transform_code = nullptr;
  GpuMemory* lds_code = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &transform_output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &lds_output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &completion));
  ASSERT_EQ(arguments->device_address % transform.arguments.alignment, 0u);
  ASSERT_EQ(arguments->device_address % lds.arguments.alignment, 0u);
  ASSERT_EQ(completion->device_address % sizeof(uint32_t), 0u);
  Pm4ComputeProgram transform_program = {0,
                                         transform.program.resource1,
                                         transform.program.resource2,
                                         transform.program.resource3,
                                         transform.group_segment_byte_length,
                                         transform.wavefront_size,
                                         {transform.workgroup_size(), 1, 1}};
  Pm4ComputeProgram lds_program = {0,
                                   lds.program.resource1,
                                   lds.program.resource2,
                                   lds.program.resource3,
                                   lds.group_segment_byte_length,
                                   lds.wavefront_size,
                                   {lds.workgroup_size(), 1, 1}};
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      transform.executable, transform.entry_byte_offset, &transform_program,
      "pm4_mixed_transform", &transform_code));
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(lds.executable, lds.entry_byte_offset,
                                         &lds_program, "pm4_mixed_lds",
                                         &lds_code));

  std::array<uint32_t, kWordCount> expected_input, observed_input;
  std::array<uint32_t, kWordCount> expected_transform, observed_transform;
  std::array<uint32_t, kWordCount> expected_lds, observed_lds;
  std::array<uint32_t, kWordCount> expected_output, observed_output;
  std::array<uint8_t, kPageByteLength> expected_arguments, observed_arguments;
  std::array<uint32_t, kControlWordCount> expected_control, observed_control;
  std::vector<uint8_t> expected_transform_code(transform_code->info.byte_length,
                                               0);
  std::vector<uint8_t> expected_lds_code(lds_code->info.byte_length, 0);
  std::vector<uint8_t> observed_transform_code(
      transform_code->info.byte_length);
  std::vector<uint8_t> observed_lds_code(lds_code->info.byte_length);
  std::memcpy(expected_transform_code.data(), transform.executable.words,
              transform.executable.byte_length);
  std::memcpy(expected_lds_code.data(), lds.executable.words,
              lds.executable.byte_length);
  expected_control.fill(0x68d329b7u);
  expected_control[kCompletionWordIndex] = 0;
  // Only the GPU advances this word after its once-only initialization.
  std::memcpy(completion->host.pointer, expected_control.data(),
              sizeof(expected_control));
  const uintptr_t completion_address =
      reinterpret_cast<uintptr_t>(completion->host.pointer) +
      kCompletionByteOffset;

  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  const uint64_t command_capacity = queue->words().size();
  ASSERT_GT(command_capacity, kCounts.size() * kCommandWordsPerEpoch);
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  RecordProperty("pm4_mixed_program_sequence", "transform,lds,transform");
  RecordProperty(
      "pm4_mixed_bound_rsrc2_sequence",
      std::to_string(transform.program.resource2) + "," +
          std::to_string(lds.program.resource2 |
                         ((lds.group_segment_byte_length / 512u) << 15)) +
          "," + std::to_string(transform.program.resource2));
  RecordProperty("pm4_mixed_bound_rsrc3_sequence",
                 std::to_string(transform.program.resource3) + "," +
                     std::to_string(lds.program.resource3) + "," +
                     std::to_string(transform.program.resource3));
  RecordProperty("pm4_mixed_resource_limits_sequence", "0,0x00400000,0");
  RecordProperty("pm4_mixed_workgroup_sequence", "64x1x1,128x1x1,64x1x1");
  RecordProperty("pm4_mixed_grid_sequence", "1024x1x1,512x1x1,1024x1x1");
  RecordProperty("pm4_mixed_group_byte_length_sequence", "0,512,0");
  RecordProperty("pm4_mixed_argument_offsets", "0,64,128");
  RecordProperty(
      "pm4_mixed_kernarg_semantic_byte_lengths",
      std::to_string(transform.arguments.byte_length) + "," +
          std::to_string(kernels::lds_exchange::kArgumentByteLength) + "," +
          std::to_string(transform.arguments.byte_length));
  RecordProperty("pm4_mixed_kernarg_slot_byte_length", 32);
  RecordProperty(
      "pm4_mixed_lds_capacity_per_compute_unit",
      std::to_string(endpoint_info.compute.local_data_share_byte_length));
  RecordProperty("pm4_mixed_payload_offset_words", kPayloadOffset);
  RecordProperty("pm4_mixed_last_transform_count", kTransformGridSize);
  RecordProperty("pm4_mixed_checked_data_bytes_each", sizeof(expected_input));
  RecordProperty("pm4_mixed_checked_argument_bytes",
                 sizeof(expected_arguments));
  RecordProperty("pm4_mixed_checked_control_bytes", sizeof(expected_control));
  RecordProperty("pm4_mixed_checked_transform_code_bytes",
                 std::to_string(expected_transform_code.size()));
  RecordProperty("pm4_mixed_checked_lds_code_bytes",
                 std::to_string(expected_lds_code.size()));
  RecordProperty("pm4_mixed_completion_byte_offset", kCompletionByteOffset);
  RecordProperty("pm4_mixed_command_words_per_epoch", kCommandWordsPerEpoch);
  RecordProperty("pm4_mixed_first_published_word_count",
                 std::to_string(commands.word_count()));
  RecordProperty("pm4_mixed_command_capacity_dwords",
                 std::to_string(command_capacity));
  RecordProperty("pm4_mixed_completed_epochs", 0);
  RecordProperty("pm4_mixed_completed_dispatches", 0);

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    // Role and position distinguish prefix, suffix, and inactive tail words.
    for (uint32_t i = 0; i < kWordCount; ++i) {
      expected_input[i] = 0x759bf13du ^ i;
      expected_transform[i] = 0xa36cf197u ^ i;
      expected_lds[i] = 0x619b30d5u ^ i;
      expected_output[i] = 0x4e90b725u ^ i;
    }
    for (uint32_t i = 0; i < kTransformGridSize; ++i) {
      const uint32_t value = static_cast<uint32_t>(
          UINT64_C(0xfffffff0) + uint64_t{i} * 0x01030507u +
          uint64_t{epoch} * 0x11111111u);
      expected_input[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        expected_transform[kPayloadOffset + i] =
            static_cast<uint32_t>(uint64_t{value} * 3 + kFirstAddends[epoch]);
      }
    }
    for (uint32_t i = 0; i < kLdsGridSize; ++i) {
      const auto record =
          kernels::lds_exchange::ExpectedRecord(i, kSeeds[epoch]);
      expected_lds[kPayloadOffset + i * 2] = record[0];
      expected_lds[kPayloadOffset + i * 2 + 1] = record[1];
    }
    observed_transform = expected_transform;
    observed_lds = expected_lds;
    for (uint32_t i = 0; i < kTransformGridSize; ++i) {
      const uint32_t position = kPayloadOffset + i;
      // The final oracle consumes expected LDS tokens, never device results.
      expected_output[position] = static_cast<uint32_t>(
          uint64_t{expected_lds[position]} * 3 + kLastAddends[epoch]);
      if (i < kCounts[epoch]) {
        observed_transform[position] = ~expected_transform[position];
      }
      observed_lds[position] = ~expected_lds[position];
    }
    observed_output = expected_output;
    for (uint32_t i = 0; i < kTransformGridSize; ++i) {
      observed_output[kPayloadOffset + i] =
          ~expected_output[kPayloadOffset + i];
    }
    std::memcpy(input->host.pointer, expected_input.data(),
                sizeof(expected_input));
    std::memcpy(transform_output->host.pointer, observed_transform.data(),
                sizeof(observed_transform));
    std::memcpy(lds_output->host.pointer, observed_lds.data(),
                sizeof(observed_lds));
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    const kernels::transform::Arguments first_arguments = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        transform_output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch], kFirstAddends[epoch]};
    const kernels::lds_exchange::Arguments lds_arguments = {
        lds_output->device_address + kPayloadOffset * sizeof(uint32_t),
        kSeeds[epoch]};
    const kernels::transform::Arguments last_arguments = {
        lds_output->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kTransformGridSize, kLastAddends[epoch]};
    // All scalar fetches have initialized backing; C++ tail padding is not
    // copied. The three records remain immutable through final retirement.
    expected_arguments.fill(0);
    std::memcpy(expected_arguments.data(), &first_arguments,
                transform.arguments.byte_length);
    std::memcpy(expected_arguments.data() + kArgumentStride, &lds_arguments,
                kernels::lds_exchange::kArgumentByteLength);
    std::memcpy(expected_arguments.data() + kArgumentStride * 2,
                &last_arguments, transform.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                sizeof(expected_arguments));
    expected_control[kCompletionWordIndex] = epoch + 1;

    commands.SystemBarrier();
    commands.BindCompute(transform_program, arguments->device_address);
    commands.Dispatch(transform_program, kTransformGridSize, 1, 1);
    // T0 -> L changes bindings after T0 completes; L has no input buffer.
    commands.SystemBarrier();
    commands.BindCompute(lds_program,
                         arguments->device_address + kArgumentStride);
    commands.Dispatch(lds_program, kLdsGridSize, 1, 1);
    // L -> T1 is the payload dependency. Restore T's full ordinary binding.
    commands.SystemBarrier();
    commands.BindCompute(transform_program,
                         arguments->device_address + kArgumentStride * 2);
    commands.Dispatch(transform_program, kTransformGridSize, 1, 1);
    // This independent drain owns all shader lifetimes, even on a bad payload.
    commands.SystemBarrier();
    commands.WriteData32(completion->device_address + kCompletionByteOffset,
                         epoch + 1);
    commands.PadToEightWords();
    ASSERT_EQ(commands.word_count(), (epoch + 1) * kCommandWordsPerEpoch);
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    GpuWaitEqual<uint32_t>(completion_address, epoch + 1);

    // Snapshot the final consumer first, then every other complete initialized
    // extent before diagnostics or command-retirement queries can intervene.
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_lds.data(), lds_output->host.pointer,
                sizeof(observed_lds));
    std::memcpy(observed_transform.data(), transform_output->host.pointer,
                sizeof(observed_transform));
    std::memcpy(observed_input.data(), input->host.pointer,
                sizeof(observed_input));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_control.data(), completion->host.pointer,
                sizeof(observed_control));
    std::memcpy(observed_transform_code.data(), transform_code->host.pointer,
                observed_transform_code.size());
    std::memcpy(observed_lds_code.data(), lds_code->host.pointer,
                observed_lds_code.size());
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(observed_output[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(observed_lds[i], expected_lds[i]) << "LDS output word=" << i;
      EXPECT_EQ(observed_transform[i], expected_transform[i])
          << "first transform word=" << i;
      EXPECT_EQ(observed_input[i], expected_input[i]) << "input word=" << i;
    }
    for (uint32_t i = 0; i < kPageByteLength; ++i) {
      EXPECT_EQ(observed_arguments[i], expected_arguments[i])
          << "argument byte=" << i;
    }
    for (size_t i = 0; i < expected_transform_code.size(); ++i) {
      EXPECT_EQ(observed_transform_code[i], expected_transform_code[i])
          << "transform code byte=" << i;
    }
    for (size_t i = 0; i < expected_lds_code.size(); ++i) {
      EXPECT_EQ(observed_lds_code[i], expected_lds_code[i])
          << "LDS code byte=" << i;
    }
    for (uint32_t i = 0; i < kControlWordCount; ++i) {
      EXPECT_EQ(observed_control[i], expected_control[i])
          << "control word=" << i;
    }
    // Oracle failures still retire the stream and stop before any owner reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "pm4_mixed_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_first_count", kCounts[epoch]);
    RecordProperty(prefix + "_first_addend",
                   std::to_string(kFirstAddends[epoch]));
    RecordProperty(prefix + "_lds_seed", std::to_string(kSeeds[epoch]));
    RecordProperty(prefix + "_last_addend",
                   std::to_string(kLastAddends[epoch]));
    RecordProperty(prefix + "_published_word_count",
                   std::to_string(commands.word_count()));
    RecordProperty("pm4_mixed_completed_epochs", epoch + 1);
    RecordProperty("pm4_mixed_completed_dispatches", (epoch + 1) * 3);
  }
  RecordProperty("pm4_mixed_final_published_word_count",
                 std::to_string(commands.word_count()));
  RecordProperty("pm4_mixed_command_word_count",
                 std::to_string(commands.word_count()));
}

}  // namespace
