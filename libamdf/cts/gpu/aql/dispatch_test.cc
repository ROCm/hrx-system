// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

TEST_F(AqlDispatchTest, CoherentSystemPayloadChangesAcrossEpochs) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kGuard = 0x759bf13du;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kAddends = {7, 0x80000023u};
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected;
    std::array<uint32_t, kWordCount> download;
    std::array<uint32_t, kWordCount> unchanged_input;
    upload.fill(kGuard);
    expected.fill(kGuard);
    download.fill(kGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      upload[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        // The CPU oracle uses wider arithmetic, then applies uint32 wrapping.
        const uint32_t result =
            static_cast<uint32_t>(uint64_t{value} * 3 + kAddends[epoch]);
        expected[kPayloadOffset + i] = result;
        download[kPayloadOffset + i] = ~result;
      }
    }
    std::memcpy(input->host.pointer, upload.data(), sizeof(upload));
    std::memcpy(output->host.pointer, download.data(), sizeof(download));
    const kernels::transform::Arguments payload = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kAddends[epoch],
    };
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &payload,
                kernel.arguments.byte_length);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1,
                       {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
                       {kGridSize, 1, 1}},
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address, completion->device_address);
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    aql::Publish(*queue, index++, packet);
    // Snapshot completion-visible data before diagnostics or ring retirement.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(download.data(), output->host.pointer, sizeof(download));
    std::memcpy(unchanged_input.data(), input->host.pointer,
                sizeof(unchanged_input));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(download[i], expected[i]) << "epoch=" << epoch << " word=" << i;
      EXPECT_EQ(unchanged_input[i], upload[i])
          << "epoch=" << epoch << " word=" << i;
    }
    // Retire even after a failed oracle, before any next-epoch storage reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_payload_completed_epochs", kCounts.size());
}

TEST_F(AqlDispatchTest,
       HeaderBarrierOrdersProducerConsumerBeforeTrailingCompletion) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kInputGuard = 0x759bf13du;
  constexpr uint32_t kIntermediateGuard = 0xa36cf197u;
  constexpr uint32_t kOutputGuard = 0x4e90b725u;
  constexpr uint32_t kCompletionGuard = 0x68d329b7u;
  constexpr uint32_t kCompletionByteLength = 4096;
  constexpr uint32_t kCompletionGuardWordCount =
      (kCompletionByteLength - sizeof(aql::Signal)) / sizeof(uint32_t);
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint64_t kPacketsPerEpoch = 3;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kProducerAddends = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {0x101u, 0x2468ace1u};
  const aql::DispatchGeometry kGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kGridSize, 1, 1}};
  constexpr aql::FenceScopes kScopes = {aql::FenceScope::kSystem,
                                        aql::FenceScope::kSystem};

  GpuMemory* input = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kCompletionByteLength, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;
  // Native signal fields retain their ABI meaning. Guards begin after the
  // complete 64-byte block, leaving its unused fields zero-initialized.
  auto* completion_guard_address =
      static_cast<uint8_t*>(completion->host.pointer) + sizeof(aql::Signal);
  std::array<uint32_t, kCompletionGuardWordCount> completion_words;
  completion_words.fill(kCompletionGuard);
  std::memcpy(completion_guard_address, completion_words.data(),
              sizeof(completion_words));

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));
  const uint64_t first_work_packet_index = index;
  // Keep both epochs in distinct resident slots after the completed cold
  // publication. The operational frontier comes from that helper.
  ASSERT_GE(queue->host.ring_byte_length / sizeof(aql::Packet),
            index + kPacketsPerEpoch * kCounts.size());

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected_intermediate;
    std::array<uint32_t, kWordCount> expected_output;
    std::array<uint32_t, kWordCount> input_words;
    std::array<uint32_t, kWordCount> intermediate_words;
    std::array<uint32_t, kWordCount> output_words;
    upload.fill(kInputGuard);
    expected_intermediate.fill(kIntermediateGuard);
    expected_output.fill(kOutputGuard);
    intermediate_words.fill(kIntermediateGuard);
    output_words.fill(kOutputGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      upload[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        const uint32_t intermediate_result = static_cast<uint32_t>(
            uint64_t{value} * 3 + kProducerAddends[epoch]);
        // Flatten both transforms with wide CPU arithmetic. The observed
        // intermediate never participates in the final output oracle.
        const uint32_t output_result = static_cast<uint32_t>(
            uint64_t{value} * 9 + uint64_t{kProducerAddends[epoch]} * 3 +
            kConsumerAddends[epoch]);
        expected_intermediate[kPayloadOffset + i] = intermediate_result;
        expected_output[kPayloadOffset + i] = output_result;
        intermediate_words[kPayloadOffset + i] = ~intermediate_result;
        output_words[kPayloadOffset + i] = ~output_result;
      }
    }
    std::memcpy(input->host.pointer, upload.data(), sizeof(upload));
    std::memcpy(intermediate->host.pointer, intermediate_words.data(),
                sizeof(intermediate_words));
    std::memcpy(output->host.pointer, output_words.data(),
                sizeof(output_words));
    const kernels::transform::Arguments producer_payload = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kProducerAddends[epoch],
    };
    const kernels::transform::Arguments consumer_payload = {
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kConsumerAddends[epoch],
    };
    // Both disjoint records keep zero padding and remain immutable through
    // terminal completion and consumption. The compiler consumes 24 bytes.
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &producer_payload,
                kernel.arguments.byte_length);
    std::memcpy(
        static_cast<uint8_t*>(arguments->host.pointer) + kArgumentStride,
        &consumer_payload, kernel.arguments.byte_length);
    signal.value = 1;
    const auto produce = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kGeometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        descriptor_address, arguments->device_address, 0, kScopes);
    const auto consume =
        aql::Dispatch(aql::HeaderBarrier::kEnabled, kGeometry,
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address + kArgumentStride, 0, kScopes);
    const auto complete =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                     completion->device_address, {}, kScopes);

    // Reserve only after constructing every packet. The consumer's header
    // joins producer completion; null dispatch completions keep their fences.
    GpuStoreRelease(queue->host.write_index_address, index + kPacketsPerEpoch);
    aql::Publish(*queue, index++, produce);
    aql::Publish(*queue, index++, consume);
    aql::Publish(*queue, index++, complete);

    // The terminal header independently joins both dispatches, even if the
    // middle edge produced wrong data. Snapshot before any diagnostic or
    // consumption operation can add synchronization to these observations.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(output_words.data(), output->host.pointer,
                sizeof(output_words));
    std::memcpy(intermediate_words.data(), intermediate->host.pointer,
                sizeof(intermediate_words));
    std::memcpy(input_words.data(), input->host.pointer, sizeof(input_words));
    aql::Signal completed_signal = {};
    std::memcpy(&completed_signal, completion->host.pointer,
                sizeof(completed_signal));
    std::memcpy(completion_words.data(), completion_guard_address,
                sizeof(completion_words));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(output_words[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(intermediate_words[i], expected_intermediate[i])
          << "intermediate word=" << i;
      EXPECT_EQ(input_words[i], upload[i]) << "input word=" << i;
    }
    EXPECT_EQ(completed_signal.kind, 1);
    EXPECT_EQ(completed_signal.value, 0);
    for (uint32_t i = 0; i < completion_words.size(); ++i) {
      EXPECT_EQ(completion_words[i], kCompletionGuard)
          << "completion guard word=" << i;
    }
    // Retire even after an oracle failure. Signal rearming and all backing
    // reuse require successful observations and a consumed frontier.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_chain_completed_epochs", kCounts.size());
  RecordProperty("aql_chain_work_packet_count",
                 std::to_string(index - first_work_packet_index));
  RecordProperty("aql_chain_final_packet_index", std::to_string(index));
}

}  // namespace
