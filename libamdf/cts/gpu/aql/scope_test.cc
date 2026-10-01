// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

TEST_F(AqlDispatchTest, AgentScopeOrdersShaderPayloadBeforeSystemCompletion) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint64_t kPacketsPerEpoch = 3;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kProducerAddends = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {0x101u, 0x2468ace1u};
  const aql::DispatchGeometry kGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kGridSize, 1, 1}};
  constexpr aql::FenceScopes kProducerScopes = {aql::FenceScope::kSystem,
                                                aql::FenceScope::kAgent};
  constexpr aql::FenceScopes kConsumerScopes = {aql::FenceScope::kAgent,
                                                aql::FenceScope::kSystem};
  constexpr aql::FenceScopes kCompletionScopes = {aql::FenceScope::kNone,
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
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &completion));
  ASSERT_EQ(input->device_address % alignof(uint32_t), 0u);
  ASSERT_EQ(intermediate->device_address % alignof(uint32_t), 0u);
  ASSERT_EQ(output->device_address % alignof(uint32_t), 0u);
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(completion->device_address % alignof(aql::Signal), 0u);
  // Initialize the complete native block once. Reserved fields remain
  // native-owned; subsequent guard writes start outside the 64-byte ABI.
  std::memset(completion->host.pointer, 0, kPageByteLength);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(
      *queue, kernel, "aql_agent_scope_kernel", &index, &descriptor_address));
  const uint64_t first_work_packet_index = index;
  const uint64_t capacity = queue->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_GE(capacity, index + kPacketsPerEpoch * kCounts.size());

  // All three bodies precede reservation. Host-produced arguments enter
  // through SYSTEM; only the shader-to-shader dependency uses AGENT scopes.
  const auto produce = aql::Dispatch(
      aql::HeaderBarrier::kEnabled, kGeometry,
      kernel.private_segment_byte_length, kernel.group_segment_byte_length,
      descriptor_address, arguments->device_address, 0, kProducerScopes);
  const auto consume = aql::Dispatch(
      aql::HeaderBarrier::kEnabled, kGeometry,
      kernel.private_segment_byte_length, kernel.group_segment_byte_length,
      descriptor_address, arguments->device_address + kArgumentStride, 0,
      kConsumerScopes);
  const auto complete =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                   completion->device_address, {}, kCompletionScopes);

  RecordProperty("aql_agent_scope_kernel_entry_byte_offset",
                 kernel.entry_byte_offset);
  RecordProperty("aql_agent_scope_producer_acquire_scope",
                 static_cast<uint32_t>(kProducerScopes.acquire));
  RecordProperty("aql_agent_scope_producer_release_scope",
                 static_cast<uint32_t>(kProducerScopes.release));
  RecordProperty("aql_agent_scope_consumer_acquire_scope",
                 static_cast<uint32_t>(kConsumerScopes.acquire));
  RecordProperty("aql_agent_scope_consumer_release_scope",
                 static_cast<uint32_t>(kConsumerScopes.release));
  RecordProperty("aql_agent_scope_completion_acquire_scope",
                 static_cast<uint32_t>(kCompletionScopes.acquire));
  RecordProperty("aql_agent_scope_completion_release_scope",
                 static_cast<uint32_t>(kCompletionScopes.release));
  RecordProperty("aql_agent_scope_producer_header_setup", produce[0]);
  RecordProperty("aql_agent_scope_consumer_header_setup", consume[0]);
  RecordProperty("aql_agent_scope_completion_header_setup", complete[0]);
  RecordProperty("aql_agent_scope_grid_size", kGridSize);
  RecordProperty("aql_agent_scope_workgroup_size", kernel.workgroup_size());
  RecordProperty("aql_agent_scope_payload_word_offset", kPayloadOffset);
  RecordProperty("aql_agent_scope_payload_observed_byte_length",
                 kWordCount * sizeof(uint32_t));
  RecordProperty("aql_agent_scope_kernarg_byte_length",
                 kernel.arguments.byte_length);
  RecordProperty("aql_agent_scope_argument_byte_stride", kArgumentStride);
  RecordProperty("aql_agent_scope_argument_observed_byte_length",
                 kPageByteLength);
  RecordProperty("aql_agent_scope_completion_observed_byte_length",
                 kPageByteLength);
  RecordProperty("aql_agent_scope_completion_guard_byte_offset",
                 sizeof(aql::Signal));
  RecordProperty("aql_agent_scope_completion_guard_byte_length",
                 kPageByteLength - sizeof(aql::Signal));
  RecordProperty("aql_agent_scope_first_work_packet_index",
                 std::to_string(first_work_packet_index));
  RecordProperty("aql_agent_scope_ring_capacity_packets",
                 std::to_string(capacity));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t guard_delta = epoch * 0x01010101u;
    const uint32_t input_guard = 0x759bf13du ^ guard_delta;
    const uint32_t intermediate_guard = 0xa36cf197u ^ guard_delta;
    const uint32_t output_guard = 0x4e90b725u ^ guard_delta;
    const uint8_t completion_guard =
        static_cast<uint8_t>(0xb7u ^ epoch * 0x3du);
    std::array<uint32_t, kWordCount> expected_input;
    std::array<uint32_t, kWordCount> expected_intermediate;
    std::array<uint32_t, kWordCount> expected_output;
    std::array<uint32_t, kWordCount> observed_input;
    std::array<uint32_t, kWordCount> observed_intermediate;
    std::array<uint32_t, kWordCount> observed_output;
    std::array<uint8_t, kPageByteLength> expected_arguments = {};
    std::array<uint8_t, kPageByteLength> observed_arguments;
    std::array<uint8_t, kPageByteLength> observed_completion;
    expected_input.fill(input_guard);
    expected_intermediate.fill(intermediate_guard);
    expected_output.fill(output_guard);
    observed_intermediate.fill(intermediate_guard);
    observed_output.fill(output_guard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      expected_input[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        const uint32_t intermediate_result = static_cast<uint32_t>(
            uint64_t{value} * 3 + kProducerAddends[epoch]);
        // Flatten the two transforms in wider arithmetic. Observed B never
        // participates in D's oracle, including after a producer mismatch.
        const uint32_t output_result = static_cast<uint32_t>(
            uint64_t{value} * 9 + uint64_t{kProducerAddends[epoch]} * 3 +
            kConsumerAddends[epoch]);
        expected_intermediate[kPayloadOffset + i] = intermediate_result;
        expected_output[kPayloadOffset + i] = output_result;
        observed_intermediate[kPayloadOffset + i] = ~intermediate_result;
        observed_output[kPayloadOffset + i] = ~output_result;
      }
    }
    std::memcpy(input->host.pointer, expected_input.data(),
                sizeof(expected_input));
    std::memcpy(intermediate->host.pointer, observed_intermediate.data(),
                sizeof(observed_intermediate));
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
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
    // Copy only the semantic bytes, preserving initialized padding in both
    // disjoint slots and the full page through terminal completion.
    std::memcpy(expected_arguments.data(), &producer_payload,
                kernel.arguments.byte_length);
    std::memcpy(expected_arguments.data() + kArgumentStride, &consumer_payload,
                kernel.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                sizeof(expected_arguments));
    std::memset(
        static_cast<uint8_t*>(completion->host.pointer) + sizeof(aql::Signal),
        completion_guard, kPageByteLength - sizeof(aql::Signal));
    signal.value = 1;

    GpuStoreRelease(queue->host.write_index_address, index + kPacketsPerEpoch);
    aql::Publish(*queue, index++, produce);
    aql::Publish(*queue, index++, consume);
    aql::Publish(*queue, index++, complete);

    // The enabled terminal AND joins both dispatches independently of the
    // middle data edge. Capture D first, then all remaining workload backing
    // before any diagnostic or consumed-frontier wait adds synchronization.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                sizeof(observed_intermediate));
    std::memcpy(observed_input.data(), input->host.pointer,
                sizeof(observed_input));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_completion.data(), completion->host.pointer,
                sizeof(observed_completion));
    aql::Signal observed_signal;
    std::memcpy(&observed_signal, observed_completion.data(),
                sizeof(observed_signal));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(observed_output[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(observed_intermediate[i], expected_intermediate[i])
          << "intermediate word=" << i;
      EXPECT_EQ(observed_input[i], expected_input[i]) << "input word=" << i;
    }
    for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
      EXPECT_EQ(observed_arguments[byte], expected_arguments[byte])
          << "argument byte=" << byte;
    }
    EXPECT_EQ(observed_signal.kind, 1);
    EXPECT_EQ(observed_signal.value, 0);
    for (uint32_t byte = sizeof(aql::Signal); byte < kPageByteLength; ++byte) {
      EXPECT_EQ(observed_completion[byte], completion_guard)
          << "completion guard byte=" << byte;
    }
    // Nonfatal oracle failures still retire the submitted stream. A failed
    // observation or retirement ends the case before signal or backing reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "aql_agent_scope_epoch_" + std::to_string(epoch);
    RecordProperty(prefix + "_count", kCounts[epoch]);
    RecordProperty(prefix + "_producer_addend",
                   std::to_string(kProducerAddends[epoch]));
    RecordProperty(prefix + "_consumer_addend",
                   std::to_string(kConsumerAddends[epoch]));
    RecordProperty(prefix + "_final_packet_index", std::to_string(index));
  }
  RecordProperty("aql_agent_scope_completed_epochs", kCounts.size());
  RecordProperty("aql_agent_scope_dispatch_count", kCounts.size() * 2);
  RecordProperty("aql_agent_scope_work_packet_count",
                 std::to_string(index - first_work_packet_index));
  RecordProperty("aql_agent_scope_final_packet_index", std::to_string(index));
}

}  // namespace
