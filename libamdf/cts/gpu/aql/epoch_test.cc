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

class AqlEpochTest : public AqlDispatchTest {
 protected:
  AqlEpochTest() : AqlDispatchTest(AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE) {}
};

TEST_F(AqlEpochTest, BarrierValueOrdersEpochPayloadAcrossQueues) {
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
  constexpr int64_t kInitialEpoch = INT64_C(0x0000000200000000);
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
  GpuMemory* producer_arguments = nullptr;
  GpuMemory* consumer_arguments = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &producer_arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &consumer_arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &control));
  ASSERT_EQ(producer_arguments->device_address % kernel.arguments.alignment,
            0u);
  ASSERT_EQ(consumer_arguments->device_address % kernel.arguments.alignment,
            0u);
  std::memset(control->host.pointer, 0, control->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(control->host.pointer);
  auto& epoch_signal = signals[0];
  auto& completion_signal = signals[1];
  epoch_signal.kind = 1;
  completion_signal.kind = 1;
  // The host initializes this epoch only once. Its first decrement borrows
  // across bit 32, and each satisfied predicate stays satisfied through use.
  epoch_signal.value = kInitialEpoch;
  const uint64_t epoch_address = control->device_address;
  const uint64_t completion_address =
      control->device_address + sizeof(aql::Signal);

  GpuUserQueue* producer = nullptr;
  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  uint64_t producer_index = 0;
  uint64_t consumer_index = 0;
  uint64_t producer_descriptor = 0;
  uint64_t consumer_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, kernel, "aql_kernel",
                                        &producer_index, &producer_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*consumer, kernel, "aql_kernel",
                                        &consumer_index, &consumer_descriptor));
  RecordProperty("aql_epoch_initial_value", std::to_string(kInitialEpoch));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected_intermediate;
    std::array<uint32_t, kWordCount> expected_output;
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
        // Compose both transforms with wide CPU arithmetic, independently of
        // the GPU intermediate, then apply the final uint32 wrapping.
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
    // The dispatches borrow distinct argument blocks. Padding stays zero, and
    // only the compiler's 24 semantic bytes are copied into either block.
    std::memset(producer_arguments->host.pointer, 0,
                producer_arguments->info.byte_length);
    std::memcpy(producer_arguments->host.pointer, &producer_payload,
                kernel.arguments.byte_length);
    std::memset(consumer_arguments->host.pointer, 0,
                consumer_arguments->info.byte_length);
    std::memcpy(consumer_arguments->host.pointer, &consumer_payload,
                kernel.arguments.byte_length);
    completion_signal.value = 1;
    const int64_t reference = kInitialEpoch - static_cast<int64_t>(epoch);
    const int64_t completed_epoch = reference - 1;
    const auto wait =
        aql::BarrierValueLessThan(epoch_address, reference, INT64_MAX);
    const auto consume = aql::Dispatch(
        aql::HeaderBarrier::kEnabled, kGeometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        consumer_descriptor, consumer_arguments->device_address,
        completion_address, kScopes);
    const auto produce = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kGeometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        producer_descriptor, producer_arguments->device_address, epoch_address,
        kScopes);

    // Publish the consumer chain first. The producer releases its payload
    // before decrementing the epoch; the waiting consumer acquires it only
    // after its preceding BARRIER_VALUE packet completes.
    GpuStoreRelease(consumer->host.write_index_address, consumer_index + 2);
    aql::Publish(*consumer, consumer_index++, wait);
    aql::Publish(*consumer, consumer_index++, consume);
    GpuStoreRelease(producer->host.write_index_address, producer_index + 1);
    aql::Publish(*producer, producer_index++, produce);

    // Observe only the completed consumer's result before any producer wait
    // or ring-consumption wait can add synchronization to this observation.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&completion_signal.value),
                          0);
    std::memcpy(output_words.data(), output->host.pointer,
                sizeof(output_words));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(output_words[i], expected_output[i]) << "output word=" << i;
    }

    // Even a failed dependency must not authorize an early read or release of
    // producer-owned storage. Independently await its exact full-width epoch
    // before inspecting the intermediate and immutable input.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&epoch_signal.value),
                          completed_epoch);
    EXPECT_EQ(GpuLoadAcquire<int64_t>(
                  reinterpret_cast<uintptr_t>(&epoch_signal.value)),
              completed_epoch);
    std::memcpy(intermediate_words.data(), intermediate->host.pointer,
                sizeof(intermediate_words));
    const auto* unchanged_input =
        static_cast<const uint32_t*>(input->host.pointer);
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(intermediate_words[i], expected_intermediate[i])
          << "intermediate word=" << i;
      EXPECT_EQ(unchanged_input[i], upload[i]) << "input word=" << i;
    }
    // Both consumed waits run even after an oracle or first retirement failure.
    // Reuse and completion-signal rearming require both independent execution
    // completions and both consumed frontiers to have succeeded.
    EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, producer_index));
    EXPECT_NO_FATAL_FAILURE(consumer->WaitConsumed(api_, consumer_index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_epoch_completed_epochs", kCounts.size());
  RecordProperty("aql_epoch_final_value",
                 std::to_string(GpuLoadAcquire<int64_t>(
                     reinterpret_cast<uintptr_t>(&epoch_signal.value))));
}

}  // namespace
