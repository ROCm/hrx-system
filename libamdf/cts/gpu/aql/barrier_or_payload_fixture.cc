// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/barrier_or_payload_fixture.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

TEST_P(AqlOrPayloadTest, ReadsReadyPayloadWhileOtherProducersRemainPending) {
  const auto& parameters = GetParam();
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  constexpr uint32_t kProducerCount = 5;
  constexpr uint32_t kGateSignal = kProducerCount;
  constexpr uint32_t kConsumerSignal = kGateSignal + 1;
  constexpr uint32_t kSignalCount = kConsumerSignal + 1;
  constexpr uint32_t kArgumentCount = kProducerCount + 1;
  constexpr uint32_t kDataWordCount = 4096;
  constexpr uint32_t kDataByteLength = kDataWordCount * sizeof(uint32_t);
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kWorkWordCount = 512;
  constexpr uint32_t kControlGuardOffset = kSignalCount * sizeof(aql::Signal);
  constexpr uint32_t kControlGuardWordCount =
      (kPageByteLength - kControlGuardOffset) / sizeof(uint32_t);
  constexpr uint32_t kEpochCount = 2;
  constexpr aql::FenceScopes kDispatchScopes = {aql::FenceScope::kSystem,
                                                aql::FenceScope::kSystem};
  constexpr aql::FenceScopes kReadinessScopes = {aql::FenceScope::kNone,
                                                 aql::FenceScope::kNone};
  const aql::DispatchGeometry geometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kWorkWordCount, 1, 1}};

  GpuMemory* input = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* control = nullptr;
  std::array<GpuMemory*, kArgumentCount> arguments = {};
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kDataByteLength, &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataByteLength, &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataByteLength, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &control));
  for (auto& argument : arguments) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &argument));
    ASSERT_EQ(argument->device_address % kernel.arguments.alignment, 0u);
  }
  ASSERT_EQ(control->device_address % alignof(aql::Signal), 0u);
  std::memset(control->host.pointer, 0, kPageByteLength);
  auto* signals = static_cast<aql::Signal*>(control->host.pointer);
  std::array<uint64_t, kSignalCount> signal_addresses;
  for (uint32_t i = 0; i < kSignalCount; ++i) {
    signals[i] = aql::UserSignal();
    signal_addresses[i] = control->device_address + i * sizeof(aql::Signal);
  }
  auto* control_guard_address =
      static_cast<uint8_t*>(control->host.pointer) + kControlGuardOffset;
  std::array<uint32_t, kControlGuardWordCount> expected_control_guards;
  for (uint32_t i = 0; i < expected_control_guards.size(); ++i) {
    expected_control_guards[i] = 0x6935bdefu ^ (i * 0x03050709u);
  }
  std::memcpy(control_guard_address, expected_control_guards.data(),
              sizeof(expected_control_guards));

  GpuUserQueue* ready = nullptr;
  GpuUserQueue* pending = nullptr;
  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&ready));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&pending));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  uint64_t ready_index = 0;
  uint64_t pending_index = 0;
  uint64_t consumer_index = 0;
  uint64_t ready_descriptor = 0;
  uint64_t pending_descriptor = 0;
  uint64_t consumer_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*ready, kernel, "aql_or_ready_kernel",
                                        &ready_index, &ready_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*pending, kernel,
                                        "aql_or_pending_kernel", &pending_index,
                                        &pending_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*consumer, kernel,
                                        "aql_or_consumer_kernel",
                                        &consumer_index, &consumer_descriptor));
  ASSERT_GE(ready->host.ring_byte_length / sizeof(aql::Packet),
            ready_index + kEpochCount);
  ASSERT_GE(pending->host.ring_byte_length / sizeof(aql::Packet),
            pending_index + kProducerCount * kEpochCount);
  ASSERT_GE(consumer->host.ring_byte_length / sizeof(aql::Packet),
            consumer_index + 2 * kEpochCount);

  std::array<uint64_t, kProducerCount> dependencies = {};
  for (uint32_t p = 0; p < kProducerCount; ++p) {
    if (parameters.live_slots & (1u << p)) {
      dependencies[p] = signal_addresses[p];
    }
  }
  const auto select =
      aql::Barrier(aql::BarrierType::kOr, aql::HeaderBarrier::kDisabled, 0,
                   dependencies, kReadinessScopes);
  const auto hold =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                   {signal_addresses[kGateSignal]}, kReadinessScopes);

  std::vector<uint32_t> expected_input(kDataWordCount);
  std::vector<uint32_t> expected_intermediate(kDataWordCount);
  std::vector<uint32_t> expected_ready_intermediate(kDataWordCount);
  std::vector<uint32_t> expected_output(kDataWordCount);
  std::vector<uint32_t> observed_input(kDataWordCount);
  std::vector<uint32_t> observed_intermediate(kDataWordCount);
  std::vector<uint32_t> observed_ready_intermediate(kDataWordCount);
  std::vector<uint32_t> observed_output(kDataWordCount);
  std::vector<uint32_t> observed_final_output(kDataWordCount);
  std::array<std::array<uint8_t, kPageByteLength>, kArgumentCount>
      expected_arguments;
  std::array<std::array<uint8_t, kPageByteLength>, kArgumentCount>
      observed_arguments;
  std::array<aql::Signal, kSignalCount> observed_ready_signals;
  std::array<aql::Signal, kSignalCount> observed_final_signals;
  std::array<uint32_t, kControlGuardWordCount> observed_ready_guards;
  std::array<uint32_t, kControlGuardWordCount> observed_final_guards;
  RecordProperty("transform_kernel_target", kernel.target);
  RecordProperty("aql_or_live_slots", parameters.live_slots);
  RecordProperty("aql_or_ready_slot", parameters.ready_slot);
  RecordProperty("aql_or_pending_producers", kProducerCount - 1);
  RecordProperty("aql_or_work_word_count", kWorkWordCount);
  RecordProperty("aql_or_checked_words_per_data_buffer", kDataWordCount);
  RecordProperty("aql_or_checked_bytes_per_kernarg", kPageByteLength);
  RecordProperty("aql_or_control_guard_byte_offset", kControlGuardOffset);
  RecordProperty("aql_or_header_barrier", "disabled");
  RecordProperty("aql_or_barrier_scopes", "none,none");
  RecordProperty("aql_or_dispatch_scopes", "system,system");
  RecordProperty("aql_or_completed_epochs", 0);

  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t consumer_addend = 0x01234567u + epoch * 0x76543210u;
    for (uint32_t i = 0; i < kDataWordCount; ++i) {
      expected_input[i] = 0x57319badu ^ (i * 31u) ^ epoch;
      expected_intermediate[i] = 0x8ac56913u ^ (i * 43u) ^ epoch;
      expected_output[i] = 0x94bd5763u ^ (i * 71u) ^ epoch;
    }
    expected_ready_intermediate = expected_intermediate;
    observed_intermediate = expected_intermediate;
    observed_output = expected_output;
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      const uint32_t first_word = kPayloadOffset + p * kWorkWordCount;
      const uint32_t addend = 0x80000023u + p * 0x03456789u + epoch * 7;
      for (uint32_t i = 0; i < kWorkWordCount; ++i) {
        const uint32_t value = static_cast<uint32_t>(
            uint64_t{0xfffffff0u} + uint64_t{i} * 0x01030507u +
            uint64_t{p} * 0x23456789u + uint64_t{epoch} * 0x11111111u);
        const uint32_t produced =
            static_cast<uint32_t>(uint64_t{value} * 3 + addend);
        expected_input[first_word + i] = value;
        expected_intermediate[first_word + i] = produced;
        observed_intermediate[first_word + i] = ~produced;
        expected_ready_intermediate[first_word + i] =
            p == parameters.ready_slot ? produced : ~produced;
        if (p == parameters.ready_slot) {
          // Both transforms use an independent wide CPU oracle. The odd
          // multiplier makes consuming the complemented initial value differ.
          const uint32_t consumed = static_cast<uint32_t>(
              uint64_t{value} * 9 + uint64_t{addend} * 3 + consumer_addend);
          expected_output[kPayloadOffset + i] = consumed;
          observed_output[kPayloadOffset + i] = ~consumed;
        }
      }
      const kernels::transform::Arguments payload = {
          input->device_address + first_word * sizeof(uint32_t),
          intermediate->device_address + first_word * sizeof(uint32_t),
          kWorkWordCount,
          addend,
      };
      expected_arguments[p].fill(0);
      std::memcpy(expected_arguments[p].data(), &payload,
                  kernel.arguments.byte_length);
    }
    const kernels::transform::Arguments consumer_payload = {
        intermediate->device_address +
            (kPayloadOffset + parameters.ready_slot * kWorkWordCount) *
                sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kWorkWordCount,
        consumer_addend,
    };
    expected_arguments[kProducerCount].fill(0);
    std::memcpy(expected_arguments[kProducerCount].data(), &consumer_payload,
                kernel.arguments.byte_length);
    for (uint32_t i = 0; i < kArgumentCount; ++i) {
      std::memcpy(arguments[i]->host.pointer, expected_arguments[i].data(),
                  kPageByteLength);
    }
    std::memcpy(input->host.pointer, expected_input.data(), kDataByteLength);
    std::memcpy(intermediate->host.pointer, observed_intermediate.data(),
                kDataByteLength);
    std::memcpy(output->host.pointer, observed_output.data(), kDataByteLength);
    for (uint32_t i = 0; i < kSignalCount; ++i) {
      signals[i].value = 1;
    }
    std::array<aql::Packet, kProducerCount> produce;
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      produce[p] = aql::Dispatch(
          aql::HeaderBarrier::kDisabled, geometry,
          kernel.private_segment_byte_length, kernel.group_segment_byte_length,
          p == parameters.ready_slot ? ready_descriptor : pending_descriptor,
          arguments[p]->device_address, signal_addresses[p], kDispatchScopes);
    }
    const auto consume = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, geometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        consumer_descriptor, arguments[kProducerCount]->device_address,
        signal_addresses[kConsumerSignal], kDispatchScopes);

    GpuStoreRelease(consumer->host.write_index_address, consumer_index + 2);
    ASSERT_TRUE(aql::Publish(*consumer, consumer_index++, select));
    ASSERT_TRUE(aql::Publish(*consumer, consumer_index++, consume));
    GpuStoreRelease(pending->host.write_index_address,
                    pending_index + kProducerCount);
    ASSERT_TRUE(aql::Publish(*pending, pending_index++, hold));
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      if (p != parameters.ready_slot) {
        ASSERT_TRUE(aql::Publish(*pending, pending_index++, produce[p]));
      }
    }
    GpuStoreRelease(ready->host.write_index_address, ready_index + 1);
    ASSERT_TRUE(
        aql::Publish(*ready, ready_index++, produce[parameters.ready_slot]));

    ASSERT_TRUE(GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&signals[kConsumerSignal].value), 0));
    // Capture the selected result before another join or releasing held work
    // can add a synchronization edge. OR does not identify the ready slot
    // or cancel other producers.
    std::memcpy(observed_output.data(), output->host.pointer, kDataByteLength);
    std::memcpy(observed_ready_intermediate.data(), intermediate->host.pointer,
                kDataByteLength);
    std::memcpy(observed_ready_signals.data(), control->host.pointer,
                sizeof(observed_ready_signals));
    std::memcpy(observed_ready_guards.data(), control_guard_address,
                sizeof(observed_ready_guards));

    // Release every accepted producer before any diagnostic can fail. Their
    // independent completions retain inputs, arguments, signals and payloads.
    GpuStoreRelease<int64_t>(
        reinterpret_cast<uintptr_t>(&signals[kGateSignal].value), 0);
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      ASSERT_TRUE(GpuWaitEqual<int64_t>(
          reinterpret_cast<uintptr_t>(&signals[p].value), 0));
    }
    std::memcpy(observed_final_output.data(), output->host.pointer,
                kDataByteLength);
    std::memcpy(observed_input.data(), input->host.pointer, kDataByteLength);
    std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                kDataByteLength);
    for (uint32_t i = 0; i < kArgumentCount; ++i) {
      std::memcpy(observed_arguments[i].data(), arguments[i]->host.pointer,
                  kPageByteLength);
    }
    std::memcpy(observed_final_signals.data(), control->host.pointer,
                sizeof(observed_final_signals));
    std::memcpy(observed_final_guards.data(), control_guard_address,
                sizeof(observed_final_guards));

    EXPECT_EQ(observed_output, expected_output);
    EXPECT_EQ(observed_final_output, expected_output);
    EXPECT_EQ(observed_ready_intermediate, expected_ready_intermediate);
    EXPECT_EQ(observed_input, expected_input);
    EXPECT_EQ(observed_intermediate, expected_intermediate);
    EXPECT_EQ(observed_arguments, expected_arguments);
    EXPECT_EQ(observed_ready_guards, expected_control_guards);
    EXPECT_EQ(observed_final_guards, expected_control_guards);
    for (uint32_t i = 0; i < kSignalCount; ++i) {
      EXPECT_EQ(observed_ready_signals[i].kind, 1) << "signal=" << i;
      EXPECT_EQ(observed_ready_signals[i].value,
                i == parameters.ready_slot || i == kConsumerSignal ? 0 : 1)
          << "signal=" << i;
      EXPECT_EQ(observed_final_signals[i].kind, 1) << "signal=" << i;
      EXPECT_EQ(observed_final_signals[i].value, 0) << "signal=" << i;
    }
    EXPECT_NO_FATAL_FAILURE(ready->WaitConsumed(api_, ready_index));
    EXPECT_NO_FATAL_FAILURE(pending->WaitConsumed(api_, pending_index));
    EXPECT_NO_FATAL_FAILURE(consumer->WaitConsumed(api_, consumer_index));
    if (HasFailure()) {
      return;
    }
    RecordProperty("aql_or_completed_epochs", epoch + 1);
  }
}

}  // namespace

std::string OrPayloadCaseName(
    const ::testing::TestParamInfo<OrPayloadCase>& info) {
  return "Mask" + std::to_string(info.param.live_slots) + "Slot" +
         std::to_string(info.param.ready_slot);
}
