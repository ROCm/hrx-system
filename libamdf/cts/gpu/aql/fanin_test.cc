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

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

TEST_F(AqlDispatchTest, BarrierAndJoinsIndependentShaderPayloads) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kProducerCount = 2;
  constexpr uint32_t kProducerWordCount = 512;
  constexpr uint32_t kConsumerWordCount = kProducerCount * kProducerWordCount;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kSignalCount = kProducerCount + 1;
  constexpr uint32_t kControlGuardByteOffset =
      kSignalCount * sizeof(aql::Signal);
  constexpr uint32_t kControlGuardWordCount =
      (kPageByteLength - kControlGuardByteOffset) / sizeof(uint32_t);
  constexpr std::array<uint32_t, kProducerCount> kInputGuards = {0x759bf13du,
                                                                 0xc16e85a7u};
  constexpr uint32_t kIntermediateGuard = 0xa36cf197u;
  constexpr uint32_t kOutputGuard = 0x4e90b725u;
  constexpr uint32_t kControlGuard = 0x68d329b7u;
  constexpr std::array<std::array<uint32_t, kProducerCount>, 2>
      kProducerAddends = {{{7, 0x10203045u}, {0x80000023u, 0x31415927u}}};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {0x101u, 0x2468ace1u};
  constexpr uint32_t kEpochCount = kProducerAddends.size();
  const aql::DispatchGeometry kProducerGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kProducerWordCount, 1, 1}};
  const aql::DispatchGeometry kConsumerGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kConsumerWordCount, 1, 1}};
  constexpr aql::FenceScopes kDispatchScopes = {aql::FenceScope::kSystem,
                                                aql::FenceScope::kSystem};
  constexpr aql::FenceScopes kBarrierScopes = {aql::FenceScope::kNone,
                                               aql::FenceScope::kNone};

  std::array<GpuMemory*, kProducerCount> inputs = {};
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  std::array<GpuMemory*, kSignalCount> arguments = {};
  GpuMemory* control = nullptr;
  for (auto& input : inputs) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ, kWordCount * sizeof(uint32_t), &input));
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  for (auto& argument : arguments) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &argument));
    ASSERT_EQ(argument->device_address % kernel.arguments.alignment, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &control));
  ASSERT_EQ(control->device_address % alignof(aql::Signal), 0u);
  // The three native signal records borrow one allocation. Reserved fields
  // retain their ABI meaning; guards begin after all three complete blocks.
  std::memset(control->host.pointer, 0, kPageByteLength);
  auto* signals = static_cast<aql::Signal*>(control->host.pointer);
  std::array<uint64_t, kSignalCount> signal_addresses;
  for (uint32_t i = 0; i < kSignalCount; ++i) {
    signals[i].kind = 1;
    signal_addresses[i] = control->device_address + i * sizeof(aql::Signal);
  }
  auto* control_guard_address =
      static_cast<uint8_t*>(control->host.pointer) + kControlGuardByteOffset;
  std::array<uint32_t, kControlGuardWordCount> control_guards;
  control_guards.fill(kControlGuard);
  std::memcpy(control_guard_address, control_guards.data(),
              sizeof(control_guards));

  GpuUserQueue* producer = nullptr;
  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  uint64_t producer_index = 0;
  uint64_t consumer_index = 0;
  uint64_t producer_descriptor = 0;
  uint64_t consumer_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, kernel,
                                        "aql_fanin_producer_kernel",
                                        &producer_index, &producer_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*consumer, kernel,
                                        "aql_fanin_consumer_kernel",
                                        &consumer_index, &consumer_descriptor));
  const uint64_t first_producer_packet_index = producer_index;
  const uint64_t first_consumer_packet_index = consumer_index;
  const uint64_t producer_capacity =
      producer->host.ring_byte_length / sizeof(aql::Packet);
  const uint64_t consumer_capacity =
      consumer->host.ring_byte_length / sizeof(aql::Packet);
  // Both epochs fit without crossing either queue's ring boundary.
  ASSERT_GE(producer_capacity, producer_index + 2 * kEpochCount);
  ASSERT_GE(consumer_capacity, consumer_index + 2 * kEpochCount);

  RecordProperty("aql_fanin_dependency_slots", "0,1");
  RecordProperty("aql_fanin_header_barrier", "disabled");
  RecordProperty("aql_fanin_barrier_scopes", "none,none");
  RecordProperty("aql_fanin_dispatch_scopes", "system,system");
  RecordProperty("aql_fanin_producer_grid_size", kProducerWordCount);
  RecordProperty("aql_fanin_consumer_grid_size", kConsumerWordCount);
  RecordProperty("aql_fanin_workgroup_size", kernel.workgroup_size());
  RecordProperty("aql_fanin_checked_words_per_payload", kWordCount);
  RecordProperty("aql_fanin_checked_bytes_per_kernarg", kPageByteLength);
  RecordProperty("aql_fanin_control_guard_byte_offset",
                 kControlGuardByteOffset);
  RecordProperty("aql_fanin_producer_first_work_packet_index",
                 std::to_string(first_producer_packet_index));
  RecordProperty("aql_fanin_consumer_first_work_packet_index",
                 std::to_string(first_consumer_packet_index));
  RecordProperty("aql_fanin_producer_ring_capacity_packets",
                 std::to_string(producer_capacity));
  RecordProperty("aql_fanin_consumer_ring_capacity_packets",
                 std::to_string(consumer_capacity));

  std::array<std::array<uint32_t, kWordCount>, kProducerCount> expected_inputs;
  std::array<std::array<uint32_t, kWordCount>, kProducerCount> observed_inputs;
  std::array<uint32_t, kWordCount> expected_intermediate;
  std::array<uint32_t, kWordCount> expected_output;
  std::array<uint32_t, kWordCount> observed_intermediate;
  std::array<uint32_t, kWordCount> observed_output;
  std::array<std::array<uint8_t, kPageByteLength>, kSignalCount>
      expected_arguments;
  std::array<std::array<uint8_t, kPageByteLength>, kSignalCount>
      observed_arguments;
  std::array<aql::Signal, kSignalCount> observed_signals;
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    expected_intermediate.fill(kIntermediateGuard);
    expected_output.fill(kOutputGuard);
    observed_intermediate.fill(kIntermediateGuard);
    observed_output.fill(kOutputGuard);
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      expected_inputs[p].fill(kInputGuards[p]);
      for (uint32_t i = 0; i < kProducerWordCount; ++i) {
        const uint32_t value = static_cast<uint32_t>(
            uint64_t{0xfffffff0u} + uint64_t{i} * 0x01030507u +
            uint64_t{p} * 0x23456789u + uint64_t{epoch} * 0x11111111u);
        expected_inputs[p][kPayloadOffset + i] = value;
        const uint32_t intermediate_result = static_cast<uint32_t>(
            uint64_t{value} * 3 + kProducerAddends[epoch][p]);
        // The final oracle composes both transforms with wide CPU arithmetic;
        // it never derives a result from the observed intermediate storage.
        const uint32_t output_result = static_cast<uint32_t>(
            uint64_t{value} * 9 + uint64_t{kProducerAddends[epoch][p]} * 3 +
            kConsumerAddends[epoch]);
        const uint32_t word = kPayloadOffset + p * kProducerWordCount + i;
        expected_intermediate[word] = intermediate_result;
        expected_output[word] = output_result;
        observed_intermediate[word] = ~intermediate_result;
        observed_output[word] = ~output_result;
      }
      std::memcpy(inputs[p]->host.pointer, expected_inputs[p].data(),
                  sizeof(expected_inputs[p]));
      // Each producer borrows a distinct argument allocation and a disjoint,
      // cache-line-aligned half of the shared intermediate payload.
      const kernels::transform::Arguments payload = {
          inputs[p]->device_address + kPayloadOffset * sizeof(uint32_t),
          intermediate->device_address +
              (kPayloadOffset + p * kProducerWordCount) * sizeof(uint32_t),
          kProducerWordCount,
          kProducerAddends[epoch][p],
      };
      expected_arguments[p].fill(0);
      std::memcpy(expected_arguments[p].data(), &payload,
                  kernel.arguments.byte_length);
    }
    std::memcpy(intermediate->host.pointer, observed_intermediate.data(),
                sizeof(observed_intermediate));
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    const kernels::transform::Arguments consumer_payload = {
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kConsumerWordCount,
        kConsumerAddends[epoch],
    };
    expected_arguments[kProducerCount].fill(0);
    std::memcpy(expected_arguments[kProducerCount].data(), &consumer_payload,
                kernel.arguments.byte_length);
    for (uint32_t i = 0; i < arguments.size(); ++i) {
      // Full-page initialization preserves the aligned backing without
      // copying the typed object's indeterminate tail padding.
      std::memcpy(arguments[i]->host.pointer, expected_arguments[i].data(),
                  sizeof(expected_arguments[i]));
      signals[i].value = 1;
    }

    std::array<aql::Packet, kProducerCount> produce;
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      produce[p] = aql::Dispatch(
          aql::HeaderBarrier::kDisabled, kProducerGeometry,
          kernel.private_segment_byte_length, kernel.group_segment_byte_length,
          producer_descriptor, arguments[p]->device_address,
          signal_addresses[p], kDispatchScopes);
    }
    const auto wait = aql::Barrier(
        aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
        {signal_addresses[0], signal_addresses[1], 0, 0, 0}, kBarrierScopes);
    const auto consume = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kConsumerGeometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        consumer_descriptor, arguments[kProducerCount]->device_address,
        signal_addresses[kProducerCount], kDispatchScopes);
    const uint32_t first_producer = epoch;

    // All four bodies and their borrowed storage are ready before publication.
    // AND blocks later launches even with its header barrier clear; the
    // consumer supplies the SYSTEM acquire after both dependencies reach zero.
    GpuStoreRelease(consumer->host.write_index_address, consumer_index + 2);
    aql::Publish(*consumer, consumer_index++, wait);
    aql::Publish(*consumer, consumer_index++, consume);
    GpuStoreRelease(producer->host.write_index_address, producer_index + 2);
    aql::Publish(*producer, producer_index++, produce[first_producer]);
    aql::Publish(*producer, producer_index++, produce[1 - first_producer]);

    // Preserve the entire consumer result before independent producer joins
    // or consumption can add synchronization to this decisive observation.
    GpuWaitEqual<int64_t>(
        reinterpret_cast<uintptr_t>(&signals[kProducerCount].value), 0);
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signals[0].value), 0);
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signals[1].value), 0);
    // Both producer executions have joined independently of the tested edge.
    // Capture every remaining initialized extent before any diagnostics.
    for (uint32_t p = 0; p < kProducerCount; ++p) {
      std::memcpy(observed_inputs[p].data(), inputs[p]->host.pointer,
                  sizeof(observed_inputs[p]));
    }
    std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                sizeof(observed_intermediate));
    for (uint32_t i = 0; i < arguments.size(); ++i) {
      std::memcpy(observed_arguments[i].data(), arguments[i]->host.pointer,
                  sizeof(observed_arguments[i]));
    }
    std::memcpy(observed_signals.data(), control->host.pointer,
                sizeof(observed_signals));
    std::memcpy(control_guards.data(), control_guard_address,
                sizeof(control_guards));

    for (uint32_t word = 0; word < kWordCount; ++word) {
      EXPECT_EQ(observed_output[word], expected_output[word])
          << "output word=" << word;
      EXPECT_EQ(observed_intermediate[word], expected_intermediate[word])
          << "intermediate word=" << word;
      for (uint32_t p = 0; p < kProducerCount; ++p) {
        EXPECT_EQ(observed_inputs[p][word], expected_inputs[p][word])
            << "producer=" << p << " input word=" << word;
      }
    }
    for (uint32_t i = 0; i < arguments.size(); ++i) {
      for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
        EXPECT_EQ(observed_arguments[i][byte], expected_arguments[i][byte])
            << "dispatch=" << i << " kernarg byte=" << byte;
      }
      EXPECT_EQ(observed_signals[i].kind, 1) << "signal=" << i;
      EXPECT_EQ(observed_signals[i].value, 0) << "signal=" << i;
    }
    for (uint32_t word = 0; word < control_guards.size(); ++word) {
      EXPECT_EQ(control_guards[word], kControlGuard)
          << "control guard word=" << word;
    }
    // Both retirements run on any nonfatal mismatch. Dependency signals stay
    // zero through their last AND reader, and no backing is rewritten early.
    EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, producer_index));
    EXPECT_NO_FATAL_FAILURE(consumer->WaitConsumed(api_, consumer_index));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "aql_fanin_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_producer_order", epoch == 0 ? "0,1" : "1,0");
    RecordProperty(prefix + "_producer_0_addend",
                   std::to_string(kProducerAddends[epoch][0]));
    RecordProperty(prefix + "_producer_1_addend",
                   std::to_string(kProducerAddends[epoch][1]));
    RecordProperty(prefix + "_consumer_addend",
                   std::to_string(kConsumerAddends[epoch]));
    RecordProperty(prefix + "_producer_packet_index",
                   std::to_string(producer_index));
    RecordProperty(prefix + "_consumer_packet_index",
                   std::to_string(consumer_index));
  }
  RecordProperty("aql_fanin_completed_epochs", kEpochCount);
  RecordProperty("aql_fanin_dispatches", kEpochCount * kSignalCount);
  RecordProperty("aql_fanin_work_packet_count",
                 std::to_string(producer_index - first_producer_packet_index +
                                consumer_index - first_consumer_packet_index));
  RecordProperty("aql_fanin_producer_final_packet_index",
                 std::to_string(producer_index));
  RecordProperty("aql_fanin_consumer_final_packet_index",
                 std::to_string(consumer_index));
}

}  // namespace
