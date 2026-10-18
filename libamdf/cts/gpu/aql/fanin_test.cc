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
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

class AqlFanInTest : public AqlDispatchTest {};

TEST_F(AqlFanInTest, BarrierAndJoinsIndependentShaderPayloads) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;

  constexpr uint32_t kDependenciesPerPacket = 5;
  constexpr uint32_t kMaxProducerCount = 11;
  constexpr uint32_t kMaxSignalCount = kMaxProducerCount + 1;
  constexpr uint32_t kMaxBarrierPacketCount =
      (kMaxProducerCount + kDependenciesPerPacket - 1) / kDependenciesPerPacket;
  constexpr uint32_t kProducerWordCount = 512;
  constexpr uint32_t kMaxConsumerWordCount =
      kMaxProducerCount * kProducerWordCount;
  constexpr uint32_t kInputWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kPageWordCount = kPageByteLength / sizeof(uint32_t);
  constexpr uint32_t kMaxPayloadWordCount =
      ((2 * kPayloadOffset + kMaxConsumerWordCount + kPageWordCount - 1) /
       kPageWordCount) *
      kPageWordCount;
  constexpr uint32_t kMaxPayloadByteLength =
      kMaxPayloadWordCount * sizeof(uint32_t);
  constexpr uint32_t kInputGuard = 0x759bf13du;
  constexpr uint32_t kIntermediateGuard = 0xa36cf197u;
  constexpr uint32_t kOutputGuard = 0x4e90b725u;
  constexpr uint32_t kControlGuard = 0x68d329b7u;
  constexpr std::array<uint32_t, 2> kProducerEpochSeeds = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {0x101u, 0x2468ace1u};
  constexpr uint32_t kEpochCount = kProducerEpochSeeds.size();
  const aql::DispatchGeometry kProducerGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kProducerWordCount, 1, 1}};
  constexpr aql::FenceScopes kDispatchScopes = {aql::FenceScope::kSystem,
                                                aql::FenceScope::kSystem};
  constexpr aql::FenceScopes kBarrierScopes = {aql::FenceScope::kNone,
                                               aql::FenceScope::kNone};

  std::array<GpuMemory*, kMaxProducerCount> inputs = {};
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  std::array<GpuMemory*, kMaxSignalCount> arguments = {};
  GpuMemory* control = nullptr;
  for (auto& input : inputs) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ, kInputWordCount * sizeof(uint32_t), &input));
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kMaxPayloadByteLength, &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kMaxPayloadByteLength, &output));
  for (auto& argument : arguments) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &argument));
    ASSERT_EQ(argument->device_address % kernel.arguments.alignment, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &control));
  ASSERT_EQ(control->device_address % alignof(aql::Signal), 0u);
  auto* signals = static_cast<aql::Signal*>(control->host.pointer);
  std::array<uint64_t, kMaxSignalCount> signal_addresses;
  for (uint32_t i = 0; i < kMaxSignalCount; ++i) {
    signal_addresses[i] = control->device_address + i * sizeof(aql::Signal);
  }

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
  ASSERT_GE(producer_capacity, kMaxProducerCount);
  ASSERT_GE(consumer_capacity, kMaxBarrierPacketCount + 1);

  std::vector<std::array<uint32_t, kInputWordCount>> expected_inputs(
      kMaxProducerCount);
  std::vector<std::array<uint32_t, kInputWordCount>> observed_inputs(
      kMaxProducerCount);
  std::vector<uint32_t> expected_intermediate(kMaxPayloadWordCount);
  std::vector<uint32_t> expected_output(kMaxPayloadWordCount);
  std::vector<uint32_t> observed_intermediate(kMaxPayloadWordCount);
  std::vector<uint32_t> observed_output(kMaxPayloadWordCount);
  std::vector<std::array<uint8_t, kPageByteLength>> expected_arguments(
      kMaxSignalCount);
  std::vector<std::array<uint8_t, kPageByteLength>> observed_arguments(
      kMaxSignalCount);
  std::array<aql::Signal, kMaxSignalCount> observed_signals;
  std::array<uint32_t, kMaxProducerCount> producer_addends;

  RecordProperty("transform_kernel_target", kernel.target);
  RecordProperty("aql_fanin_case_count", kMaxProducerCount);
  RecordProperty("aql_fanin_header_barrier", "disabled");
  RecordProperty("aql_fanin_barrier_scopes", "none,none");
  RecordProperty("aql_fanin_dispatch_scopes", "system,system");
  RecordProperty("aql_fanin_producer_grid_size", kProducerWordCount);
  RecordProperty("aql_fanin_workgroup_size", kernel.workgroup_size());
  RecordProperty("aql_fanin_checked_words_per_input", kInputWordCount);
  RecordProperty("aql_fanin_checked_words_per_payload", kMaxPayloadWordCount);
  RecordProperty("aql_fanin_checked_bytes_per_kernarg", kPageByteLength);
  RecordProperty("aql_fanin_producer_first_work_packet_index",
                 std::to_string(first_producer_packet_index));
  RecordProperty("aql_fanin_consumer_first_work_packet_index",
                 std::to_string(first_consumer_packet_index));
  RecordProperty("aql_fanin_producer_ring_capacity_packets",
                 std::to_string(producer_capacity));
  RecordProperty("aql_fanin_consumer_ring_capacity_packets",
                 std::to_string(consumer_capacity));

  for (uint32_t producer_count = 1; producer_count <= kMaxProducerCount;
       ++producer_count) {
    SCOPED_TRACE("producer_count=" + std::to_string(producer_count));
    const uint32_t barrier_packet_count =
        (producer_count + kDependenciesPerPacket - 1) / kDependenciesPerPacket;
    const uint32_t consumer_packets_per_epoch = barrier_packet_count + 1;
    const uint32_t consumer_word_count = producer_count * kProducerWordCount;
    const uint32_t active_payload_word_count =
        ((2 * kPayloadOffset + consumer_word_count + kPageWordCount - 1) /
         kPageWordCount) *
        kPageWordCount;
    const uint32_t signal_count = producer_count + 1;
    const uint32_t control_guard_byte_offset =
        signal_count * sizeof(aql::Signal);
    const uint32_t control_guard_byte_length =
        kPageByteLength - control_guard_byte_offset;
    const uint32_t control_guard_word_count =
        control_guard_byte_length / sizeof(uint32_t);
    const aql::DispatchGeometry consumer_geometry = {
        1,
        {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
        {consumer_word_count, 1, 1}};
    const uint64_t case_first_producer_packet_index = producer_index;
    const uint64_t case_first_consumer_packet_index = consumer_index;
    const std::string case_prefix =
        "aql_fanin_case_" + std::to_string(producer_count);

    // Native signal records borrow one allocation. Reserved fields retain
    // their ABI meaning, and the guard begins after the active signal blocks.
    std::memset(control->host.pointer, 0, kPageByteLength);
    for (uint32_t i = 0; i < signal_count; ++i) {
      signals[i] = aql::UserSignal();
    }
    auto* control_guard_address = static_cast<uint8_t*>(control->host.pointer) +
                                  control_guard_byte_offset;
    std::vector<uint32_t> expected_control_guards(control_guard_word_count,
                                                  kControlGuard);
    std::vector<uint32_t> observed_control_guards(control_guard_word_count);
    std::memcpy(control_guard_address, expected_control_guards.data(),
                control_guard_byte_length);

    RecordProperty(case_prefix + "_dependency_count", producer_count);
    RecordProperty(case_prefix + "_barrier_packets_per_epoch",
                   barrier_packet_count);
    RecordProperty(case_prefix + "_last_packet_dependency_count",
                   (producer_count - 1) % kDependenciesPerPacket + 1);
    RecordProperty(case_prefix + "_consumer_grid_size", consumer_word_count);
    RecordProperty(case_prefix + "_active_payload_word_count",
                   active_payload_word_count);
    RecordProperty(case_prefix + "_control_guard_byte_offset",
                   control_guard_byte_offset);
    RecordProperty(case_prefix + "_producer_first_work_packet_index",
                   std::to_string(case_first_producer_packet_index));
    RecordProperty(case_prefix + "_consumer_first_work_packet_index",
                   std::to_string(case_first_consumer_packet_index));

    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      SCOPED_TRACE("epoch=" + std::to_string(epoch));
      std::fill(expected_intermediate.begin(), expected_intermediate.end(),
                kIntermediateGuard);
      std::fill(expected_output.begin(), expected_output.end(), kOutputGuard);
      std::fill(observed_intermediate.begin(), observed_intermediate.end(),
                kIntermediateGuard);
      std::fill(observed_output.begin(), observed_output.end(), kOutputGuard);
      for (uint32_t i = 0; i < signal_count; ++i) {
        expected_arguments[i].fill(0);
      }
      for (uint32_t p = 0; p < producer_count; ++p) {
        expected_inputs[p].fill(kInputGuard ^ (p * 0x01030709u));
        producer_addends[p] = kProducerEpochSeeds[epoch] ^ (p * 0x10203045u);
        for (uint32_t i = 0; i < kProducerWordCount; ++i) {
          const uint32_t value = static_cast<uint32_t>(
              uint64_t{0xfffffff0u} + uint64_t{i} * 0x01030507u +
              uint64_t{p} * 0x23456789u + uint64_t{epoch} * 0x11111111u);
          expected_inputs[p][kPayloadOffset + i] = value;
          const uint32_t intermediate_result =
              static_cast<uint32_t>(uint64_t{value} * 3 + producer_addends[p]);
          // The final oracle composes both transforms with wide CPU arithmetic;
          // it never uses the observed intermediate storage.
          const uint32_t output_result = static_cast<uint32_t>(
              uint64_t{value} * 9 + uint64_t{producer_addends[p]} * 3 +
              kConsumerAddends[epoch]);
          const uint32_t word = kPayloadOffset + p * kProducerWordCount + i;
          expected_intermediate[word] = intermediate_result;
          expected_output[word] = output_result;
          observed_intermediate[word] = ~intermediate_result;
          observed_output[word] = ~output_result;
        }
        // Each producer borrows a distinct argument allocation and a disjoint,
        // cache-line-aligned intermediate region.
        const kernels::transform::Arguments payload = {
            inputs[p]->device_address + kPayloadOffset * sizeof(uint32_t),
            intermediate->device_address +
                (kPayloadOffset + p * kProducerWordCount) * sizeof(uint32_t),
            kProducerWordCount,
            producer_addends[p],
        };
        std::memcpy(expected_arguments[p].data(), &payload,
                    kernel.arguments.byte_length);
        std::memcpy(inputs[p]->host.pointer, expected_inputs[p].data(),
                    sizeof(expected_inputs[p]));
      }
      std::memcpy(intermediate->host.pointer, observed_intermediate.data(),
                  kMaxPayloadByteLength);
      std::memcpy(output->host.pointer, observed_output.data(),
                  kMaxPayloadByteLength);
      const kernels::transform::Arguments consumer_payload = {
          intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
          output->device_address + kPayloadOffset * sizeof(uint32_t),
          consumer_word_count,
          kConsumerAddends[epoch],
      };
      std::memcpy(expected_arguments[producer_count].data(), &consumer_payload,
                  kernel.arguments.byte_length);
      for (uint32_t i = 0; i < signal_count; ++i) {
        // Full-page initialization preserves the aligned backing without
        // copying the typed object's indeterminate tail padding.
        std::memcpy(arguments[i]->host.pointer, expected_arguments[i].data(),
                    sizeof(expected_arguments[i]));
      }
      for (uint32_t i = 0; i < signal_count; ++i) {
        signals[i].value = 1;
      }

      std::array<aql::Packet, kMaxProducerCount> produce;
      for (uint32_t p = 0; p < producer_count; ++p) {
        produce[p] = aql::Dispatch(
            aql::HeaderBarrier::kDisabled, kProducerGeometry,
            kernel.private_segment_byte_length,
            kernel.group_segment_byte_length, producer_descriptor,
            arguments[p]->device_address, signal_addresses[p], kDispatchScopes);
      }
      std::array<aql::Packet, kMaxBarrierPacketCount> waits;
      for (uint32_t group = 0; group < barrier_packet_count; ++group) {
        std::array<uint64_t, kDependenciesPerPacket> dependencies = {};
        for (uint32_t slot = 0; slot < kDependenciesPerPacket; ++slot) {
          const uint32_t p = group * kDependenciesPerPacket + slot;
          if (p < producer_count) {
            dependencies[slot] = signal_addresses[p];
          }
        }
        waits[group] =
            aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                         0, dependencies, kBarrierScopes);
      }
      const auto consume = aql::Dispatch(
          aql::HeaderBarrier::kDisabled, consumer_geometry,
          kernel.private_segment_byte_length, kernel.group_segment_byte_length,
          consumer_descriptor, arguments[producer_count]->device_address,
          signal_addresses[producer_count], kDispatchScopes);

      // Every body and its borrowed storage is ready before publication. Each
      // AND blocks later launches even with its header barrier clear; the
      // consumer supplies SYSTEM acquire after all dependency groups reach
      // zero.
      GpuStoreRelease(consumer->host.write_index_address,
                      consumer_index + consumer_packets_per_epoch);
      for (uint32_t group = 0; group < barrier_packet_count; ++group) {
        ASSERT_TRUE(aql::Publish(*consumer, consumer_index++, waits[group]));
      }
      ASSERT_TRUE(aql::Publish(*consumer, consumer_index++, consume));
      GpuStoreRelease(producer->host.write_index_address,
                      producer_index + producer_count);
      for (uint32_t i = 0; i < producer_count; ++i) {
        const uint32_t p = epoch == 0 ? i : producer_count - i - 1;
        ASSERT_TRUE(aql::Publish(*producer, producer_index++, produce[p]));
      }

      // Preserve the entire consumer result before independent producer joins
      // or consumption can add synchronization to this decisive observation.
      ASSERT_TRUE(GpuWaitEqual<int64_t>(
          reinterpret_cast<uintptr_t>(&signals[producer_count].value), 0));
      std::memcpy(observed_output.data(), output->host.pointer,
                  kMaxPayloadByteLength);
      for (uint32_t p = 0; p < producer_count; ++p) {
        ASSERT_TRUE(GpuWaitEqual<int64_t>(
            reinterpret_cast<uintptr_t>(&signals[p].value), 0));
      }
      // All producer executions have joined independently of the tested edge.
      // Capture every initialized extent before any diagnostics.
      for (uint32_t p = 0; p < producer_count; ++p) {
        std::memcpy(observed_inputs[p].data(), inputs[p]->host.pointer,
                    sizeof(observed_inputs[p]));
      }
      std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                  kMaxPayloadByteLength);
      for (uint32_t i = 0; i < signal_count; ++i) {
        std::memcpy(observed_arguments[i].data(), arguments[i]->host.pointer,
                    sizeof(observed_arguments[i]));
      }
      std::memcpy(observed_signals.data(), control->host.pointer,
                  signal_count * sizeof(aql::Signal));
      std::memcpy(observed_control_guards.data(), control_guard_address,
                  control_guard_byte_length);

      for (uint32_t word = 0; word < kMaxPayloadWordCount; ++word) {
        EXPECT_EQ(observed_output[word], expected_output[word])
            << "output word=" << word;
        EXPECT_EQ(observed_intermediate[word], expected_intermediate[word])
            << "intermediate word=" << word;
      }
      for (uint32_t word = 0; word < kInputWordCount; ++word) {
        for (uint32_t p = 0; p < producer_count; ++p) {
          EXPECT_EQ(observed_inputs[p][word], expected_inputs[p][word])
              << "producer=" << p << " input word=" << word;
        }
      }
      for (uint32_t i = 0; i < signal_count; ++i) {
        for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
          EXPECT_EQ(observed_arguments[i][byte], expected_arguments[i][byte])
              << "dispatch=" << i << " kernarg byte=" << byte;
        }
      }
      for (uint32_t i = 0; i < signal_count; ++i) {
        EXPECT_EQ(observed_signals[i].kind, 1) << "signal=" << i;
        EXPECT_EQ(observed_signals[i].value, 0) << "signal=" << i;
      }
      for (uint32_t word = 0; word < observed_control_guards.size(); ++word) {
        EXPECT_EQ(observed_control_guards[word], expected_control_guards[word])
            << "control guard word=" << word;
      }
      // Both retirements run on any nonfatal mismatch. Dependency signals stay
      // zero through their last AND reader, and no backing is rewritten early.
      EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, producer_index));
      EXPECT_NO_FATAL_FAILURE(consumer->WaitConsumed(api_, consumer_index));
      if (HasFailure()) {
        return;
      }

      const std::string epoch_prefix =
          case_prefix + "_epoch_" + std::to_string(epoch + 1);
      std::string producer_order;
      for (uint32_t i = 0; i < producer_count; ++i) {
        if (i != 0) {
          producer_order += ",";
        }
        producer_order +=
            std::to_string(epoch == 0 ? i : producer_count - i - 1);
        RecordProperty(
            epoch_prefix + "_producer_" + std::to_string(i) + "_addend",
            std::to_string(producer_addends[i]));
      }
      RecordProperty(epoch_prefix + "_producer_order", producer_order);
      RecordProperty(epoch_prefix + "_consumer_addend",
                     std::to_string(kConsumerAddends[epoch]));
      RecordProperty(epoch_prefix + "_producer_packet_index",
                     std::to_string(producer_index));
      RecordProperty(epoch_prefix + "_consumer_packet_index",
                     std::to_string(consumer_index));
      RecordProperty(case_prefix + "_completed_epochs", epoch + 1);
    }

    RecordProperty(case_prefix + "_dispatches", kEpochCount * signal_count);
    RecordProperty(
        case_prefix + "_work_packet_count",
        std::to_string(producer_index - case_first_producer_packet_index +
                       consumer_index - case_first_consumer_packet_index));
    RecordProperty(case_prefix + "_producer_final_packet_index",
                   std::to_string(producer_index));
    RecordProperty(case_prefix + "_consumer_final_packet_index",
                   std::to_string(consumer_index));
  }

  RecordProperty("aql_fanin_work_packet_count",
                 std::to_string(producer_index - first_producer_packet_index +
                                consumer_index - first_consumer_packet_index));
  RecordProperty("aql_fanin_producer_final_packet_index",
                 std::to_string(producer_index));
  RecordProperty("aql_fanin_consumer_final_packet_index",
                 std::to_string(consumer_index));
}

}  // namespace
