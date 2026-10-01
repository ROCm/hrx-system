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
#include "libamdf/cts/gpu/kernels/private_roundtrip.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip_kernels.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

// These handles borrow independent allocations retained by the case fixture.
struct Workset {
  // Private-producer output borrowed by the final consumer.
  GpuMemory* intermediate = nullptr;
  // Final consumer output, including both outer guards.
  GpuMemory* output = nullptr;
  // Separate producer and consumer argument allocations.
  std::array<GpuMemory*, 2> arguments = {};
  // Producer completion, consumer completion and host gate, plus outer guards.
  GpuMemory* control = nullptr;
  // Native queue borrowing its own complete fixed scratch pool.
  GpuUserQueue* queue = nullptr;
  // Immutable producer and consumer descriptor addresses.
  std::array<uint64_t, 2> descriptors = {};
  // Next packet reservation, including completed cold publications.
  uint64_t next_packet_index = 0;
  // First workload packet after both cold publications.
  uint64_t first_packet_index = 0;
  // Number of 64-byte packets in this queue's ring.
  uint64_t capacity = 0;
};

TEST_F(AqlDispatchTest, IndependentWorksetReusePreservesPendingConsumer) {
  const auto* producer_kernel_product =
      kernels::private_roundtrip::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(producer_kernel_product, nullptr)
      << "missing compiled private_roundtrip kernel for endpoint";
  const auto& producer_kernel = *producer_kernel_product;
  RecordProperty("private_roundtrip_kernel_target", producer_kernel.target);

  const auto* consumer_kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(consumer_kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& consumer_kernel = *consumer_kernel_product;
  RecordProperty("transform_kernel_target", consumer_kernel.target);

  constexpr uint32_t kWorksetCount = 2;
  constexpr uint32_t kGenerationCount = 3;
  constexpr uint32_t kReusedGenerationCount = 2;
  constexpr uint32_t kProducerGridSize = 512;
  constexpr uint32_t kRecordWordCount = 9;
  constexpr uint32_t kPayloadWordCount = kProducerGridSize * kRecordWordCount;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kWordCount = kPayloadWordCount + 2 * kGuardWordCount;
  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kControlWordCount = kPageByteLength / sizeof(uint32_t);
  constexpr uint32_t kSignalCount = 3;
  constexpr uint32_t kControlGuardByteOffset =
      kSignalCount * sizeof(aql::Signal);
  constexpr uint32_t kControlGuardWordOffset =
      kControlGuardByteOffset / sizeof(uint32_t);
  constexpr std::array<uint32_t, kGenerationCount> kSeeds = {
      0x13579bdfu, 0xa5c31f27u, 0x2468ace1u};
  constexpr std::array<uint32_t, kGenerationCount> kRotations = {1, 7, 3};
  constexpr std::array<uint32_t, kGenerationCount> kAddends = {7, 0x80000023u,
                                                               0x10203045u};
  constexpr std::array<std::array<uint32_t, 2>, kWorksetCount>
      kIntermediateGuards = {
          {{0x619b30d5u, 0xe270c84bu}, {0x4b6d9083u, 0xb792e15fu}}};
  constexpr std::array<std::array<uint32_t, 2>, kWorksetCount> kOutputGuards = {
      {{0x7391a5d7u, 0xa83ec069u}, {0x2dc74091u, 0xc58a36e7u}}};
  constexpr std::array<uint32_t, kWorksetCount> kControlGuards = {0x68d329b7u,
                                                                  0x94e61ca5u};
  const aql::DispatchGeometry kProducerGeometry = {
      1,
      {static_cast<uint16_t>(producer_kernel.workgroup_size()), 1, 1},
      {kProducerGridSize, 1, 1}};
  const aql::DispatchGeometry kConsumerGeometry = {
      1,
      {static_cast<uint16_t>(consumer_kernel.workgroup_size()), 1, 1},
      {kPayloadWordCount, 1, 1}};
  constexpr aql::FenceScopes kDispatchScopes = {aql::FenceScope::kSystem,
                                                aql::FenceScope::kSystem};
  constexpr aql::FenceScopes kBarrierScopes = {aql::FenceScope::kNone,
                                               aql::FenceScope::kNone};

  std::array<Workset, kWorksetCount> worksets;
  for (uint32_t i = 0; i < kWorksetCount; ++i) {
    auto& workset = worksets[i];
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        uint64_t{kWordCount} * sizeof(uint32_t), &workset.intermediate));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     uint64_t{kWordCount} * sizeof(uint32_t), &workset.output));
    for (auto& argument : workset.arguments) {
      ASSERT_NO_FATAL_FAILURE(
          CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &argument));
      ASSERT_EQ(argument->device_address % producer_kernel.arguments.alignment,
                0u);
      ASSERT_EQ(argument->device_address % consumer_kernel.arguments.alignment,
                0u);
    }
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kPageByteLength, &workset.control));
    ASSERT_EQ(workset.control->device_address % alignof(aql::Signal), 0u);
    // Native reserved fields start zero and retain their ABI meaning. Only
    // bytes after all three complete signal blocks are control guards.
    std::memset(workset.control->host.pointer, 0, kPageByteLength);
    auto* signals = static_cast<aql::Signal*>(workset.control->host.pointer);
    for (uint32_t signal = 0; signal < kSignalCount; ++signal) {
      signals[signal].kind = 1;
    }
    signals[2].value = i;
    auto* control_words = static_cast<uint32_t*>(workset.control->host.pointer);
    for (uint32_t word = kControlGuardWordOffset; word < kControlWordCount;
         ++word) {
      control_words[word] = kControlGuards[i];
    }

    ASSERT_NO_FATAL_FAILURE(CreateFixedScratchQueue(
        producer_kernel.private_segment_byte_length, &workset.queue));
    const std::string prefix = "aql_workset_" + std::to_string(i);
    ASSERT_NO_FATAL_FAILURE(PublishKernel(
        *workset.queue, producer_kernel, (prefix + "_private_kernel").c_str(),
        &workset.next_packet_index, &workset.descriptors[0]));
    ASSERT_NO_FATAL_FAILURE(PublishKernel(
        *workset.queue, consumer_kernel, (prefix + "_consumer_kernel").c_str(),
        &workset.next_packet_index, &workset.descriptors[1]));
    workset.first_packet_index = workset.next_packet_index;
    workset.capacity =
        workset.queue->host.ring_byte_length / sizeof(aql::Packet);
    const uint32_t generations = i == 0 ? kReusedGenerationCount : 1;
    ASSERT_GE(workset.capacity, workset.first_packet_index + 3 * generations);
  }

  std::array<std::array<uint32_t, kWordCount>, kGenerationCount>
      expected_intermediate;
  std::array<std::array<uint32_t, kWordCount>, kGenerationCount>
      expected_output;
  std::array<std::array<std::array<uint8_t, kPageByteLength>, 2>,
             kGenerationCount>
      expected_arguments;
  std::array<std::array<aql::Packet, 3>, kGenerationCount> packets;
  for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
    const uint32_t workset_index = generation < kReusedGenerationCount ? 0 : 1;
    const auto& workset = worksets[workset_index];
    expected_intermediate[generation].fill(
        kIntermediateGuards[workset_index][1]);
    expected_output[generation].fill(kOutputGuards[workset_index][1]);
    for (uint32_t word = 0; word < kGuardWordCount; ++word) {
      expected_intermediate[generation][word] =
          kIntermediateGuards[workset_index][0];
      expected_output[generation][word] = kOutputGuards[workset_index][0];
    }
    for (uint32_t workitem = 0; workitem < kProducerGridSize; ++workitem) {
      const auto record = kernels::private_roundtrip::ExpectedRecord(
          workitem, kSeeds[generation], kRotations[generation]);
      for (uint32_t slot = 0; slot < kRecordWordCount; ++slot) {
        const uint32_t word =
            kGuardWordCount + workitem * kRecordWordCount + slot;
        expected_intermediate[generation][word] = record[slot];
        // The consumer oracle depends only on the independent CPU record.
        expected_output[generation][word] = static_cast<uint32_t>(
            uint64_t{record[slot]} * 3 + kAddends[generation]);
      }
    }
    const kernels::private_roundtrip::Arguments producer_arguments = {
        workset.intermediate->device_address +
            kGuardWordCount * sizeof(uint32_t),
        kSeeds[generation], kRotations[generation]};
    const kernels::transform::Arguments consumer_arguments = {
        workset.intermediate->device_address +
            kGuardWordCount * sizeof(uint32_t),
        workset.output->device_address + kGuardWordCount * sizeof(uint32_t),
        kPayloadWordCount, kAddends[generation]};
    expected_arguments[generation][0].fill(0);
    expected_arguments[generation][1].fill(0);
    std::memcpy(expected_arguments[generation][0].data(), &producer_arguments,
                producer_kernel.arguments.byte_length);
    std::memcpy(expected_arguments[generation][1].data(), &consumer_arguments,
                consumer_kernel.arguments.byte_length);
    const uint64_t signal_address = workset.control->device_address;
    packets[generation][0] = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kProducerGeometry,
        producer_kernel.private_segment_byte_length,
        producer_kernel.group_segment_byte_length, workset.descriptors[0],
        workset.arguments[0]->device_address, signal_address, kDispatchScopes);
    packets[generation][1] = aql::Barrier(
        aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
        {signal_address, signal_address + 2 * sizeof(aql::Signal), 0, 0, 0},
        kBarrierScopes);
    packets[generation][2] = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kConsumerGeometry,
        consumer_kernel.private_segment_byte_length,
        consumer_kernel.group_segment_byte_length, workset.descriptors[1],
        workset.arguments[1]->device_address,
        signal_address + sizeof(aql::Signal), kDispatchScopes);
  }

  std::array<uint32_t, kWordCount> observed_intermediate;
  std::array<uint32_t, kWordCount> observed_output;
  std::array<std::array<uint8_t, kPageByteLength>, 2> observed_arguments;
  std::array<uint32_t, kControlWordCount> observed_control;
  std::array<int64_t, kGenerationCount> observed_pending_gate = {};
  std::array<int64_t, kGenerationCount> observed_pending_completion = {};
  uint32_t completed_reused_generations = 0;

  // This oracle consumes snapshots only. Both callsites acquire and capture
  // every initialized extent before invoking it, and retire unconditionally.
  const auto check_observations = [&](uint32_t generation,
                                      uint32_t workset_index) {
    for (uint32_t word = 0; word < kWordCount; ++word) {
      EXPECT_EQ(observed_output[word], expected_output[generation][word])
          << "generation=" << generation << " output word=" << word;
      EXPECT_EQ(observed_intermediate[word],
                expected_intermediate[generation][word])
          << "generation=" << generation << " intermediate word=" << word;
    }
    for (uint32_t argument = 0; argument < observed_arguments.size();
         ++argument) {
      for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
        EXPECT_EQ(observed_arguments[argument][byte],
                  expected_arguments[generation][argument][byte])
            << "generation=" << generation << " argument=" << argument
            << " byte=" << byte;
      }
    }
    for (uint32_t signal = 0; signal < kSignalCount; ++signal) {
      aql::Signal observed_signal;
      std::memcpy(&observed_signal,
                  observed_control.data() +
                      signal * sizeof(aql::Signal) / sizeof(uint32_t),
                  sizeof(observed_signal));
      EXPECT_EQ(observed_signal.kind, 1) << "signal=" << signal;
      EXPECT_EQ(observed_signal.value, 0) << "signal=" << signal;
    }
    for (uint32_t word = kControlGuardWordOffset; word < kControlWordCount;
         ++word) {
      EXPECT_EQ(observed_control[word], kControlGuards[workset_index])
          << "control guard word=" << word;
    }
  };

  // Initialize both worksets before either is published. Generation 1 uses
  // only Q0's retired backing later; all expected bytes and packets exist now.
  for (uint32_t i = 0; i < kWorksetCount; ++i) {
    auto& workset = worksets[i];
    const uint32_t generation = i == 0 ? 0 : 2;
    std::memcpy(workset.intermediate->host.pointer,
                expected_intermediate[generation].data(),
                sizeof(expected_intermediate[generation]));
    std::memcpy(workset.output->host.pointer,
                expected_output[generation].data(),
                sizeof(expected_output[generation]));
    auto* intermediate_words =
        static_cast<uint32_t*>(workset.intermediate->host.pointer);
    auto* output_words = static_cast<uint32_t*>(workset.output->host.pointer);
    for (uint32_t word = kGuardWordCount;
         word < kGuardWordCount + kPayloadWordCount; ++word) {
      intermediate_words[word] = ~expected_intermediate[generation][word];
      output_words[word] = ~expected_output[generation][word];
    }
    for (uint32_t argument = 0; argument < workset.arguments.size();
         ++argument) {
      std::memcpy(workset.arguments[argument]->host.pointer,
                  expected_arguments[generation][argument].data(),
                  kPageByteLength);
    }
    auto* signals = static_cast<aql::Signal*>(workset.control->host.pointer);
    signals[0].value = 1;
    signals[1].value = 1;
  }

  auto& parked = worksets[1];
  auto* parked_signals =
      static_cast<aql::Signal*>(parked.control->host.pointer);
  GpuStoreRelease(parked.queue->host.write_index_address,
                  parked.next_packet_index + packets[2].size());
  for (const auto& packet : packets[2]) {
    aql::Publish(*parked.queue, parked.next_packet_index++, packet);
  }
  // Join the finite producer before relying on independent queue progress.
  // G1 never passed through zero: its consumer remains pending, without any
  // claim that the CP has already begun monitoring the AND.
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&parked_signals[0].value),
                        0);
  observed_pending_gate[0] = GpuLoadAcquire<int64_t>(
      reinterpret_cast<uintptr_t>(&parked_signals[2].value));
  observed_pending_completion[0] = GpuLoadAcquire<int64_t>(
      reinterpret_cast<uintptr_t>(&parked_signals[1].value));
  EXPECT_EQ(observed_pending_gate[0], 1);
  EXPECT_EQ(observed_pending_completion[0], 1);

  auto& reused = worksets[0];
  auto* reused_signals =
      static_cast<aql::Signal*>(reused.control->host.pointer);
  for (uint32_t generation = 0;
       generation < kReusedGenerationCount && !HasFailure(); ++generation) {
    if (generation != 0) {
      // Only the completed and consumed workset is rewritten. The parked
      // workset's payload, arguments and dependency signals remain retained.
      std::memcpy(reused.intermediate->host.pointer,
                  expected_intermediate[generation].data(),
                  sizeof(expected_intermediate[generation]));
      std::memcpy(reused.output->host.pointer,
                  expected_output[generation].data(),
                  sizeof(expected_output[generation]));
      auto* intermediate_words =
          static_cast<uint32_t*>(reused.intermediate->host.pointer);
      auto* output_words = static_cast<uint32_t*>(reused.output->host.pointer);
      for (uint32_t word = kGuardWordCount;
           word < kGuardWordCount + kPayloadWordCount; ++word) {
        intermediate_words[word] = ~expected_intermediate[generation][word];
        output_words[word] = ~expected_output[generation][word];
      }
      for (uint32_t argument = 0; argument < reused.arguments.size();
           ++argument) {
        std::memcpy(reused.arguments[argument]->host.pointer,
                    expected_arguments[generation][argument].data(),
                    kPageByteLength);
      }
      GpuStoreRelease<int64_t>(
          reinterpret_cast<uintptr_t>(&reused_signals[0].value), 1);
      GpuStoreRelease<int64_t>(
          reinterpret_cast<uintptr_t>(&reused_signals[1].value), 1);
    }
    GpuStoreRelease(reused.queue->host.write_index_address,
                    reused.next_packet_index + packets[generation].size());
    for (const auto& packet : packets[generation]) {
      aql::Publish(*reused.queue, reused.next_packet_index++, packet);
    }
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&reused_signals[1].value),
                          0);
    std::memcpy(observed_output.data(), reused.output->host.pointer,
                sizeof(observed_output));
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&reused_signals[0].value),
                          0);
    std::memcpy(observed_intermediate.data(), reused.intermediate->host.pointer,
                sizeof(observed_intermediate));
    for (uint32_t argument = 0; argument < observed_arguments.size();
         ++argument) {
      std::memcpy(observed_arguments[argument].data(),
                  reused.arguments[argument]->host.pointer, kPageByteLength);
    }
    std::memcpy(observed_control.data(), reused.control->host.pointer,
                sizeof(observed_control));
    observed_pending_gate[generation + 1] = GpuLoadAcquire<int64_t>(
        reinterpret_cast<uintptr_t>(&parked_signals[2].value));
    observed_pending_completion[generation + 1] = GpuLoadAcquire<int64_t>(
        reinterpret_cast<uintptr_t>(&parked_signals[1].value));

    check_observations(generation, 0);
    EXPECT_EQ(observed_pending_gate[generation + 1], 1);
    EXPECT_EQ(observed_pending_completion[generation + 1], 1);
    EXPECT_NO_FATAL_FAILURE(
        reused.queue->WaitConsumed(api_, reused.next_packet_index));
    if (!HasFailure()) {
      ++completed_reused_generations;
    }
  }

  // Every live-phase path reaches this terminal drain, including a failed Q0
  // oracle or retirement. Opening the gate completes the already-published
  // valid graph; no failing generation is retried and Q1 is never rewritten.
  GpuStoreRelease<int64_t>(
      reinterpret_cast<uintptr_t>(&parked_signals[2].value), 0);
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&parked_signals[1].value),
                        0);
  std::memcpy(observed_output.data(), parked.output->host.pointer,
              sizeof(observed_output));
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&parked_signals[0].value),
                        0);
  std::memcpy(observed_intermediate.data(), parked.intermediate->host.pointer,
              sizeof(observed_intermediate));
  for (uint32_t argument = 0; argument < observed_arguments.size();
       ++argument) {
    std::memcpy(observed_arguments[argument].data(),
                parked.arguments[argument]->host.pointer, kPageByteLength);
  }
  std::memcpy(observed_control.data(), parked.control->host.pointer,
              sizeof(observed_control));
  check_observations(2, 1);
  EXPECT_NO_FATAL_FAILURE(
      parked.queue->WaitConsumed(api_, parked.next_packet_index));
  if (HasFailure()) {
    return;
  }

  RecordProperty("aql_workset_count", kWorksetCount);
  RecordProperty("aql_workset_scratch_pool_count", kWorksetCount);
  RecordProperty("aql_workset_private_segment_byte_length",
                 producer_kernel.private_segment_byte_length);
  RecordProperty("aql_workset_producer_grid_size", kProducerGridSize);
  RecordProperty("aql_workset_consumer_grid_size", kPayloadWordCount);
  RecordProperty("aql_workset_workgroup_size",
                 producer_kernel.workgroup_size());
  RecordProperty("aql_workset_checked_bytes_per_payload",
                 kWordCount * sizeof(uint32_t));
  RecordProperty("aql_workset_checked_bytes_per_kernarg", kPageByteLength);
  RecordProperty("aql_workset_checked_bytes_per_control", kPageByteLength);
  RecordProperty("aql_workset_control_guard_byte_offset",
                 kControlGuardByteOffset);
  RecordProperty("aql_workset_header_barrier", "disabled");
  RecordProperty("aql_workset_dispatch_scopes", "system,system");
  RecordProperty("aql_workset_barrier_scopes", "none,none");
  RecordProperty("aql_workset_dependency_slots", "0,1");
  RecordProperty("aql_workset_0_completed_generations",
                 completed_reused_generations);
  RecordProperty("aql_workset_1_completed_generations", 1);
  RecordProperty("aql_workset_dispatches", 2 * kGenerationCount);
  RecordProperty(
      "aql_workset_work_packet_count",
      std::to_string(reused.next_packet_index - reused.first_packet_index +
                     parked.next_packet_index - parked.first_packet_index));
  for (uint32_t i = 0; i < kWorksetCount; ++i) {
    const std::string prefix = "aql_workset_" + std::to_string(i);
    RecordProperty(prefix + "_first_work_packet_index",
                   std::to_string(worksets[i].first_packet_index));
    RecordProperty(prefix + "_final_packet_index",
                   std::to_string(worksets[i].next_packet_index));
    RecordProperty(prefix + "_ring_capacity_packets",
                   std::to_string(worksets[i].capacity));
  }
  for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
    const std::string prefix =
        "aql_workset_generation_" + std::to_string(generation);
    RecordProperty(prefix + "_seed", std::to_string(kSeeds[generation]));
    RecordProperty(prefix + "_rotation", kRotations[generation]);
    RecordProperty(prefix + "_addend", std::to_string(kAddends[generation]));
    const std::string pending_prefix =
        "aql_workset_pending_" + std::to_string(generation);
    RecordProperty(pending_prefix + "_gate",
                   std::to_string(observed_pending_gate[generation]));
    RecordProperty(pending_prefix + "_completion",
                   std::to_string(observed_pending_completion[generation]));
  }
}

}  // namespace
