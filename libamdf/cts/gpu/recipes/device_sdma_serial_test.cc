// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "libamdf/cts/gpu/kernels/device_sdma.h"
#include "libamdf/cts/gpu/kernels/device_sdma_kernels.h"
#include "libamdf/cts/gpu/recipes/device_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

struct AqlProducer {
  // Borrowed case-owned native queue, retained through checked teardown.
  GpuUserQueue* queue = nullptr;
  // Next packet after cold executable publication and workload dispatch.
  uint64_t packet_index = 0;
  // Descriptor in the dispatch fixture's immutable executable backing.
  uint64_t descriptor_address = 0;
};

struct Pm4Producer {
  // Borrowed case-owned command stream, retained through checked teardown.
  GpuCommandQueue* queue = nullptr;
  // Compiler launch facts with the case-owned executable's entry address.
  Pm4ComputeProgram program = {};
};

// The launch carrier varies; the shader, data-dependent choices, ownership
// protocol and complete CPU oracle remain identical.
template <typename DispatchTest>
class DeviceGeneratedSdmaSerialTest : public DeviceSdmaTest<DispatchTest> {
 protected:
  using Producer =
      std::conditional_t<std::is_same_v<DispatchTest, AqlDispatchTest>,
                         AqlProducer, Pm4Producer>;

  void RunDependentCopies();
};

template <typename DispatchTest>
void DeviceGeneratedSdmaSerialTest<DispatchTest>::RunDependentCopies() {
  const auto* kernel =
      kernels::device_sdma::kKernels.Find(this->gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing device SDMA kernel for endpoint";
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_LE(kernel->arguments.byte_length,
            sizeof(kernels::device_sdma::Arguments));
  this->RecordProperty("device_sdma_kernel_target", kernel->target);
  this->RecordProperty("device_sdma_kernel_sha256", kernel->hsaco_sha256);
  this->RecordProperty("sdma_format_features",
                       std::to_string(this->sdma_family_.format_features));

  constexpr uint32_t kRoundCount = 257;
  constexpr uint32_t kPageWordCount = 128;
  constexpr uint32_t kPayloadWordOffset = 16;
  constexpr uint32_t kPayloadWordCount = 64;
  constexpr uint32_t kRecordWordCount = 72;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kAllocationWordCount = 4096 / sizeof(uint32_t);
  constexpr uint32_t kCompletionByteOffset = 64;
  constexpr uint32_t kSeed = 0x91e10da5u;
  constexpr uint32_t kSourceGuard = 0x75db8163u;
  constexpr uint32_t kDestinationGuard = 0x32a49bc7u;
  constexpr uint32_t kControlGuard = 0x69ca714bu;
  constexpr uint32_t kRecordGuard = 0xab3265c9u;
  constexpr uint64_t kRecordByteLength =
      ((uint64_t{kRoundCount} * kRecordWordCount + 2 * kGuardWordCount) *
           sizeof(uint32_t) +
       4095) &
      ~UINT64_C(4095);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* source = nullptr;
  GpuMemory* destination = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* records = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(this->CreateMemory(kReadWrite, 4096, &destination));
  ASSERT_NO_FATAL_FAILURE(this->CreateMemory(kReadWrite, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(kReadWrite, kRecordByteLength, &records));
  ASSERT_NO_FATAL_FAILURE(
      this->CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));

  uint32_t cache_flags = 0;
  ASSERT_NO_FATAL_FAILURE(this->QueryTransferCacheFlags(source, destination,
                                                        records, &cache_flags));
  this->RecordProperty("device_sdma_cache_flags", cache_flags);

  Producer producer;
  GpuUserQueue* transfer = nullptr;
  ASSERT_NO_FATAL_FAILURE(this->CreateQueue(&producer.queue));
  ASSERT_NO_FATAL_FAILURE(this->CreateQueue(
      this->sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
          AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  ASSERT_GE(mapping.ring_byte_length, 4096u);
  ASSERT_LE(mapping.ring_byte_length, UINT64_C(1) << 32);
  ASSERT_EQ(mapping.ring_byte_length & (mapping.ring_byte_length - 1), 0u);
  if constexpr (std::is_same_v<DispatchTest, AqlDispatchTest>) {
    ASSERT_NO_FATAL_FAILURE(this->PublishKernel(
        *producer.queue, *kernel, "device_sdma", &producer.packet_index,
        &producer.descriptor_address));
  } else {
    producer.program = {
        0,
        kernel->program.resource1,
        kernel->program.resource2,
        kernel->program.resource3,
        kernel->group_segment_byte_length,
        kernel->wavefront_size,
        {kernel->workgroup_size(), 1, 1},
    };
    ASSERT_NO_FATAL_FAILURE(
        this->PrepareProgram(kernel->executable, kernel->entry_byte_offset,
                             &producer.program, "device_sdma"));
  }

  std::array<uint32_t, kAllocationWordCount> source_words;
  source_words.fill(kSourceGuard);
  for (uint32_t page = 0; page < 8; ++page) {
    for (uint32_t word = 0; word < kPayloadWordCount; ++word) {
      source_words[page * kPageWordCount + kPayloadWordOffset + word] =
          0x31415927u + page * 0x243f6a89u + word * 0x1020305u;
    }
  }
  std::array<uint32_t, kAllocationWordCount> expected_destination;
  expected_destination.fill(kDestinationGuard);
  std::array<uint32_t, kAllocationWordCount> expected_control;
  expected_control.fill(kControlGuard);
  if constexpr (std::is_same_v<DispatchTest, AqlDispatchTest>) {
    const aql::Signal final_signal = {.kind = 1};
    std::memcpy(expected_control.data(), &final_signal, sizeof(final_signal));
  } else {
    expected_control[0] = 1;
  }
  expected_control[kCompletionByteOffset / sizeof(uint32_t)] = kRoundCount;
  std::vector<uint32_t> expected_records(kRecordByteLength / sizeof(uint32_t),
                                         kRecordGuard);
  std::memcpy(source->host.pointer, source_words.data(), sizeof(source_words));
  std::memcpy(destination->host.pointer, expected_destination.data(),
              sizeof(expected_destination));
  std::memcpy(control->host.pointer, expected_control.data(),
              sizeof(expected_control));
  if constexpr (std::is_same_v<DispatchTest, AqlDispatchTest>) {
    static_cast<aql::Signal*>(control->host.pointer)->value = 1;
  } else {
    *static_cast<uint32_t*>(control->host.pointer) = 0;
  }
  auto* completion = reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(control->host.pointer) + kCompletionByteOffset);
  *completion = 0;
  std::memcpy(records->host.pointer, expected_records.data(),
              kRecordByteLength);

  // Only family-selected invariant fields come from the host encoder. The
  // shader authors every published packet and computes its addresses/length.
  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(),
                            this->sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, destination->device_address, 4);
  encoder.Fence32(control->device_address + kCompletionByteOffset, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());
  const kernels::device_sdma::Arguments payload = {
      .ring = mapping.ring_address,
      .read_index = mapping.read_index_address,
      .write_index = mapping.write_index_address,
      .notification = mapping.doorbell_address,
      .destinations = destination->device_address,
      .completion = control->device_address + kCompletionByteOffset,
      .records = records->device_address + kGuardWordCount * sizeof(uint32_t),
      .source_address = source->device_address,
      .destination_address = destination->device_address,
      .completion_address = control->device_address + kCompletionByteOffset,
      .capacity = mapping.ring_byte_length,
      .round_count = kRoundCount,
      .seed = kSeed,
      .copy_control = encoding[2],
      .fence_header = encoding[7],
      .cache_flags = cache_flags,
  };
  ASSERT_EQ(arguments->device_address % kernel->arguments.alignment, 0u);
  std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
  std::memcpy(arguments->host.pointer, &payload, kernel->arguments.byte_length);

  uint64_t expected_frontier = 0;
  uint32_t state = kSeed;
  uint32_t selected_pages = 0;
  uint32_t selected_slots = 0;
  std::array<bool, kPayloadWordCount> selected_lengths = {};
  const uint64_t chain_byte_length = 44 + ((cache_flags & 1) != 0 ? 20 : 0) +
                                     ((cache_flags & 2) != 0 ? 20 : 0);
  for (uint32_t round = 0; round < kRoundCount; ++round) {
    const uint32_t page = state & 7;
    const uint32_t slot = (state >> 3) & 3;
    const uint32_t word_count = ((state >> 8) & 63) + 1;
    selected_pages |= 1u << page;
    selected_slots |= 1u << slot;
    selected_lengths[word_count - 1] = true;
    const uint32_t source_offset = page * kPageWordCount + kPayloadWordOffset;
    const uint32_t destination_offset =
        slot * kPageWordCount + kPayloadWordOffset;
    std::copy_n(source_words.begin() + source_offset, word_count,
                expected_destination.begin() + destination_offset);
    const uint32_t next_state =
        expected_destination[destination_offset] ^
        static_cast<uint32_t>(
            uint64_t{
                expected_destination[destination_offset + word_count - 1]} +
            uint64_t{round + 1} * 0x9e3779b1u);
    const uint64_t tail =
        mapping.ring_byte_length - expected_frontier % mapping.ring_byte_length;
    if (tail < chain_byte_length) {
      expected_frontier += tail;
    }
    expected_frontier += chain_byte_length;
    const uint32_t row = kGuardWordCount + round * kRecordWordCount;
    expected_records[row] = page;
    expected_records[row + 1] = slot;
    expected_records[row + 2] = word_count;
    expected_records[row + 3] = next_state;
    expected_records[row + 4] = static_cast<uint32_t>(expected_frontier);
    expected_records[row + 5] = static_cast<uint32_t>(expected_frontier >> 32);
    expected_records[row + 6] = state;
    expected_records[row + 7] = round + 1;
    std::copy_n(expected_destination.begin() + destination_offset,
                kPayloadWordCount, expected_records.begin() + row + 8);
    state = next_state;
  }
  ASSERT_EQ(selected_pages, 0xffu);
  ASSERT_EQ(selected_slots, 0xfu);
  ASSERT_EQ(std::count(selected_lengths.begin(), selected_lengths.end(), true),
            kPayloadWordCount);
  ASSERT_GT(expected_frontier / mapping.ring_byte_length, 1u);

  std::vector<uint32_t> observed_records(expected_records.size());
  std::array<uint32_t, kAllocationWordCount> observed_destination;
  std::array<uint32_t, kAllocationWordCount> observed_source;
  std::array<uint32_t, kAllocationWordCount> observed_control;
  if constexpr (std::is_same_v<DispatchTest, AqlDispatchTest>) {
    const auto dispatch = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {1, 1, 1}},
        kernel->private_segment_byte_length, kernel->group_segment_byte_length,
        producer.descriptor_address, arguments->device_address,
        control->device_address);
    GpuStoreRelease(producer.queue->host.write_index_address,
                    producer.packet_index + 1);
    aql::Publish(*producer.queue, producer.packet_index++, dispatch);
    auto& signal = *static_cast<aql::Signal*>(control->host.pointer);
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
  } else {
    ASSERT_GE(producer.queue->words().size(), 128u);
    Pm4CommandWriter commands(producer.queue->words().data(),
                              *this->pm4_profile_);
    commands.SystemBarrier();
    commands.BindCompute(producer.program, arguments->device_address);
    commands.Dispatch(producer.program, 1, 1, 1);
    commands.ReleaseSystem32(control->device_address, 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(producer.queue->Publish(this->api_, this->gpu_api_,
                                                    commands.word_count()));
    GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(control->host.pointer),
                           1);
    this->RecordProperty("device_sdma_pm4_command_bytes",
                         commands.word_count() * sizeof(uint32_t));
  }
  // This is the only workload join. Snapshot the GPU's actual copied-data
  // transcript before command-consumption queries or native teardown.
  std::memcpy(observed_records.data(), records->host.pointer,
              kRecordByteLength);
  std::memcpy(observed_destination.data(), destination->host.pointer,
              sizeof(observed_destination));
  std::memcpy(observed_source.data(), source->host.pointer,
              sizeof(observed_source));
  std::memcpy(observed_control.data(), control->host.pointer,
              sizeof(observed_control));
  EXPECT_EQ(observed_records, expected_records);
  EXPECT_EQ(observed_destination, expected_destination);
  EXPECT_EQ(observed_source, source_words);
  EXPECT_EQ(observed_control, expected_control);
  EXPECT_EQ(GpuLoadAcquire<uint64_t>(transfer->host.write_index_address),
            expected_frontier);
  if constexpr (std::is_same_v<DispatchTest, AqlDispatchTest>) {
    EXPECT_NO_FATAL_FAILURE(
        producer.queue->WaitConsumed(this->api_, producer.packet_index));
  } else {
    EXPECT_NO_FATAL_FAILURE(producer.queue->WaitRetired(this->api_));
  }
  EXPECT_NO_FATAL_FAILURE(
      transfer->WaitConsumed(this->api_, expected_frontier));
  this->RecordProperty("device_sdma_completed_transfers", kRoundCount);
  this->RecordProperty("device_sdma_published_bytes",
                       std::to_string(expected_frontier));
  this->RecordProperty(
      "device_sdma_ring_wraps",
      std::to_string(expected_frontier / mapping.ring_byte_length));
  this->RecordProperty(
      "device_sdma_payload_lengths",
      std::count(selected_lengths.begin(), selected_lengths.end(), true));
}

using DeviceGeneratedSdmaAqlTest =
    DeviceGeneratedSdmaSerialTest<AqlDispatchTest>;
using DeviceGeneratedSdmaPm4Test =
    DeviceGeneratedSdmaSerialTest<Pm4DispatchTest>;

TEST_F(DeviceGeneratedSdmaAqlTest, DependentCopiesReuseRingAndPayload) {
  ASSERT_NO_FATAL_FAILURE(RunDependentCopies());
}

TEST_F(DeviceGeneratedSdmaPm4Test, DependentCopiesReuseRingAndPayload) {
  ASSERT_NO_FATAL_FAILURE(RunDependentCopies());
}

}  // namespace
