// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/device_sdma.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/device_sdma_batched.h"
#include "libamdf/cts/gpu/kernels/device_sdma_batched_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_consumer_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_staged.h"
#include "libamdf/cts/gpu/kernels/device_sdma_transfer_kernels.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

class DeviceGeneratedSdmaTest : public AqlDispatchTest {
 protected:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    const GpuQueueRequirements requirements = {
        .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        .roles = AMDF_QUEUE_ROLE_TRANSFER,
        .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
    };
    amdf_queue_family_info_t sdma_family = {};
    bool matches = false;
    amdf_status_t status = FindGpuQueueFamily(api_, endpoint, requirements,
                                              &sdma_family, &matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!matches) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    status = AqlDispatchTest::MatchGpuEndpoint(endpoint, &matches);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (matches) {
      sdma_family_ = sdma_family;
    }
    *out_matches = matches;
    return AMDF_STATUS_OK;
  }

  void QueryPair(const amdf_memory_site_t& producer,
                 const amdf_memory_site_t& consumer,
                 amdf_memory_pair_info_t* out_pair) {
    out_pair->type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    out_pair->structure_size = sizeof(*out_pair);
    ASSERT_EQ(api_->memory_query_pair_info(&producer, &consumer, out_pair),
              AMDF_STATUS_OK);
    ASSERT_NE(out_pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
              0u);
  }

  void CheckTransition(const amdf_cache_transition_t& transition,
                       amdf_cache_operation_t operation) {
    ASSERT_EQ(transition.kind, operation == AMDF_CACHE_OPERATION_NONE
                                   ? AMDF_CACHE_TRANSITION_KIND_NONE
                                   : AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, operation == AMDF_CACHE_OPERATION_NONE
                                       ? AMDF_CACHE_TRANSITION_EXECUTOR_NONE
                                       : AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    ASSERT_EQ(transition.operation, operation);
    ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
    ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
    ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
    ASSERT_EQ(transition.range_granularity, 0u);
  }

  void ResolveSdmaTransition(const amdf_cache_transition_t& transition,
                             amdf_cache_operation_t operation,
                             uint32_t* inout_cache_flags) {
    if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
      ASSERT_NO_FATAL_FAILURE(
          CheckTransition(transition, AMDF_CACHE_OPERATION_NONE));
      return;
    }
    ASSERT_NO_FATAL_FAILURE(CheckTransition(transition, operation));
    ASSERT_NE(
        sdma_family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
        0u);
    ASSERT_NE(sdma_family_.cache_operations & (UINT64_C(1) << operation), 0u);
    ASSERT_NE(sdma_family_.cache_transition_kinds &
                  AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
              0u);
    *inout_cache_flags |=
        operation == AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM ? 1u : 2u;
  }

  void QueryTransferCacheFlags(GpuMemory* source, GpuMemory* destination,
                               GpuMemory* records, uint32_t* out_flags) {
    amdf_memory_pair_info_t ingress = {};
    amdf_memory_pair_info_t copied = {};
    amdf_memory_pair_info_t egress = {};
    ASSERT_NO_FATAL_FAILURE(QueryPair(source->HostSite(),
                                      source->DeviceSite(sdma_family_.ordinal),
                                      &ingress));
    ASSERT_NO_FATAL_FAILURE(
        QueryPair(destination->DeviceSite(sdma_family_.ordinal),
                  destination->DeviceSite(family_.ordinal), &copied));
    ASSERT_NO_FATAL_FAILURE(QueryPair(records->DeviceSite(family_.ordinal),
                                      records->HostSite(), &egress));
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(ingress.release, AMDF_CACHE_OPERATION_NONE));
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        copied.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(egress.acquire, AMDF_CACHE_OPERATION_NONE));
    ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
        ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM, out_flags));
    ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
        copied.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
  }

  void QueryDownloadCacheFlags(GpuMemory* output, GpuMemory* readback,
                               uint32_t* out_flags) {
    amdf_memory_pair_info_t produced = {};
    amdf_memory_pair_info_t copied = {};
    amdf_memory_pair_info_t egress = {};
    ASSERT_NO_FATAL_FAILURE(QueryPair(output->DeviceSite(family_.ordinal),
                                      output->DeviceSite(sdma_family_.ordinal),
                                      &produced));
    ASSERT_NO_FATAL_FAILURE(
        QueryPair(readback->DeviceSite(sdma_family_.ordinal),
                  readback->DeviceSite(family_.ordinal), &copied));
    ASSERT_NO_FATAL_FAILURE(
        QueryPair(readback->DeviceSite(sdma_family_.ordinal),
                  readback->HostSite(), &egress));
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        produced.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(CheckTransition(
        copied.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
    ASSERT_NO_FATAL_FAILURE(
        CheckTransition(egress.acquire, AMDF_CACHE_OPERATION_NONE));
    ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
        produced.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM, out_flags));
    ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
        copied.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
    ASSERT_NO_FATAL_FAILURE(ResolveSdmaTransition(
        egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, out_flags));
  }

  // Exact transfer family selected before borrowing the cached native device.
  amdf_queue_family_info_t sdma_family_ = {};
};

TEST_F(DeviceGeneratedSdmaTest, DependentCopiesReuseRingAndPayload) {
  const auto* kernel = kernels::device_sdma::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing device SDMA kernel for endpoint";
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_LE(kernel->arguments.byte_length,
            sizeof(kernels::device_sdma::Arguments));
  RecordProperty("device_sdma_kernel_target", kernel->target);
  RecordProperty("device_sdma_kernel_sha256", kernel->hsaco_sha256);
  RecordProperty("sdma_format_features",
                 std::to_string(sdma_family_.format_features));

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
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &destination));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kRecordByteLength, &records));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));

  uint32_t cache_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, destination, records, &cache_flags));
  RecordProperty("device_sdma_cache_flags", cache_flags);

  GpuUserQueue* producer = nullptr;
  GpuUserQueue* transfer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  ASSERT_GE(mapping.ring_byte_length, 4096u);
  ASSERT_LE(mapping.ring_byte_length, UINT64_C(1) << 32);
  ASSERT_EQ(mapping.ring_byte_length & (mapping.ring_byte_length - 1), 0u);
  uint64_t packet_index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, *kernel, "device_sdma",
                                        &packet_index, &descriptor_address));

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
  aql::Signal final_signal = {};
  final_signal.kind = 1;
  std::memcpy(expected_control.data(), &final_signal, sizeof(final_signal));
  expected_control[kCompletionByteOffset / sizeof(uint32_t)] = kRoundCount;
  std::vector<uint32_t> expected_records(kRecordByteLength / sizeof(uint32_t),
                                         kRecordGuard);
  std::memcpy(source->host.pointer, source_words.data(), sizeof(source_words));
  std::memcpy(destination->host.pointer, expected_destination.data(),
              sizeof(expected_destination));
  std::memcpy(control->host.pointer, expected_control.data(),
              sizeof(expected_control));
  auto& signal = *static_cast<aql::Signal*>(control->host.pointer);
  signal.value = 1;
  auto* completion = reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(control->host.pointer) + kCompletionByteOffset);
  *completion = 0;
  std::memcpy(records->host.pointer, expected_records.data(),
              kRecordByteLength);

  // Only family-selected invariant fields come from the host encoder. The
  // shader authors every published packet and computes its addresses/length.
  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
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
  const auto dispatch = aql::Dispatch(
      aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {1, 1, 1}},
      kernel->private_segment_byte_length, kernel->group_segment_byte_length,
      descriptor_address, arguments->device_address, control->device_address);
  GpuStoreRelease(producer->host.write_index_address, packet_index + 1);
  aql::Publish(*producer, packet_index++, dispatch);
  // This is the only workload join. Snapshot the GPU's actual copied-data
  // transcript before command-consumption queries or native teardown.
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
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
  EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, packet_index));
  EXPECT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, expected_frontier));
  RecordProperty("device_sdma_completed_transfers", kRoundCount);
  RecordProperty("device_sdma_published_bytes",
                 std::to_string(expected_frontier));
  RecordProperty("device_sdma_ring_wraps",
                 std::to_string(expected_frontier / mapping.ring_byte_length));
  RecordProperty(
      "device_sdma_payload_lengths",
      std::count(selected_lengths.begin(), selected_lengths.end(), true));
}

// Each parameter uses the cached device and case-owned queues/backing. Small
// workloads cover drain boundaries; long workloads distinguish the two credits.
struct DeviceSdmaBatchCase {
  // Stable name for required native execution and structured result receipts.
  const char* name;
  // Power-of-two number of independent destination slots.
  uint32_t credit_count;
  // Finite total, including zero and partial final batches.
  uint32_t round_count;
};

class DeviceGeneratedSdmaBatchTest
    : public DeviceGeneratedSdmaTest,
      public ::testing::WithParamInterface<DeviceSdmaBatchCase> {};

TEST_P(DeviceGeneratedSdmaBatchTest, SeparateCommandAndPayloadCredits) {
  const auto test_case = GetParam();
  const auto* kernel =
      kernels::device_sdma_batched::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing batched SDMA kernel for endpoint";
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_LE(kernel->arguments.byte_length,
            sizeof(kernels::device_sdma_batched::Arguments));
  RecordProperty("device_sdma_kernel_target", kernel->target);
  RecordProperty("device_sdma_kernel_sha256", kernel->hsaco_sha256);
  RecordProperty("device_sdma_credits", test_case.credit_count);
  RecordProperty("device_sdma_transfers", test_case.round_count);

  constexpr uint32_t kBatchSize = 193;
  constexpr uint32_t kPageWordCount = 128;
  constexpr uint32_t kPayloadWordOffset = 16;
  constexpr uint32_t kPayloadWordCount = 64;
  constexpr uint32_t kRecordWordCount = 80;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kAllocationWordCount = 4096 / sizeof(uint32_t);
  constexpr uint32_t kCompletionByteOffset = 64;
  constexpr uint32_t kSeed = 0x91e10da5u;
  constexpr uint32_t kSourceGuard = 0x75db8163u;
  constexpr uint32_t kDestinationGuard = 0x32a49bc7u;
  constexpr uint32_t kControlGuard = 0x69ca714bu;
  constexpr uint32_t kRecordGuard = 0xab3265c9u;
  constexpr uint32_t kStatisticsGuard = 0x28b704edu;
  const uint64_t record_byte_length =
      ((uint64_t{test_case.round_count} * kRecordWordCount +
        2 * kGuardWordCount) *
           sizeof(uint32_t) +
       4095) &
      ~UINT64_C(4095);
  const uint64_t destination_byte_length =
      (uint64_t{test_case.credit_count} * 512 + 4095) & ~UINT64_C(4095);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* source = nullptr;
  GpuMemory* destination = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* records = nullptr;
  GpuMemory* statistics = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, destination_byte_length, &destination));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, record_byte_length, &records));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &statistics));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  uint32_t cache_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, destination, records, &cache_flags));
  RecordProperty("device_sdma_cache_flags", cache_flags);

  GpuUserQueue* producer = nullptr;
  GpuUserQueue* transfer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  ASSERT_GE(mapping.ring_byte_length, 4096u);
  ASSERT_LE(mapping.ring_byte_length, UINT64_C(1) << 32);
  ASSERT_EQ(mapping.ring_byte_length & (mapping.ring_byte_length - 1), 0u);
  uint64_t packet_index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, *kernel, "device_sdma_batch",
                                        &packet_index, &descriptor_address));

  std::array<uint32_t, kAllocationWordCount> source_words;
  source_words.fill(kSourceGuard);
  for (uint32_t page = 0; page < 8; ++page) {
    for (uint32_t word = 0; word < kPayloadWordCount; ++word) {
      source_words[page * kPageWordCount + kPayloadWordOffset + word] =
          0x31415927u + page * 0x243f6a89u + word * 0x1020305u;
    }
  }
  std::vector<uint32_t> expected_destination(
      destination_byte_length / sizeof(uint32_t), kDestinationGuard);
  std::array<uint32_t, kAllocationWordCount> expected_control;
  expected_control.fill(kControlGuard);
  aql::Signal final_signal = {};
  final_signal.kind = 1;
  std::memcpy(expected_control.data(), &final_signal, sizeof(final_signal));
  expected_control[kCompletionByteOffset / sizeof(uint32_t)] =
      test_case.round_count;
  std::vector<uint32_t> expected_records(record_byte_length / sizeof(uint32_t),
                                         kRecordGuard);
  std::array<uint32_t, kAllocationWordCount> initial_statistics;
  initial_statistics.fill(kStatisticsGuard);
  std::memcpy(source->host.pointer, source_words.data(), sizeof(source_words));
  std::memcpy(destination->host.pointer, expected_destination.data(),
              destination_byte_length);
  std::memcpy(control->host.pointer, expected_control.data(),
              sizeof(expected_control));
  std::memcpy(records->host.pointer, expected_records.data(),
              record_byte_length);
  std::memcpy(statistics->host.pointer, initial_statistics.data(),
              sizeof(initial_statistics));
  auto& signal = *static_cast<aql::Signal*>(control->host.pointer);
  signal.value = 1;
  auto* completion = reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(control->host.pointer) + kCompletionByteOffset);
  *completion = 0;

  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, destination->device_address, 4);
  encoder.Fence32(control->device_address + kCompletionByteOffset, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());
  const kernels::device_sdma_batched::Arguments payload = {
      .ring = mapping.ring_address,
      .read_index = mapping.read_index_address,
      .write_index = mapping.write_index_address,
      .notification = mapping.doorbell_address,
      .destinations = destination->device_address,
      .completion = control->device_address + kCompletionByteOffset,
      .records = records->device_address + kGuardWordCount * sizeof(uint32_t),
      .statistics =
          statistics->device_address + kGuardWordCount * sizeof(uint32_t),
      .source_address = source->device_address,
      .destination_address = destination->device_address,
      .completion_address = control->device_address + kCompletionByteOffset,
      .capacity = mapping.ring_byte_length,
      .round_count = test_case.round_count,
      .batch_size = kBatchSize,
      .credit_count = test_case.credit_count,
      .seed = kSeed,
      .copy_control = encoding[2],
      .fence_header = encoding[7],
      .cache_flags = cache_flags,
  };
  ASSERT_EQ(arguments->device_address % kernel->arguments.alignment, 0u);
  std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
  std::memcpy(arguments->host.pointer, &payload, kernel->arguments.byte_length);

  // The oracle computes only dataflow and packet extents. Actual scheduling
  // observations are checked against ownership inequalities after completion.
  const uint64_t chain_byte_length = 44 + ((cache_flags & 1) != 0 ? 20 : 0) +
                                     ((cache_flags & 2) != 0 ? 20 : 0);
  std::vector<uint64_t> frontiers(test_case.round_count + 1);
  uint32_t state = kSeed;
  uint32_t batch_seed = kSeed;
  uint32_t selected_pages = 0;
  std::array<bool, kPayloadWordCount> selected_lengths = {};
  for (uint32_t round = 0; round < test_case.round_count; ++round) {
    if (round % kBatchSize == 0) {
      batch_seed = state;
    }
    const uint32_t selection =
        batch_seed ^ static_cast<uint32_t>(uint64_t{round + 1} * 0x9e3779b1u);
    const uint32_t page = selection & 7;
    const uint32_t slot = round & (test_case.credit_count - 1);
    const uint32_t word_count = ((selection >> 8) & 63) + 1;
    selected_pages |= 1u << page;
    selected_lengths[word_count - 1] = true;
    const uint32_t source_offset = page * kPageWordCount + kPayloadWordOffset;
    const uint32_t destination_offset =
        slot * kPageWordCount + kPayloadWordOffset;
    std::copy_n(source_words.begin() + source_offset, word_count,
                expected_destination.begin() + destination_offset);
    const uint32_t copied_mix =
        expected_destination[destination_offset] ^
        static_cast<uint32_t>(
            uint64_t{
                expected_destination[destination_offset + word_count - 1]} +
            uint64_t{round + 1} * 0x9e3779b1u);
    const uint32_t next_state = state + copied_mix;
    const uint64_t tail =
        mapping.ring_byte_length - frontiers[round] % mapping.ring_byte_length;
    const uint64_t padding = tail < chain_byte_length ? tail : 0;
    frontiers[round + 1] = frontiers[round] + padding + chain_byte_length;
    const uint32_t row = kGuardWordCount + round * kRecordWordCount;
    expected_records[row] = page;
    expected_records[row + 1] = slot;
    expected_records[row + 2] = word_count;
    expected_records[row + 3] = batch_seed;
    expected_records[row + 4] = next_state;
    expected_records[row + 6] = round + 1;
    expected_records[row + 7] = static_cast<uint32_t>(padding);
    expected_records[row + 8] = static_cast<uint32_t>(frontiers[round + 1]);
    expected_records[row + 9] =
        static_cast<uint32_t>(frontiers[round + 1] >> 32);
    std::copy_n(expected_destination.begin() + destination_offset,
                kPayloadWordCount, expected_records.begin() + row + 16);
    state = next_state;
  }
  if (test_case.round_count == 417) {
    ASSERT_EQ(selected_pages, 0xffu);
    ASSERT_EQ(
        std::count(selected_lengths.begin(), selected_lengths.end(), true),
        kPayloadWordCount);
    ASSERT_GT(frontiers.back() / mapping.ring_byte_length, 1u);
  }

  std::vector<uint32_t> observed_records(expected_records.size());
  std::vector<uint32_t> observed_destination(expected_destination.size());
  std::array<uint32_t, kAllocationWordCount> observed_source;
  std::array<uint32_t, kAllocationWordCount> observed_control;
  std::array<uint32_t, kAllocationWordCount> observed_statistics;
  const auto dispatch = aql::Dispatch(
      aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {1, 1, 1}},
      kernel->private_segment_byte_length, kernel->group_segment_byte_length,
      descriptor_address, arguments->device_address, control->device_address);
  GpuStoreRelease(producer->host.write_index_address, packet_index + 1);
  aql::Publish(*producer, packet_index++, dispatch);
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
  std::memcpy(observed_records.data(), records->host.pointer,
              record_byte_length);
  std::memcpy(observed_destination.data(), destination->host.pointer,
              destination_byte_length);
  std::memcpy(observed_source.data(), source->host.pointer,
              sizeof(observed_source));
  std::memcpy(observed_control.data(), control->host.pointer,
              sizeof(observed_control));
  std::memcpy(observed_statistics.data(), statistics->host.pointer,
              sizeof(observed_statistics));
  EXPECT_EQ(observed_destination, expected_destination);
  EXPECT_EQ(observed_source, source_words);
  EXPECT_EQ(observed_control, expected_control);

  uint64_t previous_read = 0;
  uint64_t previous_published = 0;
  uint32_t previous_consumed = 0;
  uint32_t previous_completion = 0;
  uint32_t skipped_generations = 0;
  uint32_t peak_live = 0;
  uint64_t peak_bytes = 0;
  for (uint32_t round = 0; round < test_case.round_count; ++round) {
    SCOPED_TRACE(round);
    const uint32_t row = kGuardWordCount + round * kRecordWordCount;
    const uint32_t completion_generation = observed_records[row + 5];
    EXPECT_GE(completion_generation, round + 1);
    EXPECT_GE(completion_generation, previous_completion);
    EXPECT_LE(completion_generation, test_case.round_count);
    skipped_generations += completion_generation > round + 1;
    previous_completion = completion_generation;
    const uint64_t read = uint64_t{observed_records[row + 10]} |
                          (uint64_t{observed_records[row + 11]} << 32);
    const uint64_t published = uint64_t{observed_records[row + 12]} |
                               (uint64_t{observed_records[row + 13]} << 32);
    const uint32_t live = observed_records[row + 14];
    const uint32_t consumed = observed_records[row + 15];
    EXPECT_GE(read, previous_read);
    EXPECT_LE(read, published);
    EXPECT_GE(published, previous_published);
    EXPECT_LE(published, frontiers[round]);
    const auto published_position =
        std::lower_bound(frontiers.begin(), frontiers.end(), published);
    EXPECT_TRUE(published_position != frontiers.end() &&
                *published_position == published);
    EXPECT_GE(consumed, previous_consumed);
    EXPECT_LE(consumed, round);
    EXPECT_LE(consumed,
              static_cast<uint32_t>(published_position - frontiers.begin()));
    EXPECT_EQ(live, round + 1 - consumed);
    EXPECT_LE(live, test_case.credit_count);
    EXPECT_GT(live, 0u);
    EXPECT_LT(frontiers[round + 1] - read, mapping.ring_byte_length);
    peak_live = std::max(peak_live, live);
    peak_bytes = std::max(peak_bytes, frontiers[round + 1] - read);
    previous_read = read;
    previous_published = published;
    previous_consumed = consumed;
    // Completion and reservation observations depend on native progress and
    // were checked above. Every remaining word, including backing, is exact.
    expected_records[row + 5] = completion_generation;
    std::copy_n(observed_records.begin() + row + 10, 6,
                expected_records.begin() + row + 10);
  }
  EXPECT_EQ(observed_records, expected_records);
  const auto* stats = observed_statistics.data() + kGuardWordCount;
  const uint32_t batch_count =
      (test_case.round_count + kBatchSize - 1) / kBatchSize;
  EXPECT_GE(stats[0], batch_count);
  EXPECT_LE(stats[0], test_case.round_count);
  EXPECT_EQ(stats[3], peak_live);
  EXPECT_EQ(stats[4], peak_bytes);
  EXPECT_LT(peak_bytes, mapping.ring_byte_length);
  EXPECT_EQ(stats[5], batch_count);
  EXPECT_EQ(stats[6], state);
  EXPECT_EQ(stats[7], test_case.round_count);
  EXPECT_EQ(stats[8], test_case.round_count);
  EXPECT_LE(stats[9], stats[0]);
  EXPECT_EQ(uint64_t{stats[10]} | (uint64_t{stats[11]} << 32),
            frontiers.back());
  const uint64_t ring_window =
      (mapping.ring_byte_length - 1) / chain_byte_length;
  const uint32_t request_window = std::min(test_case.round_count, kBatchSize);
  const uint32_t first_window = static_cast<uint32_t>(std::min<uint64_t>(
      {request_window, test_case.credit_count, ring_window}));
  EXPECT_GE(peak_live, first_window);
  if (test_case.credit_count <= ring_window &&
      request_window > test_case.credit_count) {
    EXPECT_GT(stats[2], 0u)
        << "small payload windows must encounter credit pressure";
  }
  if (test_case.credit_count > ring_window && request_window > ring_window) {
    EXPECT_GT(stats[1], 0u)
        << "unpublished prefix must encounter ring pressure";
  }
  if (first_window > 1) {
    EXPECT_LT(stats[0], test_case.round_count)
        << "first window must publish as a batch";
    if (test_case.round_count > first_window) {
      EXPECT_GT(stats[9], 0u)
          << "publication must resume with live payload credits";
    }
  }
  if (test_case.round_count == 0) {
    EXPECT_EQ(stats[1], 0u);
    EXPECT_EQ(stats[2], 0u);
  }
  std::copy_n(stats, 12, initial_statistics.begin() + kGuardWordCount);
  EXPECT_EQ(observed_statistics, initial_statistics);
  EXPECT_EQ(GpuLoadAcquire<uint64_t>(transfer->host.write_index_address),
            frontiers.back());
  EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, packet_index));
  EXPECT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontiers.back()));
  RecordProperty("device_sdma_publications", stats[0]);
  RecordProperty("device_sdma_ring_blocks", stats[1]);
  RecordProperty("device_sdma_credit_blocks", stats[2]);
  RecordProperty("device_sdma_peak_payloads", stats[3]);
  RecordProperty("device_sdma_peak_reserved_bytes", stats[4]);
  RecordProperty("device_sdma_appends_with_live_payloads", stats[9]);
  RecordProperty("device_sdma_skipped_completion_generations",
                 skipped_generations);
  RecordProperty("device_sdma_published_bytes",
                 std::to_string(frontiers.back()));
  RecordProperty("device_sdma_ring_wraps",
                 std::to_string(frontiers.back() / mapping.ring_byte_length));
}

INSTANTIATE_TEST_SUITE_P(
    Credits, DeviceGeneratedSdmaBatchTest,
    ::testing::Values(DeviceSdmaBatchCase{"Empty", 4, 0},
                      DeviceSdmaBatchCase{"Single", 4, 1},
                      DeviceSdmaBatchCase{"BeforeCreditWindow", 4, 3},
                      DeviceSdmaBatchCase{"ExactCreditWindow", 4, 4},
                      DeviceSdmaBatchCase{"AfterCreditWindow", 4, 5},
                      DeviceSdmaBatchCase{"ExactBatch", 4, 193},
                      DeviceSdmaBatchCase{"AfterBatch", 4, 194},
                      DeviceSdmaBatchCase{"Credit1", 1, 417},
                      DeviceSdmaBatchCase{"Credit2", 2, 417},
                      DeviceSdmaBatchCase{"Credit4", 4, 417},
                      DeviceSdmaBatchCase{"Credit8", 8, 417},
                      DeviceSdmaBatchCase{"Credit16", 16, 417},
                      DeviceSdmaBatchCase{"Credit32", 32, 417},
                      DeviceSdmaBatchCase{"Credit64", 64, 417},
                      DeviceSdmaBatchCase{"Credit128", 128, 417}),
    [](const ::testing::TestParamInfo<DeviceSdmaBatchCase>& info) {
      return info.param.name;
    });

struct DeviceSdmaStagedCase {
  // Memory placement required for both reusable payload allocations.
  amdf_memory_class_t memory_class;
  // Positive payload lengths selected by GPU computation, in words.
  std::array<uint32_t, 4> word_counts;
  // Number of separately dispatched upload/compute/download jobs.
  uint32_t job_count;
  // Number of reusable input/output pairs, a power of two.
  uint32_t slot_count;
  // Stable parameter name including placement, size family and slot count.
  std::string name;
};

std::vector<DeviceSdmaStagedCase> StagedCases() {
  std::vector<DeviceSdmaStagedCase> cases;
  for (const amdf_memory_class_t memory_class :
       {AMDF_MEMORY_CLASS_SYSTEM, AMDF_MEMORY_CLASS_LOCAL}) {
    for (uint32_t slot_count : {1u, 2u, 4u}) {
      const std::string placement =
          memory_class == AMDF_MEMORY_CLASS_LOCAL ? "Local" : "System";
      const std::string suffix = "Slots" + std::to_string(slot_count);
      // Packed index/embedding/main-cache rows and one 24-row gather, followed
      // by page, cache-block and partial-projection extents. Arithmetic stays
      // integer and exact; this case tests movement rather than model math.
      cases.push_back({memory_class,
                       {17, 66, 72, 1584},
                       97,
                       slot_count,
                       placement + "Rows" + suffix});
      cases.push_back({memory_class,
                       {1024, 4352, 18432, 262144},
                       17,
                       slot_count,
                       placement + "Blocks" + suffix});
    }
  }
  return cases;
}

class DeviceGeneratedSdmaStagedTest
    : public DeviceGeneratedSdmaTest,
      public ::testing::WithParamInterface<DeviceSdmaStagedCase> {};

TEST_P(DeviceGeneratedSdmaStagedTest, IndependentConsumerReusesSelectedSlots) {
  namespace staged = kernels::device_sdma_staged;
  const auto& test_case = GetParam();
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
      (features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "LOCAL placement is not advertised by this device";
  }
  constexpr uint64_t kComputeRingByteLength = 65536;
  ASSERT_NE(family_.ring_byte_length_alignment, 0u);
  if (kComputeRingByteLength < family_.minimum_ring_byte_length ||
      kComputeRingByteLength > family_.maximum_ring_byte_length ||
      kComputeRingByteLength % family_.ring_byte_length_alignment != 0) {
    GTEST_SKIP() << "finite batch requires a 64-KiB AQL ring";
  }
  const auto* transfer_kernel =
      kernels::device_sdma_transfer::kKernels.Find(gpu_endpoint_info_);
  const auto* consumer_kernel =
      kernels::device_sdma_consumer::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(transfer_kernel, nullptr);
  ASSERT_NE(consumer_kernel, nullptr);
  ASSERT_EQ(transfer_kernel->workgroup_size(), 1u);
  ASSERT_EQ(consumer_kernel->workgroup_size(), 64u);
  ASSERT_LE(transfer_kernel->arguments.byte_length,
            sizeof(staged::TransferArguments));
  ASSERT_LE(consumer_kernel->arguments.byte_length,
            sizeof(staged::ConsumerArguments));
  for (const auto* kernel : {transfer_kernel, consumer_kernel}) {
    ASSERT_EQ(kernel->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  }

  constexpr uint32_t kSourceGuard = 0x75db8163u;
  constexpr uint32_t kPayloadGuard = 0x32a49bc7u;
  constexpr uint32_t kControlGuard = 0x69ca714bu;
  constexpr uint32_t kRecordGuard = 0xab3265c9u;
  constexpr uint32_t kSeed = 0x91e10da5u;
  constexpr uint32_t kPayloadWordOffset = 16;
  constexpr uint64_t kCompletionByteOffset = 64;
  constexpr uint64_t kStateByteOffset = 128;
  constexpr uint64_t kLengthsByteOffset = 192;
  constexpr uint64_t kArgumentStride = 512;
  constexpr uint64_t kConsumerArgumentOffset = 160;
  constexpr uint64_t kDownloadArgumentOffset = 256;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const auto align_page = [](uint64_t bytes) {
    return (bytes + 4095) & ~UINT64_C(4095);
  };
  const uint32_t maximum_word_count = *std::max_element(
      test_case.word_counts.begin(), test_case.word_counts.end());
  ASSERT_LE(maximum_word_count, 262144u);
  const uint64_t slot_bytes =
      align_page(uint64_t{maximum_word_count} * 4 + 128);
  ASSERT_LT(slot_bytes, UINT64_C(1) << 22);
  const uint64_t slot_words = slot_bytes / sizeof(uint32_t);
  const uint64_t payload_bytes = test_case.slot_count * slot_bytes;
  const uint64_t readback_bytes = test_case.job_count * slot_bytes;
  GpuMemory* source = nullptr;
  GpuMemory* seed = nullptr;
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* readback = nullptr;
  GpuMemory* selections = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 8 * slot_bytes, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, slot_bytes, &seed));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, readback_bytes, &readback));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite, align_page(test_case.job_count * sizeof(staged::Selection)),
      &selections));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ,
      align_page(test_case.job_count * kArgumentStride), &arguments));
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL) {
    ASSERT_NE(local_scope_, nullptr);
    const amdf_memory_device_access_t attachment = {
        device_,
        {.access = kReadWrite, .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    const uint32_t profile = FindGpuMemoryProfileOrdinal(
        api_, local_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL, attachment.requirements);
    ASSERT_NE(profile, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = profile,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
        .byte_length = payload_bytes,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &input));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &output));
    for (const auto* memory : {input, output}) {
      ASSERT_EQ(memory->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
      ASSERT_NE(memory->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
      ASSERT_EQ(memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
      ASSERT_EQ(memory->host.pointer, nullptr);
      ASSERT_EQ(memory->mapping, nullptr);
    }
  } else {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, payload_bytes, &input));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, payload_bytes, &output));
  }
  uint32_t upload_flags = 0;
  uint32_t download_flags = 0;
  uint32_t seed_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, input, selections, &upload_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryDownloadCacheFlags(output, readback, &download_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, input, selections, &seed_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, output, selections, &seed_flags));
  RecordProperty("staged_payload_memory_class", input->info.memory_class);
  RecordProperty("staged_payload_flags", std::to_string(input->info.flags));
  RecordProperty("staged_upload_cache_flags", upload_flags);
  RecordProperty("staged_download_cache_flags", download_flags);
  RecordProperty("staged_slot_count", test_case.slot_count);
  RecordProperty("staged_job_count", test_case.job_count);
  RecordProperty("staged_slot_byte_length", std::to_string(slot_bytes));

  GpuUserQueue* compute = nullptr;
  GpuUserQueue* transfer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(
      family_, &compute, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kComputeRingByteLength));
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  uint64_t packet_index = 0;
  uint64_t transfer_descriptor = 0;
  uint64_t consumer_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*compute, *transfer_kernel,
                                        "staged_transfer", &packet_index,
                                        &transfer_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*compute, *consumer_kernel,
                                        "staged_consumer", &packet_index,
                                        &consumer_descriptor));

  std::vector<uint32_t> expected_source(8 * slot_words, kSourceGuard);
  for (uint32_t page = 0; page < 8; ++page) {
    for (uint32_t word = 0; word < maximum_word_count; ++word) {
      expected_source[page * slot_words + kPayloadWordOffset + word] =
          0x31415927u + page * 0x243f6a89u + word * 0x1020305u;
    }
  }
  std::memcpy(source->host.pointer, expected_source.data(),
              source->info.byte_length);
  std::fill_n(static_cast<uint32_t*>(seed->host.pointer), slot_words,
              kPayloadGuard);
  std::fill_n(static_cast<uint32_t*>(readback->host.pointer),
              readback_bytes / sizeof(uint32_t), kRecordGuard);
  std::vector<uint32_t> expected_selections(
      selections->info.byte_length / sizeof(uint32_t), kRecordGuard);
  std::memcpy(selections->host.pointer, expected_selections.data(),
              selections->info.byte_length);
  std::array<uint32_t, 1024> expected_control;
  expected_control.fill(kControlGuard);
  aql::Signal final_signal = {};
  final_signal.kind = 1;
  final_signal.value = 1;
  std::memcpy(expected_control.data(), &final_signal, sizeof(final_signal));
  expected_control[kCompletionByteOffset / 4] = 0;
  expected_control[kStateByteOffset / 4 + 2] = kSeed;
  std::memcpy(expected_control.data() + kLengthsByteOffset / 4,
              test_case.word_counts.data(), sizeof(test_case.word_counts));
  std::memcpy(control->host.pointer, expected_control.data(),
              control->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(control->host.pointer);
  const uint64_t completion_host =
      reinterpret_cast<uintptr_t>(control->host.pointer) +
      kCompletionByteOffset;
  const uint64_t completion_address =
      control->device_address + kCompletionByteOffset;

  // Cold initialization is the host's only SDMA publication. Its completion
  // and RPTR join transfer ownership to the GPU before any workload starts.
  std::array<uint32_t, 96> seed_commands = {};
  SdmaCommandWriter seed_encoder(seed_commands.data(),
                                 sdma_family_.format_features);
  if ((seed_flags & 1) != 0) {
    seed_encoder.AcquireFromSystem();
  }
  for (uint32_t slot = 0; slot < test_case.slot_count; ++slot) {
    for (const auto* memory : {input, output}) {
      seed_encoder.CopyLinear(seed->device_address,
                              memory->device_address + slot * slot_bytes,
                              static_cast<uint32_t>(slot_bytes));
    }
  }
  if ((seed_flags & 2) != 0) {
    seed_encoder.ReleaseToSystem();
  }
  seed_encoder.Fence32(completion_address, 1);
  uint64_t frontier = seed_encoder.word_count() * sizeof(uint32_t);
  ASSERT_LT(frontier, mapping.ring_byte_length);
  std::memcpy(reinterpret_cast<void*>(transfer->host.ring_address),
              seed_commands.data(), frontier);
  transfer->PublishStream(frontier);
  GpuWaitEqual<uint32_t>(completion_host, 1);
  ASSERT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  GpuStoreRelease<uint32_t>(completion_host, 0);
  std::memcpy(static_cast<uint8_t*>(control->host.pointer) + kStateByteOffset,
              &frontier, sizeof(frontier));

  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, input->device_address, 4);
  encoder.Fence32(completion_address, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());
  std::vector<uint8_t> expected_arguments(arguments->info.byte_length, 0);
  std::vector<aql::Packet> packets;
  packets.reserve(test_case.job_count * 3);
  std::vector<uint32_t> expected_payload(payload_bytes / 4, kPayloadGuard);
  std::vector<uint32_t> expected_readback(readback_bytes / 4, kRecordGuard);
  uint32_t state = kSeed;
  uint32_t selected_pages = 0;
  uint32_t selected_slots = 0;
  uint32_t selected_lengths = 0;
  uint64_t uploaded_bytes = 0;
  auto advance_frontier = [&](uint32_t flags) {
    const uint64_t chain_bytes =
        44 + ((flags & 1) != 0 ? 20 : 0) + ((flags & 2) != 0 ? 20 : 0);
    const uint64_t tail =
        mapping.ring_byte_length - frontier % mapping.ring_byte_length;
    if (tail < chain_bytes) {
      frontier += tail;
    }
    frontier += chain_bytes;
  };
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    const uint64_t selection_address =
        selections->device_address + job * sizeof(staged::Selection);
    const uint64_t readback_address =
        readback->device_address + job * slot_bytes;
    const uint64_t argument_offset = job * kArgumentStride;
    staged::TransferArguments transfer_arguments = {
        .ring = mapping.ring_address,
        .read_index = mapping.read_index_address,
        .write_index = mapping.write_index_address,
        .notification = mapping.doorbell_address,
        .completion = completion_address,
        .state = control->device_address + kStateByteOffset,
        .selection = selection_address,
        .lengths = control->device_address + kLengthsByteOffset,
        .readback = readback_address,
        .source_address = source->device_address,
        .input_address = input->device_address,
        .output_address = output->device_address,
        .readback_address = readback_address,
        .completion_address = completion_address,
        .capacity = mapping.ring_byte_length,
        .slot_byte_length = slot_bytes,
        .job_index = job,
        .slot_count = test_case.slot_count,
        .phase = staged::Phase::kUpload,
        .copy_control = encoding[2],
        .fence_header = encoding[7],
        .cache_flags = upload_flags,
    };
    std::memcpy(expected_arguments.data() + argument_offset,
                &transfer_arguments, transfer_kernel->arguments.byte_length);
    const staged::ConsumerArguments consumer_arguments = {
        .input = input->device_address,
        .output = output->device_address,
        .selection = selection_address,
        .slot_byte_length = slot_bytes,
    };
    std::memcpy(
        expected_arguments.data() + argument_offset + kConsumerArgumentOffset,
        &consumer_arguments, consumer_kernel->arguments.byte_length);
    transfer_arguments.phase = staged::Phase::kDownload;
    transfer_arguments.cache_flags = download_flags;
    std::memcpy(
        expected_arguments.data() + argument_offset + kDownloadArgumentOffset,
        &transfer_arguments, transfer_kernel->arguments.byte_length);
    const uint64_t argument_address =
        arguments->device_address + argument_offset;
    packets.push_back(aql::Dispatch(aql::HeaderBarrier::kEnabled,
                                    {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
                                    transfer_descriptor, argument_address, 0));
    const uint32_t grid_words = (maximum_word_count + 63) & ~63u;
    packets.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {64, 1, 1}, {grid_words, 1, 1}}, 0, 0,
        consumer_descriptor, argument_address + kConsumerArgumentOffset, 0));
    packets.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
        transfer_descriptor, argument_address + kDownloadArgumentOffset,
        job + 1 == test_case.job_count ? control->device_address : 0));

    staged::Selection selection = {};
    std::fill_n(selection.guards, 6, kRecordGuard);
    selection.page = state & 7;
    selection.slot = (state >> 3) & (test_case.slot_count - 1);
    const uint32_t length_index = (state >> 8) & 3;
    selection.word_count = test_case.word_counts[length_index];
    selection.prior_state = state;
    selection.addend = state ^ ((job + 1) * 0x1020305u);
    selected_pages |= 1u << selection.page;
    selected_slots |= 1u << selection.slot;
    selected_lengths |= 1u << length_index;
    const uint64_t source_start =
        selection.page * slot_words + kPayloadWordOffset;
    const uint64_t output_start =
        selection.slot * slot_words + kPayloadWordOffset;
    for (uint32_t word = 0; word < selection.word_count; ++word) {
      expected_payload[output_start + word] =
          expected_source[source_start + word] * 3u + selection.addend;
    }
    std::copy_n(expected_payload.begin() + selection.slot * slot_words,
                slot_words, expected_readback.begin() + job * slot_words);
    state = expected_payload[output_start] ^
            (expected_payload[output_start + selection.word_count - 1] +
             (job + 1) * 0x9e3779b1u);
    selection.next_state = state;
    advance_frontier(upload_flags);
    selection.upload_frontier = frontier;
    advance_frontier(download_flags);
    selection.download_frontier = frontier;
    std::memcpy(expected_selections.data() + job * sizeof(selection) / 4,
                &selection, sizeof(selection));
    uploaded_bytes += uint64_t{selection.word_count} * sizeof(uint32_t);
  }
  ASSERT_EQ(selected_pages, 0xffu);
  ASSERT_EQ(selected_slots, (1u << test_case.slot_count) - 1);
  ASSERT_EQ(selected_lengths, 0xfu);
  if (test_case.job_count == 97) {
    ASSERT_GT(frontier / mapping.ring_byte_length, 1u);
  }
  std::memcpy(arguments->host.pointer, expected_arguments.data(),
              expected_arguments.size());
  expected_control[2] = 0;  // Low word of the final AQL signal value.
  expected_control[kCompletionByteOffset / 4] = test_case.job_count * 2;
  std::memcpy(expected_control.data() + kStateByteOffset / 4, &frontier,
              sizeof(frontier));
  expected_control[kStateByteOffset / 4 + 2] = state;

  // Write every later packet while the first header is still INVALID. This
  // prevents hardware polling from starting work before the complete finite
  // batch is ready, independently of when its doorbell is observed.
  const uint64_t capacity =
      compute->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_LT(packets.size(), capacity);
  ASSERT_EQ(GpuLoadAcquire<uint64_t>(compute->host.read_index_address),
            packet_index);
  auto* ring = reinterpret_cast<aql::Packet*>(compute->host.ring_address);
  auto& first_packet = ring[packet_index & (capacity - 1)];
  ASSERT_EQ(first_packet[0] & 0xffu, 1u);
  std::memcpy(first_packet.data() + 1, packets.front().data() + 1,
              sizeof(aql::Packet) - sizeof(uint32_t));
  for (size_t i = 1; i < packets.size(); ++i) {
    auto& target = ring[(packet_index + i) & (capacity - 1)];
    std::memcpy(target.data(), packets[i].data(), sizeof(target));
  }
  const uint64_t end_index = packet_index + packets.size();
  GpuStoreRelease(compute->host.write_index_address, end_index);
  GpuStoreRelease(reinterpret_cast<uintptr_t>(first_packet.data()),
                  packets.front()[0]);
  GpuStoreRelease(compute->host.doorbell_address, end_index - 1);

  // No per-job CPU service or joins. Capture the actual result before native
  // consumption/teardown can obscure the GPU-to-SDMA-to-consumer dataflow.
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
  std::vector<uint32_t> observed_readback(expected_readback.size());
  std::memcpy(observed_readback.data(), readback->host.pointer, readback_bytes);
  EXPECT_EQ(observed_readback, expected_readback);
  EXPECT_EQ(std::memcmp(selections->host.pointer, expected_selections.data(),
                        selections->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(source->host.pointer, expected_source.data(),
                        source->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(arguments->host.pointer, expected_arguments.data(),
                        arguments->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(control->host.pointer, expected_control.data(),
                        control->info.byte_length),
            0);
  EXPECT_EQ(GpuLoadAcquire<uint64_t>(transfer->host.write_index_address),
            frontier);
  EXPECT_NO_FATAL_FAILURE(compute->WaitConsumed(api_, end_index));
  EXPECT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  RecordProperty("staged_uploaded_bytes", std::to_string(uploaded_bytes));
  RecordProperty("staged_readback_bytes", std::to_string(readback_bytes));
  RecordProperty("staged_sdma_frontier", std::to_string(frontier));
  RecordProperty("staged_sdma_ring_wraps",
                 std::to_string(frontier / mapping.ring_byte_length));
}

INSTANTIATE_TEST_SUITE_P(
    Placement, DeviceGeneratedSdmaStagedTest,
    ::testing::ValuesIn(StagedCases()),
    [](const ::testing::TestParamInfo<DeviceSdmaStagedCase>& info) {
      return info.param.name;
    });

}  // namespace
