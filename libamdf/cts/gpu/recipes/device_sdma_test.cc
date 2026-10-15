// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/device_sdma_batched.h"
#include "libamdf/cts/gpu/kernels/device_sdma_batched_kernels.h"
#include "libamdf/cts/gpu/recipes/device_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

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
  const aql::Signal final_signal = aql::UserSignal();
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
  ASSERT_TRUE(aql::Publish(*producer, packet_index++, dispatch));
  ASSERT_TRUE(
      GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
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

}  // namespace
