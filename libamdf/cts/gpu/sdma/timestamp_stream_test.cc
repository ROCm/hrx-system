// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <thread>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "util/wait.h"

namespace {

constexpr uint32_t kRecordCount = 16387;
constexpr uint32_t kSlotCount = 4;
constexpr uint32_t kSlotBytes = 128;
constexpr uint32_t kGuard = 0x39c571a7;
constexpr uint32_t kPublishedOffset = 0;
constexpr uint32_t kConsumedOffset = 64;
constexpr uint32_t kDoneOffset = 128;

uint32_t SourceWord(uint32_t index) {
  return 0x712ac593u ^ (index * 0x9e3779b9u);
}

class SdmaTimestampStreamTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<amdf_queue_publication_modes_t> {
 protected:
  SdmaTimestampStreamTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = GetParam(),
        }) {}

  void CheckHostPair(const GpuMemory& memory, bool host_producer) {
    const auto host = memory.HostSite();
    const auto device = memory.DeviceSite(family_.ordinal);
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair),
    };
    ASSERT_EQ(
        api_->memory_query_pair_info(host_producer ? &host : &device,
                                     host_producer ? &device : &host, &pair),
        AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    ASSERT_EQ(pair.acquire.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    const auto& transition = host_producer ? pair.acquire : pair.release;
    if (transition.kind != AMDF_CACHE_TRANSITION_KIND_NONE) {
      ASSERT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      ASSERT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
      ASSERT_EQ(transition.operation,
                host_producer ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                              : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
      ASSERT_NE(family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
                0u);
    }
  }

  void TearDown() override {
    // Oracle failure releases all finite producer waits before checked native
    // teardown. The outer fixture retains backing if queue removal fails.
    if (control_ != nullptr) {
      GpuStoreRelease<uint32_t>(
          reinterpret_cast<uintptr_t>(control_->host.pointer) + kConsumedOffset,
          kRecordCount);
    }
    GpuCommandTest::TearDown();
  }

  // Control storage borrowed from the fixture. SDMA publishes produced records;
  // the CPU returns credits.
  GpuMemory* control_ = nullptr;
};

TEST_P(SdmaTimestampStreamTest,
       ConsumesAndReusesSamplesBeforeStreamRetirement) {
  if ((family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR) &&
      !(family_.format_features &
        (AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR |
         AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE))) {
    GTEST_SKIP() << "streaming samples before retirement requires in-stream "
                    "release operations, not a native submission epilogue";
  }
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* source = nullptr;
  GpuMemory* observations = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &observations));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control_));
  auto* input = static_cast<uint32_t*>(source->host.pointer);
  for (uint32_t i = 0; i < 1024; ++i) {
    input[i] = SourceWord(i);
  }
  std::array<uint32_t, 1024> expected_observations;
  expected_observations.fill(kGuard);
  std::memcpy(observations->host.pointer, expected_observations.data(), 4096);
  std::array<uint32_t, 1024> expected_control;
  expected_control.fill(kGuard);
  expected_control[kPublishedOffset / 4] = 0;
  expected_control[kConsumedOffset / 4] = 0;
  expected_control[kDoneOffset / 4] = 0;
  std::memcpy(control_->host.pointer, expected_control.data(), 4096);
  for (const auto* memory : {source, observations, control_}) {
    ASSERT_NO_FATAL_FAILURE(CheckHostPair(*memory, true));
    if (memory != source) {
      ASSERT_NO_FATAL_FAILURE(CheckHostPair(*memory, false));
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  memory->mapping, memory->host.flush.host_operation, 0, 4096),
              AMDF_STATUS_OK);
  }

  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(4 * 1024 * 1024, &queue));
  // The longest form is 38 DWORDs per record, plus the final wait/fence.
  ASSERT_GT(queue->words().size(), kRecordCount * 38u + 10u);
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  for (uint32_t ordinal = 0; ordinal < kRecordCount; ++ordinal) {
    const uint32_t required_credit =
        ordinal < kSlotCount ? 0 : ordinal + 1 - kSlotCount;
    commands.WaitMemory32(control_->device_address + kConsumedOffset,
                          required_credit,
                          SdmaMemoryComparison::kGreaterOrEqual);
    if (user_gcr) {
      commands.AcquireFromSystem();
    }
    const uint64_t slot =
        observations->device_address + (ordinal % kSlotCount) * kSlotBytes;
    commands.WriteGlobalTimestamp(slot);
    const std::array<uint32_t, 1> sequence = {ordinal + 1};
    commands.WriteLinear(slot + 8, sequence);
    const uint32_t source_word = (ordinal * 13) % (1024 - 16);
    commands.CopyLinear(source->device_address + source_word * 4, slot + 64,
                        64);
    commands.WriteGlobalTimestamp(slot + 32);
    if (user_gcr) {
      commands.ReleaseToSystem();
    }
    // Readiness follows the ending sample. The sample itself is never polled
    // as a commit flag and need not be an atomic 64-bit publication operation.
    commands.Fence32(control_->device_address + kPublishedOffset, ordinal + 1);
  }
  commands.WaitMemory32(control_->device_address + kConsumedOffset,
                        kRecordCount, SdmaMemoryComparison::kGreaterOrEqual);
  commands.Fence32(control_->device_address + kDoneOffset, 1);
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));

  const uintptr_t control_host =
      reinterpret_cast<uintptr_t>(control_->host.pointer);
  uint64_t first_ticks = 0;
  uint64_t previous_ticks = 0;
  uint64_t digest = 0;
  uint64_t expected_digest = 0;
  const amdf::cts::WaitDeadline deadline;
  for (uint32_t ordinal = 0; ordinal < kRecordCount; ++ordinal) {
    ASSERT_TRUE(amdf::cts::WaitUntil(
        "SDMA timestamp record",
        [&] {
          return GpuLoadAcquire<uint32_t>(control_host + kPublishedOffset) >=
                 ordinal + 1;
        },
        deadline))
        << "record " << ordinal + 1 << ", published "
        << GpuLoadAcquire<uint32_t>(control_host + kPublishedOffset);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(control_host + kDoneOffset), 0u);
    std::array<uint32_t, kSlotBytes / 4> record;
    const uint32_t slot = (ordinal % kSlotCount) * kSlotBytes;
    std::memcpy(record.data(),
                static_cast<const uint8_t*>(observations->host.pointer) + slot,
                kSlotBytes);
    const uint64_t begin = (uint64_t{record[1]} << 32) | record[0];
    const uint64_t end = (uint64_t{record[9]} << 32) | record[8];
    EXPECT_NE(begin, 0u);
    EXPECT_LE(previous_ticks, begin);
    EXPECT_LE(begin, end);
    if (ordinal == 0) {
      first_ticks = begin;
    }
    previous_ticks = end;
    EXPECT_EQ(record[2], ordinal + 1);
    const uint32_t source_word = (ordinal * 13) % (1024 - 16);
    for (uint32_t i = 0; i < record.size(); ++i) {
      if (i == 0 || i == 1 || i == 2 || i == 8 || i == 9) {
        continue;
      }
      const uint32_t expected =
          i < 16 ? kGuard : SourceWord(source_word + i - 16);
      EXPECT_EQ(record[i], expected) << "record " << ordinal << " word " << i;
      if (i >= 16) {
        digest = digest * 33 + record[i];
        expected_digest = expected_digest * 33 + expected;
      }
    }
    std::copy(record.begin(), record.end(),
              expected_observations.begin() + slot / 4);
    // The consumer has finished reading this complete record before allowing
    // the producer to reuse its slot.
    GpuStoreRelease<uint32_t>(control_host + kConsumedOffset, ordinal + 1);
    if (HasFailure()) {
      break;
    }
  }
  const bool valid = !HasFailure();
  GpuStoreRelease<uint32_t>(control_host + kConsumedOffset, kRecordCount);
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(control_host + kDoneOffset, 1));
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  if (!valid) {
    return;
  }
  EXPECT_GT(previous_ticks, first_ticks);
  EXPECT_EQ(digest, expected_digest);
  EXPECT_EQ(std::memcmp(observations->host.pointer,
                        expected_observations.data(), 4096),
            0);
  expected_control[kPublishedOffset / 4] = kRecordCount;
  expected_control[kConsumedOffset / 4] = kRecordCount;
  expected_control[kDoneOffset / 4] = 1;
  EXPECT_EQ(std::memcmp(control_->host.pointer, expected_control.data(), 4096),
            0);
  for (uint32_t i = 0; i < 1024; ++i) {
    EXPECT_EQ(input[i], SourceWord(i));
  }
  // Raw correctness observations, with no rate, calibration or overhead claim.
  RecordProperty("stream_record_count", kRecordCount);
  RecordProperty("stream_slot_count", kSlotCount);
  RecordProperty("stream_payload_capacity_bytes", kSlotCount * kSlotBytes);
  RecordProperty("stream_begin_ticks", std::to_string(first_ticks));
  RecordProperty("stream_end_ticks", std::to_string(previous_ticks));
  RecordProperty("stream_payload_digest", std::to_string(digest));
}

INSTANTIATE_TEST_SUITE_P(Publication, SdmaTimestampStreamTest,
                         ::testing::Values(AMDF_QUEUE_PUBLICATION_MODE_USER,
                                           AMDF_QUEUE_PUBLICATION_MODE_KERNEL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_PUBLICATION_MODE_USER
                                      ? "User"
                                      : "Kernel";
                         });

}  // namespace
