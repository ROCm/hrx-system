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

#include "libamdf/cts/gpu/pm4/command_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/util/wait.h"

namespace {

class Pm4EventTest : public Pm4CommandTest,
                     public ::testing::WithParamInterface<uint32_t> {
 protected:
  Pm4EventTest() : Pm4CommandTest(0, AMDF_QUEUE_PUBLICATION_MODE_USER) {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Pm4CommandTest::SetUp());
    if (IsSkipped()) {
      return;
    }
    if ((features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) == 0) {
      GTEST_SKIP() << "device has no native GPU event producer";
    }
    const amdf_gpu_event_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    ASSERT_EQ(gpu_api_->event_create(device_, &create_info, &event_),
              AMDF_STATUS_OK);
    ASSERT_EQ(gpu_api_->event_query_info(event_, &event_info_), AMDF_STATUS_OK);
    if (event_info_.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
        event_info_.format_version != 1 ||
        (event_info_.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ==
            0) {
      GTEST_SKIP() << "PM4 case has no recipe for this event producer format";
    }
  }

  void TearDown() override {
    // Native queue removal joins notification tails before releasing their ID.
    ASSERT_NO_FATAL_FAILURE(Pm4CommandTest::TearDown());
    if (event_) {
      ASSERT_EQ(gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    }
  }

  // Event retained beyond all native queues that may write its mailbox.
  amdf_gpu_event_t* event_ = nullptr;
  // Immutable producer operands qualified before recording.
  amdf_gpu_event_info_t event_info_ = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(event_info_),
  };
};

// Each record has independent authoritative readiness; event wakes may
// coalesce.
struct CompletionRecord {
  // Opaque caller payload transported by the command processor.
  uint64_t payload;
  // Device-published generation, including wrap, never reset by the host.
  uint32_t generation;
  // Surrounding storage that must remain unchanged.
  uint32_t guard;
};
static_assert(sizeof(CompletionRecord) == 16);

TEST_P(Pm4EventTest, WakesForEveryFiniteBatchIncludingTheIdleTail) {
  const uint32_t record_count = GetParam();
  constexpr uint32_t kEpochCount = 8;
  constexpr uint32_t kMaximumRecordCount = 32;
  constexpr uint32_t kInitialGeneration = UINT32_MAX - 3;
  constexpr uint32_t kGuard = 0x62bfb981;
  GpuMemory* storage = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_EQ(storage->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  std::array<CompletionRecord, kMaximumRecordCount> expected = {};
  for (auto& record : expected) {
    record = {0, kInitialGeneration, kGuard};
  }
  std::memcpy(storage->host.pointer, expected.data(), sizeof(expected));
  const uint64_t host_records =
      reinterpret_cast<uintptr_t>(storage->host.pointer);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, 65536));
  ASSERT_GE(queue->words().size(), kEpochCount * (record_count * 16u + 24u));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t generation = kInitialGeneration + epoch + 1;
    commands.SystemBarrier();
    for (uint32_t i = 0; i < record_count; ++i) {
      const uint64_t payload =
          UINT64_C(0xfedcba9876543210) ^ (uint64_t{generation} << 32) ^ i;
      expected[i] = {payload, generation, kGuard};
      const uint32_t words[] = {static_cast<uint32_t>(payload),
                                static_cast<uint32_t>(payload >> 32)};
      const uint64_t address =
          storage->device_address + i * sizeof(CompletionRecord);
      commands.WriteData(address, words, 2);
      commands.ReleaseSystem32(address + offsetof(CompletionRecord, generation),
                               generation);
    }
    commands.ReleaseSystemInterrupt32(event_info_.mailbox_address,
                                      event_info_.trigger_data,
                                      event_info_.interrupt_context);
    commands.PadToEightWords();
    // Only the GPU can wake this wait. A host signal here could conceal a
    // broken interrupt recipe when command execution wins the race to wait.
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    const uint64_t tail = host_records +
                          (record_count - 1) * sizeof(CompletionRecord) +
                          offsetof(CompletionRecord, generation);
    const amdf::cts::WaitDeadline deadline;
    do {
      const uint64_t remaining_ns = deadline.remaining_ns();
      ASSERT_NE(remaining_ns, 0u)
          << "native event did not publish the batch tail";
      ASSERT_EQ(gpu_api_->event_wait(event_, remaining_ns), AMDF_STATUS_OK);
    } while (GpuLoadAcquire<uint32_t>(tail) != generation);
    // There is no polling wait, native queue wait, or further publication to
    // rescue notification delivery, including for the final epoch.
    for (uint32_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(
          GpuLoadAcquire<uint32_t>(host_records + i * sizeof(CompletionRecord) +
                                   offsetof(CompletionRecord, generation)),
          generation);
    }
    EXPECT_EQ(
        std::memcmp(storage->host.pointer, expected.data(), sizeof(expected)),
        0);
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("native_event_batches", kEpochCount);
  RecordProperty("native_event_records_per_batch", record_count);
}

INSTANTIATE_TEST_SUITE_P(Batches, Pm4EventTest, ::testing::Values(1u, 4u, 32u),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                           return "Records" + std::to_string(info.param);
                         });

}  // namespace
