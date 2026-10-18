// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "libamdf/cts/util/wait.h"

namespace {

class SdmaEventTest : public GpuCommandTest {
 protected:
  SdmaEventTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
            .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        }) {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(GpuCommandTest::SetUp());
    if (IsSkipped()) {
      return;
    }
    if (!(features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS)) {
      GTEST_SKIP() << "device has no directly produced native event";
    }
    const amdf_gpu_event_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
        .structure_size = sizeof(create),
    };
    ASSERT_EQ(gpu_api_->event_create(device_, &create, &event_),
              AMDF_STATUS_OK);
    event_info_.type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO;
    event_info_.structure_size = sizeof(event_info_);
    ASSERT_EQ(gpu_api_->event_query_info(event_, &event_info_), AMDF_STATUS_OK);
    if (event_info_.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
        event_info_.format_version != 1 ||
        (event_info_.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ==
            0) {
      GTEST_SKIP() << "SDMA case has no recipe for this event producer format";
    }
    ASSERT_LT(event_info_.interrupt_context, 1u << 28);
  }

  void TearDown() override {
    // Queue removal joins every notification tail before its mailbox is freed.
    ASSERT_NO_FATAL_FAILURE(GpuCommandTest::TearDown());
    if (event_) {
      EXPECT_EQ(gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    }
  }

  // Shared native wake used only as a hint to inspect payload completion.
  amdf_gpu_event_t* event_ = nullptr;
  // Immutable mailbox and interrupt operands, dominated by event_.
  amdf_gpu_event_info_t event_info_ = {};
};

TEST_F(SdmaEventTest, CopyCompletionAndRingReuseWithEarlyHostHints) {
  constexpr uint32_t kPayloadWords = 256;
  constexpr uint32_t kOutputOffset = 2048;
  constexpr uint32_t kCompletionOffset = 4096;
  constexpr uint64_t kRingBytes = 4096;
  constexpr uint32_t kRounds = 256;
  if (kRingBytes < family_.minimum_ring_byte_length ||
      kRingBytes > family_.maximum_ring_byte_length ||
      kRingBytes % family_.ring_byte_length_alignment != 0) {
    GTEST_SKIP() << "native family does not admit the 4-KiB reuse ring";
  }
  GpuMemory* payload = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 8192, &payload,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  const auto host = payload->HostSite();
  const auto engine = payload->DeviceSite(family_.ordinal);
  amdf_memory_pair_info_t upload = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(upload),
  };
  amdf_memory_pair_info_t download = upload;
  ASSERT_EQ(api_->memory_query_pair_info(&host, &engine, &upload),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_query_pair_info(&engine, &host, &download),
            AMDF_STATUS_OK);
  ASSERT_EQ(upload.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(download.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  const bool acquire = upload.acquire.kind != AMDF_CACHE_TRANSITION_KIND_NONE;
  const bool release = download.release.kind != AMDF_CACHE_TRANSITION_KIND_NONE;
  if (acquire) {
    ASSERT_EQ(upload.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(upload.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    ASSERT_EQ(upload.acquire.operation,
              AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
  }
  if (release) {
    ASSERT_EQ(download.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(download.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    ASSERT_EQ(download.release.operation,
              AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
  }
  if (acquire || release) {
    ASSERT_NE(family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
              0u);
  }
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kRingBytes));
  auto* ring = reinterpret_cast<uint32_t*>(queue->host.ring_address);
  auto* input = static_cast<uint32_t*>(payload->host.pointer);
  auto* output = input + kOutputOffset / sizeof(uint32_t);
  const auto completion =
      reinterpret_cast<uint64_t>(payload->host.pointer) + kCompletionOffset;
  GpuStoreRelease<uint32_t>(completion, 0);
  uint64_t published_bytes = 0;
  uint32_t wrap_count = 0;
  for (uint32_t round = 1; round <= kRounds; ++round) {
    for (uint32_t i = 0; i < kPayloadWords; ++i) {
      input[i] = round * 0x9e3779b1u + i * 0x01020305u;
      output[i] = ~input[i];
    }
    std::array<uint32_t, 32> words;
    SdmaCommandWriter commands(words.data(), family_.format_features);
    if (acquire) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(payload->device_address,
                        payload->device_address + kOutputOffset,
                        kPayloadWords * sizeof(uint32_t));
    if (release) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(payload->device_address + kCompletionOffset, round);
    commands.NotifyKfdEvent(event_info_);
    const uint64_t byte_length = commands.word_count() * sizeof(uint32_t);
    uint64_t ring_offset = published_bytes % kRingBytes;
    if (ring_offset + byte_length > kRingBytes) {
      std::memset(reinterpret_cast<uint8_t*>(ring) + ring_offset, 0,
                  kRingBytes - ring_offset);
      published_bytes += kRingBytes - ring_offset;
      ring_offset = 0;
      ++wrap_count;
    }
    std::memcpy(reinterpret_cast<uint8_t*>(ring) + ring_offset, words.data(),
                byte_length);
    published_bytes += byte_length;
    // A stale/coalesced host hint is deliberately pending before publication.
    // Only the GPU-authored completion word permits consuming or reusing data.
    ASSERT_EQ(gpu_api_->event_signal(event_), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(queue->PublishStream(published_bytes));
    const amdf::cts::WaitDeadline deadline;
    do {
      const uint64_t remaining_ns = deadline.remaining_ns();
      ASSERT_NE(remaining_ns, 0u)
          << "native event did not publish copy completion for round " << round;
      ASSERT_EQ(gpu_api_->event_wait(event_, remaining_ns), AMDF_STATUS_OK);
    } while (GpuLoadAcquire<uint32_t>(completion) != round);
    EXPECT_EQ(std::memcmp(output, input, kPayloadWords * sizeof(uint32_t)), 0)
        << "round " << round;
    // Completion precedes the notification tail. Retire that native use before
    // ring wrap can overwrite any word of the preceding publication.
    ASSERT_EQ(api_->user_queue_wait_consumed(queue->queue, published_bytes,
                                             amdf::cts::kWaitTimeoutNs, 0),
              AMDF_STATUS_OK);
  }
  EXPECT_GT(wrap_count, 1u);
  RecordProperty("native_event_copy_rounds", kRounds);
  RecordProperty("native_event_ring_wraps", wrap_count);
}

TEST_F(SdmaEventTest, IndependentQueuesShareOneNativeWake) {
  constexpr uint32_t kRounds = 64;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  std::array<GpuUserQueue*, 2> queues;
  for (auto& queue : queues) {
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
    ASSERT_GE(queue->host.ring_byte_length, kRounds * 10u * sizeof(uint32_t));
  }
  const uint64_t host = reinterpret_cast<uint64_t>(completion->host.pointer);
  const std::array<uint64_t, 2> addresses = {host, host + 64};
  for (const auto address : addresses) {
    GpuStoreRelease<uint32_t>(address, 0);
  }
  std::array<uint64_t, 2> published_bytes = {};
  for (uint32_t round = 1; round <= kRounds; ++round) {
    for (uint32_t i = 0; i < queues.size(); ++i) {
      auto* queue = queues[i];
      auto* words = reinterpret_cast<uint32_t*>(queue->host.ring_address +
                                                published_bytes[i]);
      SdmaCommandWriter commands(words, family_.format_features);
      commands.Fence32(completion->device_address + i * 64, round);
      commands.NotifyKfdEvent(event_info_);
      published_bytes[i] += commands.word_count() * sizeof(uint32_t);
      ASSERT_NO_FATAL_FAILURE(queue->PublishStream(published_bytes[i]));
    }
    const amdf::cts::WaitDeadline deadline;
    do {
      const uint64_t remaining_ns = deadline.remaining_ns();
      ASSERT_NE(remaining_ns, 0u)
          << "shared event did not publish both queue completions for round "
          << round;
      ASSERT_EQ(gpu_api_->event_wait(event_, remaining_ns), AMDF_STATUS_OK);
    } while (GpuLoadAcquire<uint32_t>(addresses[0]) != round ||
             GpuLoadAcquire<uint32_t>(addresses[1]) != round);
    for (uint32_t i = 0; i < queues.size(); ++i) {
      ASSERT_EQ(
          api_->user_queue_wait_consumed(queues[i]->queue, published_bytes[i],
                                         amdf::cts::kWaitTimeoutNs, 0),
          AMDF_STATUS_OK);
    }
  }
  RecordProperty("native_event_shared_rounds", kRounds);
  RecordProperty("native_event_queue_count", queues.size());
}

}  // namespace
