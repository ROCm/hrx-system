// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/file_notification.h"

#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/file_notification_kernels.h"
#include "libamdf/cts/gpu/linux/io_uring/file_io_fixture.h"
#include "libamdf/cts/gpu/util/event_fixture.h"

namespace {

namespace protocol = kernels::file_notification;

constexpr uint32_t kGuard = 0xa5a5a5a5u;
constexpr uint32_t kFileBlocks = 16;
constexpr size_t kControlOffset = 64;

uint32_t SourceWord(uint32_t block, uint32_t word) {
  return (block + 1) * 0x01010307u + word * 0x00010001u;
}

enum class Outcome { kComplete, kEof, kStopBeforeAdmission };

class GpuFileNotificationTest : public GpuEventTest<GpuFileIoFixture> {
 protected:
  void TearDown() override {
    // A fatal host assertion releases the resident shader's wait. Queue-first
    // teardown then retires native events, ring references, and caller pages.
    if (stop_address_) {
      GpuStoreRelease<uint32_t>(stop_address_, 1);
    }
    GpuEventTest<GpuFileIoFixture>::TearDown();
  }

  void Run(FileMode mode, Outcome outcome);

  // Borrowed coherent control word, retained until queue-first teardown.
  uintptr_t stop_address_ = 0;
};

void GpuFileNotificationTest::Run(FileMode mode, Outcome outcome) {
  const auto* kernel = protocol::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing compiled file-notification kernel";
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  const uint32_t word_count = page_byte_length_ / sizeof(uint32_t);
  ASSERT_LE(word_count, 16384u);
  const uint32_t source_blocks = outcome == Outcome::kEof ? 2 : kFileBlocks;
  std::vector<uint32_t> expected_file(source_blocks * word_count);
  for (uint32_t block = 0; block < source_blocks; ++block) {
    for (uint32_t word = 0; word < word_count; ++word) {
      expected_file[block * word_count + word] = SourceWord(block, word);
    }
  }
  ASSERT_NO_FATAL_FAILURE(CreateFile(expected_file, mode));
  if (IsSkipped()) {
    return;
  }
  GpuMemory* payload = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateRegisteredPages(3 * page_byte_length_, kGuard, &payload));
  ASSERT_NO_FATAL_FAILURE(CreateRing(payload, FileIoPath::kDeviceWait, 1, 2));
  if (IsSkipped()) {
    return;
  }

  GpuMemory* records = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &records));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096,
                                       AMDF_MEMORY_FLAG_HOST_COHERENT,
                                       &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &completion));
  std::vector<uint32_t> expected_records(
      records->host.byte_length / sizeof(uint32_t), kGuard);
  protocol::Control expected = {};
  for (auto& round : expected.rounds) {
    round = {UINT64_C(0xa5a5a5a5a5a5a5a5), kGuard, kGuard};
  }
  std::memcpy(expected_records.data() + kControlOffset / sizeof(uint32_t),
              &expected, sizeof(expected));
  std::memcpy(records->host.pointer, expected_records.data(),
              records->host.byte_length);
  std::memset(completion->host.pointer, 0, completion->host.byte_length);
  const uintptr_t host_control =
      reinterpret_cast<uintptr_t>(records->host.pointer) + kControlOffset;
  stop_address_ = host_control + offsetof(protocol::Control, stop);
  const protocol::Arguments device_arguments = {
      .submission_entries = device_ring_memory_->device_address,
      .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
      .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
      .completion_head = RingAddress(ring_->parameters.cq_off.head),
      .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
      .payload = payload->device_address + page_byte_length_,
      .control = records->device_address + kControlOffset,
      .host_payload = reinterpret_cast<uintptr_t>(payload->host.pointer) +
                      page_byte_length_,
      .mailbox = event_info_.mailbox_address,
      .trigger = event_info_.trigger_data,
      .submission_mask = ring_->parameters.sq_entries - 1,
      .completion_mask = ring_->parameters.cq_entries - 1,
      .word_count = word_count,
      .file_block_mask = kFileBlocks - 1,
      .round_count = protocol::kRoundCount,
      .interrupt_context = event_info_.interrupt_context,
  };
  ASSERT_EQ(arguments->device_address % kernel->arguments.alignment, 0u);
  ASSERT_LE(kernel->arguments.byte_length, sizeof(device_arguments));
  std::memset(arguments->host.pointer, 0, arguments->host.byte_length);
  std::memcpy(arguments->host.pointer, &device_arguments,
              sizeof(device_arguments));

  Pm4ComputeProgram program = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {1, 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                         kernel->entry_byte_offset, &program,
                                         "file_notification"));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  commands.SystemBarrier();
  commands.BindCompute(program, arguments->device_address,
                       family_.format_features);
  commands.Dispatch(program, 1, 1, 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));

  uint64_t observed_generation = 0;
  uint64_t native_wakes = 0;
  uint64_t enter_calls = 0;
  uint64_t wait_calls = 0;
  for (;;) {
    // No file operation is guaranteed pending here. The shader interrupt is
    // the only wake source; no sleep loop or CP completion can rescue it.
    ASSERT_EQ(gpu_api_->event_wait(event_, AMDF_TIMEOUT_INFINITE),
              AMDF_STATUS_OK);
    ++native_wakes;
    const uint64_t published = GpuLoadAcquire<uint64_t>(host_control);
    if (published == observed_generation) {
      continue;
    }
    EXPECT_GT(published, observed_generation);
    observed_generation = published;
    if (published & protocol::kTerminal) {
      break;
    }
    if (outcome == Outcome::kStopBeforeAdmission) {
      GpuStoreRelease<uint32_t>(stop_address_, 1);
      continue;
    }
    // Notifications may coalesce while native I/O is progressing. Service
    // published work using native frontiers, never one request per hint.
    while (GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail)) !=
           GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail))) {
      ASSERT_NO_FATAL_FAILURE(ServiceDeviceIo(&enter_calls, &wait_calls));
    }
  }
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  // Join shader execution after observing its terminal notification. Native
  // queue consumption alone does not retire the producer's interrupt tail.
  commands.SystemBarrier();
  commands.WriteData32(completion->device_address, 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));

  std::vector<uint32_t> expected_payload(3 * word_count, kGuard);
  uint32_t cause = protocol::kInitialCause;
  if (outcome == Outcome::kComplete) {
    for (uint32_t round = 0; round < protocol::kRoundCount; ++round) {
      const uint32_t block = cause & (kFileBlocks - 1);
      uint32_t checksum = 0;
      for (uint32_t word = 0; word < word_count; ++word) {
        const uint32_t value = SourceWord(block, word);
        expected_payload[word_count + word] = value;
        checksum += value;
      }
      expected.rounds[round] = {round, block, checksum};
      cause = SourceWord(block, 0);
    }
    expected.completed = protocol::kRoundCount;
    expected.requests = protocol::kRoundCount;
    expected.result = page_byte_length_;
  } else {
    expected.requests = 1;
  }
  expected.generation = protocol::kTerminal | (expected.requests + 1);
  expected.stop = outcome == Outcome::kStopBeforeAdmission ? 1 : 0;
  expected.cause = cause;
  std::memcpy(expected_records.data() + kControlOffset / sizeof(uint32_t),
              &expected, sizeof(expected));
  const auto* actual_records =
      static_cast<const uint32_t*>(records->host.pointer);
  for (size_t word = 0; word < expected_records.size(); ++word) {
    EXPECT_EQ(actual_records[word], expected_records[word])
        << "control or guard word=" << word;
  }
  const auto* actual_payload =
      static_cast<const uint32_t*>(payload->host.pointer);
  for (size_t word = 0; word < expected_payload.size(); ++word) {
    EXPECT_EQ(actual_payload[word], expected_payload[word])
        << "payload or guard word=" << word;
  }
  EXPECT_EQ(std::memcmp(arguments->host.pointer, &device_arguments,
                        sizeof(device_arguments)),
            0);
  const uint32_t native_requests =
      outcome == Outcome::kStopBeforeAdmission ? 0 : expected.requests;
  EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail)),
            expected.requests);
  for (uint32_t offset :
       {ring_->parameters.sq_off.head, ring_->parameters.cq_off.head,
        ring_->parameters.cq_off.tail}) {
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(offset)), native_requests);
  }
  EXPECT_EQ(
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)), 0u);
  EXPECT_EQ(
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
      0u);
  ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, mode));
  RecordProperty("file_notification_dispatches", 1);
  RecordProperty("file_notification_native_wakes",
                 std::to_string(native_wakes));
  RecordProperty("file_notification_io_submits", std::to_string(enter_calls));
  RecordProperty("file_notification_io_waits", std::to_string(wait_calls));
}

TEST_F(GpuFileNotificationTest, BufferedReadsWakeAndFeedResidentSuccessors) {
  Run(FileMode::kBuffered, Outcome::kComplete);
}

TEST_F(GpuFileNotificationTest, DirectReadsWakeAndFeedResidentSuccessors) {
  Run(FileMode::kDirect, Outcome::kComplete);
}

TEST_F(GpuFileNotificationTest, EofWakesWithoutConsumingPayload) {
  Run(FileMode::kBuffered, Outcome::kEof);
}

TEST_F(GpuFileNotificationTest, HostStopRetiresBeforeNativeAdmission) {
  Run(FileMode::kBuffered, Outcome::kStopBeforeAdmission);
}

}  // namespace
