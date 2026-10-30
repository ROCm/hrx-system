// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class CommandBufferTimestampTest : public CtsTestBase<> {
 protected:
  // Two transfer tiles exercise completion across participating CPU workers.
  static constexpr iree_device_size_t kPayloadLength = 128 * 1024;
  static constexpr iree_device_size_t kRangeLength = 64 + kPayloadLength;
  static constexpr uint64_t kGuard = UINT64_C(0xCA7ECA7ECA7ECA7E);
  static constexpr uint32_t kPattern = 0x12345678;

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    const auto required =
        indirect() ? IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS
                   : IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS;
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      if (queues->families[i].provisioned_queue_count &&
          iree_all_bits_set(queues->families[i].flags, required) &&
          iree_all_bits_set(queues->families[i].role_flags,
                            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        queue_ = iree_hal_device_queue(device_, i, 0);
        break;
      }
    }
    if (!queue_) {
      GTEST_SKIP() << "No queue family supports these command timestamps";
    }
    const auto* spec =
        iree_hal_queue_family_spec(iree_hal_queue_family(queue_));
    ASSERT_GT(spec->timestamp_frequency_hz, 0u);
    ASSERT_GE(spec->timestamp_valid_bits, 1u);
    ASSERT_LE(spec->timestamp_valid_bits, 64u);
    tick_mask_ = UINT64_MAX >> (64 - spec->timestamp_valid_bits);
    // Pure transfer buffers must suffice, including for native polyfills.
    IREE_ASSERT_OK(
        CreateDevicePool(IREE_HAL_BUFFER_USAGE_TRANSFER, pool_.out()));
  }

  bool indirect() const { return recording_mode() == RecordingMode::kIndirect; }

  iree_hal_buffer_ref_t RefAt(iree_hal_buffer_t* buffer,
                              iree_device_size_t offset,
                              iree_device_size_t length = 8) const {
    return indirect() ? iree_hal_make_indirect_buffer_ref(0, offset, length)
                      : iree_hal_make_buffer_ref(buffer, offset, length);
  }

  void Allocate(iree_device_size_t length, iree_hal_buffer_t** out_buffer) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(length, 8),
        iree_infinite_timeout(), out_buffer));
    std::vector<uint64_t> initial(length / 8, kGuard);
    IREE_ASSERT_OK(UploadBufferData(initial.data(), *out_buffer, 0, length));
  }

  void Begin(iree_hal_command_buffer_t** out_command_buffer,
             iree_hal_command_buffer_mode_t mode = 0) {
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_), mode, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        indirect() ? 1 : 0, out_command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_command_buffer));
  }

  void Record(iree_hal_buffer_t* buffer,
              iree_hal_command_buffer_t** out_command_buffer,
              iree_hal_command_buffer_mode_t mode = 0) {
    Begin(out_command_buffer, mode);
    if (HasFatalFailure()) {
      return;
    }
    auto* command_buffer = *out_command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
        command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_TRANSFER, RefAt(buffer, 56), 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, RefAt(buffer, 64, kPayloadLength), &kPattern,
        sizeof(kPattern), IREE_HAL_FILL_FLAG_NONE));
    const iree_hal_memory_barrier_t overwrite_memory = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
    };
    const iree_hal_barrier_t overwrite = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .memory_barrier_count = 1,
        .memory_barriers = &overwrite_memory,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &overwrite));
    IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
        command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_TRANSFER, RefAt(buffer, 64), 0));
    const iree_hal_memory_barrier_t memory_barrier = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
    };
    const iree_hal_barrier_t consume = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .memory_barrier_count = 1,
        .memory_barriers = &memory_barrier,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &consume));
    // The fill overwrites the end timestamp's location: capture must follow
    // its completion. Resolving only at command-buffer end would also make
    // this consumer stale.
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        command_buffer, RefAt(buffer, 56, 16), RefAt(buffer, 24, 16), 0));
    const iree_hal_barrier_t publish = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &publish));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  }

  iree_status_t Submit(iree_hal_queue_t* queue,
                       iree_hal_command_buffer_t* command_buffer,
                       iree_hal_buffer_t* buffer, iree_device_size_t offset,
                       iree_hal_semaphore_list_t waits,
                       iree_hal_semaphore_list_t signals) {
    const iree_hal_buffer_binding_t binding = {buffer, offset, kRangeLength};
    const iree_hal_buffer_binding_table_t table = {
        indirect() ? 1u : 0u, indirect() ? &binding : nullptr};
    return iree_hal_queue_execute(queue, waits, signals, command_buffer, table,
                                  0);
  }

  std::array<uint64_t, 2> CheckRange(iree_hal_buffer_t* buffer,
                                     iree_device_size_t offset) {
    std::vector<uint64_t> words(kRangeLength / 8);
    IREE_EXPECT_OK(
        DownloadBufferData(buffer, offset, words.data(), kRangeLength));
    return CheckWords(words);
  }

  std::array<uint64_t, 2> CheckWords(const std::vector<uint64_t>& words) {
    EXPECT_EQ(words[0], kGuard);
    EXPECT_EQ(words[1], kGuard);
    EXPECT_EQ(words[2], kGuard);
    EXPECT_NE(words[7], kGuard);
    EXPECT_EQ(words[3], words[7]);
    EXPECT_EQ(words[4], words[8]);
    for (size_t i = 5; i < 7; ++i) {
      EXPECT_EQ(words[i], kGuard);
    }
    const uint64_t pattern = (uint64_t{kPattern} << 32) | kPattern;
    EXPECT_NE(words[8], pattern);
    EXPECT_THAT(std::vector<uint64_t>(words.begin() + 9, words.end()),
                ::testing::Each(pattern));
    return {words[7], words[8]};
  }

  // Pool backing retained until every test-local buffer is released.
  Ref<iree_hal_pool_t> pool_;
  // Borrowed provisioned queue in a timestamp-capable family.
  iree_hal_queue_t* queue_ = nullptr;
  // Valid bits used for modulo clock differences.
  uint64_t tick_mask_ = 0;
};

TEST_P(CommandBufferTimestampTest, TransferConsumerAndQueueClock) {
  Ref<iree_hal_buffer_t> buffer;
  Allocate(kRangeLength, buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  Ref<iree_hal_command_buffer_t> command_buffer;
  Record(buffer, command_buffer.out(),
         indirect() ? 0 : IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT);
  ASSERT_FALSE(HasFatalFailure());
  Ref<iree_hal_buffer_t> outer;
  Allocate(16, outer.out());
  ASSERT_FALSE(HasFatalFailure());
  SemaphoreList before(device_, {0}, {1});
  SemaphoreList after(device_, {0}, {1});
  SemaphoreList done(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_timestamp(queue_, {}, before, outer, 0, nullptr, 0));
  IREE_ASSERT_OK(Submit(queue_, command_buffer, buffer, 0, before, after));
  IREE_ASSERT_OK(
      iree_hal_queue_timestamp(queue_, after, done, outer, 8, nullptr, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(done, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  const auto ticks = CheckRange(buffer, 0);
  std::array<uint64_t, 2> bounds;
  IREE_ASSERT_OK(DownloadBufferData(outer, 0, bounds.data(), sizeof(bounds)));
  const uint64_t duration = (bounds[1] - bounds[0]) & tick_mask_;
  EXPECT_LE((ticks[0] - bounds[0]) & tick_mask_, duration);
  EXPECT_LE((ticks[1] - ticks[0]) & tick_mask_, duration);
  EXPECT_LE((bounds[1] - ticks[1]) & tick_mask_, duration);
}

TEST_P(CommandBufferTimestampTest, ConcurrentBindingsAndReuse) {
  if (!indirect()) {
    GTEST_SKIP() << "Independent targets require indirect bindings";
  }
  const auto* family = iree_hal_queue_family(queue_);
  if (!(family->spec->flags &
        IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION)) {
    GTEST_SKIP() << "Independent queue acquisition is unavailable";
  }
  Ref<iree_hal_queue_t> second_queue;
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, second_queue.out()));
  Ref<iree_hal_buffer_t> buffer;
  Allocate(3 * kRangeLength, buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  Ref<iree_hal_command_buffer_t> command_buffer;
  Record(buffer, command_buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList first_done(device_, {0}, {1});
  SemaphoreList second_done(device_, {0}, {1});
  IREE_ASSERT_OK(Submit(queue_, command_buffer, buffer, 0, gate, first_done));
  // Keep the first accepted invocation pending while another binding executes.
  // Release its gate even on a submission failure so teardown can retire it.
  iree_status_t status = Submit(second_queue, command_buffer, buffer,
                                kRangeLength, {}, second_done);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(second_done, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    CheckRange(buffer, kRangeLength);
    std::vector<uint64_t> pending(kRangeLength / 8);
    IREE_EXPECT_OK(DownloadBufferData(buffer, 0, pending.data(), kRangeLength));
    EXPECT_THAT(pending, ::testing::Each(kGuard));
  }
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
      first_done, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(status);
  CheckRange(buffer, 0);
  const auto second = ReadBufferBytes(buffer, kRangeLength, kRangeLength);
  SemaphoreList reused_done(device_, {0}, {1});
  IREE_ASSERT_OK(Submit(queue_, command_buffer, buffer, 2 * kRangeLength, {},
                        reused_done));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      reused_done, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  CheckRange(buffer, 2 * kRangeLength);
  EXPECT_EQ(ReadBufferBytes(buffer, kRangeLength, kRangeLength), second);
}

TEST_P(CommandBufferTimestampTest, PendingAllocationAndReleasedOwners) {
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList allocated(device_, {0}, {1});
  SemaphoreList initialized(device_, {0}, {1});
  SemaphoreList executed(device_, {0}, {1});
  SemaphoreList downloaded(device_, {0}, {1});
  SemaphoreList retired(device_, {0}, {1});
  Ref<iree_hal_buffer_t> buffer;
  Ref<iree_hal_command_buffer_t> command_buffer;
  const auto request = iree_hal_make_pool_reservation_request(kRangeLength, 8);
  IREE_ASSERT_OK(iree_hal_queue_alloca(queue_, gate, allocated, pool_, 1,
                                       &request, buffer.out()));
  std::vector<uint64_t> output(kRangeLength / 8);
  iree_hal_semaphore_list_t last_completion = allocated;
  // Every accepted operation has a completion to join even if a later recording
  // or submission fails. No assertion exits with the allocation gate held.
  iree_status_t status = [&]() -> iree_status_t {
    Record(buffer, command_buffer.out());
    if (HasFatalFailure()) {
      return iree_make_status(IREE_STATUS_INTERNAL);
    }
    const uint32_t guard = static_cast<uint32_t>(kGuard);
    IREE_RETURN_IF_ERROR(iree_hal_queue_fill(queue_, allocated, initialized,
                                             buffer, 0, kRangeLength, &guard,
                                             sizeof(guard), nullptr, 0));
    last_completion = initialized;
    IREE_RETURN_IF_ERROR(
        Submit(queue_, command_buffer, buffer, 0, initialized, executed));
    last_completion = executed;
    IREE_RETURN_IF_ERROR(iree_hal_queue_download(queue_, executed, downloaded,
                                                 buffer, 0, output.data(),
                                                 kRangeLength, nullptr));
    last_completion = downloaded;
    return iree_ok_status();
  }();
  iree_hal_buffer_t* raw_buffer = buffer;
  iree_status_t retire_status =
      iree_hal_queue_dealloca(queue_, last_completion, retired, 1, &raw_buffer);
  if (iree_status_is_ok(retire_status)) {
    last_completion = retired;
    buffer.reset();
  }
  command_buffer.reset();
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
      last_completion, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(retire_status);
  IREE_ASSERT_OK(status);
  CheckWords(output);
}

TEST_P(CommandBufferTimestampTest, FailedDependencyDoesNotWrite) {
  Ref<iree_hal_buffer_t> buffer;
  Allocate(kRangeLength, buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  Ref<iree_hal_command_buffer_t> command_buffer;
  Record(buffer, command_buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList done(device_, {0}, {1});
  IREE_ASSERT_OK(Submit(queue_, command_buffer, buffer, 0, gate, done));
  command_buffer.reset();
  iree_hal_semaphore_fail(gate.semaphores[0],
                          iree_make_status(IREE_STATUS_ABORTED));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, iree_hal_semaphore_list_wait(
                                                 done, iree_infinite_timeout(),
                                                 IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_THAT(ReadBufferData<uint64_t>(buffer), ::testing::Each(kGuard));
}

TEST_P(CommandBufferTimestampTest, RejectsInvalidRecordingAndBinding) {
  Ref<iree_hal_buffer_t> buffer;
  Allocate(kRangeLength, buffer.out());
  ASSERT_FALSE(HasFatalFailure());
  Ref<iree_hal_command_buffer_t> command_buffer;
  Begin(command_buffer.out());
  ASSERT_FALSE(HasFatalFailure());
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  for (iree_device_size_t length : {0, 4, 16}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_command_buffer_timestamp(command_buffer, 0, 0,
                                          RefAt(buffer, 0, length), 0));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_timestamp(command_buffer, 0, 0,
                                                          RefAt(buffer, 1), 0));
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_timestamp(command_buffer, 0, 0,
                                                          RefAt(buffer, 0), 1));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_timestamp(
                            command_buffer, IREE_HAL_EXECUTION_STAGE_HOST, 0,
                            RefAt(buffer, 0), 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_command_buffer_timestamp(command_buffer, 0, UINT32_MAX,
                                        RefAt(buffer, 0), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(command_buffer, 0, 0,
                                                   RefAt(buffer, 8), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  if (indirect()) {
    const iree_hal_buffer_binding_t misaligned = {buffer, 1, 16};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_hal_command_buffer_validate_submission(
                              command_buffer, {1, &misaligned}));
    const iree_hal_buffer_binding_t short_binding = {buffer, 0, 8};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          iree_hal_command_buffer_validate_submission(
                              command_buffer, {1, &short_binding}));
  }
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferTimestampTest);

}  // namespace iree::hal::cts
