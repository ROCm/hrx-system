// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/testing/queue_test_util.h"

namespace iree::hal::amd::xdna::testing {
namespace {

void RecordDispatches(QueueHarness& harness, iree_host_size_t count,
                      iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(ImageFixture(), &executable, &function));
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(harness.queue),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      1, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const auto binding = iree_hal_make_indirect_buffer_ref(0, 0, 64);
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        IREE_HAL_DISPATCH_FLAG_NONE));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  iree_hal_executable_release(executable);
  *out_command_buffer = command_buffer;
}

TEST(XdnaCommandBufferTest, WindowedPublicationDoesNotSignalAnAcceptedPrefix) {
  QueueHarness harness;
  harness.native.pending_capacity = 16;
  harness.native.maximum_command_count = 8;
  harness.native.point_stride = 7;
  harness.native.HoldRetirement();
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.SealDeviceGroup());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(RecordDispatches(harness, 19, &command_buffer));
  iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  // The caller publishes the entire free window without a proactor turn.
  EXPECT_EQ(harness.native.submission_count, 2u);
  iree_hal_submitted_signal_flags_t flags = 0;
  iree_async_axis_t axis = 0;
  uint64_t epoch = 0, signal_value = 0;
  EXPECT_FALSE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(harness.done), &flags, &axis,
      &epoch, &signal_value));
  binding = {};
  iree_hal_buffer_release(buffer);
  iree_hal_command_buffer_release(command_buffer);
  // Retiring the first array admits the short singleton tail through
  // wraparound.
  harness.native.retirement_limit = 7;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilSubmitted(5));
  uint64_t completed_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &completed_value));
  EXPECT_EQ(completed_value, 0u);
  harness.native.retirement_limit = UINT64_MAX;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, 1));
}

class XdnaCommandBufferReplayTest : public ::testing::TestWithParam<uint32_t> {
};

TEST_P(XdnaCommandBufferReplayTest,
       ConcurrentReplaysRetainIndependentBindingsWithoutWarmAllocations) {
  const uint32_t dispatch_count = GetParam();
  HostAllocationCounters counters;
  QueueHarness harness;
  harness.native.pending_capacity = 16;
  harness.native.maximum_command_count = 8;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize(TrackingAllocator(&counters)));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordDispatches(harness, dispatch_count, &command_buffer));
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  std::array<iree_hal_semaphore_t*, 2> completions = {};
  for (size_t i = 0; i < buffers.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffers[i]));
    IREE_ASSERT_OK(iree_hal_semaphore_create(harness.device,
                                             IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                             0, 0, &completions[i]));
  }
  HostAllocationSnapshot warmed = {};
  for (uint64_t iteration = 1; iteration <= 5; ++iteration) {
    harness.native.HoldRetirement();
    for (size_t i = 0; i < buffers.size(); ++i) {
      iree_hal_buffer_binding_t binding = {buffers[(i + iteration) % 2], 0, 64};
      IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                            {1, &completions[i], &iteration},
                                            command_buffer, {1, &binding}, 0));
    }
    EXPECT_EQ(harness.native.submission_count,
              iteration * 2 * (dispatch_count == 2 ? 2 : 1));
    {
      std::lock_guard<std::mutex> lock(harness.native.mutex);
      ASSERT_EQ(harness.native.pending_commands.size(), dispatch_count * 2);
      EXPECT_NE(harness.native.pending_commands[0].bytes,
                harness.native.pending_commands[dispatch_count].bytes);
      if (dispatch_count == 2) {
        EXPECT_NE(harness.native.pending_commands[0].submission,
                  harness.native.pending_commands[1].submission);
      } else {
        EXPECT_EQ(
            harness.native.pending_commands[0].submission,
            harness.native.pending_commands[dispatch_count - 1].submission);
      }
      EXPECT_NE(harness.native.pending_commands[dispatch_count - 1].submission,
                harness.native.pending_commands[dispatch_count].submission);
      for (size_t i = 1; i < dispatch_count * 2; ++i) {
        EXPECT_NE(harness.native.pending_commands[i - 1].offset,
                  harness.native.pending_commands[i].offset);
      }
    }
    harness.native.retirement_limit = UINT64_MAX;
    harness.native.Wake();
    for (auto* completion : completions) {
      ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(completion, iteration));
    }
    if (iteration == 2) {
      warmed = SnapshotHostAllocations(counters);
    }
    if (iteration > 2) {
      ExpectHostAllocationsUnchanged(warmed, counters);
    }
  }
  iree_hal_command_buffer_release(command_buffer);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  for (auto* completion : completions) {
    iree_hal_semaphore_release(completion);
  }
}

INSTANTIATE_TEST_SUITE_P(ShortAndBatched, XdnaCommandBufferReplayTest,
                         ::testing::Values(2u, 8u));

TEST(XdnaCommandBufferTest,
     NativeBusyKeepsPreparedSuffixAndResumesAfterProgress) {
  QueueHarness harness;
  harness.native.pending_capacity = 24;
  harness.native.maximum_command_count = 8;
  harness.native.HoldRetirement();
  harness.native.busy_submission_attempt = 2;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(RecordDispatches(harness, 24, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  EXPECT_EQ(harness.native.submission_count, 1u);
  EXPECT_EQ(harness.native.submission_attempt_count, 2u);
  const size_t flush_count = harness.native.FlushCount();
  harness.native.retirement_limit = UINT64_MAX;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, 1));
  EXPECT_EQ(harness.native.submission_count, 3u);
  EXPECT_EQ(harness.native.submission_attempt_count, 4u);
  // The retry keeps the entire prepared array; the final array reuses the
  // retired first array's invocations with the same address tuples.
  EXPECT_EQ(harness.native.FlushCount(), flush_count);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaCommandBufferTest,
     RejectedSuffixRetainsAcceptedPrefixUntilRetirement) {
  QueueHarness harness;
  harness.native.pending_capacity = 32;
  harness.native.maximum_command_count = 8;
  harness.native.HoldRetirement();
  harness.native.reject_submission_attempt = 3;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(RecordDispatches(harness, 25, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  EXPECT_EQ(harness.native.submission_count, 2u);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_GT(harness.native.live_memories, 1u);
  harness.native.retirement_limit = UINT64_MAX;
  harness.native.Wake();
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                            iree_infinite_timeout(), nullptr));
    status = iree_hal_semaphore_query(harness.done, &observed);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(harness.native.submission_attempt_count, 3u);
}

TEST(XdnaCommandBufferTest, PreparationFailureReleasesUnacceptedArray) {
  QueueHarness harness;
  harness.native.pending_capacity = 8;
  harness.native.maximum_command_count = 8;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(RecordDispatches(harness, 8, &command_buffer));
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  for (auto& buffer : buffers) {
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  }
  iree_hal_buffer_binding_t binding = {buffers[0], 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  // Changing bindings requires each retired instruction range to be repatched.
  // Fail publication of the second range after the first has been prepared.
  harness.native.fail_cache_publication_countdown = 2;
  const size_t flush_count = harness.native.FlushCount();
  binding.buffer = buffers[1];
  ++value;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  iree_hal_command_buffer_release(command_buffer);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  EXPECT_EQ(harness.native.submission_attempt_count, 1u);
  EXPECT_EQ(harness.native.FlushCount(), flush_count + 2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_wait(harness.done, value, iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
}

static void AbandonWindowedRecording() {
  QueueHarness harness;
  harness.native.pending_capacity = 2;
  harness.native.HoldRetirement();
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(RecordDispatches(harness, 5, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  EXPECT_EQ(harness.native.submission_count, 2u);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
  const size_t live_memories = harness.native.live_memories;
  harness.native.outcome = Outcome::kRefreshFailure;
  harness.native.Wake();
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    IREE_ASSERT_OK(iree_async_proactor_poll(harness.proactor,
                                            iree_infinite_timeout(), nullptr));
    status = iree_hal_semaphore_query(harness.done, &value);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);
  EXPECT_EQ(harness.native.live_memories, live_memories);
  EXPECT_EQ(harness.native.submission_count, 2u);
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

TEST(XdnaCommandBufferDeathTest,
     UnobservablePrefixRetainsRecordingAndDoesNotIssueSuffix) {
  EXPECT_EXIT(AbandonWindowedRecording(), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace iree::hal::amd::xdna::testing
