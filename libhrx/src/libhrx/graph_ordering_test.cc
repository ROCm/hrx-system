// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <memory>
#include <thread>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Captures the actual queue boundary while leaving command recording, memory,
// semaphores, and execution to Task. Tests choose when to forward each accepted
// submission, making dependency checks independent of worker timing.
struct RecordedSubmission {
  ~RecordedSubmission() {
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_semaphore_list_free(waits, iree_allocator_system());
    iree_hal_semaphore_list_free(signals, iree_allocator_system());
  }

  // Retained dependencies copied before the queue call returns.
  iree_hal_semaphore_list_t waits = {};
  // Retained completion points copied before the queue call returns.
  iree_hal_semaphore_list_t signals = {};
  // Retained command buffer, or null for a barrier.
  iree_hal_command_buffer_t* command_buffer = nullptr;
  // Whether a barrier requests the conservative payload visibility defaults.
  bool default_barriers = true;
  // Whether this captured operation has been forwarded to Task.
  bool submitted = false;
};

struct RecordingQueue {
  // HAL queue identity, using the underlying Task queue's family.
  iree_hal_queue_t base;
  // Borrowed Task queue, owned by the fixture's device.
  iree_hal_queue_t* target = nullptr;
  // Captured calls in acceptance order.
  std::vector<std::unique_ptr<RecordedSubmission>> submissions;
  // Optional dependency failure returned before accepting a selected call.
  size_t reject_at = SIZE_MAX;

  static iree_status_t Capture(iree_hal_queue_t* base,
                               iree_hal_semaphore_list_t waits,
                               iree_hal_semaphore_list_t signals,
                               iree_hal_command_buffer_t* command_buffer) {
    auto& queue = *reinterpret_cast<RecordingQueue*>(base);
    if (queue.submissions.size() == queue.reject_at) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "submission storage exhausted");
    }
    auto submission = std::make_unique<RecordedSubmission>();
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_clone(
        &waits, iree_allocator_system(), &submission->waits));
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_clone(
        &signals, iree_allocator_system(), &submission->signals));
    iree_hal_command_buffer_retain(command_buffer);
    submission->command_buffer = command_buffer;
    queue.submissions.push_back(std::move(submission));
    return iree_ok_status();
  }

  void Initialize(iree_hal_queue_t* queue) {
    static const iree_hal_queue_vtable_t vtable = {
        .destroy = [](iree_hal_queue_t*) {},
        .barrier =
            [](iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals,
               const iree_hal_queue_barriers_t* barriers,
               iree_hal_queue_barrier_flags_t flags) {
              EXPECT_EQ(flags, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
              if (barriers) {
                EXPECT_NE(barriers->before, nullptr);
                EXPECT_NE(barriers->after, nullptr);
                EXPECT_EQ(barriers->before->count, 0u);
                EXPECT_EQ(barriers->after->count, 0u);
              }
              IREE_RETURN_IF_ERROR(Capture(queue, waits, signals, nullptr));
              reinterpret_cast<RecordingQueue*>(queue)
                  ->submissions.back()
                  ->default_barriers = !barriers;
              return iree_ok_status();
            },
        .execute =
            [](iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
               iree_hal_semaphore_list_t signals,
               iree_hal_command_buffer_t* command_buffer,
               iree_hal_buffer_binding_table_t bindings,
               iree_hal_queue_execute_flags_t flags) {
              EXPECT_EQ(bindings.count, 0u);
              EXPECT_EQ(flags, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
              return Capture(queue, waits, signals, command_buffer);
            },
    };
    target = queue;
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    iree_hal_queue_initialize(iree_hal_queue_family(queue), &params, &vtable,
                              &base);
  }

  iree_status_t Submit(size_t index) {
    auto& submission = *submissions[index];
    submission.submitted = true;
    if (submission.command_buffer) {
      return iree_hal_queue_execute(
          target, submission.waits, submission.signals,
          submission.command_buffer, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    }
    const iree_hal_barrier_list_t empty_list = {};
    const iree_hal_queue_barriers_t barriers = {&empty_list, &empty_list};
    return iree_hal_queue_barrier(
        target, submission.waits, submission.signals,
        submission.default_barriers ? nullptr : &barriers,
        IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  }
};

class GraphOrderingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_device_get(0, &device_)));
    for (auto& stream : streams_) {
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_stream_create(hrx_device_get_queue(device_, 0, 0), 0, &stream)));
    }
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
        streams_[0], sizeof(uint32_t) * kWordCount,
        HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buffer_)));
    const uint32_t zero = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        streams_[0], buffer_, 0, sizeof(uint32_t) * kWordCount, &zero,
        sizeof(zero))));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_semaphore_create(device_, 0, &gate_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_create(device_, 0, &graph_)));
    queue_.Initialize(device_->dispatch_queue);
    for (auto stream : streams_) {
      stream->hal_queue = &queue_.base;
    }
  }

  void TearDown() override {
    uint64_t gate_value = 0;
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_semaphore_query(gate_, &gate_value)));
    if (!gate_value) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate_, 1)));
    }
    for (size_t i = 0; i < queue_.submissions.size(); ++i) {
      if (!queue_.submissions[i]->submitted) {
        IREE_EXPECT_OK(queue_.Submit(i));
      }
    }
    for (auto stream : streams_) {
      stream->hal_queue = queue_.target;
      if (queue_.reject_at == SIZE_MAX) {
        IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream)));
      }
    }
    queue_.submissions.clear();
    hrx_graph_exec_release(exec_);
    hrx_graph_release(graph_);
    hrx_event_release(event_);
    hrx_semaphore_release(gate_);
    hrx_buffer_release(buffer_);
    for (auto stream : streams_) {
      hrx_stream_release(stream);
    }
    iree_hal_queue_release(&queue_.base);
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  void AddFills(size_t count, hrx_graph_node_t dependency = nullptr) {
    for (size_t i = 0; i < count; ++i) {
      hrx_graph_fill_buffer_node_attrs_t attrs = {
          .dst = {buffer_, i * sizeof(uint32_t), sizeof(uint32_t)},
          .pattern = static_cast<uint32_t>(i + 1),
          .pattern_size = sizeof(uint32_t),
      };
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_fill_buffer_node(
          graph_, dependency ? &dependency : nullptr, dependency ? 1 : 0,
          &attrs, nullptr)));
    }
  }

  void Instantiate() {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
        graph_, hrx_stream_get_queue(streams_[0]), 0, &exec_)));
  }

  void SubmitReverse() {
    for (size_t i = queue_.submissions.size(); i > 0; --i) {
      if (!queue_.submissions[i - 1]->submitted) {
        IREE_ASSERT_OK(queue_.Submit(i - 1));
      }
    }
  }

  void VerifyContents(size_t count) {
    std::array<uint32_t, kWordCount> contents;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), buffer_, 0,
                            contents.data(), sizeof(contents))));
    for (size_t i = 0; i < contents.size(); ++i) {
      EXPECT_EQ(contents[i], i < count ? i + 1 : 0) << i;
    }
  }

  static constexpr size_t kWordCount = 64;
  // Borrowed CPU device, live until shutdown after all resource releases.
  hrx_device_t device_ = nullptr;
  // Independently serialized submission timelines sharing one Task device.
  std::array<hrx_stream_t, 2> streams_ = {};
  // Output storage retained until all captured work completes.
  hrx_buffer_t buffer_ = nullptr;
  // External dependency released even after a test assertion fails.
  hrx_semaphore_t gate_ = nullptr;
  // Optional recorded event kept live through its stream wait.
  hrx_event_t event_ = nullptr;
  // Public graph constructed by each test.
  hrx_graph_t graph_ = nullptr;
  // Public executable owning all graph resources through completion.
  hrx_graph_exec_t exec_ = nullptr;
  // Queue boundary controlling submission order without replacing execution.
  RecordingQueue queue_;
};

TEST_F(GraphOrderingTest, EveryRootWaitsForIncomingStreamWork) {
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_wait_on(streams_[0], {gate_, 1})));
  const uint64_t entry_value = streams_[0]->timepoint;
  AddFills(17);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  ASSERT_EQ(queue_.submissions.size(), 4u);  // Entry wait, two roots, join.
  for (size_t i = 1; i < 3; ++i) {
    const auto& waits = queue_.submissions[i]->waits;
    ASSERT_EQ(waits.count, 1u);
    EXPECT_EQ(waits.semaphores[0], streams_[0]->semaphore->hal_semaphore);
    EXPECT_EQ(waits.payload_values[0], entry_value);
  }
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate_, 1)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, CompletionJoinsEveryTerminalWorkstream) {
  AddFills(17);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  ASSERT_EQ(queue_.submissions.size(), 3u);
  auto& join = *queue_.submissions[2];
  ASSERT_EQ(join.command_buffer, nullptr);
  EXPECT_FALSE(join.default_barriers);
  ASSERT_EQ(join.waits.count, 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto& signals = queue_.submissions[i]->signals;
    ASSERT_EQ(signals.count, 1u);
    EXPECT_EQ(join.waits.semaphores[i], signals.semaphores[0]);
    EXPECT_EQ(join.waits.payload_values[i], signals.payload_values[0]);
  }
  // Complete the last recorded branch while deliberately holding the first.
  IREE_ASSERT_OK(queue_.Submit(2));
  IREE_ASSERT_OK(queue_.Submit(1));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue_.submissions[1]->signals,
                                              iree_infinite_timeout(), 0));
  bool complete = true;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_query(streams_[0], &complete)));
  EXPECT_FALSE(complete);
  IREE_ASSERT_OK(queue_.Submit(0));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, SingleExitNeedsNoExtraQueueOperation) {
  AddFills(1);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  ASSERT_EQ(queue_.submissions.size(), 1u);
  EXPECT_NE(queue_.submissions[0]->command_buffer, nullptr);
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(1);
}

TEST_F(GraphOrderingTest, EmptyGraphPreservesStreamPosition) {
  Instantiate();
  const uint64_t position = streams_[0]->timepoint;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  EXPECT_TRUE(queue_.submissions.empty());
  EXPECT_EQ(streams_[0]->timepoint, position);
}

TEST_F(GraphOrderingTest, QueuedReplaysReserveDistinctGenerations) {
  AddFills(64);
  Instantiate();
  constexpr size_t kReplayCount = 16;
  for (size_t i = 0; i < kReplayCount; ++i) {
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  }
  ASSERT_EQ(queue_.submissions.size(), kReplayCount * 9u);
  for (size_t replay = 0; replay < kReplayCount; ++replay) {
    for (size_t branch = 0; branch < 8; ++branch) {
      const auto& operation = *queue_.submissions[replay * 9 + branch];
      EXPECT_EQ(operation.signals.payload_values[0], replay + 1);
    }
  }
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(64);
}

TEST_F(GraphOrderingTest, CrossStreamReplayWaitsForPreviousLaunch) {
  AddFills(17);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  const uint64_t previous_completion = streams_[0]->timepoint;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[1])));
  ASSERT_EQ(queue_.submissions.size(), 6u);
  for (size_t i = 3; i < 5; ++i) {
    const auto& waits = queue_.submissions[i]->waits;
    ASSERT_EQ(waits.count, 1u);
    EXPECT_EQ(waits.semaphores[0], streams_[0]->semaphore->hal_semaphore);
    EXPECT_EQ(waits.payload_values[0], previous_completion);
  }
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[1])));
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, CrossStreamReplayPreservesBothEntryDependencies) {
  AddFills(17);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  const uint64_t previous_completion = streams_[0]->timepoint;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_wait_on(streams_[1], {gate_, 1})));
  const uint64_t stream_entry = streams_[1]->timepoint;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[1])));
  ASSERT_EQ(queue_.submissions.size(), 7u);
  for (size_t i = 4; i < 6; ++i) {
    const auto& waits = queue_.submissions[i]->waits;
    ASSERT_EQ(waits.count, 2u);
    EXPECT_EQ(waits.semaphores[0], streams_[1]->semaphore->hal_semaphore);
    EXPECT_EQ(waits.payload_values[0], stream_entry);
    EXPECT_EQ(waits.semaphores[1], streams_[0]->semaphore->hal_semaphore);
    EXPECT_EQ(waits.payload_values[1], previous_completion);
  }
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate_, 1)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[1])));
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, IntermediatePartitionJoinsAllPredecessors) {
  AddFills(17);
  std::array<hrx_graph_node_t, 17> predecessors;
  size_t node_count = predecessors.size();
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_get_nodes(graph_, predecessors.data(), &node_count)));
  hrx_graph_node_t join = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_add_empty_node(
      graph_, predecessors.data(), node_count, &join)));
  AddFills(1, join);
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  ASSERT_EQ(queue_.submissions.size(), 4u);
  const auto& waits = queue_.submissions[2]->waits;
  ASSERT_EQ(waits.count, 2u);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(waits.semaphores[i],
              queue_.submissions[i]->signals.semaphores[0]);
  }
  EXPECT_EQ(queue_.submissions[3]->waits.semaphores[0],
            queue_.submissions[2]->signals.semaphores[0]);
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, ConcurrentStreamsReserveOrderedLaunches) {
  AddFills(17);
  Instantiate();
  std::array<std::thread, 2> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (size_t replay = 0; replay < 8; ++replay) {
        IREE_EXPECT_OK(
            hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[i])));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  ASSERT_EQ(queue_.submissions.size(), 16u * 3u);
  for (size_t replay = 0; replay < 16; ++replay) {
    EXPECT_EQ(queue_.submissions[replay * 3]->signals.payload_values[0],
              replay + 1);
  }
  SubmitReverse();
  for (auto stream : streams_) {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream)));
  }
  VerifyContents(17);
}

TEST_F(GraphOrderingTest, RejectedSuffixFailsReservedCompletion) {
  AddFills(17);
  Instantiate();
  queue_.reject_at = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  ASSERT_EQ(queue_.submissions.size(), 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_query(queue_.submissions[0]->signals.semaphores[0],
                               &value));
  // The dependency rejected this launch before any captured operation was
  // forwarded. No device retirement claim is made from the failed semaphore.
  queue_.submissions.clear();
}

class GraphStreamWaitTest : public GraphOrderingTest,
                            public ::testing::WithParamInterface<bool> {};

TEST_P(GraphStreamWaitTest, CompletedExternalWaitPreservesPendingStreamWork) {
  if (GetParam()) {
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_event_create(device_, HRX_EVENT_FLAG_DISABLE_TIMING, &event_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_record(event_, streams_[1])));
    IREE_ASSERT_OK(queue_.Submit(0));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_)));
  } else {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate_, 1)));
  }

  const size_t first = queue_.submissions.size();
  const uint32_t pattern = 0x12345678;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
      streams_[0], buffer_, 0, sizeof(pattern), &pattern, sizeof(pattern))));
  if (GetParam()) {
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_wait_event(streams_[0], event_)));
  } else {
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_wait_on(streams_[0], {gate_, 1})));
  }
  ASSERT_EQ(queue_.submissions.size(), first + 2);
  const auto& wait = *queue_.submissions[first + 1];
  ASSERT_EQ(wait.waits.count, 2u);
  EXPECT_EQ(wait.waits.semaphores[0],
            (GetParam() ? event_->semaphore : gate_)->hal_semaphore);
  EXPECT_EQ(wait.waits.payload_values[0], 1u);
  EXPECT_EQ(wait.waits.semaphores[1], streams_[0]->semaphore->hal_semaphore);
  EXPECT_EQ(wait.waits.payload_values[1], streams_[0]->timepoint - 1);

  const hrx_graph_copy_buffer_node_attrs_t copy = {
      .src = {buffer_, 0, sizeof(pattern)},
      .dst = {buffer_, sizeof(pattern), sizeof(pattern)},
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_copy_buffer_node(graph_, nullptr, 0, &copy, nullptr)));
  Instantiate();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_exec_launch(exec_, streams_[0])));
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  uint32_t contents[2] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), buffer_, 0,
                          contents, sizeof(contents))));
  EXPECT_EQ(contents[0], pattern);
  EXPECT_EQ(contents[1], pattern);
}

INSTANTIATE_TEST_SUITE_P(TimelineAndEvent, GraphStreamWaitTest,
                         ::testing::Bool());

TEST_F(GraphOrderingTest, SelfTimelineWaitCoalescesWithPendingStreamWork) {
  const uint32_t pattern = 1;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
      streams_[0], buffer_, 0, sizeof(pattern), &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_wait_on(streams_[0], {streams_[0]->semaphore, 0})));
  ASSERT_EQ(queue_.submissions.size(), 2u);
  const auto& waits = queue_.submissions[1]->waits;
  ASSERT_EQ(waits.count, 1u);
  EXPECT_EQ(waits.semaphores[0], streams_[0]->semaphore->hal_semaphore);
  EXPECT_EQ(waits.payload_values[0], streams_[0]->timepoint - 1);
  SubmitReverse();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(streams_[0])));
  VerifyContents(1);
}

}  // namespace
