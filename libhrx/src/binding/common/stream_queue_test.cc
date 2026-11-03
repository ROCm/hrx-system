// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_queue.h"

#include <atomic>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/graph.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

struct ObserverCountCondition {
  iree_hal_streaming_context_t* context;
  int32_t expected_count;
};

static bool ObserverCountEquals(void* user_data) {
  auto* condition = static_cast<ObserverCountCondition*>(user_data);
  iree_slim_mutex_lock(&condition->context->value_wait_lane_mutex);
  const bool matches =
      iree_atomic_load(&condition->context->active_value_wait_observer_count,
                       iree_memory_order_acquire) == condition->expected_count;
  iree_slim_mutex_unlock(&condition->context->value_wait_lane_mutex);
  return matches;
}

enum ObserverFinishPhase : int32_t {
  OBSERVER_FINISH_PHASE_INITIAL = 0,
  OBSERVER_FINISH_PHASE_PAUSED = 1,
  OBSERVER_FINISH_PHASE_RELEASED = 2,
};

struct ObserverFinishState {
  iree_hal_streaming_context_t* context = nullptr;
  iree_notification_t notification;
  iree_atomic_int32_t phase;
  std::atomic<bool> mutex_was_held{false};
};

static bool ObserverFinishPaused(void* user_data) {
  auto* state = static_cast<ObserverFinishState*>(user_data);
  return iree_atomic_load(&state->phase, iree_memory_order_acquire) >=
         OBSERVER_FINISH_PHASE_PAUSED;
}

static bool ObserverFinishReleased(void* user_data) {
  auto* state = static_cast<ObserverFinishState*>(user_data);
  return iree_atomic_load(&state->phase, iree_memory_order_acquire) >=
         OBSERVER_FINISH_PHASE_RELEASED;
}

static void PauseFinalObserverFinish(void* user_data) {
  auto* state = static_cast<ObserverFinishState*>(user_data);
  const bool acquired =
      iree_slim_mutex_try_lock(&state->context->value_wait_lane_mutex);
  if (acquired) {
    iree_slim_mutex_unlock(&state->context->value_wait_lane_mutex);
  }
  state->mutex_was_held.store(!acquired, std::memory_order_release);
  iree_atomic_store(&state->phase, OBSERVER_FINISH_PHASE_PAUSED,
                    iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
  iree_notification_await(&state->notification, ObserverFinishReleased, state,
                          iree_infinite_timeout());
}

class StreamValueWaitObserverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));
    device_ = hrx_device_hal(hrx_device);
    ASSERT_NE(nullptr, device_);
    primary_queue_ = iree_hal_device_queue(device_, /*family_ordinal=*/0,
                                           /*queue_ordinal=*/0);
    ASSERT_NE(nullptr, primary_queue_);
    context_.device = device_;
    context_.host_allocator = iree_allocator_system();
    iree_hal_streaming_value_wait_lanes_initialize(&context_);
    context_initialized_ = true;
  }

  void TearDown() override {
    if (context_initialized_) {
      iree_hal_streaming_value_wait_lanes_deinitialize(&context_);
    }
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  iree_hal_streaming_value_wait_lane_t* CreateLane() {
    const iree_hal_queue_family_t* family =
        iree_hal_queue_family(primary_queue_);
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    iree_hal_queue_t* queue = nullptr;
    IREE_EXPECT_OK(iree_hal_queue_acquire(family, &queue_params, &queue));
    if (!queue) {
      return nullptr;
    }
    iree_hal_streaming_value_wait_lane_t* lane = nullptr;
    IREE_EXPECT_OK(iree_allocator_malloc(context_.host_allocator, sizeof(*lane),
                                         (void**)&lane));
    if (!lane) {
      iree_hal_queue_release(queue);
      return nullptr;
    }
    memset(lane, 0, sizeof(*lane));
    lane->queue = queue;
    lane->family = iree_hal_queue_family(queue);
    lane->priority = iree_hal_queue_priority(queue);
    lane->execution_resources = iree_hal_queue_execution_resources(queue);
    lane->owner_stream_id = 1;
    iree_slim_mutex_initialize(&lane->submission_mutex);
    return lane;
  }

  void DetachPendingLane(iree_hal_streaming_value_wait_lane_t* lane) {
    iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
    ASSERT_EQ(IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING,
              lane->list_state);
    if (lane->prev) {
      lane->prev->next = lane->next;
    } else {
      ASSERT_EQ(lane, context_.pending_value_wait_lanes);
      context_.pending_value_wait_lanes = lane->next;
    }
    if (lane->next) {
      lane->next->prev = lane->prev;
    }
    lane->next = nullptr;
    lane->prev = nullptr;
    lane->list_state = IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE;
    lane->restore_pending = true;
    iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
  }

  void PrepareSubmission(
      iree_hal_streaming_value_wait_lane_t* lane,
      iree_hal_streaming_value_wait_submission_t** out_submission,
      iree_hal_semaphore_t** out_completion) {
    IREE_ASSERT_OK(iree_hal_streaming_prepare_value_wait_submission(
        &context_, lane, nullptr, out_submission));
    *out_completion = (*out_submission)->completion_semaphore;
    iree_hal_semaphore_retain(*out_completion);
  }

  void PublishSubmission(
      iree_hal_streaming_value_wait_lane_t* lane,
      iree_hal_streaming_value_wait_submission_t* submission) {
    iree_slim_mutex_lock(&lane->submission_mutex);
    ASSERT_TRUE(
        iree_hal_streaming_value_wait_lane_accepts_submission(&context_, lane));
    iree_hal_streaming_publish_pending_value_wait_lane(&context_, lane,
                                                       submission);
    iree_slim_mutex_unlock(&lane->submission_mutex);
  }

  void WaitForObserverCount(int32_t expected_count) {
    ObserverCountCondition condition = {&context_, expected_count};
    ASSERT_TRUE(iree_notification_await(
        &context_.value_wait_observer_notification, ObserverCountEquals,
        &condition, iree_infinite_timeout()));
  }

  iree_hal_streaming_context_t context_ = {};
  iree_hal_device_t* device_ = nullptr;
  iree_hal_queue_t* primary_queue_ = nullptr;
  bool context_initialized_ = false;
};

TEST_F(StreamValueWaitObserverTest,
       RollingDepthHasBoundedRecordsAndLinearCompletionWork) {
  constexpr int32_t kDepth = 4;
  constexpr int32_t kSubmissionCount = 128;
  iree_hal_streaming_value_wait_lane_t* lane = CreateLane();
  ASSERT_NE(nullptr, lane);
  std::deque<iree_hal_semaphore_t*> completions;

  for (int32_t i = 0; i < kDepth; ++i) {
    if (i != 0) {
      DetachPendingLane(lane);
    }
    iree_hal_streaming_value_wait_submission_t* submission = nullptr;
    iree_hal_semaphore_t* completion = nullptr;
    PrepareSubmission(lane, &submission, &completion);
    PublishSubmission(lane, submission);
    completions.push_back(completion);
  }
  for (int32_t i = kDepth; i < kSubmissionCount; ++i) {
    IREE_ASSERT_OK(iree_hal_semaphore_signal(completions.front(), 1,
                                             /*frontier=*/nullptr));
    iree_hal_semaphore_release(completions.front());
    completions.pop_front();
    WaitForObserverCount(kDepth - 1);

    iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
    EXPECT_EQ(kDepth - 1, context_.live_value_wait_submission_count);
    EXPECT_LE(context_.peak_value_wait_submission_count, kDepth);
    iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);

    DetachPendingLane(lane);
    iree_hal_streaming_value_wait_submission_t* submission = nullptr;
    iree_hal_semaphore_t* completion = nullptr;
    PrepareSubmission(lane, &submission, &completion);
    PublishSubmission(lane, submission);
    completions.push_back(completion);
  }

  int32_t expected_count = kDepth;
  while (!completions.empty()) {
    IREE_ASSERT_OK(iree_hal_semaphore_signal(completions.front(), 1,
                                             /*frontier=*/nullptr));
    iree_hal_semaphore_release(completions.front());
    completions.pop_front();
    WaitForObserverCount(--expected_count);
  }

  // The final callback must recycle the lane without another wait-bearing API
  // call. Each callback queries and unlinks exactly its own two memberships.
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(nullptr, context_.pending_value_wait_lanes);
  EXPECT_EQ(lane, context_.idle_value_wait_lanes);
  EXPECT_EQ(IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE,
            lane->list_state);
  EXPECT_EQ(0u, lane->submission_count);
  EXPECT_EQ(nullptr, lane->retired_failure_head);
  EXPECT_EQ(0u, context_.live_value_wait_submission_count);
  EXPECT_LE(context_.peak_value_wait_submission_count, kDepth);
  EXPECT_EQ(kSubmissionCount, context_.value_wait_completion_query_count);
  EXPECT_EQ(kSubmissionCount, context_.value_wait_record_visit_count);
  EXPECT_EQ(kSubmissionCount, context_.value_wait_observer_removal_count);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
}

TEST_F(StreamValueWaitObserverTest,
       FailedTerminalHoleWaitsForEveryExactSubmission) {
  iree_hal_streaming_value_wait_lane_t* lane = CreateLane();
  ASSERT_NE(nullptr, lane);
  iree_hal_streaming_value_wait_submission_t* first = nullptr;
  iree_hal_semaphore_t* first_completion = nullptr;
  PrepareSubmission(lane, &first, &first_completion);
  PublishSubmission(lane, first);
  DetachPendingLane(lane);
  iree_hal_streaming_value_wait_submission_t* hole = nullptr;
  iree_hal_semaphore_t* hole_completion = nullptr;
  PrepareSubmission(lane, &hole, &hole_completion);
  PublishSubmission(lane, hole);

  iree_hal_semaphore_fail(
      hole_completion,
      iree_make_status(IREE_STATUS_ABORTED, "later exact submission failed"));
  WaitForObserverCount(1);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(lane, context_.pending_value_wait_lanes);
  EXPECT_EQ(first, lane->submission_head);
  EXPECT_EQ(first, lane->submission_tail);
  EXPECT_EQ(1u, lane->submission_count);
  EXPECT_EQ(hole, lane->retired_failure_head);
  EXPECT_EQ(1u, lane->retired_failure_count);
  EXPECT_EQ(2u, context_.live_value_wait_submission_count);
  EXPECT_TRUE(lane->has_failed_submission);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
  iree_hal_semaphore_release(hole_completion);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(first_completion, 1,
                                           /*frontier=*/nullptr));
  iree_hal_semaphore_release(first_completion);
  WaitForObserverCount(0);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(nullptr, context_.pending_value_wait_lanes);
  EXPECT_EQ(nullptr, context_.idle_value_wait_lanes);
  EXPECT_EQ(0u, context_.live_value_wait_submission_count);
  EXPECT_EQ(2u, context_.value_wait_completion_query_count);
  EXPECT_EQ(2u, context_.value_wait_record_visit_count);
  EXPECT_EQ(2u, context_.value_wait_observer_removal_count);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
}

TEST_F(StreamValueWaitObserverTest,
       RejectedPreparedObserverRollsBackWithoutPublication) {
  iree_hal_streaming_value_wait_lane_t* lane = CreateLane();
  ASSERT_NE(nullptr, lane);
  iree_hal_streaming_value_wait_submission_t* submission = nullptr;
  iree_hal_semaphore_t* completion = nullptr;
  PrepareSubmission(lane, &submission, &completion);

  iree_hal_streaming_reject_value_wait_submission(&context_, submission);
  WaitForObserverCount(0);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(completion, &value));
  iree_hal_semaphore_release(completion);

  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(nullptr, context_.active_value_wait_observers);
  EXPECT_EQ(nullptr, context_.pending_value_wait_lanes);
  EXPECT_EQ(0u, context_.live_value_wait_submission_count);
  EXPECT_EQ(1u, context_.value_wait_completion_query_count);
  EXPECT_EQ(0u, context_.value_wait_record_visit_count);
  EXPECT_EQ(1u, context_.value_wait_observer_removal_count);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);

  iree_hal_streaming_release_value_wait_lane(&context_, lane);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(lane, context_.idle_value_wait_lanes);
  EXPECT_EQ(IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_IDLE,
            lane->list_state);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
}

TEST_F(StreamValueWaitObserverTest,
       TeardownJoinsFinalObserverThroughNotificationPost) {
  iree_hal_streaming_value_wait_lane_t* lane = CreateLane();
  ASSERT_NE(nullptr, lane);

  ObserverFinishState finish_state;
  finish_state.context = &context_;
  iree_notification_initialize(&finish_state.notification);
  iree_atomic_store(&finish_state.phase, OBSERVER_FINISH_PHASE_INITIAL,
                    iree_memory_order_relaxed);
  context_.value_wait_observer_finish_hook = PauseFinalObserverFinish;
  context_.value_wait_observer_finish_hook_user_data = &finish_state;

  iree_hal_streaming_value_wait_submission_t* submission = nullptr;
  iree_hal_semaphore_t* completion = nullptr;
  PrepareSubmission(lane, &submission, &completion);
  PublishSubmission(lane, submission);

  std::atomic<bool> teardown_completed{false};
  context_initialized_ = false;
  std::thread teardown_thread([&] {
    iree_hal_streaming_value_wait_lanes_deinitialize(&context_);
    teardown_completed.store(true, std::memory_order_release);
  });

  const bool observer_paused =
      iree_notification_await(&finish_state.notification, ObserverFinishPaused,
                              &finish_state, iree_infinite_timeout());
  EXPECT_TRUE(observer_paused);
  if (observer_paused) {
    EXPECT_TRUE(finish_state.mutex_was_held.load(std::memory_order_acquire));
    EXPECT_FALSE(teardown_completed.load(std::memory_order_acquire));
  }

  iree_atomic_store(&finish_state.phase, OBSERVER_FINISH_PHASE_RELEASED,
                    iree_memory_order_release);
  iree_notification_post(&finish_state.notification, IREE_ALL_WAITERS);
  teardown_thread.join();
  EXPECT_TRUE(teardown_completed.load(std::memory_order_acquire));

  iree_notification_deinitialize(&finish_state.notification);
  iree_hal_semaphore_release(completion);
}

TEST_F(StreamValueWaitObserverTest,
       AcquiredLaneFailureLinearizesBeforeOrAfterAppend) {
  // Append wins the lane gate: the older failure becomes sticky only after the
  // accepted append is published, so the append remains tracked destroy-only.
  iree_hal_streaming_value_wait_lane_t* lane = CreateLane();
  ASSERT_NE(nullptr, lane);
  iree_hal_streaming_value_wait_submission_t* old_submission = nullptr;
  iree_hal_semaphore_t* old_completion = nullptr;
  PrepareSubmission(lane, &old_submission, &old_completion);
  PublishSubmission(lane, old_submission);
  DetachPendingLane(lane);
  iree_hal_streaming_value_wait_submission_t* appended_submission = nullptr;
  iree_hal_semaphore_t* appended_completion = nullptr;
  PrepareSubmission(lane, &appended_submission, &appended_completion);

  iree_slim_mutex_lock(&lane->submission_mutex);
  iree_hal_semaphore_fail(
      old_completion,
      iree_make_status(IREE_STATUS_ABORTED, "old exact submission failed"));
  ASSERT_TRUE(
      iree_hal_streaming_value_wait_lane_accepts_submission(&context_, lane));
  iree_hal_streaming_publish_pending_value_wait_lane(&context_, lane,
                                                     appended_submission);
  iree_slim_mutex_unlock(&lane->submission_mutex);
  WaitForObserverCount(1);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(appended_submission, lane->submission_head);
  EXPECT_EQ(old_submission, lane->retired_failure_head);
  EXPECT_EQ(2u, context_.live_value_wait_submission_count);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
  iree_hal_semaphore_release(old_completion);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(appended_completion, 1,
                                           /*frontier=*/nullptr));
  iree_hal_semaphore_release(appended_completion);
  WaitForObserverCount(0);

  // Failure wins the same gate while the lane is acquired: its exact proof is
  // retained on the lane until release performs queue-first destruction.
  lane = CreateLane();
  ASSERT_NE(nullptr, lane);
  old_submission = nullptr;
  old_completion = nullptr;
  PrepareSubmission(lane, &old_submission, &old_completion);
  PublishSubmission(lane, old_submission);
  DetachPendingLane(lane);
  iree_hal_semaphore_fail(old_completion,
                          iree_make_status(IREE_STATUS_ABORTED,
                                           "acquired exact submission failed"));
  WaitForObserverCount(0);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_NONE,
            lane->list_state);
  EXPECT_EQ(old_submission, lane->retired_failure_head);
  EXPECT_EQ(1u, lane->retired_failure_count);
  EXPECT_EQ(1u, context_.live_value_wait_submission_count);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
  iree_slim_mutex_lock(&lane->submission_mutex);
  EXPECT_FALSE(
      iree_hal_streaming_value_wait_lane_accepts_submission(&context_, lane));
  iree_slim_mutex_unlock(&lane->submission_mutex);
  iree_hal_streaming_release_value_wait_lane(&context_, lane);
  iree_hal_semaphore_release(old_completion);
  iree_slim_mutex_lock(&context_.value_wait_lane_mutex);
  EXPECT_EQ(0u, context_.live_value_wait_submission_count);
  EXPECT_EQ(nullptr, context_.pending_value_wait_lanes);
  EXPECT_EQ(nullptr, context_.idle_value_wait_lanes);
  iree_slim_mutex_unlock(&context_.value_wait_lane_mutex);
}

class StreamQueueLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(0)));
    hrx_device_t device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(0, &device)));
    device_entry_.hrx_device = device;
    device_entry_.hal_device = hrx_device_hal(device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &device_entry_.block_pool);
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_,
        {.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO},
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    // Release test dependencies before any assertion-driven cleanup can join
    // pending work. Tests signal value one; cleanup advances to two.
    for (auto* gate : gates_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 2, nullptr));
    }
    for (auto* exec : graph_execs_) {
      iree_hal_streaming_graph_exec_release(exec);
    }
    for (auto* graph : graphs_) {
      iree_hal_streaming_graph_release(graph);
    }
    for (auto* event : events_) {
      iree_hal_streaming_event_release(event);
    }
    iree_hal_streaming_context_release(context_);
    for (auto* stream : streams_) {
      iree_hal_streaming_stream_release(stream);
    }
    for (auto* gate : gates_) {
      iree_hal_semaphore_release(gate);
    }
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  iree_hal_streaming_stream_t* CreateStream() {
    iree_hal_streaming_stream_t* stream = nullptr;
    IREE_EXPECT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0,
        iree_allocator_system(), &stream));
    if (stream) {
      streams_.push_back(stream);
    }
    return stream;
  }

  iree_hal_semaphore_t* CreateGate() {
    iree_hal_semaphore_t* gate = nullptr;
    IREE_EXPECT_OK(iree_hal_semaphore_create(
        context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
    if (gate) {
      gates_.push_back(gate);
    }
    return gate;
  }

  // The CPU queue uses a semaphore dependency for externally controlled work.
  // GPU callers request the same lease for native memory predicates. Queue
  // selection, acceptance, and completion below use the shipping
  // implementation.
  iree_status_t StartLease(iree_hal_streaming_stream_t* stream,
                           iree_hal_semaphore_t* gate = nullptr) {
    iree_slim_mutex_lock(&stream->mutex);
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
    iree_hal_streaming_submission_t submission = {};
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_submission_prepare_locked(
          stream, stream->queue, signal_value,
          IREE_HAL_STREAMING_SUBMISSION_FLAG_INDEPENDENT, &submission);
    }
    iree_hal_semaphore_t* waits[] = {stream->timeline_semaphore, gate};
    uint64_t values[] = {wait_value, 1};
    const iree_hal_semaphore_list_t wait_list = {gate ? 2u : 1u, waits, values};
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_barrier(submission.queue, wait_list,
                                      submission.signals, nullptr,
                                      IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    }
    iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                iree_status_is_ok(status));
    iree_slim_mutex_unlock(&stream->mutex);
    return status;
  }

  void AwaitObserverCount(int32_t count) {
    ObserverCountCondition condition = {context_, count};
    ASSERT_TRUE(iree_notification_await(
        &context_->value_wait_observer_notification, ObserverCountEquals,
        &condition, iree_infinite_timeout()));
  }

  // CPU device registry entry used by the real streaming context.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context owning submitted work and completion observers.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Caller-owned stream references released after context teardown.
  std::vector<iree_hal_streaming_stream_t*> streams_;
  // Dependency gates released before joining accepted test work.
  std::vector<iree_hal_semaphore_t*> gates_;
  // Callback output kept alive through fixture teardown on assertion failure.
  std::atomic<iree_hal_queue_t*> observed_queue_{nullptr};
  // Independent producer callback output with the same lifetime.
  std::atomic<iree_hal_queue_t*> producer_queue_{nullptr};
  // Event handles kept alive until dependency gates have been released.
  std::vector<iree_hal_streaming_event_t*> events_;
  // Graph templates released after all dependency gates are open.
  std::vector<iree_hal_streaming_graph_t*> graphs_;
  // Instantiated graphs released before their templates and contexts.
  std::vector<iree_hal_streaming_graph_exec_t*> graph_execs_;
};

static iree_status_t ObserveExecutionQueue(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  static_cast<std::atomic<iree_hal_queue_t*>*>(user_data)->store(
      context->queue, std::memory_order_release);
  return iree_ok_status();
}

TEST_F(StreamQueueLeaseTest, OrdinaryWorkDoesNotAllocateACompletionObserver) {
  auto* stream = CreateStream();
  ASSERT_NE(nullptr, stream);
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      stream, iree_hal_make_host_call(ObserveExecutionQueue, &observed_queue_),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
  EXPECT_EQ(stream->queue, observed_queue_.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, stream->value_wait_lane);
  EXPECT_EQ(0u, context_->peak_value_wait_submission_count);
}

TEST_F(StreamQueueLeaseTest,
       DependentWorkKeepsLeaseAfterOriginalWaitCompletes) {
  auto* stream = CreateStream();
  auto* producer = CreateStream();
  auto* first_gate = CreateGate();
  auto* tail_gate = CreateGate();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, producer);
  ASSERT_NE(nullptr, first_gate);
  ASSERT_NE(nullptr, tail_gate);
  IREE_ASSERT_OK(StartLease(stream, first_gate));
  auto* leased_queue = stream->value_wait_lane->queue;
  EXPECT_NE(stream->queue, leased_queue);

  iree_hal_streaming_event_t* raw_event = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &raw_event));
  std::unique_ptr<iree_hal_streaming_event_t,
                  decltype(&iree_hal_streaming_event_release)>
      event(raw_event, iree_hal_streaming_event_release);
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      event.get(), stream, /*capture_external_record=*/false));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_wait_semaphores(
      stream, {1, &tail_gate, &gate_value}));
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      stream, iree_hal_make_host_call(ObserveExecutionQueue, &observed_queue_),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(first_gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(event.get()));
  AwaitObserverCount(2);
  iree_slim_mutex_lock(&stream->mutex);
  EXPECT_NE(nullptr, stream->value_wait_lane);
  if (stream->value_wait_lane) {
    EXPECT_EQ(leased_queue, stream->value_wait_lane->queue);
  }
  iree_slim_mutex_unlock(&stream->mutex);

  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      producer,
      iree_hal_make_host_call(ObserveExecutionQueue, &producer_queue_), args,
      IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(producer));
  EXPECT_NE(leased_queue, producer_queue_.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, observed_queue_.load(std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      producer->queue, iree_hal_semaphore_list_empty(),
      {1, &tail_gate, &gate_value}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
  AwaitObserverCount(0);
  EXPECT_EQ(leased_queue, observed_queue_.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, stream->value_wait_lane);
}

TEST_F(StreamQueueLeaseTest, RejectedAppendPreservesAcceptedLeaseAndTail) {
  auto* stream = CreateStream();
  auto* gate = CreateGate();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, gate);
  IREE_ASSERT_OK(StartLease(stream, gate));
  auto* lane = stream->value_wait_lane;
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_submission_t submission;
  iree_status_t status = iree_hal_streaming_submission_prepare_locked(
      stream, stream->queue, stream->pending_value + 1,
      IREE_HAL_STREAMING_SUBMISSION_FLAG_NONE, &submission);
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_submission_finish_locked(stream, &submission,
                                                /*accepted=*/false);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_ASSERT_OK(status);
  AwaitObserverCount(1);
  EXPECT_EQ(1u, stream->pending_value);
  EXPECT_EQ(lane, stream->value_wait_lane);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
  AwaitObserverCount(0);
  EXPECT_EQ(nullptr, stream->value_wait_lane);
}

TEST_F(StreamQueueLeaseTest, CompletedLeasesRecycleAcrossLiveStreams) {
  iree_hal_queue_t* reused_queue = nullptr;
  for (int i = 0; i < 16; ++i) {
    auto* stream = CreateStream();
    auto* gate = CreateGate();
    ASSERT_NE(nullptr, stream);
    ASSERT_NE(nullptr, gate);
    IREE_ASSERT_OK(StartLease(stream, gate));
    auto* queue = stream->value_wait_lane->queue;
    if (reused_queue) {
      EXPECT_EQ(reused_queue, queue);
    }
    reused_queue = queue;
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
    AwaitObserverCount(0);
    EXPECT_EQ(nullptr, stream->value_wait_lane);
  }
  EXPECT_EQ(1u, context_->peak_value_wait_submission_count);
  EXPECT_EQ(16u, context_->value_wait_completion_query_count);
}

TEST_F(StreamQueueLeaseTest, PublicTimelineFailureDoesNotReleasePendingWork) {
  auto* stream = CreateStream();
  auto* other_stream = CreateStream();
  auto* gate = CreateGate();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, other_stream);
  ASSERT_NE(nullptr, gate);
  IREE_ASSERT_OK(StartLease(stream, gate));
  auto* lane = stream->value_wait_lane;
  iree_hal_semaphore_fail(
      stream->timeline_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "public stream timeline failed"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_streaming_stream_synchronize(stream));
  EXPECT_EQ(lane, stream->value_wait_lane);

  // The failed public signal is not proof that the accepted queue operation
  // has stopped using its private completion storage or its stream.
  auto* other_gate = CreateGate();
  ASSERT_NE(nullptr, other_gate);
  IREE_ASSERT_OK(StartLease(other_stream, other_gate));
  EXPECT_NE(lane, other_stream->value_wait_lane);
  iree_hal_streaming_context_unregister_stream(context_, stream);
  EXPECT_EQ(nullptr, stream->context);
  EXPECT_EQ(nullptr, stream->queue);
  EXPECT_EQ(lane, stream->value_wait_lane);
  // Transfer the last non-observer reference away before completing the work.
  streams_.erase(streams_.begin());
  iree_hal_streaming_stream_release(stream);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(other_gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(other_stream));
  AwaitObserverCount(0);
  EXPECT_EQ(nullptr, context_->pending_value_wait_lanes);
  EXPECT_EQ(0u, context_->live_value_wait_submission_count);
}

TEST_F(StreamQueueLeaseTest, ContextTeardownJoinsDetachedStreamObserver) {
  auto* stream = CreateStream();
  auto* gate = CreateGate();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, gate);
  IREE_ASSERT_OK(StartLease(stream, gate));

  ObserverFinishState finish_state;
  finish_state.context = context_;
  iree_notification_initialize(&finish_state.notification);
  iree_atomic_store(&finish_state.phase, OBSERVER_FINISH_PHASE_INITIAL,
                    iree_memory_order_relaxed);
  context_->value_wait_observer_finish_hook = PauseFinalObserverFinish;
  context_->value_wait_observer_finish_hook_user_data = &finish_state;
  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  EXPECT_TRUE(iree_notification_await(&finish_state.notification,
                                      ObserverFinishPaused, &finish_state,
                                      iree_infinite_timeout()));

  // HIP stream destruction can remove registry ownership after its public
  // timeline completes while the completion observer still touches context
  // storage. Context destruction must join that final callback.
  iree_hal_streaming_context_unregister_stream(context_, stream);
  EXPECT_EQ(nullptr, stream->context);
  streams_.clear();
  iree_hal_streaming_stream_release(stream);
  std::atomic<bool> teardown_completed{false};
  auto* context = context_;
  context_ = nullptr;
  std::thread teardown_thread([&] {
    iree_hal_streaming_context_release(context);
    teardown_completed.store(true, std::memory_order_release);
  });
  EXPECT_TRUE(finish_state.mutex_was_held.load(std::memory_order_acquire));
  EXPECT_FALSE(teardown_completed.load(std::memory_order_acquire));
  iree_atomic_store(&finish_state.phase, OBSERVER_FINISH_PHASE_RELEASED,
                    iree_memory_order_release);
  iree_notification_post(&finish_state.notification, IREE_ALL_WAITERS);
  teardown_thread.join();
  EXPECT_TRUE(teardown_completed.load(std::memory_order_acquire));
  iree_notification_deinitialize(&finish_state.notification);
}

TEST_F(StreamQueueLeaseTest, GraphLaunchExtendsTheAcceptedStreamLease) {
  auto* stream = CreateStream();
  auto* gate = CreateGate();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, gate);
  iree_hal_streaming_graph_t* graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  std::unique_ptr<iree_hal_streaming_graph_t,
                  decltype(&iree_hal_streaming_graph_release)>
      owned_graph(graph, iree_hal_streaming_graph_release);
  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0, &node));
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
  std::unique_ptr<iree_hal_streaming_graph_exec_t,
                  decltype(&iree_hal_streaming_graph_exec_release)>
      owned_exec(exec, iree_hal_streaming_graph_exec_release);

  IREE_ASSERT_OK(StartLease(stream, gate));
  auto* lane = stream->value_wait_lane;
  auto result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream, &result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, result);
  EXPECT_EQ(lane, stream->value_wait_lane);
  EXPECT_EQ(2u, lane->submission_count);
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      stream, iree_hal_make_host_call(ObserveExecutionQueue, &observed_queue_),
      args, IREE_HAL_HOST_CALL_FLAG_NONE));
  auto* queue = lane->queue;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream));
  AwaitObserverCount(0);
  EXPECT_EQ(queue, observed_queue_.load(std::memory_order_acquire));
}

enum class DependencyKind {
  Stream,
  Event,
  ContextEvent,
  GraphEvent,
  Semaphore
};

class StreamQueueDependencyTest
    : public StreamQueueLeaseTest,
      public ::testing::WithParamInterface<DependencyKind> {};

TEST_P(StreamQueueDependencyTest, TransitiveWaitLeavesProducerQueueAvailable) {
  auto* source = CreateStream();
  auto* consumer = CreateStream();
  auto* producer = CreateStream();
  auto* gate = CreateGate();
  ASSERT_NE(nullptr, source);
  ASSERT_NE(nullptr, consumer);
  ASSERT_NE(nullptr, producer);
  ASSERT_NE(nullptr, gate);

  iree_hal_streaming_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &event));
  events_.push_back(event);
  IREE_ASSERT_OK(StartLease(source, gate));
  auto* source_queue = source->value_wait_lane->queue;
  if (GetParam() == DependencyKind::ContextEvent) {
    IREE_ASSERT_OK(iree_hal_streaming_event_record_after_streams(event, &source,
                                                                 1, nullptr));
    EXPECT_NE(context_->queue, context_->event_record_queue);
  } else {
    IREE_ASSERT_OK(iree_hal_streaming_event_record(
        event, source, /*capture_external_record=*/false));
  }

  switch (GetParam()) {
    case DependencyKind::Stream:
      IREE_ASSERT_OK(iree_hal_streaming_stream_wait_stream(consumer, source));
      break;
    case DependencyKind::Event:
    case DependencyKind::ContextEvent:
      IREE_ASSERT_OK(iree_hal_streaming_stream_wait_event(
          consumer, event, /*capture_external_wait=*/false));
      break;
    case DependencyKind::GraphEvent: {
      iree_hal_streaming_graph_t* graph = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_create(
          context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
          &graph));
      graphs_.push_back(graph);
      iree_hal_streaming_graph_node_t* node = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
          graph, nullptr, 0, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
          event, &node));
      iree_hal_streaming_graph_exec_t* exec = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
          graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));
      graph_execs_.push_back(exec);
      auto result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
      IREE_ASSERT_OK(
          iree_hal_streaming_graph_exec_launch(exec, consumer, &result));
      EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, result);
      break;
    }
    case DependencyKind::Semaphore: {
      iree_hal_streaming_recorded_point_t point;
      iree_hal_streaming_event_acquire_recorded_point(event, &point);
      iree_status_t status = iree_hal_streaming_stream_wait_semaphores(
          consumer, {1, &point.semaphore, &point.value});
      iree_hal_streaming_event_release_recorded_point(&point);
      IREE_ASSERT_OK(status);
      break;
    }
  }
  ASSERT_NE(nullptr, consumer->value_wait_lane);
  auto* consumer_queue = consumer->value_wait_lane->queue;
  EXPECT_NE(source_queue, consumer_queue);
  EXPECT_NE(context_->queue, consumer_queue);
  const uint64_t args[4] = {};
  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      consumer,
      iree_hal_make_host_call(ObserveExecutionQueue, &observed_queue_), args,
      IREE_HAL_HOST_CALL_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_streaming_queue_host_call(
      producer,
      iree_hal_make_host_call(ObserveExecutionQueue, &producer_queue_), args,
      IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(producer));
  EXPECT_EQ(context_->queue, producer_queue_.load(std::memory_order_acquire));
  EXPECT_EQ(nullptr, observed_queue_.load(std::memory_order_acquire));
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      producer->queue, iree_hal_semaphore_list_empty(), {1, &gate, &value},
      nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(consumer));
  AwaitObserverCount(0);
  EXPECT_EQ(consumer_queue, observed_queue_.load(std::memory_order_acquire));
}

INSTANTIATE_TEST_SUITE_P(Dependencies, StreamQueueDependencyTest,
                         ::testing::Values(DependencyKind::Stream,
                                           DependencyKind::Event,
                                           DependencyKind::ContextEvent,
                                           DependencyKind::GraphEvent,
                                           DependencyKind::Semaphore));

}  // namespace
