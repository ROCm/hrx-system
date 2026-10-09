// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_value.h"

#include <atomic>
#include <deque>
#include <thread>
#include <utility>

#include "common/context.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  Cleanup cleanup_;
};
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

iree_hal_queue_family_spec_t MakeValueWaitFamily(uint32_t queue_count = 1) {
  iree_hal_queue_family_spec_t family =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  family.provisioned_queue_count = queue_count;
  family.physical_device_affinity = UINT64_C(1) << 0;
  family.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  family.zero_compute_atomic_capabilities.operations.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.device_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.system_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.system_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.wait_conditions.device_scope_32 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.device_scope_64 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_32 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_64 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.atomic_capabilities = family.zero_compute_atomic_capabilities;
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION;
  return family;
}

TEST(StreamQueueCapabilitiesTest, AcceptsDynamicZeroComputeWaitQueue) {
  const iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  EXPECT_TRUE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresDynamicWaitQueueAcquisition) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily(2);
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresEveryWaitCapability) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();

  family.zero_compute_atomic_capabilities.operations.system_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));

  family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_32 &=
      ~IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NOT_EQUAL;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));

  family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities.operations.device_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresOnePhysicalDevicePerFamily) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.physical_device_affinity = (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RejectsComputeBackedWaits) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities = {};
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, DoesNotRequireHostCallRole) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.role_flags |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL;
  EXPECT_TRUE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresAtomicRole) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.role_flags &= ~IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresIndependentProducerLane) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily(1);
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RejectsMissingFamilySpecification) {
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(nullptr));
}

TEST(StreamValueTimelineTest, OverflowDoesNotReserveDuplicateValue) {
  iree_hal_streaming_stream_t stream = {.pending_value =
                                            IREE_HAL_SEMAPHORE_MAX_VALUE};
  uint64_t wait_value = 17;
  uint64_t signal_value = 23;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_stream_reserve_next_value_locked(
                            &stream, &wait_value, &signal_value));
  EXPECT_EQ(IREE_HAL_SEMAPHORE_MAX_VALUE, stream.pending_value);
  EXPECT_EQ(17u, wait_value);
  EXPECT_EQ(23u, signal_value);
}

TEST(StreamValueWaitLaneTest, TerminalHoleDoesNotRetireBlockedLane) {
  iree_hal_streaming_context_t context = {};
  iree_slim_mutex_initialize(&context.value_wait_lane_mutex);

  iree_hal_streaming_value_wait_lane_t lane =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  lane.list_state = IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING;
  iree_hal_streaming_value_wait_submission_t pending =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  pending.state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED;
  pending.lane = &lane;
  iree_hal_streaming_value_wait_submission_t failed_hole =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  failed_hole.state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED;
  failed_hole.lane = &lane;
  failed_hole.is_terminal = true;
  failed_hole.has_failed = true;
  iree_hal_streaming_value_wait_submission_t completed_hole =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
           // differs from declaration order.
  completed_hole.state =
      IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED;
  completed_hole.lane = &lane;
  completed_hole.is_terminal = true;
  pending.next = &failed_hole;
  failed_hole.prev = &pending;
  failed_hole.next = &completed_hole;
  completed_hole.prev = &failed_hole;
  lane.submission_head = &pending;
  lane.submission_tail = &completed_hole;
  lane.submission_count = 3;
  context.pending_value_wait_lanes = &lane;
  context.live_value_wait_submission_count = 3;

  iree_hal_streaming_value_wait_submission_t* reclaimed = nullptr;
  iree_hal_streaming_value_wait_lane_t* completed_lanes = nullptr;
  iree_hal_streaming_value_wait_lane_t* failed_lanes = nullptr;
  iree_slim_mutex_lock(&context.value_wait_lane_mutex);
  iree_hal_streaming_detach_resolved_value_wait_lanes_locked(
      &context, &lane, &reclaimed, &completed_lanes, &failed_lanes);
  iree_slim_mutex_unlock(&context.value_wait_lane_mutex);

  EXPECT_EQ(&lane, context.pending_value_wait_lanes);
  EXPECT_EQ(&pending, lane.submission_head);
  EXPECT_EQ(&pending, lane.submission_tail);
  EXPECT_EQ(1u, lane.submission_count);
  EXPECT_EQ(&completed_hole, reclaimed);
  EXPECT_EQ(&failed_hole, lane.retired_failure_head);
  EXPECT_EQ(1u, lane.retired_failure_count);
  EXPECT_EQ(2u, context.live_value_wait_submission_count);
  EXPECT_TRUE(lane.has_failed_submission);
  EXPECT_EQ(nullptr, completed_lanes);
  EXPECT_EQ(nullptr, failed_lanes);
  EXPECT_EQ(3u, context.value_wait_record_visit_count);

  pending.is_terminal = true;
  reclaimed = nullptr;
  iree_slim_mutex_lock(&context.value_wait_lane_mutex);
  iree_hal_streaming_detach_resolved_value_wait_lanes_locked(
      &context, &lane, &reclaimed, &completed_lanes, &failed_lanes);
  iree_slim_mutex_unlock(&context.value_wait_lane_mutex);
  EXPECT_EQ(nullptr, context.pending_value_wait_lanes);
  EXPECT_EQ(nullptr, completed_lanes);
  EXPECT_EQ(&lane, failed_lanes);
  EXPECT_EQ(&pending, reclaimed);
  EXPECT_EQ(0u, lane.submission_count);
  EXPECT_EQ(0u, context.live_value_wait_submission_count);
  EXPECT_EQ(4u, context.value_wait_record_visit_count);

  iree_slim_mutex_deinitialize(&context.value_wait_lane_mutex);
}

struct ObserverCountCondition {
  iree_hal_streaming_context_t* context;
  int32_t expected_count;
};

static bool ObserverCountEquals(void* user_data) {
  auto* condition = static_cast<ObserverCountCondition*>(user_data);
  return iree_atomic_load(&condition->context->active_value_wait_observer_count,
                          iree_memory_order_acquire) ==
         condition->expected_count;
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
        &context_, lane, out_submission));
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
        &condition, iree_make_timeout_ms(10000)));
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
                              &finish_state, iree_make_timeout_ms(10000));
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

TEST(StreamValueWaitLaneTest,
     OwnerFailureDoesNotRetireBlockedAcceptedSubmission) {
  IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
  iree_hal_queue_t* wait_queue = nullptr;
  iree_hal_buffer_t* target_buffer = nullptr;
  iree_hal_buffer_mapping_t target_mapping = {};
  iree_hal_semaphore_t* owner_timeline = nullptr;
  iree_hal_semaphore_t* lane_completion = nullptr;
  iree_hal_semaphore_t* foreign_completion = nullptr;
  iree_hal_streaming_context_t context = {};
  iree_hal_streaming_value_wait_lane_t lane = {};
  bool context_mutex_initialized = false;
  bool lane_mutex_initialized = false;
  bool wait_submission_accepted = false;
  ScopeExit cleanup([&] {
    if (wait_submission_accepted && target_mapping.contents.data) {
      iree_atomic_store(
          reinterpret_cast<iree_atomic_int32_t*>(target_mapping.contents.data),
          1, iree_memory_order_release);
      iree_status_ignore(iree_hal_semaphore_wait(lane_completion, /*value=*/1,
                                                 iree_infinite_timeout(),
                                                 IREE_ASYNC_WAIT_FLAG_NONE));
    }
    if (target_mapping.contents.data) {
      IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&target_mapping));
    }
    iree_hal_semaphore_release(foreign_completion);
    iree_hal_semaphore_release(lane_completion);
    iree_hal_semaphore_release(owner_timeline);
    iree_hal_buffer_release(target_buffer);
    iree_hal_queue_release(wait_queue);
    if (lane_mutex_initialized) {
      iree_slim_mutex_deinitialize(&lane.submission_mutex);
    }
    if (context_mutex_initialized) {
      iree_slim_mutex_deinitialize(&context.value_wait_lane_mutex);
    }
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  });

  hrx_device_t hrx_device = nullptr;
  IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));
  iree_hal_device_t* device = hrx_device_hal(hrx_device);
  ASSERT_NE(nullptr, device);
  iree_hal_queue_t* foreign_queue =
      iree_hal_device_queue(device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, foreign_queue);
  const iree_hal_queue_family_t* family = iree_hal_queue_family(foreign_queue);
  iree_hal_queue_params_t queue_params;
  iree_hal_queue_params_initialize(&queue_params);
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &queue_params, &wait_queue));
  ASSERT_NE(foreign_queue, wait_queue);

  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), buffer_params, sizeof(uint32_t),
      &target_buffer));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      target_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_MAP_FLAG_NONE,
      /*local_byte_offset=*/0, sizeof(uint32_t), &target_mapping));
  ASSERT_NE(nullptr, target_mapping.contents.data);
  iree_atomic_store(
      reinterpret_cast<iree_atomic_int32_t*>(target_mapping.contents.data), 0,
      iree_memory_order_release);

  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &owner_timeline));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &lane_completion));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &foreign_completion));

  iree_hal_semaphore_t* wait_signal_semaphores[] = {
      owner_timeline,
      lane_completion,
  };
  uint64_t wait_signal_values[] = {1, 1};
  const iree_hal_semaphore_list_t wait_signal_list = {
      IREE_ARRAYSIZE(wait_signal_semaphores), wait_signal_semaphores,
      wait_signal_values};
  IREE_ASSERT_OK(iree_hal_queue_atomic_wait(
      wait_queue, iree_hal_semaphore_list_empty(), wait_signal_list,
      target_buffer, /*target_offset=*/0,
      iree_hal_atomic_wait_params_t{
          .value = 1,
          .mask = UINT32_MAX,
          .flags =
              IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          .width = IREE_HAL_ATOMIC_WIDTH_32,
          .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      },
      /*barriers=*/NULL));
  wait_submission_accepted = true;

  uint64_t value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(lane_completion, &value));
  ASSERT_EQ(0u, value);
  iree_hal_semaphore_fail(
      owner_timeline,
      iree_make_status(IREE_STATUS_ABORTED, "independent owner failure"));

  iree_hal_streaming_value_wait_submission_t submission =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment sequencing
           // spans intervening work.
  submission.completion_semaphore = lane_completion;
  submission.state = IREE_HAL_STREAMING_VALUE_WAIT_SUBMISSION_STATE_PUBLISHED;
  lane.queue = wait_queue;
  lane.list_state = IREE_HAL_STREAMING_VALUE_WAIT_LANE_LIST_STATE_PENDING;
  lane.submission_head = &submission;
  lane.submission_tail = &submission;
  lane.submission_count = 1;
  submission.lane = &lane;
  iree_slim_mutex_initialize(&lane.submission_mutex);
  lane_mutex_initialized = true;
  iree_slim_mutex_initialize(&context.value_wait_lane_mutex);
  context_mutex_initialized = true;
  context.pending_value_wait_lanes = &lane;
  context.live_value_wait_submission_count = 1;

  iree_hal_streaming_value_wait_submission_t* reclaimed = nullptr;
  iree_hal_streaming_value_wait_lane_t* completed_lanes = nullptr;
  iree_hal_streaming_value_wait_lane_t* failed_lanes = nullptr;
  iree_slim_mutex_lock(&context.value_wait_lane_mutex);
  iree_hal_streaming_detach_resolved_value_wait_lanes_locked(
      &context, &lane, &reclaimed, &completed_lanes, &failed_lanes);
  iree_slim_mutex_unlock(&context.value_wait_lane_mutex);
  EXPECT_EQ(&lane, context.pending_value_wait_lanes);
  EXPECT_EQ(nullptr, completed_lanes);
  EXPECT_EQ(nullptr, failed_lanes);

  // A distinct exact queue must still make progress while the accepted lane
  // remains blocked and its owner's unrelated timeline is failed.
  iree_hal_semaphore_t* foreign_signal_semaphores[] = {foreign_completion};
  uint64_t foreign_signal_values[] = {1};
  const iree_hal_semaphore_list_t foreign_signal_list = {
      IREE_ARRAYSIZE(foreign_signal_semaphores), foreign_signal_semaphores,
      foreign_signal_values};
  IREE_EXPECT_OK(iree_hal_queue_barrier(
      foreign_queue, iree_hal_semaphore_list_empty(), foreign_signal_list,
      /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_semaphore_wait(foreign_completion, /*value=*/1,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(&lane, context.pending_value_wait_lanes);

  iree_atomic_store(
      reinterpret_cast<iree_atomic_int32_t*>(target_mapping.contents.data), 1,
      iree_memory_order_release);
  wait_submission_accepted = false;
  iree_status_t completion_status = iree_hal_semaphore_wait(
      lane_completion, /*value=*/1, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE);
  const bool lane_failed =
      iree_status_code(completion_status) != IREE_STATUS_OK;
  if (!lane_failed) {
    IREE_EXPECT_OK(completion_status);
  } else {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, completion_status);
  }
  submission.is_terminal = true;
  submission.has_failed = lane_failed;

  reclaimed = nullptr;
  completed_lanes = nullptr;
  failed_lanes = nullptr;
  iree_slim_mutex_lock(&context.value_wait_lane_mutex);
  iree_hal_streaming_detach_resolved_value_wait_lanes_locked(
      &context, &lane, &reclaimed, &completed_lanes, &failed_lanes);
  iree_slim_mutex_unlock(&context.value_wait_lane_mutex);
  EXPECT_EQ(nullptr, context.pending_value_wait_lanes);
  EXPECT_TRUE(completed_lanes == &lane || failed_lanes == &lane);
  EXPECT_FALSE(completed_lanes == &lane && failed_lanes == &lane);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_query(owner_timeline, &value));

  // Every submission on the lane is terminal, so releasing the exact queue
  // cannot wait on an unresolved predicate.
  iree_hal_queue_release(wait_queue);
  wait_queue = nullptr;
}

}  // namespace
