// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/membership.h"

#include <atomic>
#include <functional>
#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// The protocol queue is a dependency. The actual membership implementation
// owns construction, worker/source handoffs, final release, and terminal joins.
class MembershipTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  struct Queue {
    // Real refcounted HAL identity, destroyed by public final release.
    iree_hal_queue_t base;
    // Borrowed dependency controller, live through release and worker joins.
    MembershipTest* owner = nullptr;
    // Production membership's installed final-release callback.
    bool (*release)(void*, iree_hal_queue_t*) = nullptr;
    // Borrowed stable slot passed to the release callback.
    void* release_data = nullptr;
    // Accepted-work lifetime, distinct from the first execution error.
    std::atomic<bool> idle{false};
  };

  void SetUp() override {
    context_.gpu = &gpu_;
    context_.event = reinterpret_cast<amdf_gpu_event_t*>(this);
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    gpu_.event_wait = [](amdf_gpu_event_t* event, uint64_t) {
      reinterpret_cast<MembershipTest*>(event)->event_.acquire();
      return AMDF_STATUS_OK;
    };
    gpu_.event_signal = [](amdf_gpu_event_t* event) {
      reinterpret_cast<MembershipTest*>(event)->event_.release();
      return AMDF_STATUS_OK;
    };
    vtable_.destroy = [](iree_hal_queue_t* base) {
      auto* queue = reinterpret_cast<Queue*>(base);
      if (queue->release(queue->release_data, base)) {
        ++queue->owner->destroyed_;
        delete queue;
      }
    };
    lifecycle_ = {
        .close =
            [](iree_hal_queue_t* base) {
              reinterpret_cast<Queue*>(base)->owner->closed_.release();
            },
        .is_idle =
            [](iree_hal_queue_t* base) {
              return reinterpret_cast<Queue*>(base)->idle.load();
            },
        .shutdown =
            [](iree_hal_queue_t* base) {
              auto* queue = reinterpret_cast<Queue*>(base);
              EXPECT_TRUE(queue->idle.load());
              ++queue->owner->shutdowns_;
              return AMDF_STATUS_OK;
            },
        .assign_frontier = [](iree_hal_queue_t*, iree_async_frontier_tracker_t*,
                              iree_async_axis_t) {},
        .reserve_frontiers =
            [](iree_hal_queue_t*, uint32_t, iree_notification_t*) {
              return iree_ok_status();
            },
        .set_release =
            [](iree_hal_queue_t* base,
               bool (*release)(void*, iree_hal_queue_t*), void* data) {
              auto* queue = reinterpret_cast<Queue*>(base);
              queue->release = release;
              queue->release_data = data;
            },
        .completed_epoch = [](iree_hal_queue_t*) -> uint64_t { return 1; },
        .publication_source =
            [](iree_hal_queue_t* base) {
              return iree_hal_amd_gpu_publication_source_t{
                  .publish = [](void*) { return false; },
                  .user_data = base,
              };
            },
        .completion_source =
            [](iree_hal_queue_t* base) {
              return iree_hal_amd_gpu_completion_source_t{
                  .fn =
                      [](void* data, amdf_status_t,
                         iree_hal_amd_gpu_completion_scope_t* scope,
                         iree_hal_amd_gpu_completion_wait_t*,
                         iree_hal_amd_gpu_completion_copy_t*) {
                        auto* queue = static_cast<Queue*>(data);
                        if (queue == &queue->owner->provisioned_ &&
                            queue->owner->run_callout_.exchange(false)) {
                          iree_hal_amd_gpu_completion_scope_unlock(scope);
                          queue->owner->callout_();
                        }
                        return false;
                      },
                  .user_data = base,
              };
            },
    };
    // This dependency has no resident recordings. Membership still uses the
    // real program registry's ordinary coordinate-growth operation.
    programs_.host_allocator = iree_allocator_system();
    programs_.context = &context_;
    programs_.queue_count = 1;
    iree_slim_mutex_initialize(&programs_.registry_mutex);
    membership_.context = &context_;
    membership_.host_allocator = iree_allocator_system();
    membership_.publication = &publication_;
    membership_.completion = &completion_;
    membership_.lifecycle = &lifecycle_;
    membership_.programs = &programs_;
    membership_.create = [](void* data, uint32_t, uint32_t, uint32_t, uint64_t,
                            iree_hal_amd_gpu_signal_cache_t*,
                            const iree_hal_queue_params_t* params,
                            iree_hal_queue_t** out_queue) {
      auto* self = static_cast<MembershipTest*>(data);
      auto* queue = new Queue;
      queue->owner = self;
      iree_hal_queue_initialize(&self->family_, params, &self->vtable_,
                                &queue->base);
      *out_queue = &queue->base;
      return iree_ok_status();
    };
    membership_.user_data = this;
    // One provisioned source establishes the device's initial coordinate.
    provisioned_.owner = this;
    provisioned_.idle = true;
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    iree_hal_queue_initialize(&family_, &params, &vtable_, &provisioned_.base);
    provisioned_ptr_ = &provisioned_.base;
    membership_.provisioned_count = 1;
    membership_.provisioned = &provisioned_ptr_;
    iree_hal_amd_gpu_queue_membership_initialize(&membership_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        0, nullptr, {}, iree_allocator_system(), &publication_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
        &context_, 0, nullptr, 0, {}, iree_allocator_system(), &completion_));
    iree_hal_amd_gpu_failure_initialize(&context_.endpoint_info, nullptr,
                                        context_.event_sink, GetParam(),
                                        &failure_);
    membership_.failure = &failure_;
  }

  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_deinitialize(&membership_));
    iree_hal_amd_gpu_failure_deinitialize(&failure_);
    iree_hal_amd_gpu_program_pool_deinitialize(&programs_);
  }

  // Native wake dependency; no GPU is accessed.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed native owner and diagnostic destination.
  iree_hal_amd_gpu_context_t context_ = {};
  // Actual source-lifetime owners joined by the fixture.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Actual completion observation and sticky execution failure.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Actual membership implementation, the subject under test.
  iree_hal_amd_gpu_queue_membership_t membership_ = {};
  // Actual shared software failure deadline used by acquired release.
  iree_hal_amd_gpu_failure_t failure_ = {};
  // Empty resident-program registry used during coordinate expansion.
  iree_hal_amd_gpu_program_pool_t programs_ = {};
  // Exact mocked protocol lifecycle; native behavior is tested separately.
  iree_hal_amd_gpu_queue_lifecycle_t lifecycle_ = {};
  // Refcounted queue's ordinary public destructor dispatch.
  iree_hal_queue_vtable_t vtable_ = {};
  // Borrowed public queue-family identity.
  iree_hal_queue_family_t family_ = {};
  // One idle provisioned dependency, never publicly released.
  Queue provisioned_;
  // Borrowed provisioned array entry.
  iree_hal_queue_t* provisioned_ptr_ = nullptr;
  // Persistent native event notifications.
  std::counting_semaphore<> event_{0};
  // Admission closure receipt, without elapsed-time readiness assumptions.
  std::counting_semaphore<> closed_{0};
  // Completed native destruction count.
  std::atomic<uint32_t> shutdowns_{0};
  // Final host queue destructor count.
  std::atomic<uint32_t> destroyed_{0};
  // Publishes the dependency callout after its function has been initialized.
  std::atomic<bool> run_callout_{false};
  // Scripted queue callback; real membership and worker ownership surround it.
  std::function<void()> callout_;
};

TEST_P(MembershipTest, CompletionCallbackCanAcquireAndReleaseAnIdleQueue) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* initial = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &initial));
  reinterpret_cast<Queue*>(initial)->idle = true;
  iree_hal_queue_release(initial);
  std::binary_semaphore completed{0};
  callout_ = [&] {
    iree_hal_queue_t* acquired = nullptr;
    iree_status_t status = iree_hal_amd_gpu_queue_membership_acquire(
        &membership_, &params, &acquired);
    IREE_EXPECT_OK(status);
    if (acquired) {
      reinterpret_cast<Queue*>(acquired)->idle = true;
      iree_hal_queue_release(acquired);
    }
    completed.release();
  };
  run_callout_.store(true);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  completed.acquire();
  EXPECT_EQ(shutdowns_.load(), 2u);
  EXPECT_EQ(destroyed_.load(), 2u);
  EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_is_empty(&membership_));
}

TEST_P(MembershipTest, FailedObservationWaitsForTerminalOwnership) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* base = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &base));
  auto* queue = reinterpret_cast<Queue*>(base);
  std::jthread releaser([&] { iree_hal_queue_release(base); });
  closed_.acquire();
  const amdf_status_t error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  ASSERT_TRUE(iree_hal_amd_gpu_failure_begin(&failure_, error, "native wait",
                                             iree_time_now()));
  iree_hal_amd_gpu_completion_request_failure(&completion_, error);
  EXPECT_EQ(iree_hal_amd_gpu_completion_request_stop(&completion_),
            AMDF_STATUS_OK);
  iree_notification_post(&membership_.notification, IREE_ALL_WAITERS);
  // Terminal ownership can take arbitrarily long. The waiting release retains
  // its own queue, with neither membership nor terminal-traversal exclusion.
  iree_slim_mutex_lock(&membership_.mutex);
  iree_slim_mutex_lock(&membership_.slots_mutex);
  EXPECT_EQ(shutdowns_.load(), 0u);
  EXPECT_EQ(destroyed_.load(), 0u);
  queue->idle = true;
  iree_slim_mutex_unlock(&membership_.slots_mutex);
  iree_slim_mutex_unlock(&membership_.mutex);
  iree_hal_amd_gpu_queue_membership_finish_failure(&membership_,
                                                   AMDF_STATUS_OK);
  releaser.join();
  EXPECT_EQ(shutdowns_.load(), 1u);
  EXPECT_EQ(destroyed_.load(), 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_is_empty(&membership_));
}

TEST_P(MembershipTest, WaitingReleaseDoesNotBlockCallbackAcquisition) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* pending = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &pending));
  std::jthread releaser([&] { iree_hal_queue_release(pending); });
  closed_.acquire();
  // Join enrollment before the independent callback begins. The pending queue
  // is still not idle, so this is a progress check, not a scheduling delay.
  iree_slim_mutex_lock(&membership_.mutex);
  iree_slim_mutex_unlock(&membership_.mutex);
  std::binary_semaphore completed{0};
  callout_ = [&] {
    iree_hal_queue_t* acquired = nullptr;
    iree_status_t status = iree_hal_amd_gpu_queue_membership_acquire(
        &membership_, &params, &acquired);
    IREE_EXPECT_OK(status);
    if (acquired) {
      reinterpret_cast<Queue*>(acquired)->idle = true;
      iree_hal_queue_release(acquired);
    }
    completed.release();
  };
  run_callout_.store(true);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  completed.acquire();
  EXPECT_EQ(destroyed_.load(), 1u);
  reinterpret_cast<Queue*>(pending)->idle = true;
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  releaser.join();
  EXPECT_EQ(destroyed_.load(), 2u);
}

TEST_P(MembershipTest, PendingReleasesJoinIndependently) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  iree_hal_queue_t* second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &second));
  std::jthread first_releaser([&] { iree_hal_queue_release(first); });
  closed_.acquire();
  std::jthread second_releaser([&] { iree_hal_queue_release(second); });
  closed_.acquire();
  iree_slim_mutex_lock(&membership_.mutex);
  iree_slim_mutex_unlock(&membership_.mutex);
  reinterpret_cast<Queue*>(second)->idle = true;
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  second_releaser.join();
  EXPECT_EQ(destroyed_.load(), 1u);
  reinterpret_cast<Queue*>(first)->idle = true;
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  first_releaser.join();
  EXPECT_EQ(destroyed_.load(), 2u);
}

INSTANTIATE_TEST_SUITE_P(Deadline, MembershipTest,
                         ::testing::Values(0u, 30000u));

}  // namespace
