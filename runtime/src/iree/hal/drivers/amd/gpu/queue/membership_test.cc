// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/membership.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Native events coalesce signals. A counting semaphore would preserve stale
// wakes and could hide a missing notification after a source-table handoff.
class NativeEvent {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    waiting_ = true;
    condition_.notify_all();
    condition_.wait(lock, [&] { return signaled_; });
    signaled_ = false;
    waiting_ = false;
  }

  void Signal() {
    std::lock_guard<std::mutex> lock(mutex_);
    signaled_ = true;
    condition_.notify_all();
  }

  void AwaitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return waiting_ && !signaled_; });
  }

 private:
  // Serializes native event state and dependency observation.
  std::mutex mutex_;
  // Observes actual wait entry without elapsed-time readiness assumptions.
  std::condition_variable condition_;
  // A pending wake, consumed once regardless of how many signals preceded it.
  bool signaled_ = false;
  // Whether the sole completion worker is inside the native wait boundary.
  bool waiting_ = false;
};

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
    std::atomic<bool> idle{true};
    // Stable native coordinate supplied to the constructor.
    uint32_t ordinal = 0;
    // Installed dependency width, changed only by the mocked publisher.
    uint32_t frontier_count = 0;
    // Starting epoch inherited from the preceding coordinate owner.
    uint64_t initial_epoch = 0;
  };

  void SetUp() override {
    context_.gpu = &gpu_;
    context_.event = reinterpret_cast<amdf_gpu_event_t*>(this);
    context_.event_sink = {[](void*, const iree_hal_device_event_t*) {},
                           nullptr};
    gpu_.event_wait = [](amdf_gpu_event_t* event, uint64_t) {
      reinterpret_cast<MembershipTest*>(event)->event_.Wait();
      return AMDF_STATUS_OK;
    };
    gpu_.event_signal = [](amdf_gpu_event_t* event) {
      reinterpret_cast<MembershipTest*>(event)->event_.Signal();
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
              auto* queue = reinterpret_cast<Queue*>(base);
              const bool idle = queue->idle.load();
              if (!idle) {
                queue->owner->busy_.release();
              }
              return idle;
            },
        .is_host_idle = [](iree_hal_queue_t*) { return true; },
        .try_inactivate =
            [](iree_hal_queue_t* base, amdf_status_t* out_status) {
              auto* queue = reinterpret_cast<Queue*>(base);
              if (queue->owner->inactivate_) {
                queue->owner->inactivate_(queue);
              }
              *out_status = AMDF_STATUS_OK;
              return true;
            },
        .retire_failed =
            [](iree_hal_queue_t* base, iree_status_code_t) {
              reinterpret_cast<Queue*>(base)->idle = true;
              return true;
            },
        .shutdown =
            [](iree_hal_queue_t* base) {
              auto* queue = reinterpret_cast<Queue*>(base);
              EXPECT_TRUE(queue->idle.load());
              if (queue->owner->shutdown_) {
                queue->owner->shutdown_(queue);
              }
              ++queue->owner->shutdowns_;
              return AMDF_STATUS_OK;
            },
        .trim = [](iree_hal_queue_t*) {},
        .trim_scratch =
            [](iree_hal_queue_t* base) {
              auto* queue = reinterpret_cast<Queue*>(base);
              if (queue->owner->trim_) {
                queue->owner->trim_(queue);
              }
              return AMDF_STATUS_OK;
            },
        .notify_failure =
            [](iree_hal_queue_t* base) {
              auto* queue = reinterpret_cast<Queue*>(base);
              if (queue->owner->notify_) {
                queue->owner->notify_(queue);
              }
            },
        .assign_frontier = [](iree_hal_queue_t*, iree_async_frontier_tracker_t*,
                              iree_async_axis_t) {},
        .reserve_frontiers =
            [](iree_hal_queue_t* base, uint32_t count, iree_notification_t*) {
              reinterpret_cast<Queue*>(base)->frontier_count = count;
              return iree_ok_status();
            },
        .set_release =
            [](iree_hal_queue_t* base,
               bool (*release)(void*, iree_hal_queue_t*), void* data) {
              auto* queue = reinterpret_cast<Queue*>(base);
              queue->release = release;
              queue->release_data = data;
            },
        .completed_epoch = [](iree_hal_queue_t* base) -> uint64_t {
          return reinterpret_cast<Queue*>(base)->initial_epoch + 1;
        },
        .publication_source =
            [](iree_hal_queue_t* base) {
              return iree_hal_amd_gpu_publication_source_t{
                  .publish =
                      [](void* data) {
                        auto* queue = static_cast<Queue*>(data);
                        if (queue->owner->publish_) {
                          queue->owner->publish_(queue);
                        }
                        return false;
                      },
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
    programs_.registry.queue_count = 1;
    iree_slim_mutex_initialize(&programs_.registry.mutex);
    iree_notification_initialize(&programs_.registry.notification);
    iree_slim_mutex_initialize(&programs_.constants.growth_mutex);
    iree_slim_mutex_initialize(&programs_.constants.mutex);
    membership_.context = &context_;
    membership_.host_allocator = iree_allocator_system();
    membership_.publication = &publication_;
    membership_.completion = &completion_;
    membership_.lifecycle = &lifecycle_;
    membership_.programs = &programs_;
    membership_.create = [](void* data, uint32_t ordinal, uint32_t,
                            uint32_t frontier_count, uint64_t initial_epoch,
                            iree_hal_amd_gpu_signal_cache_t*,
                            const iree_hal_queue_params_t* params,
                            iree_hal_queue_t** out_queue) {
      auto* self = static_cast<MembershipTest*>(data);
      if (self->construct_) {
        IREE_RETURN_IF_ERROR(self->construct_(ordinal));
      }
      auto* queue = new Queue;
      queue->owner = self;
      queue->ordinal = ordinal;
      queue->frontier_count = frontier_count;
      queue->initial_epoch = initial_epoch;
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
  // Coalescing native notification shared by GPU completion and host wakes.
  NativeEvent event_;
  // Admission closure receipt, without elapsed-time readiness assumptions.
  std::counting_semaphore<> closed_{0};
  // Observation of accepted work that still requires its retirement proof.
  std::counting_semaphore<> busy_{0};
  // Completed native destruction count.
  std::atomic<uint32_t> shutdowns_{0};
  // Final host queue destructor count.
  std::atomic<uint32_t> destroyed_{0};
  // Publishes the dependency callout after its function has been initialized.
  std::atomic<bool> run_callout_{false};
  // Scripted queue callback; real membership and worker ownership surround it.
  std::function<void()> callout_;
  // Native constructor dependency, before the complete queue is returned.
  std::function<iree_status_t(uint32_t)> construct_;
  // Native shutdown dependency, after accepted work is idle.
  std::function<void(Queue*)> shutdown_;
  // Native stop dependency, with terminal traversal's lifetime ownership.
  std::function<void(Queue*)> inactivate_;
  // Admission notification dependency, borrowing a live queue.
  std::function<void(Queue*)> notify_;
  // Native scratch detachment dependency, borrowing a live queue.
  std::function<void(Queue*)> trim_;
  // Native publication dependency, outside worker source-table bookkeeping.
  std::function<void(Queue*)> publish_;
};

TEST_P(MembershipTest, IdleQueueReleaseRearmsRetainedDownload) {
  struct Download {
    // Arms preemption inside the next bounded completion metadata check.
    std::atomic<bool> pause{false};
    // Observes the worker before a membership change requests replacement.
    std::binary_semaphore entered{0};
    // Resumes observation with ready bytes while replacement is pending.
    std::binary_semaphore resume{0};
    // Receives final CPU-copy completion without another device event.
    std::binary_semaphore copied{0};
    // Worker-owned ready-copy predicate, cleared after the transform.
    bool ready = false;
  } download;
  membership_.completion_tail_count = 1;
  membership_.completion_tail[0] = {
      .fn =
          [](void* user_data, amdf_status_t,
             iree_hal_amd_gpu_completion_scope_t*,
             iree_hal_amd_gpu_completion_wait_t*,
             iree_hal_amd_gpu_completion_copy_t* out_copy) {
            auto* download = static_cast<Download*>(user_data);
            if (download->pause.exchange(false)) {
              // Model scheduler preemption during the short metadata check,
              // before the worker can accept the copy outside exclusion.
              download->entered.release();
              download->resume.acquire();
              download->ready = true;
            }
            if (download->ready && out_copy) {
              *out_copy = {
                  .fn =
                      [](void* user_data) {
                        auto* download = static_cast<Download*>(user_data);
                        download->ready = false;
                        download->copied.release();
                      },
                  .user_data = download,
              };
            }
            return false;
          },
      .user_data = &download,
  };
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* queue = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &queue));
  event_.AwaitIdle();
  download.pause.store(true);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  download.entered.acquire();
  const int32_t observing_state =
      iree_atomic_load(&completion_.state, iree_memory_order_acquire);
  std::jthread releaser([&] { iree_hal_queue_release(queue); });
  while (iree_atomic_load(&completion_.state, iree_memory_order_acquire) ==
         observing_state) {
    std::this_thread::yield();
  }
  // Replacement excludes this round's new copy borrow. The membership owner
  // must wake retained sources after swapping tables, even for idle removal.
  download.resume.release();
  releaser.join();
  download.copied.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(destroyed_.load(), 1u);
}

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
  queue->idle = false;
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
  reinterpret_cast<Queue*>(pending)->idle = false;
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
  reinterpret_cast<Queue*>(first)->idle = false;
  reinterpret_cast<Queue*>(second)->idle = false;
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

TEST_P(MembershipTest, RetainedPublisherDoesNotBlockIndependentMembership) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  auto* queue = reinterpret_cast<Queue*>(first);
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  publish_ = [&](Queue* publishing) {
    if (publishing == queue) {
      entered.release();
      resume.acquire();
      publishing->idle = true;
      EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
    }
  };
  queue->idle = false;
  iree_hal_amd_gpu_publication_notify(&publication_, queue->ordinal + 3);
  entered.acquire();

  // These acquisitions grow and replace the actual immutable source arrays.
  // Native frontier allocation is a separate dependency in this fixture.
  iree_hal_queue_t* independent[4] = {};
  for (auto& acquired : independent) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(
        &membership_, &params, &acquired));
  }
  for (auto* acquired : independent) {
    iree_hal_queue_release(acquired);
  }
  EXPECT_EQ(destroyed_.load(), 4u);
  EXPECT_FALSE(queue->idle.load());

  resume.release();
  iree_hal_queue_release(first);
  EXPECT_EQ(destroyed_.load(), 5u);
}

TEST_P(MembershipTest, NativeConstructionAllowsIndependentGrowthAndRelease) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  construct_ = [&](uint32_t ordinal) {
    if (ordinal == 2) {
      entered.release();
      resume.acquire();
    }
    return iree_ok_status();
  };
  iree_hal_queue_t* second = nullptr;
  std::jthread constructor([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                             &params, &second));
  });
  entered.acquire();
  // The fourth dynamic coordinate widens the vector while the second queue
  // is inside its native constructor. Neither that growth nor another queue's
  // final release may wait for the constructor.
  iree_hal_queue_t* third = nullptr;
  iree_hal_queue_t* fourth = nullptr;
  IREE_EXPECT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &third));
  IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &fourth));
  EXPECT_EQ(reinterpret_cast<Queue*>(third)->ordinal, 3u);
  EXPECT_EQ(reinterpret_cast<Queue*>(fourth)->ordinal, 4u);
  EXPECT_EQ(membership_.frontier_count, 8u);
  iree_hal_queue_release(first);
  iree_hal_queue_release(third);
  iree_hal_queue_release(fourth);
  EXPECT_EQ(destroyed_.load(), 3u);
  resume.release();
  constructor.join();
  EXPECT_EQ(reinterpret_cast<Queue*>(second)->frontier_count, 8u);
  iree_hal_queue_release(second);
  EXPECT_EQ(destroyed_.load(), 4u);
}

TEST_P(MembershipTest, NativeShutdownDoesNotReserveOtherCoordinates) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  iree_hal_queue_t* second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &second));
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  shutdown_ = [&](Queue* queue) {
    if (queue->ordinal == 1) {
      entered.release();
      resume.acquire();
    }
  };
  std::jthread releaser([&] { iree_hal_queue_release(first); });
  entered.acquire();
  iree_hal_queue_t* third = nullptr;
  IREE_EXPECT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &third));
  EXPECT_EQ(reinterpret_cast<Queue*>(third)->ordinal, 3u);
  iree_hal_queue_release(second);
  iree_hal_queue_release(third);
  EXPECT_EQ(destroyed_.load(), 2u);
  EXPECT_FALSE(iree_hal_amd_gpu_queue_membership_is_empty(&membership_));
  resume.release();
  releaser.join();
  shutdown_ = {};
  // Once joined, the same stable coordinate is reusable with its continued
  // epoch. The native shutdown did not reset or lend out its signal cache.
  iree_hal_queue_t* reused[3] = {};
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(
        &membership_, &params, &reused[i]));
    EXPECT_EQ(reinterpret_cast<Queue*>(reused[i])->ordinal, 3u - i);
    EXPECT_EQ(reinterpret_cast<Queue*>(reused[i])->initial_epoch, 1u);
  }
  for (auto* queue : reused) {
    iree_hal_queue_release(queue);
  }
  EXPECT_EQ(destroyed_.load(), 6u);
}

TEST_P(MembershipTest, NativeConstructorFailureReturnsItsCoordinate) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  construct_ = [](uint32_t) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "native backing");
  };
  iree_hal_queue_t* failed = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_queue_membership_acquire(
                            &membership_, &params, &failed));
  EXPECT_EQ(failed, nullptr);
  EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_is_empty(&membership_));
  construct_ = {};
  iree_hal_queue_t* recovered = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(
      &membership_, &params, &recovered));
  EXPECT_EQ(reinterpret_cast<Queue*>(recovered)->ordinal, 1u);
  EXPECT_EQ(reinterpret_cast<Queue*>(recovered)->initial_epoch, 0u);
  iree_hal_queue_release(recovered);
}

TEST_P(MembershipTest, ReleaseJoinsOnlyItsOwnLifecycleBorrower) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  notify_ = [&](Queue* queue) {
    if (queue->ordinal == 1) {
      entered.release();
      resume.acquire();
    }
  };
  std::jthread notifier(
      [&] { iree_hal_amd_gpu_queue_membership_notify_failure(&membership_); });
  entered.acquire();
  std::jthread releaser([&] { iree_hal_queue_release(first); });
  closed_.acquire();
  iree_hal_queue_t* independent = nullptr;
  IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(
      &membership_, &params, &independent));
  iree_hal_queue_release(independent);
  EXPECT_EQ(destroyed_.load(), 1u);
  EXPECT_EQ(shutdowns_.load(), 1u);
  resume.release();
  notifier.join();
  releaser.join();
  EXPECT_EQ(destroyed_.load(), 2u);
}

TEST_P(MembershipTest, ScratchTrimAllowsIndependentAcquisitionAndRelease) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  trim_ = [&](Queue* queue) {
    if (queue->ordinal == 1) {
      entered.release();
      resume.acquire();
      // Native scratch detachment can begin asynchronous backing release.
      // Final queue release must observe and join this accepted owner even
      // though it began after release closed admission to subsequent trims.
      queue->idle = false;
    }
  };
  std::jthread trimmer([&] {
    EXPECT_EQ(iree_hal_amd_gpu_queue_membership_trim(&membership_),
              AMDF_STATUS_OK);
  });
  entered.acquire();
  std::jthread releaser([&] { iree_hal_queue_release(first); });
  closed_.acquire();
  iree_hal_queue_t* independent = nullptr;
  IREE_EXPECT_OK(iree_hal_amd_gpu_queue_membership_acquire(
      &membership_, &params, &independent));
  iree_hal_queue_release(independent);
  EXPECT_EQ(destroyed_.load(), 1u);
  resume.release();
  trimmer.join();
  busy_.acquire();
  EXPECT_EQ(shutdowns_.load(), 1u);
  reinterpret_cast<Queue*>(first)->idle = true;
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  releaser.join();
  EXPECT_EQ(destroyed_.load(), 2u);
}

TEST_P(MembershipTest, TerminalTraversalJoinsAConcurrentConstructor) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  construct_ = [&](uint32_t) {
    entered.release();
    resume.acquire();
    return iree_ok_status();
  };
  std::jthread constructor([&] {
    iree_hal_queue_t* failed = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          iree_hal_amd_gpu_queue_membership_acquire(
                              &membership_, &params, &failed));
    EXPECT_EQ(failed, nullptr);
  });
  entered.acquire();
  const amdf_status_t error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  ASSERT_TRUE(iree_hal_amd_gpu_failure_begin(&failure_, error, "native wait",
                                             iree_time_now()));
  iree_hal_amd_gpu_completion_request_failure(&completion_, error);
  EXPECT_EQ(iree_hal_amd_gpu_completion_request_stop(&completion_),
            AMDF_STATUS_OK);
  amdf_status_t status = AMDF_STATUS_OK;
  EXPECT_FALSE(iree_hal_amd_gpu_queue_membership_try_failure(
      &membership_, IREE_HAL_AMD_GPU_FAILURE_CLOSING, IREE_STATUS_UNAVAILABLE,
      &status));
  // Existing owners were closed despite the constructor still being pending.
  closed_.acquire();
  closed_.acquire();
  EXPECT_EQ(status, AMDF_STATUS_OK);
  resume.release();
  // The failed acquisition enters its synchronous final release after native
  // construction returns. Its idle queue can stop independently instead of
  // waiting for the terminal traversal that is joining this constructor.
  closed_.acquire();
  constructor.join();
  inactivate_ = [&](Queue*) {
    const bool acquired = iree_slim_mutex_try_lock(&membership_.slots_mutex);
    EXPECT_TRUE(acquired);
    if (acquired) {
      iree_slim_mutex_unlock(&membership_.slots_mutex);
    }
  };
  for (auto phase :
       {IREE_HAL_AMD_GPU_FAILURE_CLOSING, IREE_HAL_AMD_GPU_FAILURE_INACTIVATING,
        IREE_HAL_AMD_GPU_FAILURE_RETIRING, IREE_HAL_AMD_GPU_FAILURE_JOINING}) {
    EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_try_failure(
        &membership_, phase, IREE_STATUS_UNAVAILABLE, &status));
    EXPECT_EQ(status, AMDF_STATUS_OK);
  }
  iree_hal_amd_gpu_queue_membership_finish_failure(&membership_,
                                                   AMDF_STATUS_OK);
  iree_hal_queue_release(first);
  EXPECT_EQ(destroyed_.load(), 2u);
}

TEST_P(MembershipTest, TerminalCalloutCanReleaseAnIndependentIdleQueue) {
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* first = nullptr;
  iree_hal_queue_t* second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_queue_membership_acquire(&membership_, &params, &first));
  IREE_ASSERT_OK(iree_hal_amd_gpu_queue_membership_acquire(&membership_,
                                                           &params, &second));
  const amdf_status_t error =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  ASSERT_TRUE(iree_hal_amd_gpu_failure_begin(&failure_, error, "native wait",
                                             iree_time_now()));
  iree_hal_amd_gpu_completion_request_failure(&completion_, error);
  EXPECT_EQ(iree_hal_amd_gpu_completion_request_stop(&completion_),
            AMDF_STATUS_OK);
  inactivate_ = [&](Queue* queue) {
    if (queue->ordinal == 2) {
      iree_hal_queue_release(first);
      EXPECT_EQ(destroyed_.load(), 1u);
    }
  };
  amdf_status_t status = AMDF_STATUS_OK;
  EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_try_failure(
      &membership_, IREE_HAL_AMD_GPU_FAILURE_INACTIVATING,
      IREE_STATUS_DATA_LOSS, &status));
  EXPECT_EQ(status, AMDF_STATUS_OK);
  iree_hal_amd_gpu_queue_membership_finish_failure(&membership_,
                                                   AMDF_STATUS_OK);
  iree_hal_queue_release(second);
  EXPECT_EQ(destroyed_.load(), 2u);
}

INSTANTIATE_TEST_SUITE_P(Deadline, MembershipTest,
                         ::testing::Values(0u, 30000u));

}  // namespace
