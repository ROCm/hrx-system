// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/device_failure.h"

#include <array>
#include <atomic>
#include <barrier>
#include <semaphore>
#include <thread>

#include "iree/async/proactor_platform.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"
#include "iree/hal/driver/amd/gpu/queue/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only native queues/events are dependencies. The terminal coordinator, both
// workers, concrete PM4 publishers/retirement, captures, and buffers are real.
class DeviceFailureTest : public ::testing::Test {
 protected:
  struct NativeQueue {
    // Native dependency controller, live through both worker joins.
    DeviceFailureTest* owner = nullptr;
    // Exact queue whose removal result this dependency supplies.
    uint32_t ordinal = 0;
  };

  void SetUp() override {
    context_.gpu = &gpu_;
    context_.event = reinterpret_cast<amdf_gpu_event_t*>(this);
    context_.event_sink = {
        .fn =
            [](void* data, const iree_hal_device_event_t* event) {
              auto* self = static_cast<DeviceFailureTest*>(data);
              const auto* payload = reinterpret_cast<
                  const iree_hal_device_driver_failure_event_t*>(
                  event->payload.data);
              if (payload->status_code == IREE_STATUS_DEADLINE_EXCEEDED) {
                ++self->deadlines_;
                self->deadline_reported_.release();
              }
            },
        .user_data = this,
    };
    gpu_.event_wait = [](amdf_gpu_event_t* event, uint64_t) {
      auto* self = reinterpret_cast<DeviceFailureTest*>(event);
      self->event_.acquire();
      return self->event_status_.load();
    };
    gpu_.event_signal = [](amdf_gpu_event_t* event) {
      reinterpret_cast<DeviceFailureTest*>(event)->event_.release();
      return AMDF_STATUS_OK;
    };
    gpu_.user_queue_inactivate = [](amdf_user_queue_t* queue) {
      auto& native = *reinterpret_cast<NativeQueue*>(queue);
      auto& self = *native.owner;
      EXPECT_EQ(self.released_.load(), 0u);
      if (native.ordinal == 0 && self.hold_stop_) {
        self.stop_entered_.release();
        self.resume_stop_.acquire();
      }
      if (native.ordinal == 1 && !amdf_status_is_ok(self.stop_status_)) {
        return self.stop_status_;
      }
      ++self.stopped_;
      return AMDF_STATUS_OK;
    };
    lifecycle_ = {
        .close = iree_hal_amd_gpu_pm4_queue_close,
        .is_idle = iree_hal_amd_gpu_pm4_queue_is_idle,
        .is_host_idle =
            [](iree_hal_queue_t* base) {
              const bool idle = iree_hal_amd_gpu_pm4_queue_is_host_idle(base);
              auto* queue =
                  reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(base);
              auto* native =
                  reinterpret_cast<NativeQueue*>(queue->storage.queue);
              if (!idle && native->owner->observe_host_join_) {
                native->owner->host_join_entered_.release();
              }
              return idle;
            },
        .try_inactivate = iree_hal_amd_gpu_pm4_queue_try_inactivate,
        .retire_failed = iree_hal_amd_gpu_pm4_queue_retire_failed,
        .notify_failure = iree_hal_amd_gpu_pm4_queue_notify_failure,
    };
    membership_.context = &context_;
    membership_.host_allocator = iree_allocator_system();
    membership_.lifecycle = &lifecycle_;
    membership_.provisioned_count = queues_.size();
    membership_.provisioned = provisioned_.data();
    iree_hal_amd_gpu_queue_membership_initialize(&membership_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0},
                                                                   &barriers_));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    for (uint32_t i = 0; i < queues_.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
          proactor_, nullptr, 0, 2, iree_allocator_system(), &outputs_[i]));
      auto& queue = queues_[i];
      native_[i] = {.owner = this, .ordinal = i};
      queue.context = &context_;
      queue.options.publication = &publication_;
      queue.options.source_ordinal = i;
      queue.options.retirement_capacity = issued_[i].size();
      queue.options.signals = &signals_[i];
      queue.retirement.capacity_mask = issued_[i].size() - 1;
      queue.issued = issued_[i].data();
      queue.storage.context = &context_;
      queue.storage.queue = reinterpret_cast<amdf_user_queue_t*>(&native_[i]);
      queue.storage.mapping =
          reinterpret_cast<amdf_user_queue_mapping_t*>(&native_[i]);
      iree_hal_amd_gpu_signal_cache_initialize(&publication_, i, &signals_[i]);
      iree_hal_amd_gpu_publisher_initialize(&queue.publisher);
      iree_hal_amd_gpu_capture_pool_initialize(2048, iree_allocator_system(),
                                               &queue.captures);
      iree_slim_mutex_initialize(&queue.pending.completion_mutex);
      iree_async_semaphore_wait_context_initialize(&queue.wait_context);
      iree_hal_amd_gpu_host_wait_set_initialize(&queue.wait_context,
                                                &queue.pending.waits);
      provisioned_[i] = &queue.base;
      publication_sources_[i] =
          iree_hal_amd_gpu_pm4_queue_publication_source(&queue.base);
      completion_sources_[i] =
          iree_hal_amd_gpu_pm4_queue_completion_source(&queue.base);
    }
  }

  void TearDown() override {
    Join();
    if (!amdf_status_is_ok(membership_.terminal.status)) {
      // The subject must retain every capture on failed native removal. The
      // dependency ends native use explicitly before fixture storage expires.
      EXPECT_EQ(released_.load(), 0u);
      stopped_ = queues_.size();
      for (auto& queue : queues_) {
        iree_hal_amd_gpu_pm4_submission_release(
            iree_hal_amd_gpu_pm4_retirement_submission(
                &queue.issued[0].action));
      }
    }
    EXPECT_EQ(released_.load(), 6u);
    for (auto& queue : queues_) {
      iree_hal_amd_gpu_capture_pool_deinitialize(&queue.captures);
      iree_hal_amd_gpu_publisher_deinitialize(&queue.publisher);
      iree_hal_amd_gpu_host_wait_set_deinitialize(&queue.pending.waits);
      iree_async_semaphore_wait_context_deinitialize(&queue.wait_context);
      iree_slim_mutex_deinitialize(&queue.pending.completion_mutex);
    }
    EXPECT_TRUE(iree_hal_amd_gpu_queue_membership_deinitialize(&membership_));
    for (auto* semaphore : outputs_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_async_proactor_release(proactor_);
  }

  void Start(uint32_t timeout_ms = 30000, bool notify_failure = true) {
    for (uint32_t i = 0; i < queues_.size(); ++i) {
      std::array<iree_hal_buffer_ref_t, 3> refs;
      for (uint32_t j = 0; j < refs.size(); ++j) {
        iree_hal_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
            {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
            IREE_HAL_BUFFER_USAGE_STORAGE, bytes_[i][j].size(),
            iree_make_byte_span(bytes_[i][j].data(), bytes_[i][j].size()),
            {[](void* data, iree_hal_buffer_t*) {
               auto* self = static_cast<DeviceFailureTest*>(data);
               EXPECT_EQ(self->stopped_.load(), self->queues_.size());
               ++self->released_;
             },
             this},
            iree_allocator_system(), &buffer));
        refs[j] = iree_hal_make_buffer_ref(buffer, 0, bytes_[i][j].size());
      }
      iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
      uint64_t value = 1;
      IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
          &queues_[i].captures, false, {}, {1, &outputs_[i], &value}, nullptr,
          &function_, {}, {}, {}, {refs.size(), refs.data()}, &barriers_,
          nullptr, 0, &capture));
      for (auto ref : refs) {
        iree_hal_buffer_release(ref.buffer);
      }
      issued_[i][0].action = {
          .tagged_submission = reinterpret_cast<uintptr_t>(capture) | 1,
      };
      iree_atomic_store(&queues_[i].producer.count, 1,
                        iree_memory_order_release);
      // A prior native observation has failed. No GPU completion word exists;
      // the real terminal worker must use the trusted host issued record.
      iree_atomic_store(&queues_[i].native_failure, native_error_,
                        iree_memory_order_release);
    }
    iree_hal_amd_gpu_device_failure_initialize(
        &context_, nullptr, timeout_ms, &publication_, &completion_,
        &membership_, nullptr, &failure_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        publication_sources_.size(), publication_sources_.data(), {},
        iree_allocator_system(), &publication_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
        &context_, completion_sources_.size(), completion_sources_.data(), 0,
        {}, iree_allocator_system(), &completion_));
    if (notify_failure) {
      iree_hal_amd_gpu_context_fail_execution(&context_, native_error_,
                                              "native completion");
    }
  }

  void Join() {
    if (completion_.thread) {
      iree_thread_release(completion_.thread);
      completion_.thread = nullptr;
      iree_slim_mutex_deinitialize(&completion_.drain_mutex);
    }
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    iree_hal_amd_gpu_device_failure_deinitialize(&failure_);
  }

  // Native dependency tables; no hardware is used or faulted.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed native execution identity and installed production callbacks.
  iree_hal_amd_gpu_context_t context_ = {};
  // Exact cold protocol contract also used by device construction.
  iree_hal_amd_gpu_queue_lifecycle_t lifecycle_ = {};
  // Actual registry and native ownership result seen by final releases.
  iree_hal_amd_gpu_queue_membership_t membership_ = {};
  // Actual production coordinator and its diagnostic state.
  iree_hal_amd_gpu_device_failure_t failure_ = {};
  // Actual existing publication worker and callback table.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Actual existing completion worker and callback table.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Two concrete independent compute queues.
  std::array<iree_hal_amd_gpu_pm4_queue_t, 2> queues_ = {};
  // Stable native dependency identities for each compute queue.
  std::array<NativeQueue, 2> native_ = {};
  // Borrowed provisioned queue identities consumed by membership.
  std::array<iree_hal_queue_t*, 2> provisioned_ = {};
  // Host issued records, independent of GPU retirement words.
  std::array<std::array<iree_hal_amd_gpu_pm4_retirement_slot_t, 4>, 2> issued_ =
      {};
  // Actual native-signal capacity metadata, with no backing needed here.
  std::array<iree_hal_amd_gpu_signal_cache_t, 2> signals_ = {};
  // Exact protocol publication callbacks, consumed by the real worker.
  std::array<iree_hal_amd_gpu_publication_source_t, 2> publication_sources_ =
      {};
  // Exact protocol retirement callbacks, consumed by the real worker.
  std::array<iree_hal_amd_gpu_completion_source_t, 2> completion_sources_ = {};
  // Immutable dispatch metadata needed only by capture construction.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Public barrier policy consumed only during capture construction.
  iree_hal_amd_gpu_pm4_barrier_profile_t barriers_ = {};
  // Six independently retained application buffers.
  std::array<std::array<std::array<uint8_t, 16>, 3>, 2> bytes_ = {};
  // Independent operation outputs; neither is failed by the test controller.
  std::array<iree_hal_semaphore_t*, 2> outputs_ = {};
  // Actual async service backing the public completion identities.
  iree_async_proactor_t* proactor_ = nullptr;
  // First native execution error, preserved separately from native cleanup.
  amdf_status_t native_error_ =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  // Optional rejection of the second native queue's removal.
  amdf_status_t stop_status_ = AMDF_STATUS_OK;
  // Explicitly holds a native removal to test independent deadline reporting.
  bool hold_stop_ = false;
  // Dependency event wake, with persistent notification semantics.
  std::counting_semaphore<> event_{0};
  // Native observation can fail concurrently with an independent publisher.
  std::atomic<amdf_status_t> event_status_{AMDF_STATUS_OK};
  // Native removal entry, used instead of sleeping to infer progress.
  std::binary_semaphore stop_entered_{0};
  // Test controller permits the held native call to return.
  std::binary_semaphore resume_stop_{0};
  // Error sink receipt, independent of the blocked native removal.
  std::binary_semaphore deadline_reported_{0};
  // Immutable test observation policy, installed before either worker starts.
  bool observe_host_join_ = false;
  // Receipts from the real idle predicate, independently of monitor scheduling.
  std::counting_semaphore<> host_join_entered_{0};
  // Native owners whose access has ended.
  std::atomic<uint32_t> stopped_{0};
  // Final application buffer callbacks after all native owners stopped.
  std::atomic<uint32_t> released_{0};
  // Number of deadline events, including concurrent cleanup observations.
  std::atomic<uint32_t> deadlines_{0};
};

TEST_F(DeviceFailureTest, StopsAllQueuesBeforeReleasingAnyCapture) {
  Start();
  Join();
  EXPECT_EQ(stopped_.load(), 2u);
  EXPECT_EQ(released_.load(), 6u);
  EXPECT_EQ(membership_.terminal.status, AMDF_STATUS_OK);
  EXPECT_EQ(iree_hal_amd_gpu_completion_query_status(&completion_),
            native_error_);
  EXPECT_EQ(deadlines_.load(), 0u);
}

TEST_F(DeviceFailureTest, PublicFailureWaitsForNativeStop) {
  hold_stop_ = true;
  Start();
  stop_entered_.acquire();
  EXPECT_EQ(stopped_.load(), 0u);
  EXPECT_EQ(released_.load(), 0u);
  for (auto* semaphore : outputs_) {
    uint64_t value = UINT64_MAX;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(value, 0u);
  }
  resume_stop_.release();
  Join();
  EXPECT_EQ(stopped_.load(), 2u);
  EXPECT_EQ(released_.load(), 6u);
  for (auto* semaphore : outputs_) {
    uint64_t value = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          iree_hal_semaphore_query(semaphore, &value));
  }
}

TEST_F(DeviceFailureTest, NativeRemovalFailureRetainsEveryCapture) {
  stop_status_ = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  Start();
  Join();
  EXPECT_EQ(stopped_.load(), 1u);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(membership_.terminal.status, stop_status_);
  EXPECT_EQ(iree_hal_amd_gpu_completion_query_status(&completion_),
            native_error_);
  for (auto* semaphore : outputs_) {
    uint64_t value = UINT64_MAX;
    IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore, &value));
    EXPECT_EQ(value, 0u);
  }
}

TEST_F(DeviceFailureTest, ExistingPublisherReportsBlockedNativeStop) {
  hold_stop_ = true;
  Start(0);
  stop_entered_.acquire();
  deadline_reported_.acquire();
  EXPECT_EQ(stopped_.load(), 0u);
  EXPECT_EQ(released_.load(), 0u);
  resume_stop_.release();
  Join();
  EXPECT_EQ(membership_.terminal.status, AMDF_STATUS_OK);
  EXPECT_EQ(released_.load(), 6u);
  EXPECT_EQ(deadlines_.load(), 1u);
}

TEST_F(DeviceFailureTest, PublisherJoinExpiryRetainsUnprovenCaptures) {
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_try_acquire(&queues_[0].publisher));
  observe_host_join_ = true;
  Start(0);
  host_join_entered_.acquire();
  deadline_reported_.acquire();
  EXPECT_EQ(iree_atomic_load(&membership_.terminal.completed,
                             iree_memory_order_acquire),
            0);
  // The borrowed publisher prevents its own stop, not the other queue's.
  // Capture retirement still requires every native owner to have stopped.
  EXPECT_EQ(stopped_.load(), 1u);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(deadlines_.load(), 1u);
  // The caller ends its explicit publisher borrow only after the timeout.
  // No queued entries exist and the subject did not steal that ownership.
  EXPECT_FALSE(
      iree_hal_amd_gpu_publisher_handoff(&queues_[0].publisher, nullptr));
  // Production publishers end this borrow with the closed-queue wake. This
  // direct borrow has no submitted entry, so its controller supplies the wake.
  EXPECT_TRUE(iree_hal_amd_gpu_context_wake_failure(&context_));
  Join();
  EXPECT_EQ(membership_.terminal.status,
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(stopped_.load(), 1u);
  EXPECT_EQ(released_.load(), 0u);
}

TEST_F(DeviceFailureTest, NativeRemovalFailureStillJoinsHostCallbacks) {
  // An accepted external host callback has an independent lifetime from the
  // unread native captures. Its dependency keeps that borrower live until the
  // test explicitly returns it through the production count/notification pair.
  iree_atomic_store(&queues_[0].pending.host_owners, 1,
                    iree_memory_order_release);
  observe_host_join_ = true;
  stop_status_ = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  Start(0);
  host_join_entered_.acquire();
  deadline_reported_.acquire();
  EXPECT_EQ(iree_atomic_load(&membership_.terminal.completed,
                             iree_memory_order_acquire),
            0);
  EXPECT_EQ(stopped_.load(), 1u);
  EXPECT_EQ(released_.load(), 0u);
  iree_slim_mutex_lock(&queues_[0].pending.completion_mutex);
  iree_atomic_store(&queues_[0].pending.host_owners, 0,
                    iree_memory_order_release);
  EXPECT_TRUE(iree_hal_amd_gpu_context_wake_failure(&context_));
  iree_slim_mutex_unlock(&queues_[0].pending.completion_mutex);
  Join();
  EXPECT_EQ(membership_.terminal.status, stop_status_);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(deadlines_.load(), 1u);
}

TEST_F(DeviceFailureTest, HostJoinPastDeadlinePreservesProvenNativeCleanup) {
  iree_atomic_store(&queues_[0].pending.host_owners, 1,
                    iree_memory_order_release);
  observe_host_join_ = true;
  Start(0);
  host_join_entered_.acquire();
  deadline_reported_.acquire();
  EXPECT_EQ(iree_atomic_load(&membership_.terminal.completed,
                             iree_memory_order_acquire),
            0);
  EXPECT_EQ(stopped_.load(), 2u);
  EXPECT_EQ(released_.load(), 6u);
  iree_slim_mutex_lock(&queues_[0].pending.completion_mutex);
  iree_atomic_store(&queues_[0].pending.host_owners, 0,
                    iree_memory_order_release);
  EXPECT_TRUE(iree_hal_amd_gpu_context_wake_failure(&context_));
  iree_slim_mutex_unlock(&queues_[0].pending.completion_mutex);
  Join();
  EXPECT_EQ(membership_.terminal.status, AMDF_STATUS_OK);
  EXPECT_EQ(deadlines_.load(), 1u);
}

TEST_F(DeviceFailureTest,
       ConcurrentPublisherAndNativeWaitShareOneTerminalJoin) {
  Start(30000, false);
  std::barrier<> start(2);
  std::jthread publisher([&] {
    start.arrive_and_wait();
    iree_hal_amd_gpu_context_fail_execution(&context_, native_error_,
                                            "native publication");
    EXPECT_EQ(iree_hal_amd_gpu_completion_query_status(&completion_),
              native_error_);
  });
  event_status_ = native_error_;
  start.arrive_and_wait();
  event_.release();
  publisher.join();
  Join();
  EXPECT_EQ(membership_.terminal.status, AMDF_STATUS_OK);
  EXPECT_EQ(stopped_.load(), 2u);
  EXPECT_EQ(released_.load(), 6u);
}

}  // namespace
