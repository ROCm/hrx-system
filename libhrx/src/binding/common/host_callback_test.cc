// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/host_callback.h"

#include <array>
#include <atomic>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/semaphore.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Allocator dependency with an explicit gate on one capture's terminal free.
struct CaptureAllocator {
  // Number of allocation requests served by this owner.
  std::atomic<int> allocations{0};
  // Reject allocations while preserving infallible deallocation.
  std::atomic<bool> reject_allocations{false};
  // Exact callback capture whose terminal free is held, or nullptr.
  std::atomic<void*> held_capture{nullptr};
  // Publishes entry into the held terminal free.
  std::binary_semaphore free_entered{0};
  // Releases the held terminal free.
  std::binary_semaphore free_release{0};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* allocator = static_cast<CaptureAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      if (allocator->reject_allocations.load(std::memory_order_acquire)) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      ++allocator->allocations;
    } else if (*inout_pointer &&
               *inout_pointer ==
                   allocator->held_capture.load(std::memory_order_acquire)) {
      allocator->free_entered.release();
      allocator->free_release.acquire();
    }
    const iree_allocator_t system = iree_allocator_system();
    return system.ctl(system.self, command, params, inout_pointer);
  }

  iree_allocator_t value() { return {this, Control}; }
};

struct CapturedState {
  // Callback resource whose destruction is observed independently of the job.
  iree_hal_resource_t resource;
  // Number of times the adapter releases its retained payload.
  int destroyed = 0;

  CapturedState() {
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            +[](iree_hal_resource_t* base) {
              ++reinterpret_cast<CapturedState*>(base)->destroyed;
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }
};

class HostCallbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    iree_hal_streaming_host_callback_queue_initialize(allocator_.value(),
                                                      &queue_);
  }

  void TearDown() override {
    iree_hal_streaming_host_callback_queue_deinitialize(&queue_);
    iree_async_proactor_release(proactor_);
  }

  iree_hal_semaphore_t* MakeSemaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_host_semaphore_create(
        proactor_, 0, iree_allocator_system(), &semaphore));
    return semaphore;
  }

  static void Increment(void* user_data) { ++*static_cast<int*>(user_data); }

  static iree_status_t Ready(
      iree_hal_host_call_t call,
      iree_hal_semaphore_list_t signals = iree_hal_semaphore_list_empty()) {
    iree_hal_host_call_context_t context = {.signal_semaphore_list = signals};
    const uint64_t args[4] = {};
    return call.fn(call.user_data, args, &context);
  }

  // Caller-driven native proactor borrowed by real host semaphores.
  iree_async_proactor_t* proactor_ = nullptr;
  // Allocator observing only the callback subsystem under test.
  CaptureAllocator allocator_;
  // Frontend-owned callback execution state.
  iree_hal_streaming_host_callback_queue_t queue_;
};

TEST_F(HostCallbackTest, UnusedQueueStartsNoThreadAndAllocatesNothing) {
  EXPECT_EQ(queue_.thread, nullptr);
  EXPECT_EQ(allocator_.allocations, 0);
}

TEST_F(HostCallbackTest, CancelledInvocationReleasesPayloadWithoutCallingIt) {
  int calls = 0;
  CapturedState captured;
  iree_hal_host_call_t call;
  IREE_ASSERT_OK(iree_hal_streaming_host_callback_prepare(
      &queue_, Increment, &calls, &captured.resource, &call));
  iree_hal_resource_release(&captured.resource);
  EXPECT_EQ(captured.destroyed, 0);
  // Releasing the prepared call models queue rejection or pre-invocation
  // cancellation, both of which end HAL ownership without running the function.
  iree_hal_resource_release(call.resource);
  EXPECT_EQ(captured.destroyed, 1);
  EXPECT_EQ(calls, 0);
}

TEST_F(HostCallbackTest, AllocationFailurePreservesOutputAndStartsNoWorker) {
  allocator_.reject_allocations = true;
  int calls = 0;
  const iree_hal_host_call_t expected =
      iree_hal_make_host_call(nullptr, &calls);
  iree_hal_host_call_t call = expected;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_host_callback_prepare(
                            &queue_, Increment, &calls, nullptr, &call));
  EXPECT_EQ(call.fn, expected.fn);
  EXPECT_EQ(call.user_data, expected.user_data);
  EXPECT_EQ(call.resource, expected.resource);
  EXPECT_EQ(queue_.thread, nullptr);
}

TEST_F(HostCallbackTest,
       ReadyCaptureFailureRunsNothingAndPublishesNoCompletion) {
  int calls = 0;
  iree_hal_host_call_t call;
  IREE_ASSERT_OK(iree_hal_streaming_host_callback_prepare(
      &queue_, Increment, &calls, nullptr, &call));
  auto* semaphore = MakeSemaphore();
  uint64_t value = 1;
  allocator_.reject_allocations = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Ready(call, {1, &semaphore, &value}));
  iree_hal_resource_release(call.resource);
  iree_hal_streaming_host_callback_queue_stop(&queue_);
  uint64_t actual = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &actual));
  EXPECT_EQ(actual, 0u);
  EXPECT_EQ(calls, 0);
  iree_hal_semaphore_release(semaphore);
}

TEST_F(HostCallbackTest, LateNativeReadyCallbackIsCancelledAfterWorkerStop) {
  int calls = 0;
  iree_hal_host_call_t call;
  IREE_ASSERT_OK(iree_hal_streaming_host_callback_prepare(
      &queue_, Increment, &calls, nullptr, &call));
  iree_hal_streaming_host_callback_queue_stop(&queue_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, Ready(call));
  iree_hal_resource_release(call.resource);
  EXPECT_EQ(calls, 0);
}

TEST_F(HostCallbackTest, ReadyFifoClonesCompletionListsAndPreservesCallOrder) {
  std::binary_semaphore entered(0), release(0);
  std::array<int, 2> order = {};
  int count = 0;
  struct Call {
    // Shared ordered result, written only on the callback worker.
    std::array<int, 2>* order;
    // Next result slot, written only on the callback worker.
    int* count;
    // This callback's observable identity.
    int value;
    // Optional gates holding the first callback while the second is enqueued.
    std::binary_semaphore* entered;
    // Releases a callback held at entry, or nullptr.
    std::binary_semaphore* release;
  } arguments[] = {
      {&order, &count, 17, &entered, &release},
      {&order, &count, 23, nullptr, nullptr},
  };
  auto invoke = +[](void* user_data) {
    auto* call = static_cast<Call*>(user_data);
    if (call->entered) {
      call->entered->release();
      call->release->acquire();
    }
    (*call->order)[(*call->count)++] = call->value;
  };
  iree_hal_host_call_t calls[2];
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_host_callback_prepare(
        &queue_, invoke, &arguments[i], nullptr, &calls[i]));
  }
  iree_hal_semaphore_t* semaphores[] = {MakeSemaphore(), MakeSemaphore(),
                                        MakeSemaphore()};
  uint64_t values[] = {3, 5, 7};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        Ready(calls[0], {1, &semaphores[0], &values[0]}));
  entered.acquire();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        Ready(calls[1], {2, &semaphores[1], &values[1]}));
  iree_hal_resource_release(calls[0].resource);
  iree_hal_resource_release(calls[1].resource);
  // The borrowed native callback arrays end at return from Ready.
  values[0] = values[1] = values[2] = 99;
  release.release();
  IREE_EXPECT_OK(
      iree_hal_semaphore_wait(semaphores[2], 7, iree_infinite_timeout(), 0));
  iree_hal_streaming_host_callback_queue_stop(&queue_);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(order, (std::array<int, 2>{17, 23}));
  for (auto* semaphore : semaphores) {
    iree_hal_semaphore_release(semaphore);
  }
}

TEST_F(HostCallbackTest, StopJoinsCpuTailAfterCompletionIsAlreadyVisible) {
  int calls = 0;
  struct Invocation {
    // Number of observed callback entries.
    int* calls;
    // Holds the worker until the submitting reference has been released.
    std::binary_semaphore release{0};
  } invocation = {&calls};
  iree_hal_host_call_t call;
  IREE_ASSERT_OK(iree_hal_streaming_host_callback_prepare(
      &queue_,
      +[](void* user_data) {
        auto* invocation = static_cast<Invocation*>(user_data);
        ++*invocation->calls;
        invocation->release.acquire();
      },
      &invocation, nullptr, &call));
  allocator_.held_capture = call.resource;
  auto* semaphore = MakeSemaphore();
  uint64_t value = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        Ready(call, {1, &semaphore, &value}));
  iree_hal_resource_release(call.resource);
  invocation.release.release();
  allocator_.free_entered.acquire();
  // The worker is now inside allocator cleanup after signaling completion.
  IREE_EXPECT_OK(
      iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(), 0));
  std::atomic<bool> stopped = false;
  std::thread shutdown([&] {
    iree_hal_streaming_host_callback_queue_stop(&queue_);
    stopped.store(true, std::memory_order_release);
  });
  iree_notification_await(
      &queue_.notification,
      +[](void* user_data) {
        auto* queue =
            static_cast<iree_hal_streaming_host_callback_queue_t*>(user_data);
        iree_slim_mutex_lock(&queue->mutex);
        const bool stopping = queue->stopping;
        iree_slim_mutex_unlock(&queue->mutex);
        return stopping;
      },
      &queue_, iree_infinite_timeout());
  EXPECT_FALSE(stopped.load(std::memory_order_acquire));
  allocator_.free_release.release();
  shutdown.join();
  EXPECT_TRUE(stopped.load(std::memory_order_acquire));
  EXPECT_EQ(calls, 1);
  iree_hal_semaphore_release(semaphore);
}

TEST_F(HostCallbackTest, ConcurrentReadyProducersRetireEveryCallBeforeJoin) {
  constexpr size_t kProducerCount = 4;
  constexpr uint64_t kCallsPerProducer = 32;
  std::atomic<int> calls = 0;
  std::array<iree_hal_semaphore_t*, kProducerCount> semaphores;
  std::vector<std::thread> producers;
  for (size_t i = 0; i < kProducerCount; ++i) {
    semaphores[i] = MakeSemaphore();
    producers.emplace_back([&, i] {
      for (uint64_t value = 1; value <= kCallsPerProducer; ++value) {
        iree_hal_host_call_t call;
        iree_status_t status = iree_hal_streaming_host_callback_prepare(
            &queue_,
            +[](void* data) { ++*static_cast<std::atomic<int>*>(data); },
            &calls, nullptr, &call);
        if (iree_status_is_ok(status)) {
          status = Ready(call, {1, &semaphores[i], &value});
          iree_hal_resource_release(call.resource);
          if (iree_status_is_deferred(status)) {
            iree_status_free(status);
            continue;
          }
        }
        iree_hal_semaphore_fail(semaphores[i], status);
        return;
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  for (auto* semaphore : semaphores) {
    IREE_EXPECT_OK(iree_hal_semaphore_wait(semaphore, kCallsPerProducer,
                                           iree_infinite_timeout(), 0));
  }
  iree_hal_streaming_host_callback_queue_stop(&queue_);
  EXPECT_EQ(calls, kProducerCount * kCallsPerProducer);
  for (auto* semaphore : semaphores) {
    iree_hal_semaphore_release(semaphore);
  }
}

}  // namespace
