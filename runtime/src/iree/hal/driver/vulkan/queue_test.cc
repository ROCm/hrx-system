// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/queue.h"

#include <atomic>
#include <thread>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/vulkan/util/libvulkan.h"

namespace iree::hal::cts {
namespace {

struct CompletionState {
  // Number of times the queue invoked the completion action.
  std::atomic<int> callback_count{0};
  // Status observed by the completion action.
  std::atomic<iree_status_code_t> status_code{IREE_STATUS_OK};
  // Number of action resources released by their final owner.
  std::atomic<int> destroy_count{0};
};

struct CompletionResource {
  // Ref-counted storage held by the caller and accepted submission.
  iree_hal_resource_t resource;
  // Borrowed observations that outlive submission completion.
  CompletionState* state;
};

static void DestroyCompletionResource(iree_hal_resource_t* base_resource) {
  auto* resource = reinterpret_cast<CompletionResource*>(base_resource);
  ++resource->state->destroy_count;
  delete resource;
}

static const iree_hal_resource_vtable_t kCompletionResourceVTable = {
    .destroy = DestroyCompletionResource,
};

static iree_hal_vulkan_queue_completion_action_t MakeCompletionAction(
    CompletionState* state) {
  auto* resource = new CompletionResource;
  iree_hal_resource_initialize(&kCompletionResourceVTable, &resource->resource);
  resource->state = state;
  return {
      .fn =
          [](void* user_data, const iree_status_t status) {
            auto* state = static_cast<CompletionState*>(user_data);
            state->status_code = iree_status_code(status);
            ++state->callback_count;
          },
      .user_data = state,
      .resource = &resource->resource,
  };
}

class VulkanQueueTest : public CtsTestBase<> {};

TEST_P(VulkanQueueTest, RejectedSubmissionDoesNotInvokeCompletion) {
  SemaphoreList waits(device_, {0}, {1});
  SemaphoreList signals(device_, {0}, {1});
  iree_hal_semaphore_fail(waits.semaphores[0],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  CompletionState state;
  auto action = MakeCompletionAction(&state);
  Status status(iree_hal_vulkan_queue_submit_barrier_with_action(
      reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_), waits,
      signals, /*barriers=*/nullptr, action));
  iree_hal_resource_release(action.resource);

  EXPECT_EQ(status.code(), StatusCode::kAborted);
  EXPECT_EQ(state.callback_count.load(), 0);
  EXPECT_EQ(state.destroy_count.load(), 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(VulkanQueueTest, AcceptedSubmissionCompletesOnceAndReleasesAction) {
  for (iree_status_code_t completion_status :
       {IREE_STATUS_OK, IREE_STATUS_ABORTED}) {
    SCOPED_TRACE(completion_status);
    SemaphoreList waits(device_, {0}, {1});
    SemaphoreList signals(device_, {0}, {1});
    CompletionState state;
    auto action = MakeCompletionAction(&state);
    Status status(iree_hal_vulkan_queue_submit_barrier_with_action(
        reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_), waits,
        signals, /*barriers=*/nullptr, action));
    iree_hal_resource_release(action.resource);
    IREE_ASSERT_OK(status.release());
    EXPECT_EQ(state.callback_count.load(), 0);
    EXPECT_EQ(state.destroy_count.load(), 0);

    if (completion_status == IREE_STATUS_OK) {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(waits.semaphores[0], 1,
                                               /*frontier=*/nullptr));
    } else {
      iree_hal_semaphore_fail(waits.semaphores[0],
                              iree_status_from_code(completion_status));
    }
    IREE_EXPECT_STATUS_IS(
        completion_status,
        iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(state.callback_count.load(), 1);
    EXPECT_EQ(state.status_code.load(), completion_status);
    EXPECT_EQ(state.destroy_count.load(), 1);
  }
}

TEST_P(VulkanQueueTest, ReentrantCompletionKeepsWaitProgressMonotonic) {
  SemaphoreList first_signals(device_, {0}, {1});
  SemaphoreList final_signals(device_, {0}, {1});
  struct State {
    // Queue borrowed until both completion callbacks have returned.
    iree_hal_vulkan_queue_t* queue;
    // Terminal signal for the observer submitted by the first callback.
    iree_hal_semaphore_list_t final_signals;
    // Epoch retired reentrantly before the first callback returns.
    std::atomic<uint64_t> nested_epoch{0};
    // Completion progress seen after the first callback has returned.
    std::atomic<uint64_t> observed_epoch{0};
  } state{reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_),
          final_signals};
  iree_hal_vulkan_queue_completion_action_t action = {
      .fn =
          [](void* user_data, const iree_status_t completion_status) {
            auto* state = static_cast<State*>(user_data);
            auto* queue = state->queue;
            iree_status_t status = iree_status_clone(completion_status);
            if (iree_status_is_ok(status)) {
              status = iree_hal_vulkan_queue_submit_barrier(
                  queue, iree_hal_semaphore_list_empty(),
                  iree_hal_semaphore_list_empty(), /*barriers=*/nullptr);
            }
            if (iree_status_is_ok(status)) {
              iree_slim_mutex_lock(&queue->submission_mutex);
              uint64_t epoch = queue->next_epoch_value - 1;
              iree_slim_mutex_unlock(&queue->submission_mutex);
              state->nested_epoch = epoch;
              VkSemaphoreWaitInfo wait_info = {
                  .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                  .semaphoreCount = 1,
                  .pSemaphores = &queue->epoch_semaphore,
                  .pValues = &epoch,
              };
              status = iree_status_from_vk_result(
                  __FILE__, __LINE__,
                  iree_vkWaitSemaphores_raw(&queue->syms, queue->logical_device,
                                            &wait_info, UINT64_MAX),
                  "vkWaitSemaphores");
            }
            if (iree_status_is_ok(status)) {
              // Retire the nested submission while this callback still owns
              // the earlier one, then observe progress on the worker's next
              // drain after the outer completion has finished.
              iree_hal_vulkan_queue_drain_completions(queue);
              iree_hal_vulkan_queue_completion_action_t observer = {
                  .fn =
                      [](void* user_data, const iree_status_t) {
                        auto* state = static_cast<State*>(user_data);
                        state->observed_epoch = (uint64_t)iree_atomic_load(
                            &state->queue->last_dequeued_epoch,
                            iree_memory_order_acquire);
                      },
                  .user_data = state,
              };
              status = iree_hal_vulkan_queue_submit_barrier_with_action(
                  queue, iree_hal_semaphore_list_empty(), state->final_signals,
                  /*barriers=*/nullptr, observer);
            }
            if (!iree_status_is_ok(status)) {
              iree_hal_semaphore_list_fail(state->final_signals, status);
            }
          },
      .user_data = &state,
  };
  IREE_ASSERT_OK(iree_hal_vulkan_queue_submit_barrier_with_action(
      state.queue, iree_hal_semaphore_list_empty(), first_signals,
      /*barriers=*/nullptr, action));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
      first_signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
      final_signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_GE(state.observed_epoch.load(), state.nested_epoch.load());
}

TEST_P(VulkanQueueTest, RetireWhileDeferredDependencyResolves) {
  // Retirement consumes the queue's frontier, so use a private device rather
  // than the fixture's shared device.
  DeviceCreateContext create_context;
  IREE_ASSERT_OK(create_context.Initialize(iree_allocator_system()));
  Ref<iree_hal_device_t> device;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, create_context.params(), iree_allocator_system(), device.out()));
  Ref<iree_hal_device_group_t> group;
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, create_context.frontier_tracker(), iree_allocator_system(),
      group.out()));
  auto family_ordinal =
      iree_hal_queue_family_ordinal(iree_hal_queue_family(transfer_queue_));
  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(
      iree_hal_device_queue(device, family_ordinal, /*queue_ordinal=*/0));
  ASSERT_NE(queue, nullptr);

  SemaphoreList waits(device, {0}, {1});
  SemaphoreList signals(device, {0}, {1});
  CompletionState state;
  auto action = MakeCompletionAction(&state);
  Status status(iree_hal_vulkan_queue_submit_barrier_with_action(
      queue, waits, signals, /*barriers=*/nullptr, action));
  iree_hal_resource_release(action.resource);
  IREE_ASSERT_OK(status.release());
  EXPECT_EQ(state.callback_count.load(), 0);

  // A dependency may resolve while shutdown joins accepted work. Both the
  // dependency and retirement can wake the completion worker; cancellation or
  // promotion must still complete the deferred submission exactly once.
  std::thread signal_thread([&]() {
    while (
        !iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire)) {
      std::this_thread::yield();
    }
    IREE_EXPECT_OK(iree_hal_semaphore_signal(waits.semaphores[0], 1,
                                             /*frontier=*/nullptr));
  });
  iree_hal_vulkan_queue_retire_frontier(queue);
  signal_thread.join();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(state.callback_count.load(), 1);
  EXPECT_EQ(state.status_code.load(), IREE_STATUS_CANCELLED);
  EXPECT_EQ(state.destroy_count.load(), 1);
}

CTS_REGISTER_TEST_SUITE(VulkanQueueTest);

}  // namespace
}  // namespace iree::hal::cts
