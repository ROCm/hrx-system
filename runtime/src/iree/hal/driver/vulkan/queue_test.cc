// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/queue.h"

#include <algorithm>
#include <atomic>
#include <semaphore>
#include <thread>
#include <vector>

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

TEST_P(VulkanQueueTest, RecordableTransferHasOneNativeCompletionPoint) {
  Ref<iree_hal_buffer_t> source, target;
  IREE_ASSERT_OK(
      CreateFilledDeviceBuffer<uint32_t>(64, 0xABABABABu, source.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(256, target.out()));
  const uint32_t pattern = 0xCDCDCDCDu;
  std::vector<uint8_t> update(64, 0xEF);
  iree_hal_transfer_operation_t operations[4] = {};
  operations[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  operations[0].fill.target_buffer = target;
  operations[0].fill.length = 64;
  operations[0].fill.pattern = &pattern;
  operations[0].fill.pattern_length = sizeof(pattern);
  operations[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  operations[1].update.source_buffer = update.data();
  operations[1].update.target_buffer = target;
  operations[1].update.target_offset = 64;
  operations[1].update.length = 64;
  operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  operations[2].copy.source_buffer = source;
  operations[2].copy.target_buffer = target;
  operations[2].copy.target_offset = 128;
  operations[2].copy.length = 64;
  operations[3].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  operations[3].upload.source = update.data();
  operations[3].upload.target_buffer = target;
  operations[3].upload.target_offset = 192;
  operations[3].upload.length = 64;

  auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_);
  iree_slim_mutex_lock(&queue->submission_mutex);
  const uint64_t transfer_epoch = queue->next_epoch_value;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  SemaphoreList signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      transfer_queue_, iree_hal_semaphore_list_empty(), signal,
      IREE_ARRAYSIZE(operations), operations, /*barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_vulkan_semaphore_submission_t producer;
  ASSERT_TRUE(iree_hal_vulkan_semaphore_query_submission(signal.semaphores[0],
                                                         1, &producer));
  EXPECT_EQ(producer.epoch, transfer_epoch);
  iree_slim_mutex_lock(&queue->submission_mutex);
  const uint64_t next_epoch = queue->next_epoch_value;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  EXPECT_EQ(next_epoch, transfer_epoch + 1);

  std::vector<uint8_t> expected(256, 0);
  std::fill_n(expected.begin(), 64, 0xCD);
  std::fill_n(expected.begin() + 64, 64, 0xEF);
  std::fill_n(expected.begin() + 128, 64, 0xAB);
  std::fill_n(expected.begin() + 192, 64, 0xEF);
  EXPECT_EQ(ReadBufferBytes(target, 0, 256), expected);
}

TEST_P(VulkanQueueTest, UnalignedFillBatchHasOneNativeCompletionPoint) {
  const auto* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  for (iree_host_size_t family_ordinal = 0;
       family_ordinal < queues->family_count; ++family_ordinal) {
    const auto& family = queues->families[family_ordinal];
    if (!family.provisioned_queue_count ||
        !iree_any_bit_set(family.role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
      continue;
    }
    SCOPED_TRACE(family_ordinal);
    auto* base_queue = iree_hal_device_queue(device_, family_ordinal, 0);
    auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(base_queue);
    // One fill reuses the two-set cache; three fills require a pool lease.
    // Transfer-only families instead reuse the command buffer's scratch slot.
    for (iree_host_size_t operation_count : {1u, 3u}) {
      SCOPED_TRACE(operation_count);
      std::vector<uint8_t> expected(128, 0xCD);
      Ref<iree_hal_buffer_t> target, view;
      IREE_ASSERT_OK(CreateDeviceBufferWithData(expected.data(),
                                                expected.size(), target.out()));
      IREE_ASSERT_OK(iree_hal_buffer_subspan(
          target, 1, expected.size() - 1, iree_allocator_system(), view.out()));
      const uint8_t patterns[3][4] = {
          {0xA5}, {0x39, 0xC7}, {0x42, 0xD3, 0x81, 0x6E}};
      iree_hal_transfer_operation_t operations[3] = {};
      for (iree_host_size_t i = 0; i < operation_count; ++i) {
        const iree_host_size_t pattern_length = iree_host_size_t{1} << i;
        operations[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
        operations[i].fill.target_buffer = view;
        operations[i].fill.target_offset = i * 32;
        operations[i].fill.length = pattern_length * 5;
        operations[i].fill.pattern = patterns[i];
        operations[i].fill.pattern_length = pattern_length;
        for (iree_host_size_t j = 0; j < pattern_length * 5; ++j) {
          expected[1 + i * 32 + j] = patterns[i][j % pattern_length];
        }
      }
      iree_slim_mutex_lock(&queue->submission_mutex);
      const uint64_t transfer_epoch = queue->next_epoch_value;
      iree_slim_mutex_unlock(&queue->submission_mutex);
      SemaphoreList gate(device_, {0}, {1});
      SemaphoreList signal(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_transfer(base_queue, gate, signal,
                                             operation_count, operations,
                                             /*barriers=*/nullptr));
      view.reset();
      IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      iree_hal_vulkan_semaphore_submission_t producer;
      ASSERT_TRUE(iree_hal_vulkan_semaphore_query_submission(
          signal.semaphores[0], 1, &producer));
      EXPECT_EQ(producer.epoch, transfer_epoch);
      iree_slim_mutex_lock(&queue->submission_mutex);
      const uint64_t next_epoch = queue->next_epoch_value;
      iree_slim_mutex_unlock(&queue->submission_mutex);
      EXPECT_EQ(next_epoch, transfer_epoch + 1);
      EXPECT_EQ(ReadBufferBytes(target, 0, expected.size()), expected);
    }
  }
}

TEST_P(VulkanQueueTest, UploadRunsBeforeProducerRetirement) {
  for (iree_status_code_t dependency_status :
       {IREE_STATUS_OK, IREE_STATUS_ABORTED}) {
    SCOPED_TRACE(dependency_status);
    const auto input = MakeDeterministicBytes(64);
    Ref<iree_hal_buffer_t> target;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(input.size(), target.out()));
    SemaphoreList gate(device_, {0}, {1});
    SemaphoreList producer_signals(device_, {0, 0}, {1, 1});
    SemaphoreList consumer_signals(device_, {0}, {1});
    struct State {
      // Expected outcome of the producer's second signal callback.
      iree_status_code_t terminal_status;
      // Announces that GPU completion reached caller-resource retirement.
      std::binary_semaphore entered{0};
      // Holds that retirement while the consumer demonstrates native progress.
      std::binary_semaphore resume{0};
      // Holds the completion worker after it has retired the dependency edge.
      std::binary_semaphore terminal_entered{0};
      // Allows the worker to return after another helper retires the consumer.
      std::binary_semaphore terminal_resume{0};
      // Joins the final public callback's access to this state.
      std::binary_semaphore terminal_finished{0};
    } state{dependency_status};
    struct Resource {
      // Resource retained by the producer until its native work completes.
      iree_hal_resource_t base;
      // Borrowed handshake state outliving both terminal signals.
      State* state;
    };
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            [](iree_hal_resource_t* base) {
              auto* resource = reinterpret_cast<Resource*>(base);
              resource->state->entered.release();
              resource->state->resume.acquire();
              delete resource;
            },
    };
    auto* resource = new Resource;
    iree_hal_resource_initialize(&vtable, &resource->base);
    resource->state = &state;
    auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_);
    const iree_hal_vulkan_queue_completion_action_t action = {
        .resource = &resource->base,
    };
    Status producer_status(iree_hal_vulkan_queue_submit_barrier_with_action(
        queue, gate, producer_signals, nullptr, action));
    // The unresolved gate ensures the worker owns the final resource reference.
    if (!producer_status.ok()) {
      state.resume.release();
    }
    iree_hal_resource_release(&resource->base);
    IREE_ASSERT_OK(producer_status.release());
    IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
    state.entered.acquire();

    iree_slim_mutex_lock(&queue->submission_mutex);
    const uint64_t consumer_epoch = queue->next_epoch_value;
    iree_slim_mutex_unlock(&queue->submission_mutex);
    const iree_hal_semaphore_list_t consumer_waits = {
        1, producer_signals.semaphores.data(),
        producer_signals.payload_values.data()};
    Status consumer_status(
        iree_hal_queue_upload(transfer_queue_, consumer_waits, consumer_signals,
                              input.data(), target, 0, input.size(), nullptr));
    Status availability_status;
    if (consumer_status.ok()) {
      iree_async_semaphore_t* availability = nullptr;
      availability_status = iree_hal_vulkan_semaphore_submission_timeline(
          consumer_signals.semaphores[0], &availability);
      if (availability_status.ok()) {
        availability_status = iree_async_semaphore_multi_wait(
            IREE_ASYNC_WAIT_MODE_ALL, &availability,
            consumer_signals.payload_values.data(), 1, iree_infinite_timeout(),
            IREE_ASYNC_WAIT_FLAG_NONE, iree_allocator_system());
      }
    }
    iree_slim_mutex_lock(&queue->submission_mutex);
    const bool admitted = queue->next_epoch_value == consumer_epoch + 1;
    iree_slim_mutex_unlock(&queue->submission_mutex);
    EXPECT_TRUE(admitted);
    if (consumer_status.ok() && admitted) {
      const VkSemaphoreWaitInfo wait = {
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
          .semaphoreCount = 1,
          .pSemaphores = &queue->epoch_semaphore,
          .pValues = &consumer_epoch,
      };
      IREE_EXPECT_OK(iree_status_from_vk_result(
          __FILE__, __LINE__,
          iree_vkWaitSemaphores_raw(&queue->syms, queue->logical_device, &wait,
                                    UINT64_MAX),
          "vkWaitSemaphores"));
      // A second completion helper observes the consumer's finished GPU work,
      // but cannot publish success before the producer releases its resources.
      iree_hal_vulkan_queue_drain_completions(queue);
      uint64_t value = 0;
      IREE_EXPECT_OK(
          iree_hal_semaphore_query(consumer_signals.semaphores[0], &value));
      EXPECT_EQ(value, 0u);
    }
    if (dependency_status != IREE_STATUS_OK) {
      iree_hal_semaphore_fail(producer_signals.semaphores[0],
                              iree_status_from_code(dependency_status));
    }
    // Signal-list order retires the consumer's prerequisite before this second
    // signal observer holds the completion worker. The consumer is now on the
    // retirement-ready list, and an independent completion helper must drain
    // it.
    iree_async_semaphore_timepoint_t terminal = {
        .callback =
            [](void* user_data, iree_async_semaphore_timepoint_t*,
               iree_status_t status) {
              auto* state = static_cast<State*>(user_data);
              IREE_EXPECT_STATUS_IS(state->terminal_status, status);
              state->terminal_entered.release();
              state->terminal_resume.acquire();
              state->terminal_finished.release();
            },
        .user_data = &state,
    };
    Status terminal_registration(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(
            producer_signals.semaphores[1]),
        1, &terminal));
    state.resume.release();
    if (terminal_registration.ok()) {
      state.terminal_entered.acquire();
      if (dependency_status == IREE_STATUS_OK) {
        // The consumer already consumed success. A later failure of that public
        // timeline must not rewrite its captured dependency outcome.
        iree_hal_semaphore_fail(producer_signals.semaphores[0],
                                iree_status_from_code(IREE_STATUS_ABORTED));
      }
      iree_hal_vulkan_queue_drain_completions(queue);
      if (consumer_status.ok() && admitted) {
        uint64_t value = 0;
        IREE_EXPECT_STATUS_IS(
            dependency_status,
            iree_hal_semaphore_query(consumer_signals.semaphores[0], &value));
        if (dependency_status == IREE_STATUS_OK) {
          EXPECT_EQ(value, 1u);
        } else {
          EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
        }
      }
      state.terminal_resume.release();
      state.terminal_finished.acquire();
    }
    IREE_EXPECT_OK(terminal_registration.release());
    IREE_EXPECT_OK(availability_status.release());
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_list_wait(producer_signals, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    if (consumer_status.ok()) {
      IREE_EXPECT_STATUS_IS(dependency_status,
                            iree_hal_semaphore_list_wait(
                                consumer_signals, iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    }
    IREE_EXPECT_OK(consumer_status.release());
    EXPECT_EQ(ReadBufferBytes(target, 0, input.size()), input);
  }
}

TEST_P(VulkanQueueTest, RejectedSubmissionDoesNotInvokeCompletion) {
  for (size_t failed_index : {0u, 1u}) {
    SCOPED_TRACE(failed_index);
    SemaphoreList waits(device_, {0, 0}, {1, 1});
    SemaphoreList signals(device_, {0}, {1});
    iree_hal_semaphore_fail(waits.semaphores[failed_index],
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
}

TEST_P(VulkanQueueTest, DeferredWaitPreservesConsumedSuccess) {
  // Either position may already be complete when an independent prerequisite
  // defers admission. Failure after that success cannot rewrite the edge.
  for (size_t completed_index : {0u, 1u}) {
    SCOPED_TRACE(completed_index);
    const size_t gate_index = 1 - completed_index;
    SemaphoreList waits(device_, {0, 0}, {1, 1});
    SemaphoreList signals(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_semaphore_signal(waits.semaphores[completed_index],
                                             1, nullptr));
    IREE_ASSERT_OK(iree_hal_vulkan_queue_submit_barrier(
        reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_), waits,
        signals, nullptr));
    iree_hal_semaphore_fail(waits.semaphores[completed_index],
                            iree_status_from_code(IREE_STATUS_ABORTED));
    IREE_ASSERT_OK(
        iree_hal_semaphore_signal(waits.semaphores[gate_index], 1, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
        signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
}

TEST_P(VulkanQueueTest, DeferredPublicCallbackPreservesConsumedSuccess) {
  for (size_t completed_index : {0u, 1u}) {
    SCOPED_TRACE(completed_index);
    SemaphoreList gate(device_, {0}, {1});
    SemaphoreList already_complete(device_, {1}, {1});
    SemaphoreList producer_signals(device_, {0, 0}, {2, 1});
    SemaphoreList consumer_signals(device_, {0}, {1});
    struct State {
      // Native work has completed; public signals remain unpublished.
      std::binary_semaphore entered{0};
      // Permits public completion after the consumer registers its wait.
      std::binary_semaphore resume{0};
    } state;
    struct Resource {
      // Caller resource retained through native completion.
      iree_hal_resource_t base;
      // Borrowed handshakes outliving resource retirement.
      State* state;
    };
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            [](iree_hal_resource_t* base) {
              auto* resource = reinterpret_cast<Resource*>(base);
              resource->state->entered.release();
              resource->state->resume.acquire();
              delete resource;
            },
    };
    auto* resource = new Resource;
    iree_hal_resource_initialize(&vtable, &resource->base);
    resource->state = &state;
    auto* queue = reinterpret_cast<iree_hal_vulkan_queue_t*>(transfer_queue_);
    const iree_hal_vulkan_queue_completion_action_t action = {
        .resource = &resource->base,
    };
    Status producer_status(iree_hal_vulkan_queue_submit_barrier_with_action(
        queue, gate, producer_signals, nullptr, action));
    if (!producer_status.ok()) {
      state.resume.release();
    }
    iree_hal_resource_release(&resource->base);
    IREE_ASSERT_OK(producer_status.release());
    Status gate_status(iree_hal_semaphore_list_signal(gate, nullptr));
    if (!gate_status.ok()) {
      state.resume.release();
    }
    IREE_ASSERT_OK(gate_status.release());
    state.entered.acquire();

    // The producer publishes 2, but the consumer waits for 1. The exact native
    // producer cache cannot supply that point, so admission awaits its public
    // completion. The other wait exercises callback compaction in both orders.
    iree_hal_semaphore_t* wait_semaphores[2];
    wait_semaphores[completed_index] = already_complete.semaphores[0];
    wait_semaphores[1 - completed_index] = producer_signals.semaphores[0];
    uint64_t wait_values[] = {1, 1};
    const iree_hal_semaphore_list_t consumer_waits = {2, wait_semaphores,
                                                      wait_values};
    Status consumer_status(iree_hal_vulkan_queue_submit_barrier(
        queue, consumer_waits, consumer_signals, nullptr));
    if (consumer_status.ok()) {
      auto* semaphore = reinterpret_cast<iree_async_semaphore_t*>(
          producer_signals.semaphores[0]);
      while (true) {
        iree_slim_mutex_lock(&semaphore->mutex);
        const bool registered = semaphore->timepoints.count != 0;
        iree_slim_mutex_unlock(&semaphore->mutex);
        if (registered) {
          break;
        }
        std::this_thread::yield();
      }
    }

    // Hold the admission retry while the producer publishes both signals.
    // The second signal proves the first signal's public callback has returned.
    iree_slim_mutex_lock(&queue->submission_mutex);
    state.resume.release();
    Status terminal_status(iree_hal_semaphore_wait(
        producer_signals.semaphores[1], 1, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_semaphore_fail(producer_signals.semaphores[0],
                            iree_status_from_code(IREE_STATUS_ABORTED));
    iree_slim_mutex_unlock(&queue->submission_mutex);
    IREE_ASSERT_OK(consumer_status.release());
    IREE_ASSERT_OK(terminal_status.release());
    IREE_EXPECT_OK(iree_hal_semaphore_list_wait(
        consumer_signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
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
