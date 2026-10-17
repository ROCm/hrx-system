// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/host_call.h"

#include <array>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/device.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Native submission is the dependency: the operation capture, callback,
// private timepoints, and public signals all use their production paths.
struct NativeQueue : iree_hal_queue_t {
  struct Boundary {
    // Retained native completion operand, released after signaling.
    iree_hal_semaphore_t* semaphore = nullptr;
    // Copied private completion value.
    uint64_t value = 0;
    // Native boundary's cache operation, or zero for a relaxed call.
    iree_hal_barrier_flags_t effects = 0;
    // Profile identity; only the public entry contributes an event.
    iree_hal_profile_queue_event_type_t type =
        IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
    // Count of public completion points passed to native recording.
    iree_host_size_t public_signal_count = 0;
  };

  // The host-call protocol issues an entry and an optional return boundary.
  std::array<Boundary, 2> boundaries;
  // Number of boundaries presented, including a rejected boundary.
  uint32_t count = 0;
  // One-based boundary to reject before accepting its operands.
  uint32_t reject_boundary = 0;
  // Native execution may finish before its submitting call returns.
  bool immediate = false;
  // Delivers early completion on a distinct completion owner.
  bool concurrent = false;

  void Complete(uint32_t index, iree_status_t status = iree_ok_status()) {
    auto& boundary = boundaries[index];
    auto* semaphore = boundary.semaphore;
    boundary.semaphore = nullptr;
    if (iree_status_is_ok(status)) {
      IREE_CHECK_OK(
          iree_hal_semaphore_signal(semaphore, boundary.value, nullptr));
    } else {
      iree_hal_semaphore_fail(semaphore, status);
    }
    iree_hal_semaphore_release(semaphore);
  }

  static iree_status_t Submit(iree_hal_queue_t* base,
                              iree_hal_semaphore_list_t waits,
                              iree_hal_semaphore_list_t signals,
                              const iree_hal_queue_barriers_t* barriers,
                              iree_hal_profile_queue_event_type_t type,
                              iree_host_size_t operation_count,
                              iree_host_size_t public_signal_count) {
    auto* queue = static_cast<NativeQueue*>(base);
    const uint32_t index = queue->count++;
    EXPECT_LT(index, queue->boundaries.size());
    EXPECT_EQ(signals.count, 1u);
    EXPECT_EQ(operation_count, index == 0 ? 1u : 0u);
    EXPECT_EQ(barriers->after->count, 0u);
    auto& boundary = queue->boundaries[index];
    boundary.type = type;
    boundary.public_signal_count = public_signal_count;
    if (barriers->before->count) {
      EXPECT_EQ(barriers->before->count, 1u);
      const auto& barrier = barriers->before->values[0];
      boundary.effects = barrier.flags;
      EXPECT_EQ(
          index == 0 ? barrier.target_stage_mask : barrier.source_stage_mask,
          IREE_HAL_EXECUTION_STAGE_HOST);
    }
    if (queue->reject_boundary == index + 1) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "boundary rejected");
    }
    boundary.semaphore = signals.semaphores[0];
    boundary.value = signals.payload_values[0];
    iree_hal_semaphore_retain(boundary.semaphore);
    auto* semaphore = boundary.semaphore;
    if (queue->immediate) {
      if (queue->concurrent) {
        std::thread completion([queue, index] { queue->Complete(index); });
        completion.join();
      } else {
        queue->Complete(index);
      }
      // Completion can retire the entire host call, but the producer still
      // owns this borrowed list until it returns. Oversized captures exercise
      // actual storage release as well as the reusable capture-pool path.
      EXPECT_EQ(signals.semaphores[0], semaphore);
      EXPECT_EQ(signals.payload_values[0], boundary.value);
    }
    return iree_ok_status();
  }
};

struct Device {
  // Ordinary HAL lifetime retained by accepted host calls.
  iree_hal_resource_t resource;
  // Real platform semaphore implementation used by this dependency.
  iree_async_proactor_t* proactor = nullptr;
  // Final-release observation, read after joining the completion owner.
  bool destroyed = false;

  explicit Device(iree_async_proactor_t* value) : proactor(value) {
    static const iree_hal_device_vtable_t vtable = {
        .destroy =
            [](iree_hal_device_t* base) {
              reinterpret_cast<Device*>(base)->destroyed = true;
            },
        .create_semaphore =
            [](iree_hal_device_t* base, iree_hal_queue_family_affinity_t,
               uint64_t initial_value, iree_hal_semaphore_flags_t,
               iree_hal_semaphore_t** out_semaphore) {
              return iree_hal_host_semaphore_create(
                  reinterpret_cast<Device*>(base)->proactor, initial_value,
                  iree_allocator_system(), out_semaphore);
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }
  iree_hal_device_t* get() {
    return reinterpret_cast<iree_hal_device_t*>(this);
  }
};

struct Callback {
  // Callback state follows the public optional-resource lifetime contract.
  iree_hal_resource_t resource;
  // Exact queue identity checked by the application callback.
  iree_hal_queue_t* queue = nullptr;
  // Public completion is inspected during the callback.
  iree_hal_semaphore_t* public_completion = nullptr;
  // Deferred completion list copied and retained by the application callback.
  iree_hal_semaphore_t* private_completion = nullptr;
  // Copied value paired with private_completion.
  uint64_t private_value = 0;
  // Number of application invocations.
  uint32_t count = 0;
  // Final callback-state release observation.
  bool destroyed = false;
  // Selects the public DEFERRED completion contract.
  bool deferred = false;
  // Signals a deferred completion before returning from the callback.
  bool early_completion = false;
  // Failure returned synchronously or published through deferred completion.
  iree_status_code_t failure = IREE_STATUS_OK;
  // Actual invocation owner, compared with the submitting thread.
  std::thread::id thread;

  Callback() {
    static const iree_hal_resource_vtable_t vtable = {
        .destroy =
            [](iree_hal_resource_t* base) {
              reinterpret_cast<Callback*>(base)->destroyed = true;
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }

  void Complete() {
    auto* semaphore = private_completion;
    private_completion = nullptr;
    if (failure == IREE_STATUS_OK) {
      IREE_CHECK_OK(
          iree_hal_semaphore_signal(semaphore, private_value, nullptr));
    } else {
      iree_hal_semaphore_fail(semaphore, iree_status_from_code(failure));
    }
    iree_hal_semaphore_release(semaphore);
  }

  static iree_status_t Invoke(void* user_data, const uint64_t arguments[4],
                              iree_hal_host_call_context_t* context) {
    auto* state = static_cast<Callback*>(user_data);
    EXPECT_FALSE(state->destroyed);
    ++state->count;
    state->thread = std::this_thread::get_id();
    EXPECT_EQ(context->queue, state->queue);
    for (uint32_t i = 0; i < 4; ++i) {
      EXPECT_EQ(arguments[i], 17u + i);
    }
    uint64_t value = UINT64_MAX;
    IREE_CHECK_OK(iree_hal_semaphore_query(state->public_completion, &value));
    const bool blocking = context->signal_semaphore_list.count != 0;
    EXPECT_EQ(value, blocking ? 0u : 1u);
    if (!state->deferred) {
      return iree_status_from_code(state->failure);
    }
    if (blocking) {
      const auto& signals = context->signal_semaphore_list;
      EXPECT_EQ(signals.count, 1u);
      state->private_completion = signals.semaphores[0];
      state->private_value = signals.payload_values[0];
      iree_hal_semaphore_retain(state->private_completion);
      if (state->early_completion) {
        state->Complete();
        IREE_CHECK_OK(
            iree_hal_semaphore_query(state->public_completion, &value));
        EXPECT_EQ(value, 0u);
        EXPECT_FALSE(state->destroyed);
      }
    }
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
};

class HostCallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_host_semaphore_create(
        proactor_, 0, iree_allocator_system(), &completion_));
    iree_hal_amd_gpu_capture_pool_initialize(4096, iree_allocator_system(),
                                             &captures_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_hal_semaphore_release(completion_);
    iree_async_proactor_release(proactor_);
  }

  struct Options {
    // Public cache and ordering hints.
    iree_hal_host_call_flags_t flags = IREE_HAL_HOST_CALL_FLAG_NONE;
    // Complete native work before its submitter returns.
    bool immediate = false;
    // Deliver early native completion on a distinct thread.
    bool concurrent = false;
    // Callback transfers terminal completion to its retained private point.
    bool deferred = false;
    // Deferred completion occurs before the callback returns.
    bool early = false;
    // Application callback's terminal result.
    iree_status_code_t failure = IREE_STATUS_OK;
    // One-based native boundary rejected before acceptance, or zero.
    uint32_t rejected_boundary = 0;
    // Distinct public signals, including oversized capture cases.
    iree_host_size_t signal_count = 1;
    // One-based accepted native boundary to fail during completion, or zero.
    uint32_t failed_boundary = 0;
  };

  void Run(Options options) {
    Device device(proactor_);
    iree_hal_queue_family_t family = {.device = device.get()};
    NativeQueue queue;
    queue.queue_family = &family;
    queue.immediate = options.immediate;
    queue.concurrent = options.concurrent;
    queue.reject_boundary = options.rejected_boundary;
    uint32_t failures = 0;
    iree_hal_amd_gpu_context_t context = {
        .event_sink =
            {
                .fn =
                    [](void* user_data, const iree_hal_device_event_t* event) {
                      EXPECT_EQ(event->type,
                                IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
                      ++*static_cast<uint32_t*>(user_data);
                    },
                .user_data = &failures,
            },
    };
    const iree_hal_amd_gpu_host_call_executor_t executor = {
        .queue = &queue,
        .context = &context,
        .captures = &captures_,
        .barrier = NativeQueue::Submit,
    };
    Callback callback;
    callback.queue = &queue;
    callback.public_completion = completion_;
    callback.deferred = options.deferred;
    callback.early_completion = options.early;
    callback.failure = options.failure;
    std::vector<iree_hal_semaphore_t*> semaphores(options.signal_count,
                                                  completion_);
    for (iree_host_size_t i = 1; i < options.signal_count; ++i) {
      IREE_ASSERT_OK(iree_hal_host_semaphore_create(
          proactor_, 0, iree_allocator_system(), &semaphores[i]));
    }
    std::vector<uint64_t> values(options.signal_count, 1);
    const uint64_t arguments[] = {17, 18, 19, 20};
    iree_status_t status = iree_hal_amd_gpu_host_call_submit(
        &executor, iree_hal_semaphore_list_empty(),
        {options.signal_count, semaphores.data(), values.data()},
        iree_hal_make_host_call_with_resource(Callback::Invoke, &callback,
                                              &callback.resource),
        arguments, options.flags);
    if (options.rejected_boundary == 1) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
    } else {
      IREE_ASSERT_OK(status);
    }
    iree_hal_resource_release(&callback.resource);
    iree_hal_device_release(device.get());
    if (options.rejected_boundary != 1 && !options.immediate) {
      EXPECT_FALSE(device.destroyed);
      EXPECT_FALSE(callback.destroyed);
      EXPECT_EQ(callback.count, 0u);
      queue.Complete(0, options.failed_boundary == 1
                            ? iree_status_from_code(IREE_STATUS_ABORTED)
                            : iree_ok_status());
    }
    EXPECT_TRUE(callback.destroyed);
    if (callback.private_completion) {
      EXPECT_FALSE(device.destroyed);
      uint64_t value = UINT64_MAX;
      IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &value));
      EXPECT_EQ(value, 0u);
      callback.Complete();
    }
    if (!options.immediate && queue.boundaries[1].semaphore) {
      EXPECT_FALSE(device.destroyed);
      uint64_t value = UINT64_MAX;
      IREE_ASSERT_OK(iree_hal_semaphore_query(completion_, &value));
      EXPECT_EQ(value, 0u);
      queue.Complete(1, options.failed_boundary == 2
                            ? iree_status_from_code(IREE_STATUS_ABORTED)
                            : iree_ok_status());
    }
    EXPECT_TRUE(device.destroyed);
    EXPECT_EQ(callback.count,
              options.rejected_boundary == 1 || options.failed_boundary == 1
                  ? 0u
                  : 1u);
    if (options.concurrent && callback.count) {
      EXPECT_NE(callback.thread, std::this_thread::get_id());
    }
    const bool failed = options.failure != IREE_STATUS_OK ||
                        options.rejected_boundary == 2 ||
                        options.failed_boundary != 0;
    EXPECT_EQ(failures, failed ? 1u : 0u);
    const bool has_return =
        !iree_any_bit_set(options.flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
                                             IREE_HAL_HOST_CALL_FLAG_RELAXED) &&
        options.failure == IREE_STATUS_OK && options.rejected_boundary != 1 &&
        options.failed_boundary != 1;
    EXPECT_EQ(queue.count, has_return ? 2u : 1u);
    EXPECT_EQ(queue.boundaries[0].type,
              IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL);
    EXPECT_EQ(queue.boundaries[0].public_signal_count, options.signal_count);
    EXPECT_EQ(queue.boundaries[0].effects,
              iree_any_bit_set(options.flags, IREE_HAL_HOST_CALL_FLAG_RELAXED)
                  ? 0u
                  : IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
    if (has_return) {
      EXPECT_EQ(queue.boundaries[1].type,
                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE);
      EXPECT_EQ(queue.boundaries[1].effects,
                IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE);
    }
    for (iree_host_size_t i = 1; i < options.signal_count; ++i) {
      uint64_t completed = 0;
      IREE_EXPECT_OK(iree_hal_semaphore_query(semaphores[i], &completed));
      EXPECT_EQ(completed, 1u);
      iree_hal_semaphore_release(semaphores[i]);
    }
    uint64_t value = UINT64_MAX;
    status = iree_hal_semaphore_query(completion_, &value);
    if (failed) {
      IREE_EXPECT_STATUS_IS(options.rejected_boundary == 2
                                ? IREE_STATUS_UNAVAILABLE
                                : IREE_STATUS_ABORTED,
                            status);
    } else {
      IREE_ASSERT_OK(status);
      EXPECT_EQ(value, options.rejected_boundary == 1 ? 0u : 1u);
    }
  }

  // Real asynchronous semaphore backend used throughout the protocol.
  iree_async_proactor_t* proactor_ = nullptr;
  // Public host-call completion observed by the application.
  iree_hal_semaphore_t* completion_ = nullptr;
  // Production queue capture pool, including oversized capture handling.
  iree_hal_amd_gpu_capture_pool_t captures_ = {};
};

TEST_F(HostCallTest, NativeBoundariesReleaseCallbackBeforePublicCompletion) {
  Run({});
}
TEST_F(HostCallTest, ImmediateNativeCompletionRetainsProducerOperands) {
  Run({.immediate = true});
}
TEST_F(HostCallTest, ConcurrentEarlyCompletionKeepsItsInvocationOwner) {
  Run({.immediate = true, .concurrent = true});
}
TEST_F(HostCallTest, OversizedCaptureSurvivesConcurrentEarlyCompletion) {
  Run({.immediate = true, .concurrent = true, .signal_count = 1660});
}
TEST_F(HostCallTest, DeferredCompletionRetainsDeviceAfterCallbackRelease) {
  Run({.deferred = true});
}
TEST_F(HostCallTest, DeferredCompletionBeforeCallbackReturns) {
  Run({.immediate = true, .concurrent = true, .deferred = true, .early = true});
}
TEST_F(HostCallTest, RelaxedCallbackNeedsOnlyEntryBoundary) {
  Run({.flags = IREE_HAL_HOST_CALL_FLAG_RELAXED});
}
TEST_F(HostCallTest, RelaxedDeferredCompletionRemainsOrdered) {
  Run({.flags = IREE_HAL_HOST_CALL_FLAG_RELAXED, .deferred = true});
}
TEST_F(HostCallTest, NonBlockingPublishesAtInvocation) {
  Run({.flags = IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING});
}
TEST_F(HostCallTest, NonBlockingDeferredCallbackHasNoCompletionObligation) {
  Run({
      .flags = IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING,
      .immediate = true,
      .concurrent = true,
      .deferred = true,
  });
}
TEST_F(HostCallTest, EntryRejectionRetainsNothingAndDoesNotInvoke) {
  Run({.rejected_boundary = 1});
}
TEST_F(HostCallTest, ReturnRejectionFailsPublicCompletion) {
  Run({.immediate = true, .concurrent = true, .rejected_boundary = 2});
}
TEST_F(HostCallTest, NativeEntryFailureDoesNotInvokeCallback) {
  Run({.failed_boundary = 1});
}
TEST_F(HostCallTest, NativeReturnFailureFailsPublicCompletion) {
  Run({.failed_boundary = 2});
}

TEST_F(HostCallTest, CallbackFailureEndsWithoutReturnBoundary) {
  Run({.failure = IREE_STATUS_ABORTED});
}
TEST_F(HostCallTest, DeferredCallbackFailureEndsWithoutReturnBoundary) {
  Run({.deferred = true, .failure = IREE_STATUS_ABORTED});
}
TEST_F(HostCallTest, EarlyDeferredFailureWaitsForCallbackReturn) {
  Run({
      .immediate = true,
      .concurrent = true,
      .deferred = true,
      .early = true,
      .failure = IREE_STATUS_ABORTED,
  });
}

}  // namespace
