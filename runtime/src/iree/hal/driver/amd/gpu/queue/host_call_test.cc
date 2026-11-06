// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/host_call.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/device.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only native submission is controlled. Captures, callbacks, and public
// semaphore signaling use the production implementation.
struct NativeQueue : iree_hal_queue_t {
  // Accepted invocation owners, independent of the caller's asynchronous work.
  std::atomic<int> host_owners{0};
  static void Begin(iree_hal_queue_t* queue) {
    ++static_cast<NativeQueue*>(queue)->host_owners;
  }
  static void End(iree_hal_queue_t* queue) {
    --static_cast<NativeQueue*>(queue)->host_owners;
  }

  struct Entry {
    // Accepted continuation, invoked once after native retirement.
    iree_hal_amd_gpu_retirement_callback_t callback = {};
    // Native entry's cache operation, or zero for a relaxed call.
    iree_hal_barrier_flags_t effects = 0;
    // Count of public output points used by native profile recording.
    iree_host_size_t public_signal_count = 0;
  };

  // Entries from independent calls; one call must submit exactly one entry.
  std::vector<Entry> entries;
  // Reject entry before accepting callback ownership.
  bool reject = false;
  // Native execution may finish before its submitting call returns.
  bool immediate = false;
  // Deliver immediate retirement on a distinct completion owner.
  bool concurrent = false;

  void Complete(uint32_t index, iree_status_t status = iree_ok_status()) {
    auto callback = entries[index].callback;
    entries[index].callback = {};
    callback.fn(callback.user_data, status, nullptr, true);
  }

  static iree_status_t Submit(iree_hal_queue_t* base,
                              iree_hal_semaphore_list_t waits,
                              iree_hal_amd_gpu_retirement_callback_t callback,
                              const iree_hal_queue_barriers_t* barriers,
                              iree_host_size_t public_signal_count) {
    auto* queue = static_cast<NativeQueue*>(base);
    const uint32_t index = queue->entries.size();
    EXPECT_EQ(barriers->after->count, 0u);
    Entry entry = {.public_signal_count = public_signal_count};
    if (barriers->before->count) {
      EXPECT_EQ(barriers->before->count, 1u);
      const auto& barrier = barriers->before->values[0];
      entry.effects = barrier.flags;
      EXPECT_EQ(barrier.target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST);
    }
    queue->entries.push_back(entry);
    if (queue->reject) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "entry rejected");
    }
    queue->entries[index].callback = callback;
    if (queue->immediate) {
      if (queue->concurrent) {
        std::thread completion([queue, index] { queue->Complete(index); });
        completion.join();
      } else {
        queue->Complete(index);
      }
      // Invocation completion cannot return the capture to its pool while
      // this submitting call still owns it, including oversized captures.
      EXPECT_EQ(queue->host_owners.load(), 1);
    }
    return iree_ok_status();
  }
};

struct Device {
  // Device identity borrowed by accepted host calls for diagnostic reporting.
  iree_hal_resource_t resource;
  // Real platform semaphore implementation used by this dependency.
  iree_async_proactor_t* proactor = nullptr;
  // Final-release observation, read after joining the completion owner.
  bool destroyed = false;

  // Submission must consume caller-owned semaphore identities, never create
  // one.
  uint32_t semaphore_creations = 0;

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
              ++reinterpret_cast<Device*>(base)->semaphore_creations;
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

// Completion storage belongs to the asynchronous work, not callback state.
struct AsyncWork {
  // Cloned public output points, owned until completion.
  iree_hal_semaphore_list_t signals = {};
  // Result produced by the asynchronous work.
  iree_status_code_t failure = IREE_STATUS_OK;

  void Complete() {
    if (failure == IREE_STATUS_OK) {
      IREE_CHECK_OK(iree_hal_semaphore_list_signal(signals, nullptr));
    } else {
      iree_hal_semaphore_list_fail(signals, iree_status_from_code(failure));
    }
    iree_hal_semaphore_list_free(signals, iree_allocator_system());
    signals = {};
  }
};

struct Callback {
  // Callback state follows the public optional-resource lifetime contract.
  iree_hal_resource_t resource;
  // Exact queue identity checked by the application callback.
  iree_hal_queue_t* queue = nullptr;
  // Original outputs, borrowed from the test's submitting owner.
  iree_hal_semaphore_list_t outputs = {};
  // Separately owned asynchronous operation, or NULL for a synchronous call.
  AsyncWork* work = nullptr;
  // Number of application invocations.
  uint32_t count = 0;
  // Final callback-state release observation.
  bool destroyed = false;
  // Whether the public outputs are published before invocation.
  bool non_blocking = false;
  // Signals a deferred completion before returning from the callback.
  bool early_completion = false;
  // Failure returned synchronously.
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
    const auto& signals = context->signal_semaphore_list;
    EXPECT_EQ(signals.count, state->non_blocking ? 0u : state->outputs.count);
    for (iree_host_size_t i = 0; i < state->outputs.count; ++i) {
      uint64_t value = UINT64_MAX;
      IREE_CHECK_OK(
          iree_hal_semaphore_query(state->outputs.semaphores[i], &value));
      EXPECT_EQ(value,
                state->non_blocking ? state->outputs.payload_values[i] : 0u);
      if (!state->non_blocking) {
        EXPECT_EQ(signals.semaphores[i], state->outputs.semaphores[i]);
        EXPECT_EQ(signals.payload_values[i], state->outputs.payload_values[i]);
      }
    }
    if (!state->work) {
      return iree_status_from_code(state->failure);
    }
    IREE_CHECK_OK(iree_hal_semaphore_list_clone(
        &signals, iree_allocator_system(), &state->work->signals));
    if (state->early_completion) {
      state->work->Complete();
      // Completing the original public outputs does not end this invocation's
      // ownership of callback state, arguments, or the borrowed context list.
      EXPECT_FALSE(state->destroyed);
      for (iree_host_size_t i = 0; i < signals.count; ++i) {
        EXPECT_EQ(signals.semaphores[i], state->outputs.semaphores[i]);
        uint64_t value = 0;
        iree_status_t status =
            iree_hal_semaphore_query(signals.semaphores[i], &value);
        if (state->work->failure == IREE_STATUS_OK) {
          IREE_EXPECT_OK(status);
          EXPECT_EQ(value, signals.payload_values[i]);
        } else {
          IREE_EXPECT_STATUS_IS(state->work->failure, status);
        }
      }
      EXPECT_EQ(arguments[3], 20u);
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
    iree_hal_amd_gpu_capture_pool_initialize(4096, iree_allocator_system(),
                                             &captures_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_async_proactor_release(proactor_);
  }

  struct Options {
    // Public cache and ordering hints.
    iree_hal_host_call_flags_t flags = IREE_HAL_HOST_CALL_FLAG_NONE;
    // Complete native work before its submitter returns.
    bool immediate = false;
    // Deliver early native completion on a distinct thread.
    bool concurrent = false;
    // Callback transfers terminal completion to asynchronous work.
    bool deferred = false;
    // Deferred completion occurs before the callback returns.
    bool early = false;
    // Application callback's terminal result.
    iree_status_code_t failure = IREE_STATUS_OK;
    // Reject the native entry before acceptance.
    bool reject = false;
    // Distinct public signals, including oversized capture cases.
    iree_host_size_t signal_count = 1;
    // Fail accepted entry, notifying before actual retirement.
    bool fail_entry = false;
  };

  void Run(Options options) {
    Device device(proactor_);
    iree_hal_queue_family_t family = {.device = device.get()};
    NativeQueue queue;
    queue.queue_family = &family;
    queue.immediate = options.immediate;
    queue.concurrent = options.concurrent;
    queue.reject = options.reject;
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
        .begin = NativeQueue::Begin,
        .end = NativeQueue::End,
        .enter = NativeQueue::Submit,
    };
    std::vector<iree_hal_semaphore_t*> semaphores(options.signal_count);
    std::vector<uint64_t> values(options.signal_count);
    for (iree_host_size_t i = 0; i < options.signal_count; ++i) {
      IREE_ASSERT_OK(iree_hal_host_semaphore_create(
          proactor_, 0, iree_allocator_system(), &semaphores[i]));
      values[i] = i + 1;
    }
    AsyncWork work;
    work.failure = options.failure;
    Callback callback;
    callback.queue = &queue;
    callback.outputs = {options.signal_count, semaphores.data(), values.data()};
    callback.work = options.deferred ? &work : nullptr;
    callback.early_completion = options.early;
    callback.non_blocking =
        options.flags & IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING;
    callback.failure = options.failure;
    const uint64_t arguments[] = {17, 18, 19, 20};
    iree_status_t status = iree_hal_amd_gpu_host_call_submit(
        &executor, {}, callback.outputs,
        iree_hal_make_host_call_with_resource(Callback::Invoke, &callback,
                                              &callback.resource),
        arguments, options.flags);
    if (options.reject) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
    } else {
      IREE_ASSERT_OK(status);
    }
    iree_hal_resource_release(&callback.resource);
    if (!options.reject && !options.immediate) {
      EXPECT_FALSE(callback.destroyed);
      EXPECT_EQ(callback.count, 0u);
      if (options.fail_entry) {
        auto entry = queue.entries[0].callback;
        entry.fn(entry.user_data, iree_status_from_code(IREE_STATUS_ABORTED),
                 nullptr, false);
        EXPECT_FALSE(callback.destroyed);
        EXPECT_EQ(callback.count, 0u);
        EXPECT_EQ(queue.host_owners.load(), 1);
        for (auto* semaphore : semaphores) {
          uint64_t value = UINT64_MAX;
          IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
          EXPECT_EQ(value, 0u);
        }
      }
      queue.Complete(0, options.fail_entry
                            ? iree_status_from_code(IREE_STATUS_ABORTED)
                            : iree_ok_status());
    }
    EXPECT_TRUE(callback.destroyed);
    // The host-call capture ends at invocation return. Asynchronous work owns
    // its output list independently, without keeping a backend continuation.
    EXPECT_EQ(queue.host_owners.load(), 0);
    const bool pending_work = work.signals.count != 0;
    if (work.signals.count) {
      // An independent call on the same queue must complete without waiting
      // for this callback's deferred result, including a result that will fail.
      iree_hal_semaphore_t* independent = nullptr;
      IREE_ASSERT_OK(iree_hal_host_semaphore_create(
          proactor_, 0, iree_allocator_system(), &independent));
      uint64_t one = 1;
      Callback other;
      other.queue = &queue;
      other.outputs = {1, &independent, &one};
      IREE_ASSERT_OK(iree_hal_amd_gpu_host_call_submit(
          &executor, {}, other.outputs,
          iree_hal_make_host_call_with_resource(Callback::Invoke, &other,
                                                &other.resource),
          arguments, 0));
      iree_hal_resource_release(&other.resource);
      if (!options.immediate) {
        queue.Complete(1);
      }
      uint64_t observed = 0;
      IREE_ASSERT_OK(iree_hal_semaphore_query(independent, &observed));
      EXPECT_EQ(observed, 1u);
      EXPECT_TRUE(other.destroyed);
      EXPECT_EQ(queue.host_owners.load(), 0);
      iree_hal_semaphore_release(independent);
      for (auto* semaphore : semaphores) {
        uint64_t value = UINT64_MAX;
        IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
        EXPECT_EQ(value, 0u);
      }
      work.Complete();
    }
    EXPECT_EQ(device.semaphore_creations, 0u);
    iree_hal_device_release(device.get());
    EXPECT_TRUE(device.destroyed);
    EXPECT_EQ(callback.count, options.reject || options.fail_entry ? 0u : 1u);
    if (options.concurrent && callback.count) {
      EXPECT_NE(callback.thread, std::this_thread::get_id());
    }
    const bool failed = !options.reject && (options.fail_entry ||
                                            options.failure != IREE_STATUS_OK);
    EXPECT_EQ(failures,
              options.fail_entry || (failed && !options.deferred) ? 1u : 0u);
    ASSERT_EQ(queue.entries.size(), pending_work ? 2u : 1u);
    EXPECT_EQ(queue.entries[0].public_signal_count, options.signal_count);
    EXPECT_EQ(queue.entries[0].effects,
              options.flags & IREE_HAL_HOST_CALL_FLAG_RELAXED
                  ? 0u
                  : IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
    for (iree_host_size_t i = 0; i < semaphores.size(); ++i) {
      uint64_t value = UINT64_MAX;
      status = iree_hal_semaphore_query(semaphores[i], &value);
      if (failed && (!callback.non_blocking || options.fail_entry)) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
      } else {
        IREE_EXPECT_OK(status);
        EXPECT_EQ(value, options.reject ? 0u : values[i]);
      }
      iree_hal_semaphore_release(semaphores[i]);
    }
  }

  // Real asynchronous semaphore backend used throughout the protocol.
  iree_async_proactor_t* proactor_ = nullptr;
  // Production queue capture pool, including oversized capture handling.
  iree_hal_amd_gpu_capture_pool_t captures_ = {};
};

TEST_F(HostCallTest, SynchronousCallUsesOriginalOutputs) { Run({}); }
TEST_F(HostCallTest, ImmediateNativeCompletionRetainsProducerOperands) {
  Run({.immediate = true});
}
TEST_F(HostCallTest, ConcurrentEarlyCompletionKeepsItsInvocationOwner) {
  Run({.immediate = true, .concurrent = true});
}
TEST_F(HostCallTest, OversizedCaptureSurvivesConcurrentEarlyCompletion) {
  Run({.immediate = true, .concurrent = true, .signal_count = 1660});
}
TEST_F(HostCallTest, DeferredWorkOwnsOriginalOutputsAfterInvocation) {
  Run({.deferred = true});
}
TEST_F(HostCallTest, DeferredWorkPreservesMultipleOutputValues) {
  Run({.deferred = true, .signal_count = 3});
}
TEST_F(HostCallTest, EmptyOutputsNeedNoIntermediateSemaphore) {
  Run({.signal_count = 0});
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
  Run({.reject = true});
}
TEST_F(HostCallTest, NativeEntryFailureDoesNotInvokeCallback) {
  Run({.fail_entry = true});
}

TEST_F(HostCallTest, CallbackFailureEndsWithoutReturnBoundary) {
  Run({.failure = IREE_STATUS_ABORTED});
}
TEST_F(HostCallTest, DeferredCallbackFailureEndsWithoutReturnBoundary) {
  Run({.deferred = true, .failure = IREE_STATUS_ABORTED});
}
TEST_F(HostCallTest, EarlyDeferredFailurePreservesInvocationOwnership) {
  Run({
      .immediate = true,
      .concurrent = true,
      .deferred = true,
      .early = true,
      .failure = IREE_STATUS_ABORTED,
  });
}

}  // namespace
