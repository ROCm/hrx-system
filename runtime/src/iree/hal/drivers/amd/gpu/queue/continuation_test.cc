// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/continuation.h"

#include <array>
#include <atomic>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/hal/memory/maintenance_thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Continuation = iree_hal_amd_gpu_continuation_t;
using Entry = iree_hal_memory_maintenance_entry_t;

// Models a slow cold call whose release is controlled by an external owner.
struct Gate : Entry {
  // Publishes that the service has entered the cold call.
  std::binary_semaphore entered{0};
  // Controller-owned permission to return from the cold call.
  std::binary_semaphore resume{0};

  Gate() {
    fn = [](Entry* entry) {
      auto* self = static_cast<Gate*>(entry);
      self->entered.release();
      self->resume.acquire();
    };
  }
};

class ContinuationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &maintenance_));
  }
  void TearDown() override {
    iree_hal_memory_maintenance_release(maintenance_);
  }
  // Actual shared service worker used by both continuation and cold entries.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
};

TEST_F(ContinuationTest, IdleDeinitializationRunsNoCallback) {
  uint32_t count = 0;
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        ++*static_cast<uint32_t*>(user_data);
        return false;
      },
      &count, &continuation);
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(count, 0u);
}

TEST_F(ContinuationTest, NotificationsCoalesceWhileQueuedBehindColdWork) {
  Gate gate;
  iree_hal_memory_maintenance_enqueue(maintenance_, &gate);
  gate.entered.acquire();
  uint32_t count = 0;
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        ++*static_cast<uint32_t*>(user_data);
        return false;
      },
      &count, &continuation);
  for (uint32_t i = 0; i < 1000; ++i) {
    iree_hal_amd_gpu_continuation_notify(&continuation);
  }
  gate.resume.release();
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(count, 1u);
}

TEST_F(ContinuationTest, WakeDuringFinalPredicateCheckSchedulesAnotherTurn) {
  struct State {
    // Controls the first turn after its final readiness observation.
    Gate gate;
    // Number of completed service turns.
    uint32_t count = 0;
  } state;
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        auto* state = static_cast<State*>(user_data);
        if (++state->count == 1) {
          state->gate.entered.release();
          state->gate.resume.acquire();
        }
        return false;
      },
      &state, &continuation);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  state.gate.entered.acquire();
  for (uint32_t i = 0; i < 1000; ++i) {
    iree_hal_amd_gpu_continuation_notify(&continuation);
  }
  state.gate.resume.release();
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(state.count, 2u);
}

TEST_F(ContinuationTest, NewNotificationAfterIdleStartsAnotherTurn) {
  uint32_t count = 0;
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        ++*static_cast<uint32_t*>(user_data);
        return false;
      },
      &count, &continuation);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  // The FIFO call joins the first turn without inspecting private state.
  iree_hal_memory_maintenance_call(maintenance_, [](void*) {}, nullptr);
  EXPECT_EQ(count, 1u);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(count, 2u);
}

TEST_F(ContinuationTest, BoundedTurnsYieldToColdWorkOnTheSameOwner) {
  struct State : Entry {
    // Shared executor used to interleave the cold entry.
    iree_hal_memory_maintenance_t* maintenance = nullptr;
    // Serial callback order, read only after the continuation join.
    std::vector<int> order;
    // Number of ready turns already executed.
    uint32_t turns = 0;
  } state;
  state.maintenance = maintenance_;
  state.fn = [](Entry* entry) {
    static_cast<State*>(entry)->order.push_back(2);
  };
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        auto* state = static_cast<State*>(user_data);
        if (++state->turns == 1) {
          state->order.push_back(1);
          iree_hal_memory_maintenance_enqueue(state->maintenance, state);
          return true;
        }
        state->order.push_back(3);
        return false;
      },
      &state, &continuation);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(state.order, (std::vector<int>{1, 2, 3}));
}

TEST_F(ContinuationTest, ReentrantPreparationUsesTheCurrentServiceOwner) {
  struct State {
    // Borrowed shared executor, also used by synchronous preparation.
    iree_hal_memory_maintenance_t* maintenance = nullptr;
    // Result written by the nested preparation callback.
    uint32_t value = 0;
  } state{maintenance_};
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        auto* state = static_cast<State*>(user_data);
        iree_hal_memory_maintenance_call(
            state->maintenance,
            [](void* nested_data) {
              static_cast<State*>(nested_data)->value = 123;
            },
            state);
        return false;
      },
      &state, &continuation);
  iree_hal_amd_gpu_continuation_notify(&continuation);
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(state.value, 123u);
}

TEST_F(ContinuationTest, ConcurrentNotificationsPreserveAuthoritativeWork) {
  struct State {
    // Work published before its corresponding notification.
    std::atomic<uint32_t> pending = 0;
    // Completed work owned by the sole service callback.
    uint32_t completed = 0;
  } state;
  Continuation continuation;
  iree_hal_amd_gpu_continuation_initialize(
      maintenance_,
      [](void* user_data) {
        auto* state = static_cast<State*>(user_data);
        state->completed +=
            state->pending.exchange(0, std::memory_order_acq_rel);
        return false;
      },
      &state, &continuation);
  std::array<std::thread, 4> producers;
  for (auto& producer : producers) {
    producer = std::thread([&] {
      for (uint32_t i = 0; i < 2048; ++i) {
        state.pending.fetch_add(1, std::memory_order_release);
        iree_hal_amd_gpu_continuation_notify(&continuation);
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  iree_hal_amd_gpu_continuation_deinitialize(&continuation);
  EXPECT_EQ(state.completed, 8192u);
  EXPECT_EQ(state.pending.load(), 0u);
}

}  // namespace
