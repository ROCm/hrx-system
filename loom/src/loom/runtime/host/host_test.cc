// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/host.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include "iree/base/internal/arena.h"
#include "iree/base/threading/numa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree/task_pool.h"

namespace {

TEST(HostTest, NullLifetimeOperations) {
  loom_runtime_host_retain(nullptr);
  loom_runtime_host_release(nullptr);
}

TEST(HostTest, CompilerAdapterUsesApplicationExecutor) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host));
  EXPECT_EQ(loom_runtime_host_executor(host),
            iree_task_executor_from_loomc_task_pool(
                loom_runtime_host_task_pool(host)));
  EXPECT_EQ(iree_task_executor_worker_count(loom_runtime_host_executor(host)),
            1u);
  EXPECT_NE(loom_runtime_host_task_driver(host), nullptr);
  EXPECT_EQ(loom_runtime_host_target_environment(host), nullptr);
  iree_host_size_t target_provider_count = 1;
  EXPECT_EQ(loom_runtime_host_target_providers(host, &target_provider_count),
            nullptr);
  EXPECT_EQ(target_provider_count, 0u);
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler(host));
  loom_runtime_host_retain(host);
  loom_runtime_host_release(host);
  EXPECT_NE(loom_runtime_host_proactor_pool(host), nullptr);
  loom_runtime_host_release(host);
}

TEST(HostTest, CompilerTrimUsesConfiguredAllocatorOnce) {
  struct TrimAllocator {
    // Actual storage provider for the complete host lifetime.
    iree_allocator_t backing = iree_allocator_system();
    // Number of explicit reclamation requests from the host owner.
    iree_host_size_t calls = 0;
    // Injected external control failure, leaving allocation/free unaffected.
    iree_status_code_t status_code = IREE_STATUS_OK;
  } state;
  const iree_allocator_t allocator = {
      &state,
      [](void* self, iree_allocator_command_t command, const void* parameters,
         void** pointer) -> iree_status_t {
        auto* state = static_cast<TrimAllocator*>(self);
        if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
          ++state->calls;
          if (state->status_code != IREE_STATUS_OK) {
            return iree_status_from_code(state->status_code);
          }
        }
        return state->backing.ctl(state->backing.self, command, parameters,
                                  pointer);
      },
  };
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_runtime_host_create(2, nullptr, allocator, &host));
  EXPECT_EQ(state.calls, 0u);
  IREE_EXPECT_OK(loom_runtime_host_trim_compiler(host));
  EXPECT_EQ(state.calls, 1u);
  state.status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_runtime_host_trim_compiler(host));
  EXPECT_EQ(state.calls, 2u);
  EXPECT_EQ(iree_task_executor_worker_count(loom_runtime_host_executor(host)),
            2u);
  loom_runtime_host_release(host);
}

TEST(HostTest, InvalidCapacityAndAllocationFailureReturnNoHost) {
  loom_runtime_host_t* host = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_host_create(0, nullptr, iree_allocator_system(), &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_host_create(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT + 1, nullptr,
                               iree_allocator_system(), &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_host_create(1, nullptr, iree_allocator_null(), &host));
  EXPECT_EQ(host, nullptr);
}

TEST(HostTest, DeviceIoPlacementResolvesEveryOnlineNode) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host));
  auto* pool = loom_runtime_host_proactor_pool(host);
  EXPECT_EQ(iree_async_proactor_pool_count(pool), iree_numa_node_count());
  iree_bitmap_t online_nodes = iree_numa_online_nodes();
  for (uint32_t node_id = 0; node_id < online_nodes.bit_count; ++node_id) {
    if (!iree_bitmap_test(online_nodes, node_id)) {
      continue;
    }
    iree_async_proactor_pool_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        iree_async_proactor_pool_acquire_for_node(pool, node_id, &entry));
    EXPECT_EQ(iree_async_proactor_pool_entry_node_id(entry), node_id);
    EXPECT_NE(iree_async_proactor_pool_entry_proactor(entry), nullptr);
    iree_async_proactor_pool_entry_release(entry);
  }
  loom_runtime_host_release(host);
}

TEST(HostTest, BatchArenasShareAndReturnScratchBlocks) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host));
  auto* pool = loom_runtime_host_block_pool(host);
  iree_arena_allocator_t first, second;
  iree_arena_initialize(pool, &first);
  iree_arena_initialize(pool, &second);
  void* first_bytes = nullptr;
  void* second_bytes = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&first, 256, &first_bytes));
  IREE_ASSERT_OK(iree_arena_allocate(&second, 256, &second_bytes));
  EXPECT_NE(first_bytes, second_bytes);
  iree_arena_deinitialize(&first);
  iree_arena_initialize(pool, &first);
  void* reused_bytes = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&first, 256, &reused_bytes));
  EXPECT_EQ(first_bytes, reused_bytes);
  iree_arena_deinitialize(&first);
  // Trimming free blocks leaves an independently live batch untouched.
  iree_arena_block_pool_trim(pool);
  std::memset(second_bytes, 0x5a, 256);
  iree_arena_deinitialize(&second);
  loom_runtime_host_release(host);
}

class HostReclamationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_allocator_t allocator = {this, Allocate};
    IREE_ASSERT_OK(loom_runtime_host_create(2, nullptr, allocator, &host_));
  }
  void TearDown() override { loom_runtime_host_release(host_); }
  static iree_status_t Allocate(void* payload, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* self = static_cast<HostReclamationTest*>(payload);
    if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
      ++self->trims_;
      std::unique_lock<std::mutex> lock(self->mutex_);
      if (self->blocked_) {
        self->entered_ = true;
        self->condition_.notify_one();
        self->condition_.wait(lock, [&] { return !self->blocked_; });
      }
      if (self->failure_ != IREE_STATUS_OK) {
        return iree_status_from_code(self->failure_);
      }
    }
    const auto backing = iree_allocator_system();
    return backing.ctl(backing.self, command, parameters, pointer);
  }
  // Host shared by all independently held preparation scopes.
  loom_runtime_host_t* host_ = nullptr;
  // Actual configured allocator controls, counted independently of policy.
  std::atomic<unsigned> trims_{0};
  // Injected allocator-control failure; normal allocation remains usable.
  iree_status_code_t failure_ = IREE_STATUS_OK;
  // Joins the allocator dependency gate, not the code under test.
  std::mutex mutex_;
  // Explicit dependency arrival and release.
  std::condition_variable condition_;
  // The allocator has entered its trim operation.
  bool entered_ = false;
  // The test dependency is intentionally holding that operation.
  bool blocked_ = false;
};

TEST_F(HostReclamationTest, NestedIntentNotifiesAndCoalescesAcrossOwners) {
  unsigned first_wakes = 0, second_wakes = 0;
  const auto wake = [](void* payload) { ++*static_cast<unsigned*>(payload); };
  loom_runtime_host_preparation_observer_t first{{wake, &first_wakes}};
  loom_runtime_host_preparation_observer_t second{{wake, &second_wakes}};
  loom_runtime_host_observe_preparation(host_, &first);
  loom_runtime_host_observe_preparation(host_, &second);
  EXPECT_EQ(loom_runtime_host_compiler_idle_deadline(host_, 0),
            IREE_TIME_INFINITE_FUTURE);
  loom_runtime_host_preparation_begin(host_);
  for (int i = 0; i < 4; ++i) {
    loom_runtime_host_preparation_begin(host_);
    loom_runtime_host_preparation_end(host_);
    IREE_ASSERT_OK(
        loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
    EXPECT_EQ(trims_.load(), 0u);
  }
  EXPECT_EQ(first_wakes, 0u);
  EXPECT_EQ(second_wakes, 0u);
  loom_runtime_host_preparation_end(host_);
  EXPECT_EQ(first_wakes, 1u);
  EXPECT_EQ(second_wakes, 1u);
  const auto delay = iree_make_duration_ms(1000);
  const auto deadline = loom_runtime_host_compiler_idle_deadline(host_, delay);
  ASSERT_NE(deadline, IREE_TIME_INFINITE_FUTURE);
  IREE_ASSERT_OK(
      loom_runtime_host_trim_compiler_idle(host_, delay, deadline - 1));
  EXPECT_EQ(trims_.load(), 0u);
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler_idle(host_, delay, deadline));
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler_idle(host_, delay, deadline));
  EXPECT_EQ(trims_.load(), 1u);
  loom_runtime_host_unobserve_preparation(host_, &first);
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  EXPECT_EQ(first_wakes, 1u);
  EXPECT_EQ(second_wakes, 2u);
  loom_runtime_host_unobserve_preparation(host_, &second);
}

TEST_F(HostReclamationTest, QueuedIdleObservationIsRecheckedAfterArrival) {
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  const auto old_deadline = loom_runtime_host_compiler_idle_deadline(host_, 0);
  loom_runtime_host_preparation_begin(host_);
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler_idle(host_, 0, old_deadline));
  EXPECT_EQ(trims_.load(), 0u);
  loom_runtime_host_preparation_end(host_);
  const auto deadline = loom_runtime_host_compiler_idle_deadline(host_, 1);
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler_idle(host_, 1, deadline - 1));
  EXPECT_EQ(trims_.load(), 0u);
  IREE_ASSERT_OK(loom_runtime_host_trim_compiler_idle(host_, 1, deadline));
  EXPECT_EQ(trims_.load(), 1u);
}

TEST_F(HostReclamationTest, NewWorkDoesNotWaitForAllocatorAndGetsLaterTrim) {
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  blocked_ = true;
  struct {
    // Worker-owned result transferred to the test owner by thread join.
    iree_status_t status = iree_ok_status();
  } outcome;
  std::thread worker([&] {
    outcome.status =
        loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now());
  });
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  IREE_EXPECT_OK(
      loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
  EXPECT_EQ(trims_.load(), 1u);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    blocked_ = false;
    condition_.notify_one();
  }
  worker.join();
  IREE_ASSERT_OK(outcome.status);
  IREE_ASSERT_OK(
      loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
  EXPECT_EQ(trims_.load(), 2u);
}

TEST_F(HostReclamationTest,
       FailureReturnsOnceAndDoesNotPoisonLaterPreparation) {
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  failure_ = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
  IREE_ASSERT_OK(
      loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
  EXPECT_EQ(trims_.load(), 1u);
  failure_ = IREE_STATUS_OK;
  loom_runtime_host_preparation_begin(host_);
  loom_runtime_host_preparation_end(host_);
  IREE_ASSERT_OK(
      loom_runtime_host_trim_compiler_idle(host_, 0, iree_time_now()));
  EXPECT_EQ(trims_.load(), 2u);
}

}  // namespace
