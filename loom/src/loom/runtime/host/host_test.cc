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

#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/numa.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"
#include "loomc/task.h"

namespace {

TEST(HostTest, NullLifetimeOperations) {
  loom_runtime_host_retain(nullptr);
  loom_runtime_host_release(nullptr);
}

TEST(HostTest, CompilerAdapterUsesApplicationExecutor) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_runtime_host_create(1, nullptr, nullptr,
                                          iree_allocator_system(), &host));
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
  IREE_ASSERT_OK(
      loom_runtime_host_create(2, nullptr, nullptr, allocator, &host));
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
      loom_runtime_host_create(0, nullptr, nullptr, iree_allocator_system(),
                               &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_host_create(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT + 1, nullptr,
                               nullptr, iree_allocator_system(), &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_host_create(1, nullptr, nullptr,
                                                 iree_allocator_null(), &host));
  EXPECT_EQ(host, nullptr);
}

TEST(HostTest, DeviceIoPlacementPreservesNodeIdentities) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_runtime_host_create(1, nullptr, nullptr,
                                          iree_allocator_system(), &host));
  auto* pool = loom_runtime_host_proactor_pool(host);
  EXPECT_EQ(iree_async_proactor_pool_count(pool), iree_numa_node_count());
  iree_bitmap_t online_nodes = iree_numa_online_nodes();
  iree_host_size_t index = 0;
  for (uint32_t node_id = 0; node_id < online_nodes.bit_count; ++node_id) {
    if (!iree_bitmap_test(online_nodes, node_id)) {
      continue;
    }
    EXPECT_EQ(iree_async_proactor_pool_node_id(pool, index++), node_id);
  }
  // Physical inventory does not imply permission to run on every node. The
  // constructor's current node is inside its captured budget.
  uint32_t node_id = iree_numa_node_for_current_thread();
  iree_async_proactor_pool_entry_t* entry = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_pool_acquire_for_node(pool, node_id, &entry));
  EXPECT_EQ(iree_async_proactor_pool_entry_node_id(entry), node_id);
  EXPECT_NE(iree_async_proactor_pool_entry_proactor(entry), nullptr);
  iree_async_proactor_pool_entry_release(entry);
  loom_runtime_host_release(host);
}

#if defined(IREE_PLATFORM_LINUX)
TEST(HostTest, EmptyProcessorBudgetReturnsNoHost) {
  const iree_thread_affinity_set_t budget = {};
  loom_runtime_host_t* host = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_host_create(1, &budget, nullptr, iree_allocator_system(),
                               &host));
  EXPECT_EQ(host, nullptr);
}

struct AffinityObservation {
  AffinityObservation() { iree_notification_initialize(&notification); }
  ~AffinityObservation() {
    iree_status_free(status);
    iree_notification_deinitialize(&notification);
  }

  void Capture() {
    node_id = iree_numa_node_for_current_thread();
    status = iree_thread_affinity_query_current(&processors);
  }
  void Complete() {
    completed.store(true, std::memory_order_release);
    iree_notification_post(&notification, IREE_ALL_WAITERS);
  }
  void Await() {
    iree_notification_await(
        &notification,
        [](void* data) {
          return static_cast<AffinityObservation*>(data)->completed.load(
              std::memory_order_acquire);
        },
        this, iree_infinite_timeout());
  }
  void ExpectProcessor(uint32_t processor) {
    IREE_EXPECT_OK(iree_status_clone(status));
    iree_bitmap_t mask = iree_thread_affinity_set_bitmap(&processors);
    EXPECT_EQ(iree_bitmap_count(mask), 1u);
    ASSERT_LT(processor, mask.bit_count);
    EXPECT_TRUE(iree_bitmap_test(mask, processor));
  }

  // Native mask observed by a real host task or I/O callback.
  iree_thread_affinity_set_t processors = {};
  // Actual node observed by that worker, used to request the matching I/O
  // entry.
  uint32_t node_id = IREE_NUMA_NODE_ANY;
  // Native query status owned until the observer is destroyed.
  iree_status_t status = iree_ok_status();
  // Readiness publication; the owner joins the worker before freeing storage.
  std::atomic<bool> completed{false};
  // Wakes the constructing test thread without a wall-clock assumption.
  iree_notification_t notification;
};

enum class HostBudgetMode { kCaptureConstructor, kExplicit };

static void CheckHostBudget(HostBudgetMode mode) {
  iree_thread_affinity_set_t available_set;
  IREE_ASSERT_OK(iree_thread_affinity_query_current(&available_set));
  iree_bitmap_t available = iree_thread_affinity_set_bitmap(&available_set);
  const auto constructor_cpu = iree_bitmap_find_first_set(available, 0);
  auto expected_cpu = constructor_cpu;
  if (mode == HostBudgetMode::kExplicit) {
    expected_cpu =
        constructor_cpu + 1 < available.bit_count
            ? iree_bitmap_find_first_set(available, constructor_cpu + 1)
            : available.bit_count;
    if (expected_cpu == available.bit_count) {
      GTEST_SKIP() << "Explicit override requires two available CPUs";
    }
  }
  iree_thread_affinity_set_t budget_set = {};
  struct State {
    // Caller-owned constructor input, or NULL requesting capture.
    iree_thread_affinity_set_t* budget = nullptr;
    // Startup/submission status published by joining the constructor.
    iree_status_t status = iree_ok_status();
    // Observations outlive host and entry teardown on the constructor thread.
    AffinityObservation task;
    // I/O observation after the host and its aggregate pool are released.
    AffinityObservation io;
  } state;
  if (mode == HostBudgetMode::kExplicit) {
    state.budget = &budget_set;
    iree_bitmap_set(iree_thread_affinity_set_bitmap(state.budget),
                    expected_cpu);
  }
  const iree_thread_create_params_t params = {
      .initial_affinity =
          {
              .id_assigned = 1,
              .id = static_cast<uint32_t>(constructor_cpu),
          },
      .initial_processor_set = &available_set,
  };
  iree_thread_t* constructor = nullptr;
  iree_status_t status = iree_thread_create(
      +[](void* user_data) -> int {
        auto* state = static_cast<State*>(user_data);
        loom_runtime_host_t* host = nullptr;
        iree_status_t status = loom_runtime_host_create(
            1, state->budget, nullptr, iree_allocator_system(), &host);
        // The explicit input is borrowed only until host creation returns.
        if (state->budget) {
          iree_bitmap_reset_all(iree_thread_affinity_set_bitmap(state->budget));
        }
        struct Task {
          // Intrusive task accepted by the actual compiler scheduling domain.
          loomc_task_t base;
          // Observation owned by the waiting test.
          AffinityObservation* observation;
        } task = {.observation = &state->task};
        static const loomc_task_vtable_t vtable = {
            [](loomc_task_t* base, loomc_host_size_t) {
              reinterpret_cast<Task*>(base)->observation->Capture();
            },
            [](loomc_task_t* base) {
              reinterpret_cast<Task*>(base)->observation->Complete();
            },
        };
        loomc_task_initialize(&vtable, &task.base);
        if (iree_status_is_ok(status)) {
          status = iree_status_from_loomc(loomc_task_sink_submit(
              loom_runtime_host_compiler_sink(host), &task.base));
          if (iree_status_is_ok(status)) {
            state->task.Await();
            status = iree_status_clone(state->task.status);
          } else {
            loomc_task_destroy(&task.base);
          }
        } else {
          loomc_task_destroy(&task.base);
        }
        iree_async_proactor_pool_entry_t* entry = nullptr;
        if (iree_status_is_ok(status)) {
          status = iree_async_proactor_pool_acquire_for_node(
              loom_runtime_host_proactor_pool(host), state->task.node_id,
              &entry);
        }
        loom_runtime_host_release(host);
        iree_async_nop_operation_t nop = {};
        iree_async_operation_initialize(
            &nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
            IREE_ASYNC_OPERATION_FLAG_NONE,
            [](void* user_data, iree_async_operation_t*, iree_status_t status,
               iree_async_completion_flags_t) {
              auto* observation = static_cast<AffinityObservation*>(user_data);
              if (iree_status_is_ok(status)) {
                observation->Capture();
              } else {
                observation->status = status;
              }
              observation->Complete();
            },
            &state->io);
        if (iree_status_is_ok(status)) {
          status = iree_async_proactor_submit_one(
              iree_async_proactor_pool_entry_proactor(entry), &nop.base);
          if (iree_status_is_ok(status)) {
            state->io.Await();
          }
        }
        iree_async_proactor_pool_entry_release(entry);
        state->status = status;
        return 0;
      },
      &state, params, iree_allocator_system(), &constructor);
  iree_thread_release(constructor);
  IREE_EXPECT_OK(status);
  IREE_EXPECT_OK(state.status);
  state.task.ExpectProcessor(static_cast<uint32_t>(expected_cpu));
  state.io.ExpectProcessor(static_cast<uint32_t>(expected_cpu));
}

TEST(HostTest, CapturedBudgetSurvivesHostRelease) {
  CheckHostBudget(HostBudgetMode::kCaptureConstructor);
}

TEST(HostTest, ExplicitBudgetOverridesPinnedConstructor) {
  CheckHostBudget(HostBudgetMode::kExplicit);
}
#endif  // IREE_PLATFORM_LINUX

TEST(HostTest, BatchArenasShareAndReturnScratchBlocks) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_runtime_host_create(1, nullptr, nullptr,
                                          iree_allocator_system(), &host));
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
    IREE_ASSERT_OK(
        loom_runtime_host_create(2, nullptr, nullptr, allocator, &host_));
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
