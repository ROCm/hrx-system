// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/thread.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/numa.h"
#include "iree/base/threading/thread_impl.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_LINUX)
#include <sched.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX

namespace {

//==============================================================================
// iree_thread_affinity_t
//==============================================================================

TEST(ThreadAffinityTest, SetAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0xFF, sizeof(affinity));  // Dirty memory.
  iree_thread_affinity_set_any(&affinity);
  EXPECT_TRUE(iree_thread_affinity_is_unspecified(affinity));
  EXPECT_FALSE(affinity.group_any);
  EXPECT_FALSE(affinity.id_assigned);
}

TEST(ThreadAffinityTest, SetGroupAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0xFF, sizeof(affinity));  // Dirty memory.
  iree_thread_affinity_set_group_any(0x80000105u, &affinity);
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
  EXPECT_TRUE(affinity.group_any);
  EXPECT_EQ(0x80000105u, affinity.group);
  EXPECT_FALSE(affinity.id_assigned);
}

TEST(ThreadAffinityTest, IsUnspecifiedFalseWhenGroupAny) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0, sizeof(affinity));
  affinity.group_any = 1;
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
}

TEST(ThreadAffinityTest, IsUnspecifiedFalseWhenIdAssigned) {
  iree_thread_affinity_t affinity;
  memset(&affinity, 0, sizeof(affinity));
  affinity.id_assigned = 1;
  affinity.id = 3;
  EXPECT_FALSE(iree_thread_affinity_is_unspecified(affinity));
}

//==============================================================================
// iree_thread_t
//==============================================================================

TEST(ThreadTest, Lifetime) {
  // Default parameters.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  // Our thread: do a bit of math and notify the main test thread when done.
  struct entry_data_t {
    iree_atomic_int32_t value;
    iree_notification_t barrier;
  } entry_data;
  iree_atomic_store(&entry_data.value, 123, iree_memory_order_relaxed);
  iree_notification_initialize(&entry_data.barrier);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->barrier, IREE_ALL_WAITERS);
    return 0;
  };

  // Create the thread and immediately begin running it.
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Wait for the thread to finish.
  iree_notification_await(
      &entry_data.barrier,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load(&entry_data->value,
                                iree_memory_order_relaxed) == (123 + 1);
      },
      &entry_data, iree_infinite_timeout());

  // By holding on to the thread object and releasing it here after the thread
  // has finished, we ensure that destruction occurs on the main thread,
  // avoiding data races reported by TSan.
  iree_thread_release(thread);
  iree_notification_deinitialize(&entry_data.barrier);
}

TEST(ThreadTest, CreateSuspended) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.create_suspended = true;

  struct entry_data_t {
    iree_atomic_int32_t value;
    std::atomic<bool> gate_open;
    iree_notification_t gate;
    iree_notification_t done;
  } entry_data;
  iree_atomic_store(&entry_data.value, 123, iree_memory_order_relaxed);
  entry_data.gate_open.store(false, std::memory_order_relaxed);
  iree_notification_initialize(&entry_data.gate);
  iree_notification_initialize(&entry_data.done);

  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    // Wait for gate to open before doing work.
    iree_notification_await(
        &entry_data->gate,
        +[](void* arg) {
          return static_cast<std::atomic<bool>*>(arg)->load(
              std::memory_order_acquire);
        },
        &entry_data->gate_open, iree_infinite_timeout());
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_acq_rel);
    iree_notification_post(&entry_data->done, IREE_ALL_WAITERS);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Value should be unchanged (thread created but suspended).
  ASSERT_EQ(123,
            iree_atomic_load(&entry_data.value, iree_memory_order_seq_cst));

  // Open the gate. If the thread were running (not suspended), it would now
  // pass through the gate and modify the value. Since it's suspended, the
  // value should remain unchanged.
  entry_data.gate_open.store(true, std::memory_order_release);
  iree_notification_post(&entry_data.gate, IREE_ALL_WAITERS);

  // Value should still be unchanged (thread is suspended, can't run).
  ASSERT_EQ(123,
            iree_atomic_load(&entry_data.value, iree_memory_order_seq_cst));

  // Resume the thread and wait for it to finish its work.
  iree_thread_resume(thread);
  iree_notification_await(
      &entry_data.done,
      +[](void* entry_arg) -> bool {
        auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
        return iree_atomic_load(&entry_data->value,
                                iree_memory_order_relaxed) == (123 + 1);
      },
      &entry_data, iree_infinite_timeout());

  iree_thread_release(thread);
  iree_notification_deinitialize(&entry_data.done);
  iree_notification_deinitialize(&entry_data.gate);
}

TEST(ThreadTest, RetainRelease) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  std::atomic<bool> completed{false};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* completed = reinterpret_cast<std::atomic<bool>*>(entry_arg);
    completed->store(true, std::memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &completed, params,
                                    iree_allocator_system(), &thread));

  // Retain adds a reference.
  iree_thread_retain(thread);

  // Wait for thread to complete.
  while (!completed.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // First release does not destroy the thread.
  iree_thread_release(thread);

  // Verify we can still get the thread ID (object is alive).
  EXPECT_NE(0u, iree_thread_id(thread));

  // Second release destroys the thread.
  iree_thread_release(thread);
}

TEST(ThreadTest, ReleaseWaitsForCompletion) {
  // Tests that iree_thread_release properly waits for thread completion.
  // Note: iree_thread_join + iree_thread_release cannot both be called as
  // release internally joins when destroying the thread.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  std::atomic<int32_t> value{0};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* value = reinterpret_cast<std::atomic<int32_t>*>(entry_arg);
    // Do some work.
    for (int i = 0; i < 100; ++i) {
      value->fetch_add(1, std::memory_order_relaxed);
      iree_thread_yield();
    }
    return 42;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &value, params,
                                    iree_allocator_system(), &thread));

  // Release internally joins/waits for the thread to complete.
  iree_thread_release(thread);

  // Thread has completed all work (release waited for it).
  EXPECT_EQ(100, value.load(std::memory_order_acquire));
}

TEST(ThreadTest, NamedThread) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("TestThread");

  std::atomic<bool> completed{false};
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* completed = reinterpret_cast<std::atomic<bool>*>(entry_arg);
    completed->store(true, std::memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &completed, params,
                                    iree_allocator_system(), &thread));

  // Wait for completion.
  while (!completed.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  iree_thread_release(thread);
}

// Affinity changes operate on a live suspended thread, not a thread that may
// already have exited by the time the native affinity syscall runs.
TEST(ThreadTest, RequestAffinity) {
  iree_thread_create_params_t params = {.create_suspended = true};
  uint32_t observed_node = IREE_NUMA_NODE_ANY;
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(
      +[](void* arg) -> int {
        *static_cast<uint32_t*>(arg) = iree_numa_node_for_current_thread();
        return 0;
      },
      &observed_node, params, iree_allocator_system(), &thread));

  uint32_t node = iree_numa_node_for_current_thread();
  iree_thread_affinity_t affinity;
  iree_thread_affinity_set_group_any(node, &affinity);
  IREE_EXPECT_OK(iree_thread_request_affinity(thread, affinity, {}));
  iree_thread_resume(thread);
  iree_thread_release(thread);
  EXPECT_EQ(node, observed_node);
}

TEST(ThreadTest, NumaAffinityBeforeEntry) {
  iree_thread_create_params_t params = {};
  uint32_t node = iree_numa_node_for_current_thread();
  iree_thread_affinity_set_group_any(node, &params.initial_affinity);
  uint32_t observed_node = IREE_NUMA_NODE_ANY;
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(
      +[](void* arg) -> int {
        *static_cast<uint32_t*>(arg) = iree_numa_node_for_current_thread();
        return 0;
      },
      &observed_node, params, iree_allocator_system(), &thread));
  iree_thread_release(thread);
  EXPECT_EQ(node, observed_node);
}

TEST(ThreadTest, FailedAffinityNeverCallsEntry) {
  iree_thread_create_params_t params = {};
  iree_thread_affinity_set_group_any(UINT16_MAX, &params.initial_affinity);
  bool called = false;
  iree_thread_t* thread = nullptr;
  iree_status_t status = iree_thread_create(
      +[](void* arg) -> int {
        *static_cast<bool*>(arg) = true;
        return 0;
      },
      &called, params, iree_allocator_system(), &thread);
  IREE_EXPECT_NOT_OK(status);
  iree_thread_release(thread);
  EXPECT_EQ(nullptr, thread);
  EXPECT_FALSE(called);
}

TEST(ThreadTest, InvalidAllocatorClearsOutput) {
  iree_thread_create_params_t params = {};
  iree_thread_t* thread = reinterpret_cast<iree_thread_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_thread_create(
                            +[](void*) -> int { return 0; }, nullptr, params,
                            iree_allocator_null(), &thread));
  EXPECT_EQ(nullptr, thread);
}

#if defined(IREE_PLATFORM_LINUX)
TEST(ThreadTest, CapturedBudgetSurvivesPinnedCreator) {
  iree_bitmap_t captured = {};
  IREE_ASSERT_OK(
      iree_thread_affinity_query_current(iree_allocator_system(), &captured));
  const iree_host_size_t selected_cpu = iree_bitmap_find_first_set(captured, 0);
  const iree_host_size_t captured_bits = captured.bit_count;
  std::vector<uint64_t> captured_words(
      captured.words,
      captured.words + iree_bitmap_calculate_words(captured.bit_count));
  iree_allocator_free(iree_allocator_system(), captured.words);
  ASSERT_LT(selected_cpu, captured_bits);

  // Include padding beyond cpu_set_t to exercise dynamically sized native
  // masks.
  const iree_host_size_t budget_bits =
      std::max<iree_host_size_t>(captured_bits, CPU_SETSIZE * 2);
  for (bool node_local : {false, true}) {
    SCOPED_TRACE(node_local ? "node within budget" : "captured budget");
    std::vector<uint64_t> budget_words(
        iree_bitmap_calculate_words(budget_bits));
    iree_bitmap_t budget = {
        .bit_count = budget_bits,
        .words = budget_words.data(),
    };
    if (node_local) {
      iree_bitmap_set(budget, selected_cpu);
    } else {
      std::copy(captured_words.begin(), captured_words.end(),
                budget_words.begin());
    }
    struct State {
      // Child startup input borrowing the application-owned budget.
      iree_thread_create_params_t child_params = {};
      // Effective child affinity, owned until the outer test releases it.
      iree_bitmap_t observed = {};
      // Native creation or query failure propagated to the outer test.
      iree_status_t status = iree_ok_status();
    } state;
    state.child_params.initial_processor_mask = budget;
    if (node_local) {
      iree_thread_affinity_set_group_any(0,
                                         &state.child_params.initial_affinity);
    }
    const iree_thread_create_params_t creator_params = {
        .initial_affinity =
            {
                .id_assigned = 1,
                .id = static_cast<uint32_t>(selected_cpu),
            },
        .initial_processor_mask = budget,
    };
    iree_thread_t* creator = nullptr;
    IREE_ASSERT_OK(iree_thread_create(
        +[](void* arg) -> int {
          auto* state = static_cast<State*>(arg);
          if (state->child_params.initial_affinity.group_any) {
            state->child_params.initial_affinity.group =
                iree_numa_node_for_current_thread();
          }
          iree_thread_t* child = nullptr;
          iree_status_t status = iree_thread_create(
              +[](void* arg) -> int {
                auto* state = static_cast<State*>(arg);
                state->status = iree_thread_affinity_query_current(
                    iree_allocator_system(), &state->observed);
                return 0;
              },
              state, state->child_params, iree_allocator_system(), &child);
          iree_thread_release(child);
          state->status = iree_status_join(status, state->status);
          return 0;
        },
        &state, creator_params, iree_allocator_system(), &creator));
    iree_thread_release(creator);
    IREE_EXPECT_OK(state.status);
    EXPECT_EQ(iree_bitmap_count(state.observed), iree_bitmap_count(budget));
    for (iree_host_size_t cpu = 0; cpu < state.observed.bit_count; ++cpu) {
      EXPECT_EQ(iree_bitmap_test(state.observed, cpu),
                cpu < budget.bit_count && iree_bitmap_test(budget, cpu))
          << "CPU " << cpu;
    }
    iree_allocator_free(iree_allocator_system(), state.observed.words);
  }
}

TEST(ThreadTest, EmptyProcessorBudgetNeverCallsEntry) {
  uint64_t words = 0;
  const iree_thread_create_params_t params = {
      .initial_processor_mask =
          {
              .bit_count = 64,
              .words = &words,
          },
  };
  bool called = false;
  iree_thread_t* thread = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_thread_create(
                            +[](void* arg) -> int {
                              *static_cast<bool*>(arg) = true;
                              return 0;
                            },
                            &called, params, iree_allocator_system(), &thread));
  EXPECT_EQ(thread, nullptr);
  EXPECT_FALSE(called);
}

TEST(ThreadTest, RejectedAffinityReportsTargetThread) {
  // A missing CPU is rejected by the real kernel without requiring a cgroup
  // mutation or changing the test thread's own placement.
  constexpr int rejected_cpu = CPU_SETSIZE - 2;
  ASSERT_EQ(0, access("/sys/devices/system/cpu", F_OK));
  for (int cpu = rejected_cpu; cpu < CPU_SETSIZE; ++cpu) {
    const std::string path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    if (access(path.c_str(), F_OK) == 0) {
      GTEST_SKIP() << "CPU " << cpu << " exists on this system";
    }
    ASSERT_EQ(ENOENT, errno);
  }

  cpu_set_t inherited;
  ASSERT_EQ(0, sched_getaffinity(0, sizeof(inherited), &inherited));
  int selected_cpu = 0;
  while (selected_cpu < CPU_SETSIZE && !CPU_ISSET(selected_cpu, &inherited)) {
    ++selected_cpu;
  }
  ASSERT_LT(selected_cpu, CPU_SETSIZE);
  iree_thread_create_params_t params = {
      .name = IREE_SV("affinity-test"),
      .create_suspended = true,
      .initial_affinity =
          {
              .id_assigned = 1,
              .id = static_cast<uint32_t>(selected_cpu),
          },
  };
  bool called = false;
  const iree_thread_entry_t entry = +[](void* arg) -> int {
    *static_cast<bool*>(arg) = true;
    return 0;
  };
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry, &called, params,
                                    iree_allocator_system(), &thread));

  iree_thread_affinity_t rejected = {
      .id_assigned = 1,
      .id = rejected_cpu,
      .smt = 1,
  };
  iree_status_t status = iree_thread_request_affinity(thread, rejected, {});
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status));
#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
  const std::string message = iree::Status::ToString(status);
  EXPECT_THAT(message, ::testing::HasSubstr("thread 'affinity-test'"));
  EXPECT_THAT(message, ::testing::HasSubstr(
                           "requested CPUs [" + std::to_string(rejected_cpu) +
                           "-" + std::to_string(rejected_cpu + 1) + "]"));
  EXPECT_THAT(message, ::testing::HasSubstr(
                           "observed thread affinity after rejection [" +
                           std::to_string(selected_cpu) + "]"));
  EXPECT_THAT(message, ::testing::HasSubstr("errno " + std::to_string(EINVAL)));
#endif  // IREE_STATUS_FEATURE_ANNOTATIONS
  iree_status_free(status);
  EXPECT_FALSE(called);
  iree_thread_resume(thread);
  iree_thread_release(thread);
  EXPECT_TRUE(called);

  // The same native rejection during startup must retire the suspended native
  // thread without ever admitting the user's entry routine.
  called = false;
  params.create_suspended = false;
  params.initial_affinity = rejected;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_thread_create(entry, &called, params,
                                           iree_allocator_system(), &thread));
  EXPECT_EQ(nullptr, thread);
  EXPECT_FALSE(called);
}

TEST(ThreadTest, CpuAffinityBeforeEntry) {
  cpu_set_t allowed;
  ASSERT_EQ(0, sched_getaffinity(0, sizeof(allowed), &allowed));
  int cpu = -1;
  for (int i = CPU_SETSIZE - 1; i >= 0; --i) {
    if (CPU_ISSET(i, &allowed)) {
      cpu = i;
      break;
    }
  }
  ASSERT_GE(cpu, 0);

  iree_thread_create_params_t params = {};
  params.initial_affinity.id_assigned = 1;
  params.initial_affinity.id = cpu;
  cpu_set_t observed;
  CPU_ZERO(&observed);
  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(
      +[](void* arg) -> int {
        EXPECT_EQ(0, sched_getaffinity(0, sizeof(cpu_set_t),
                                       static_cast<cpu_set_t*>(arg)));
        return 0;
      },
      &observed, params, iree_allocator_system(), &thread));
  iree_thread_release(thread);
  EXPECT_EQ(1, CPU_COUNT(&observed));
  EXPECT_TRUE(CPU_ISSET(cpu, &observed));
}

TEST(ThreadTest, FailedNativeCreationNeverCallsEntry) {
  iree_thread_create_params_t params = {.stack_size = 1};
  bool called = false;
  iree_thread_t* thread = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_thread_create(
                            +[](void* arg) -> int {
                              *static_cast<bool*>(arg) = true;
                              return 0;
                            },
                            &called, params, iree_allocator_system(), &thread));
  EXPECT_EQ(nullptr, thread);
  EXPECT_FALSE(called);
}
#endif  // IREE_PLATFORM_LINUX

// Testing whether priority took effect is hard given that on certain platforms
// the priority may not be respected or may be clamped by the system. This test
// exercises the mechanics of the priority override code on our side.
TEST(ThreadTest, PriorityOverride) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));

  struct entry_data_t {
    iree_atomic_int32_t value;
  } entry_data;
  iree_atomic_store(&entry_data.value, 0, iree_memory_order_relaxed);
  iree_thread_entry_t entry_fn = +[](void* entry_arg) -> int {
    auto* entry_data = reinterpret_cast<struct entry_data_t*>(entry_arg);
    iree_atomic_fetch_add(&entry_data->value, 1, iree_memory_order_release);
    return 0;
  };

  iree_thread_t* thread = nullptr;
  IREE_ASSERT_OK(iree_thread_create(entry_fn, &entry_data, params,
                                    iree_allocator_system(), &thread));
  EXPECT_NE(0u, iree_thread_id(thread));

  // Push a few overrides.
  // Some platforms (Apple) may ignore the request and return NULL. Code using
  // overrides needs to be tolerant of this.
  iree_thread_override_t* override0 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGH);
  iree_thread_override_t* override1 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_HIGHEST);
  iree_thread_override_t* override2 = iree_thread_priority_class_override_begin(
      thread, IREE_THREAD_PRIORITY_CLASS_LOWEST);

  // Wait for the thread to finish.
  while (iree_atomic_load(&entry_data.value, iree_memory_order_acquire) != 1) {
    iree_thread_yield();
  }

  // Pop overrides (in opposite order intentionally).
  iree_thread_override_end(override0);
  iree_thread_override_end(override1);
  iree_thread_override_end(override2);

  iree_thread_release(thread);
}

//==============================================================================
// iree_thread_override_list_t
//==============================================================================
// This is an implementation detail but useful to test on its own as it's shared
// across several platform implementations.

TEST(ThreadOverrideListTest, PriorityClass) {
  static iree_thread_t* kThreadSentinel =
      reinterpret_cast<iree_thread_t*>(0x123);
  static iree_thread_priority_class_t current_priority_class =
      IREE_THREAD_PRIORITY_CLASS_NORMAL;
  iree_thread_override_list_t list;
  iree_thread_override_list_initialize(
      +[](iree_thread_t* thread, iree_thread_priority_class_t priority_class) {
        EXPECT_EQ(kThreadSentinel, thread);
        EXPECT_NE(current_priority_class, priority_class);
        current_priority_class = priority_class;
      },
      current_priority_class, iree_allocator_system(), &list);

  // (NORMAL) -> HIGH -> [ignored LOW] -> HIGHEST
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_NORMAL, current_priority_class);
  iree_thread_override_t* override0 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_HIGH);
  EXPECT_NE(nullptr, override0);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGH, current_priority_class);
  iree_thread_override_t* override1 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_LOW);
  EXPECT_NE(nullptr, override1);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGH, current_priority_class);
  iree_thread_override_t* override2 = iree_thread_override_list_add(
      &list, kThreadSentinel, IREE_THREAD_PRIORITY_CLASS_HIGHEST);
  EXPECT_NE(nullptr, override2);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);

  // Out of order to ensure highest bit sticks.
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override1);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override0);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_HIGHEST, current_priority_class);
  iree_thread_override_remove_self(override2);
  ASSERT_EQ(IREE_THREAD_PRIORITY_CLASS_NORMAL, current_priority_class);

  iree_thread_override_list_deinitialize(&list);
}

}  // namespace
