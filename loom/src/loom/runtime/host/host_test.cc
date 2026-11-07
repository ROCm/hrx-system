// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/host.h"

#include <cstring>

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

}  // namespace
