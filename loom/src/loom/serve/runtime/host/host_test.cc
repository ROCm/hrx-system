// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/host/host.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/base/threading/numa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree/task_pool.h"

namespace {

TEST(HostTest, NullLifetimeOperations) {
  loom_serve_host_retain(nullptr);
  loom_serve_host_release(nullptr);
}

TEST(HostTest, CompilerAdapterUsesApplicationExecutor) {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  EXPECT_EQ(
      loom_serve_host_executor(host),
      iree_task_executor_from_loomc_task_pool(loom_serve_host_task_pool(host)));
  EXPECT_EQ(iree_task_executor_worker_count(loom_serve_host_executor(host)),
            1u);
  EXPECT_NE(loom_serve_host_task_driver(host), nullptr);
  EXPECT_EQ(loom_serve_host_target_environment(host), nullptr);
  iree_host_size_t target_provider_count = 1;
  EXPECT_EQ(loom_serve_host_target_providers(host, &target_provider_count),
            nullptr);
  EXPECT_EQ(target_provider_count, 0u);
  loom_serve_host_retain(host);
  loom_serve_host_release(host);
  EXPECT_NE(loom_serve_host_proactor_pool(host), nullptr);
  loom_serve_host_release(host);
}

TEST(HostTest, InvalidCapacityAndAllocationFailureReturnNoHost) {
  loom_serve_host_t* host = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(0, nullptr, iree_allocator_system(), &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT + 1, nullptr,
                             iree_allocator_system(), &host));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(1, nullptr, iree_allocator_null(), &host));
  EXPECT_EQ(host, nullptr);
}

TEST(HostTest, DeviceIoPlacementResolvesEveryOnlineNode) {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  auto* pool = loom_serve_host_proactor_pool(host);
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
  loom_serve_host_release(host);
}

TEST(HostTest, BatchArenasShareAndReturnScratchBlocks) {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  auto* pool = loom_serve_host_block_pool(host);
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
  loom_serve_host_release(host);
}

}  // namespace
