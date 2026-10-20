// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/signal.h"

#include <array>
#include <cstring>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AqlSignalTest, InitializesNativeAbiAndRecyclesOnlyAfterReadersRetire) {
  constexpr uint64_t kBaseAddress = 0x123456780000;
  std::array<iree_hal_amd_gpu_aql_signal_t, 2> native;
  std::memset(native.data(), 0xa5, sizeof(native));
  std::array<iree_hal_amd_gpu_signal_t, 2> metadata;
  std::array<uint64_t, 4> frontiers;
  iree_hal_amd_gpu_signal_pool_t pool;
  iree_hal_amd_gpu_signal_block_t block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &pool);
  const amdf_gpu_event_info_t event = {
      .mailbox_address = 0x345678900000,
      .interrupt_context = 0x123456,
  };
  iree_hal_amd_gpu_aql_signal_block_initialize(
      &pool, &pool, &event, native.data(), kBaseAddress, metadata.data(),
      native.size(), 2, frontiers.data(), &block);
  for (const auto& signal : native) {
    const iree_hal_amd_gpu_aql_signal_t expected = {
        .kind = 1,
        .event_mailbox = event.mailbox_address,
        .event_id = event.interrupt_context,
    };
    EXPECT_EQ(std::memcmp(&signal, &expected, sizeof(signal)), 0);
  }
  auto* first = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 1);
  auto* second = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 1);
  ASSERT_EQ(first, &metadata[0]);
  ASSERT_EQ(second, &metadata[1]);
  EXPECT_EQ(first->device_address, kBaseAddress);
  EXPECT_EQ(second->device_address, kBaseAddress + 64);
  EXPECT_EQ(first->host_value, &native[0].value);
  EXPECT_EQ(second->host_value, &native[1].value);
  EXPECT_EQ(native[0].value, 1u);
  EXPECT_EQ(native[1].value, 1u);
  const std::array<uint64_t, 2> published = {17, 3};
  iree_hal_amd_gpu_signal_publish_frontier(first, published.data());
  iree_hal_amd_gpu_signal_retain(first);  // Independent native reader.
  native[0].value = 0;  // Model the firmware's completed producer.
  iree_hal_amd_gpu_signal_retire(first, IREE_STATUS_OK);
  iree_hal_amd_gpu_signal_release(first);
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 1), nullptr);
  EXPECT_FALSE(iree_hal_amd_gpu_signal_block_try_deinitialize(&block));
  std::array<uint64_t, 2> joined = {9, 8};
  iree_hal_amd_gpu_signal_join_frontier(first, joined.data());
  EXPECT_EQ(joined, (std::array<uint64_t, 2>{17, 8}));
  iree_status_code_t code = IREE_STATUS_UNKNOWN;
  EXPECT_TRUE(iree_hal_amd_gpu_signal_query_retirement(first, &code));
  EXPECT_EQ(code, IREE_STATUS_OK);
  iree_hal_amd_gpu_signal_release(first);  // Reader has now retired.
  auto* reused = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 1);
  EXPECT_EQ(reused, first);
  EXPECT_EQ(native[0].value, 1u);
  EXPECT_EQ(native[0].kind, 1u);
  EXPECT_EQ(native[0].event_mailbox, event.mailbox_address);
  EXPECT_EQ(native[0].event_id, event.interrupt_context);
  iree_hal_amd_gpu_signal_release(reused);
  iree_hal_amd_gpu_signal_release(second);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
}

TEST(AqlSignalTest, SemaphoreCacheDoesNotOwnTheLastDependentReader) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, 0, 2, iree_allocator_system(), &semaphore));
  iree_hal_amd_gpu_aql_signal_t native;
  iree_hal_amd_gpu_signal_t metadata;
  uint64_t frontier;
  iree_hal_amd_gpu_signal_pool_t pool;
  iree_hal_amd_gpu_signal_block_t block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &pool);
  const amdf_gpu_event_info_t event = {};
  iree_hal_amd_gpu_aql_signal_block_initialize(&pool, &pool, &event, &native,
                                               0x4000, &metadata, 1, 1,
                                               &frontier, &block);
  auto* producer = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 1);
  const uint64_t epoch = 7;
  iree_hal_amd_gpu_signal_publish_frontier(producer, &epoch);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_publish_signal(semaphore, 5, producer));
  iree_hal_amd_gpu_semaphore_wait_t reader;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_acquire_wait(semaphore, 5, &pool, &reader));
  EXPECT_EQ(reader.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  EXPECT_EQ(reader.signal->device_address, 0x4000u);
  // A foreign protocol/address domain receives no native storage borrow.
  iree_hal_amd_gpu_semaphore_wait_t foreign;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_acquire_wait(semaphore, 5, &native, &foreign));
  EXPECT_EQ(foreign.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST);
  EXPECT_EQ(foreign.signal, nullptr);
  native.value = 0;
  iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_OK);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_publish_completed(semaphore, 5, nullptr));
  iree_hal_amd_gpu_signal_release(producer);
  iree_hal_semaphore_release(semaphore);
  EXPECT_FALSE(iree_hal_amd_gpu_signal_block_try_deinitialize(&block));
  EXPECT_EQ(native.value, 0u);  // Still visible to the dependent GPU.
  iree_hal_amd_gpu_signal_release(reader.signal);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
  iree_async_proactor_release(proactor);
}

}  // namespace
