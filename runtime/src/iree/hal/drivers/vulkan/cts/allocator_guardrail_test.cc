// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"

namespace iree::hal::cts {

class VulkanAllocatorGuardrailTest : public CtsTestBase<> {};

namespace {

constexpr iree_device_size_t kMaxSparseProbeAllocationSize =
    64ull * 1024ull * 1024ull;

}  // namespace

TEST_P(VulkanAllocatorGuardrailTest, QueueAllocaAcceptsSparseSizedAllocation) {
  iree_host_size_t heap_count = 0;
  iree_status_t count_status = iree_hal_allocator_query_memory_heaps(
      device_allocator_, /*capacity=*/0, /*heaps=*/NULL, &heap_count);
  if (iree_status_is_out_of_range(count_status)) {
    iree_status_free(count_status);
  } else {
    IREE_ASSERT_OK(count_status);
  }
  ASSERT_NE(0u, heap_count);

  std::vector<iree_hal_allocator_memory_heap_t> heaps(heap_count);
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      device_allocator_, heaps.size(), heaps.data(), &heap_count));
  heaps.resize(heap_count);

  iree_device_size_t allocation_size = 0;
  for (const auto& heap : heaps) {
    if (!iree_all_bits_set(heap.allowed_usage,
                           IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      continue;
    }
    if (iree_device_size_checked_add(heap.max_allocation_size, 1,
                                     &allocation_size)) {
      break;
    }
  }
  if (allocation_size == 0) {
    GTEST_SKIP() << "No finite Vulkan allocation limit to probe";
  }
  if (allocation_size > kMaxSparseProbeAllocationSize) {
    GTEST_SKIP() << "Sparse queue_alloca probe would allocate "
                 << allocation_size << " bytes";
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateDevicePool(IREE_HAL_BUFFER_USAGE_TRANSFER, pool.out()));
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(allocation_size,
                                             /*min_alignment=*/1);

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, iree_hal_semaphore_list_empty(), alloca_signal, pool,
      /*request_count=*/1, &request, &raw_buffer));
  buffer.reset(raw_buffer);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t pattern = 0x1234CAFEu;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, empty_wait, fill_signal, buffer.get(),
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> data =
      ReadBufferBytes(buffer.get(), /*offset=*/0, sizeof(pattern));
  uint32_t readback = 0;
  memcpy(&readback, data.data(), sizeof(readback));
  EXPECT_EQ(pattern, readback);

  SemaphoreList dealloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* dealloca_buffer = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, fill_signal,
                                         dealloca_signal,
                                         /*buffer_count=*/1, &dealloca_buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(VulkanAllocatorGuardrailTest,
       CompletedDeallocationIsImmediatelyReusable) {
  constexpr iree_device_size_t kBlockSize = 512;
  iree_hal_fixed_block_pool_options_t options = {
      .block_size = kBlockSize,
      .blocks_per_slab = 1,
      .frontier_capacity = 2,
  };
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(
      CreateDevicePool(IREE_HAL_BUFFER_USAGE_TRANSFER, backing_pool.out()));
  IREE_ASSERT_OK(CreateFiniteBlockPool(backing_pool, options,
                                       iree_allocator_system(), pool.out()));

  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(kBlockSize,
                                             /*min_alignment=*/0);
  Ref<iree_hal_buffer_t> transient_buffer;
  SemaphoreList empty_wait;
  SemaphoreList allocated(device_, {0}, {1});
  SemaphoreList released(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, empty_wait, allocated,
                                       pool, 1, &request,
                                       transient_buffer.out()));
  iree_hal_buffer_t* dealloca_buffer = transient_buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, allocated, released,
                                         1, &dealloca_buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(released, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Completion returns an empty prerequisite, so reuse needs no completion
  // query and carries no wait into the next submission.
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      pool, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_EQ(IREE_HAL_POOL_ACQUIRE_OK_FRESH, result);
  EXPECT_EQ(info.reuse_frontier, nullptr);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, 1, &request, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
      buffer.out()));
  const uint32_t pattern = 0x1234ABCDu;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, empty_wait, filled, buffer, 0, kBlockSize, &pattern,
      sizeof(pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(filled, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  for (uint32_t value : ReadBufferData<uint32_t>(buffer)) {
    EXPECT_EQ(pattern, value);
  }
}

CTS_REGISTER_TEST_SUITE(VulkanAllocatorGuardrailTest);

}  // namespace iree::hal::cts
