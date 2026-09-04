// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/passthrough_pool.h"

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

  iree_hal_buffer_params_t params = {0};
  iree_device_size_t allocation_size = 0;
  for (const auto& heap : heaps) {
    if (!iree_all_bits_set(heap.allowed_usage,
                           IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      continue;
    }
    if (iree_device_size_checked_add(heap.max_allocation_size, 1,
                                     &allocation_size)) {
      params.type = heap.type;
      params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
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

  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, &backend));
  iree_hal_passthrough_pool_options_t options = {};
  options.asan = backend.asan;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, backend.slab_provider, backend.notification,
      iree_allocator_system(), pool.out()));
  params.queue_family_affinity = iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(transfer_queue_)));
  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = allocation_size,
  };

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
  IREE_ASSERT_OK(iree_hal_queue_fill(transfer_queue_, empty_wait, fill_signal,
                                     buffer.get(), /*target_offset=*/0,
                                     sizeof(pattern), &pattern, sizeof(pattern),
                                     IREE_HAL_FILL_FLAG_NONE));
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

CTS_REGISTER_TEST_SUITE(VulkanAllocatorGuardrailTest);

}  // namespace iree::hal::cts
