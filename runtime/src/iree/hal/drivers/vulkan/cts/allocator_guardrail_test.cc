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
      device_, iree_hal_queue_family(transfer_queue_), &backend));
  iree_hal_passthrough_pool_options_t options = {};
  options.asan = backend.asan;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, backend.slab_provider, backend.notification,
      backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
      pool.out()));
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
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device_, iree_hal_queue_family(transfer_queue_), &backend));
  constexpr iree_device_size_t kBlockSize = 512;
  iree_hal_fixed_block_pool_options_t options = {};
  options.block_size = kBlockSize;
  options.blocks_per_slab = 1;
  options.frontier_capacity = 2;
  options.asan = backend.asan;
  Ref<iree_hal_pool_t> backing_pool;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateFiniteBlockPool(backend, options,
                                       iree_allocator_system(),
                                       backing_pool.out(), pool.out()));

  iree_hal_pool_reservation_request_t request = {};
  request.allocation_size = kBlockSize;
  request.params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  request.params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  request.params.queue_family_affinity = iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(transfer_queue_)));
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
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &info, &result));
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

TEST_P(VulkanAllocatorGuardrailTest, QueueAllocaRejectsAnotherNativeOwner) {
  DeviceCreateContext context;
  IREE_ASSERT_OK(context.Initialize(iree_allocator_system()));
  Ref<iree_hal_device_t> other_device;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, context.params(), iree_allocator_system(), other_device.out()));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder,
                                           context.frontier_tracker());
  iree_status_t status =
      iree_hal_device_group_builder_add_device(&builder, other_device);
  Ref<iree_hal_device_group_t> group;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(
        &builder, iree_allocator_system(), group.out());
  }
  iree_hal_device_group_builder_deinitialize(&builder);
  IREE_ASSERT_OK(status);

  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      other_device, iree_hal_device_queue_family(other_device, 0), &backend));
  iree_hal_passthrough_pool_options_t options = {};
  options.epoch_query = backend.epoch_query;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, backend.slab_provider, backend.notification,
      backend.frontier_tracker, backend.maintenance, iree_allocator_system(),
      pool.out()));

  // The semantic capabilities match, but a VkBuffer from a different VkDevice
  // cannot be published as a local native binding. Every acquired reservation
  // returns to its original pool when materialization exposes that mismatch.
  iree_hal_pool_reservation_request_t request = {};
  request.params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  request.params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  request.allocation_size = 4096;
  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList allocated(device_, {0}, {1});
  status =
      iree_hal_queue_alloca(transfer_queue_, iree_hal_semaphore_list_empty(),
                            allocated, pool, 1, &request, buffer.out());
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(allocated, iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.reservation_count, 0u);
}

CTS_REGISTER_TEST_SUITE(VulkanAllocatorGuardrailTest);

}  // namespace iree::hal::cts
