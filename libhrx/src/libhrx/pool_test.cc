// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "hrx_internal.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));
    IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
        device_->hal_device, iree_hal_queue_family(device_->transfer_queue),
        &backend_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  iree_hal_buffer_params_t BufferParams() const {
    return {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
  }

  void WaitForMaintenance() {
    // Producers have stopped. Join child returns and the cache/native work
    // they enqueue before observing retention or changing the trim phase.
    iree_hal_memory_maintenance_call(
        backend_.maintenance,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        backend_.maintenance);
  }

  iree_hal_pool_stats_t BackingStats() {
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(device_->mem_pool_backing.cache, &stats);
    return stats;
  }

  // CPU device supplying backing storage for the pools under test.
  hrx_device_t device_ = nullptr;
  // Borrowed memory and progress sources from the device's transfer family.
  iree_hal_queue_pool_backend_t backend_ = {};
};

TEST_F(CpuPoolTest, ExactPoolDefersNativeAllocationUntilGrowthIsAllowed) {
  // Observe the real heap allocator used by the HRX adapter.
  size_t allocation_count = 0;
  const iree_allocator_t counting_allocator = {
      &allocation_count,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** inout_pointer) -> iree_status_t {
        if (command != IREE_ALLOCATOR_COMMAND_FREE) {
          ++*static_cast<size_t*>(self);
        }
        iree_allocator_t allocator = iree_allocator_system();
        return allocator.ctl(allocator.self, command, params, inout_pointer);
      },
  };
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      IREE_SV("exact_pool_test"), counting_allocator, counting_allocator,
      &allocator));
  const iree_hal_buffer_params_t params = BufferParams();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(
      allocator, params, backend_.notification, backend_.frontier_tracker,
      iree_allocator_system(), &pool));

  for (iree_host_size_t request_count : {1u, 9u}) {
    iree_hal_pool_reservation_request_t requests[9];
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      requests[i] = {params, 128 + i * 16};
    }
    iree_hal_pool_reservation_t reservations[9];
    memset(reservations, 0xA5, sizeof(reservations));
    iree_hal_pool_reservation_t original_reservations[9];
    memcpy(original_reservations, reservations, sizeof(reservations));
    iree_hal_pool_acquire_info_t infos[9];
    memset(infos, 0x5A, sizeof(infos));
    iree_hal_pool_acquire_info_t original_infos[9];
    memcpy(original_infos, infos, sizeof(infos));
    iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
    const size_t original_allocation_count = allocation_count;

    // Validation precedes ordinary exhaustion and preserves all error outputs.
    requests[request_count - 1].allocation_size = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_pool_acquire_reservations(
            pool, request_count, requests, /*requester_frontier=*/nullptr,
            IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, reservations, infos,
            &result));
    EXPECT_EQ(memcmp(infos, original_infos, sizeof(infos)), 0);
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_NONE);
    requests[request_count - 1].allocation_size = 128;

    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, request_count, requests, /*requester_frontier=*/nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH |
            IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
        reservations, infos, &result));
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
    EXPECT_EQ(allocation_count, original_allocation_count);
    EXPECT_EQ(memcmp(reservations, original_reservations, sizeof(reservations)),
              0);
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      EXPECT_EQ(infos[i].result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
      EXPECT_EQ(infos[i].flags, IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);
      EXPECT_EQ(infos[i].reuse_frontier, nullptr);
    }

    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, request_count, requests, /*requester_frontier=*/nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations, infos, &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    EXPECT_GT(allocation_count, original_allocation_count);
    iree_hal_buffer_t* buffers[9];
    IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
        pool, request_count, requests, reservations,
        IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
        buffers));
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      const uint8_t pattern = static_cast<uint8_t>(i + 1);
      IREE_ASSERT_OK(iree_hal_buffer_map_fill(
          buffers[i], 0, IREE_HAL_WHOLE_BUFFER, &pattern, sizeof(pattern)));
      uint8_t actual = 0;
      IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[i],
                                              requests[i].allocation_size - 1,
                                              &actual, sizeof(actual)));
      EXPECT_EQ(actual, pattern);
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_hal_pool_release(pool);
  iree_hal_allocator_release(allocator);
}

TEST_F(CpuPoolTest, BatchedReservationsRetainPoolUntilRelease) {
  const iree_hal_buffer_params_t params = BufferParams();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(
      device_->allocator.hal_allocator, params, backend_.notification,
      backend_.frontier_tracker, iree_allocator_system(), &pool));

  const iree_hal_pool_reservation_request_t requests[2] = {
      {.params = params, .allocation_size = 4096},
      {.params = params, .allocation_size = 8192},
  };
  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t acquire_infos[2];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, acquire_infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(acquire_infos[0].result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(acquire_infos[1].result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  iree_hal_buffer_t* borrowed_buffers[2];
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool, IREE_ARRAYSIZE(reservations), requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, borrowed_buffers));
  EXPECT_EQ(iree_hal_buffer_byte_length(borrowed_buffers[0]), 4096u);
  EXPECT_EQ(iree_hal_buffer_byte_length(borrowed_buffers[1]), 8192u);

  // Model HRX releasing its public buffer wrapper while queue operations still
  // retain transient buffers backed by these reservations. Each reservation
  // keeps the exact pool alive through the later retirement callback.
  iree_hal_pool_release(pool);
  iree_hal_buffer_release(borrowed_buffers[0]);
  iree_hal_buffer_release(borrowed_buffers[1]);
  iree_hal_pool_release_reservations(pool, IREE_ARRAYSIZE(reservations),
                                     reservations,
                                     /*death_frontier=*/nullptr);
}

TEST_F(CpuPoolTest, TransferredReservationOwnsBackingBuffer) {
  const iree_hal_buffer_params_t params = BufferParams();
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(
      device_->allocator.hal_allocator, params, backend_.notification,
      backend_.frontier_tracker, iree_allocator_system(), &pool));

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_pool_allocate_buffer(pool, params, /*allocation_size=*/4096,
                                    iree_infinite_timeout(), &buffer));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
}

TEST_F(CpuPoolTest, AcceptsWeakerAndRejectsInvalidOrStrongerAlignment) {
  iree_hal_buffer_params_t params = BufferParams();
  params.min_alignment = 8;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(hrx_iree_exact_pool_create(
      device_->allocator.hal_allocator, params, backend_.notification,
      backend_.frontier_tracker, iree_allocator_system(), &pool));

  iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = 4096,
  };
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;

  request.params.min_alignment = 0;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, /*requester_frontier=*/nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));
  iree_hal_pool_release_reservations(pool, 1, &reservation,
                                     /*death_frontier=*/nullptr);

  request.params.min_alignment = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_acquire_reservations(
                            pool, 1, &request, /*requester_frontier=*/nullptr,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
                            &acquire_info, &result));

  request.params.min_alignment = 16;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_acquire_reservations(
                            pool, 1, &request, /*requester_frontier=*/nullptr,
                            IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
                            &acquire_info, &result));

  request.params.min_alignment = 8;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, &request, /*requester_frontier=*/nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(pool, 1, &reservation,
                                     /*death_frontier=*/nullptr);
  iree_hal_pool_release(pool);
}

TEST_F(CpuPoolTest, MemoryPoolServesSmallAndLargeBuffersFromOneAllocator) {
  const hrx_mem_pool_props_t properties = {};
  hrx_mem_pool_t pool = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_mem_pool_create(device_, &properties, &pool)));
  hrx_buffer_t buffers[3] = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_allocate_buffer(pool, 1024, &buffers[0])));
  uint64_t regular_backing_bytes = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &regular_backing_bytes)));
  const size_t lengths[] = {
      1024, static_cast<size_t>(regular_backing_bytes) + 256, 512};
  for (size_t i = 1; i < IREE_ARRAYSIZE(buffers); ++i) {
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_mem_pool_allocate_buffer(pool, lengths[i], &buffers[i])));
  }
  uint64_t live_backing_bytes = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &live_backing_bytes)));
  EXPECT_EQ(live_backing_bytes, regular_backing_bytes + lengths[1]);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    EXPECT_EQ(buffers[i]->hal_pool, pool->hal_pool);
    void* address = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_map(buffers[i], HRX_MAP_WRITE, 0, lengths[i], &address)));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(address) % 256, 0u);
    const uint8_t pattern = static_cast<uint8_t>(0x31 + i);
    memset(address, pattern, lengths[i]);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffers[i])));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_map(
        buffers[i], HRX_MAP_READ, lengths[i] - 256, 256, &address)));
    for (size_t j = 0; j < 256; ++j) {
      EXPECT_EQ(static_cast<const uint8_t*>(address)[j], pattern);
    }
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_unmap(buffers[i])));
  }
  for (auto* buffer : buffers) {
    hrx_buffer_release(buffer);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_trim(pool, 0)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &live_backing_bytes)));
  EXPECT_EQ(live_backing_bytes, 0u);
  WaitForMaintenance();
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
  hrx_mem_pool_release(pool);
}

TEST_F(CpuPoolTest, MemoryPoolTrimPreservesLiveBuffersAndRetentionFloor) {
  const hrx_mem_pool_props_t properties = {};
  hrx_mem_pool_t pool = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_mem_pool_create(device_, &properties, &pool)));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_mem_pool_allocate_buffer(pool, 1024, &buffer)));

  uint64_t committed_bytes = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &committed_bytes)));
  ASSERT_GE(committed_bytes, 1024u);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_trim(pool, 0)));
  WaitForMaintenance();
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer->hal_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE,
      /*byte_offset=*/0, /*byte_length=*/1024, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) % 256, 0u);
  memset(mapping.contents.data, 0x3C, mapping.contents.data_length);
  EXPECT_EQ(mapping.contents.data[1023], 0x3C);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  hrx_buffer_release(buffer);

  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_trim(pool, static_cast<size_t>(committed_bytes))));
  WaitForMaintenance();
  uint64_t retained_bytes = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, committed_bytes)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(pool)));
  WaitForMaintenance();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(pool)));
  WaitForMaintenance();
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, 0u);

  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_mem_pool_allocate_buffer(pool, 1024, &buffer)));
  hrx_buffer_release(buffer);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_trim(pool, 0)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  WaitForMaintenance();
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
  hrx_mem_pool_release(pool);
}

TEST_F(CpuPoolTest, MemoryPoolAllocationRequiresPool) {
  hrx_buffer_t buffer = reinterpret_cast<hrx_buffer_t>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_mem_pool_allocate_buffer(nullptr, 1024, &buffer)));
  EXPECT_EQ(buffer, nullptr);
}

}  // namespace
