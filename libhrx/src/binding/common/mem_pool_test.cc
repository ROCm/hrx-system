// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/mem_pool.h"

#include <cstring>

#include "common/hrx_bridge.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"

namespace {

class CpuPoolTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    device_ = hrx_device_group_get_device(group_, 0);
  }

  void TearDown() override {
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  iree_hal_pool_stats_t BackingStats() {
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(device_->mem_pool_backing.cache, &stats);
    return stats;
  }

  // CPU device supplying backing storage for the pools under test.
  hrx_device_t device_ = nullptr;
};

TEST_F(CpuPoolTest, MemoryPoolServesSmallAndLargeBuffersFromOneAllocator) {
  const iree_hal_streaming_mem_pool_props_t properties = {};
  iree_hal_streaming_mem_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_mem_pool_create(device_, &properties, &pool));
  hrx_buffer_t buffers[3] = {};
  IREE_ASSERT_OK(
      iree_hal_streaming_mem_pool_allocate_buffer(pool, 1024, &buffers[0]));
  uint64_t regular_backing_bytes = 0;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &regular_backing_bytes));
  const size_t lengths[] = {
      1024, static_cast<size_t>(regular_backing_bytes) + 256, 512};
  for (size_t i = 1; i < IREE_ARRAYSIZE(buffers); ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_mem_pool_allocate_buffer(pool, lengths[i],
                                                               &buffers[i]));
  }
  uint64_t live_backing_bytes = 0;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &live_backing_bytes));
  EXPECT_EQ(live_backing_bytes, regular_backing_bytes + lengths[1]);
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    void* address = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_buffer_map(buffers[i], HRX_MAP_WRITE, 0, lengths[i], &address)));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(address) % 256, 0u);
    const uint8_t pattern = static_cast<uint8_t>(0x31 + i);
    memset(address, pattern, lengths[i]);
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_unmap(buffers[i])));
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_map(buffers[i], HRX_MAP_READ,
                                           lengths[i] - 256, 256, &address)));
    for (size_t j = 0; j < 256; ++j) {
      EXPECT_EQ(static_cast<const uint8_t*>(address)[j], pattern);
    }
    IREE_ASSERT_OK(HRX_CALL(hrx_buffer_unmap(buffers[i])));
  }
  for (auto* buffer : buffers) {
    hrx_buffer_release(buffer);
  }
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_trim(pool, 0));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &live_backing_bytes));
  EXPECT_EQ(live_backing_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
  iree_hal_streaming_mem_pool_release(pool);
}

TEST_F(CpuPoolTest, MemoryPoolTrimPreservesLiveBuffersAndRetentionFloor) {
  const iree_hal_streaming_mem_pool_props_t properties = {};
  iree_hal_streaming_mem_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_mem_pool_create(device_, &properties, &pool));

  hrx_buffer_t buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_mem_pool_allocate_buffer(pool, 1024, &buffer));

  uint64_t committed_bytes = 0;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &committed_bytes));
  ASSERT_GE(committed_bytes, 1024u);

  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_trim(pool, 0));
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

  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_trim(
      pool, static_cast<size_t>(committed_bytes)));
  uint64_t retained_bytes = 0;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &retained_bytes));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_set_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD,
      committed_bytes));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_release_unused(pool));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &retained_bytes));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_set_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_release_unused(pool));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &retained_bytes));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, 0u);

  IREE_ASSERT_OK(
      iree_hal_streaming_mem_pool_allocate_buffer(pool, 1024, &buffer));
  hrx_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_trim(pool, 0));
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_get_attribute(
      pool, IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT,
      &retained_bytes));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
  iree_hal_streaming_mem_pool_release(pool);
}

TEST_F(CpuPoolTest, MemoryPoolAllocationRequiresPool) {
  hrx_buffer_t buffer = reinterpret_cast<hrx_buffer_t>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_streaming_mem_pool_allocate_buffer(nullptr, 1024, &buffer));
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(CpuPoolTest, FiniteDescendantsPreserveCompatibilityAllocationCharge) {
  const iree_hal_streaming_mem_pool_props_t properties = {.max_size = 65536};
  hrx::runtime::hrx_ptr<iree_hal_streaming_mem_pool_t*,
                        iree_hal_streaming_mem_pool_retain,
                        iree_hal_streaming_mem_pool_release>
      compatibility_pool;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_create(
      device_, &properties, compatibility_pool.for_output()));
  hrx::runtime::buffer_ptr backing;
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_allocate_buffer(
      compatibility_pool, 65536, backing.for_output()));
  hrx::runtime::mem_pool_ptr finite, child;
  const hrx_tlsf_pool_options_t finite_options = {};
  IREE_ASSERT_OK(HRX_CALL(hrx_tlsf_pool_create_from_buffer(
      backing, 0, UINT64_MAX, &finite_options, finite.for_output())));
  const hrx_fixed_block_pool_options_t child_options = {
      .block_size = 256,
      .blocks_per_slab = 16,
  };
  IREE_ASSERT_OK(HRX_CALL(
      hrx_fixed_block_pool_create(finite, &child_options, child.for_output())));
  backing.reset();
  finite.reset();
  hrx_buffer_t denied = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_mem_pool_allocate_buffer(
                            compatibility_pool, 1, &denied));
  EXPECT_EQ(denied, nullptr);
  hrx::runtime::semaphore_ptr completion;
  IREE_ASSERT_OK(
      HRX_CALL(hrx_semaphore_create(device_, 0, completion.for_output())));
  hrx_semaphore_t semaphore = completion;
  uint64_t allocated = 1;
  const hrx_semaphore_list_t allocation_signal = {&semaphore, &allocated, 1};
  const hrx_pool_reservation_request_t request = {256, 16};
  hrx::runtime::buffer_ptr buffer;
  hrx_queue_t queue = hrx_device_get_queue(device_, 0, 0);
  IREE_ASSERT_OK(
      HRX_CALL(hrx_queue_alloca(queue, nullptr, &allocation_signal, child, 1,
                                &request, buffer.for_output())));
  IREE_ASSERT_OK(
      HRX_CALL(hrx_semaphore_wait(completion, allocated, UINT64_MAX)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_streaming_mem_pool_allocate_buffer(
                            compatibility_pool, 1, &denied));
  EXPECT_EQ(denied, nullptr);
  uint64_t released = 2;
  const hrx_semaphore_list_t release_signal = {&semaphore, &released, 1};
  hrx_buffer_t root = buffer;
  IREE_ASSERT_OK(HRX_CALL(hrx_queue_dealloca(queue, &allocation_signal,
                                             &release_signal, 1, &root)));
  IREE_ASSERT_OK(
      HRX_CALL(hrx_semaphore_wait(completion, released, UINT64_MAX)));
  buffer.reset();
  child.reset();
  IREE_ASSERT_OK(iree_hal_streaming_mem_pool_allocate_buffer(
      compatibility_pool, 65536, backing.for_output()));
  backing.reset();
  compatibility_pool.reset();
}

}  // namespace
