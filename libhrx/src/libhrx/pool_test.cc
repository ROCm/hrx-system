// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
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
  uint64_t retained_bytes = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, committed_bytes)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(pool)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
      pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &retained_bytes)));
  EXPECT_EQ(retained_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, committed_bytes);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_attribute(
      pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, 0)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(pool)));
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
