// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "mem_pool_backing.h"

#include <array>
#include <memory>
#include <thread>

#include "hrx_internal.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using MemPool =
    std::unique_ptr<hrx_mem_pool_s, decltype(&hrx_mem_pool_release)>;
using Buffer = std::unique_ptr<hrx_buffer_s, decltype(&hrx_buffer_release)>;

class MemPoolBackingTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(HRX_TEST_GPU)
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_initialize(0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &device_)));
#else
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_device_get(0, &device_)));
#endif
  }

  void TearDown() override {
#if defined(HRX_TEST_GPU)
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
#else
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
#endif
  }

  MemPool CreatePool(size_t max_size = 0) {
    hrx_mem_pool_props_t properties = {.max_size = max_size};
    hrx_mem_pool_t pool = nullptr;
    IREE_EXPECT_OK(
        hrx_status_to_iree(hrx_mem_pool_create(device_, &properties, &pool)));
    return MemPool(pool, hrx_mem_pool_release);
  }

  Buffer Allocate(hrx_mem_pool_t pool, size_t length = 1024) {
    hrx_buffer_t buffer = nullptr;
    IREE_EXPECT_OK(hrx_status_to_iree(
        hrx_mem_pool_allocate_buffer(pool, length, &buffer)));
    return Buffer(buffer, hrx_buffer_release);
  }

  void WriteAndCheck(hrx_buffer_t buffer, uint8_t pattern) {
    std::array<uint8_t, 1024> input;
    input.fill(pattern);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_h2d(device_, input.data(), buffer, 0, input.size())));
    Check(buffer, pattern);
  }

  void Check(hrx_buffer_t buffer, uint8_t pattern) {
    std::array<uint8_t, 1024> output = {};
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(device_, buffer, 0, output.data(), output.size())));
    std::array<uint8_t, 1024> expected;
    expected.fill(pattern);
    size_t mismatch_count = 0;
    size_t zero_count = 0;
    for (uint8_t value : output) {
      mismatch_count += value != pattern;
      zero_count += value == 0;
    }
    EXPECT_EQ(output, expected) << "mismatched bytes: " << mismatch_count
                                << ", zero bytes: " << zero_count;
  }

  void JoinMaintenance() {
    auto* owner = device_->mem_pool_backing.cache->maintenance;
    iree_hal_memory_maintenance_call(
        owner,
        [](void* data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        owner);
  }

  iree_hal_pool_stats_t BackingStats() {
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(device_->mem_pool_backing.cache, &stats);
    return stats;
  }

  uint64_t ReservedBytes(hrx_mem_pool_t pool) {
    uint64_t bytes = 0;
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_mem_pool_get_attribute(
        pool, HRX_MEM_POOL_ATTR_RESERVED_MEM_CURRENT, &bytes)));
    return bytes;
  }

  void SetThreshold(hrx_mem_pool_t pool, uint64_t bytes) {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_attribute(
        pool, HRX_MEM_POOL_ATTR_RELEASE_THRESHOLD, bytes)));
  }

  void Trim(hrx_mem_pool_t pool, size_t bytes = 0) {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_trim(pool, bytes)));
    JoinMaintenance();
  }

  // Initialized runtime device owning all pools in each test.
  hrx_device_t device_ = nullptr;
};

TEST_F(MemPoolBackingTest, LivePoolsReuseBackingAndKeepIndependentAccounting) {
  auto first = CreatePool();
  auto second = CreatePool();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(device_->mem_pool_backing.cache, nullptr);

  auto buffer = Allocate(first.get());
  ASSERT_NE(buffer, nullptr);
  const auto* contract = first->hal_pool->memory_contract;
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->domain,
            iree_hal_device_group_memory_domain(device_->hal_device_group));
  EXPECT_EQ(device_->mem_pool_backing.cache->memory_contract, contract);
  const auto* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_->hal_device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    EXPECT_EQ(iree_hal_buffer_family_usage(
                  buffer->hal_buffer,
                  iree_hal_device_queue_family(device_->hal_device, i)),
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH);
  }
#if defined(HRX_TEST_GPU)
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer->hal_buffer),
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
#else
  const auto host = iree_hal_pool_query_host_access(first->hal_pool);
  EXPECT_EQ(host.access, IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(host.modes,
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT);
#endif
  WriteAndCheck(buffer.get(), 0x37);
  const auto* identity =
      iree_hal_buffer_memory_view(buffer->hal_buffer).backing;
  ASSERT_NE(identity, nullptr);
  const uint64_t backing_bytes = BackingStats().bytes_committed;
  EXPECT_EQ(ReservedBytes(first.get()), backing_bytes);
  EXPECT_EQ(ReservedBytes(second.get()), 0u);
  buffer.reset();
  JoinMaintenance();
  if (device_->mem_pool_backing.pool_options.asan.quarantine_size != 0) {
    // Native ASAN keeps returned ranges poisoned until quarantine eviction.
    // Explicit trim drains that child policy while retaining the shared slab.
    iree_hal_pool_stats_t stats = {};
    iree_hal_pool_query_stats(first->hal_pool, &stats);
    EXPECT_EQ(stats.bytes_reserved, 0u);
    EXPECT_GT(stats.bytes_quarantined, 0u);
    EXPECT_EQ(ReservedBytes(first.get()), backing_bytes);
    Trim(first.get(), backing_bytes);
  }
  EXPECT_EQ(ReservedBytes(first.get()), 0u);
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);

  buffer = Allocate(second.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_NE(first->hal_pool, second->hal_pool);
  EXPECT_EQ(second->hal_pool->memory_contract, contract);
  EXPECT_EQ(iree_hal_buffer_memory_view(buffer->hal_buffer).backing, identity);
  WriteAndCheck(buffer.get(), 0x59);
  EXPECT_EQ(ReservedBytes(first.get()), 0u);
  EXPECT_EQ(ReservedBytes(second.get()), backing_bytes);
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);
  iree_hal_slab_cache_stats_t cache_stats = {};
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(
      device_->mem_pool_backing.cache, &cache_stats));
  EXPECT_EQ(cache_stats.miss_count, 1u);
  EXPECT_EQ(cache_stats.hit_count, 1u);

  Trim(first.get());
  Check(buffer.get(), 0x59);
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);
  buffer.reset();
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

TEST_F(MemPoolBackingTest, BackingOutlivesPublicPoolAndItsLastBuffer) {
  auto first = CreatePool();
  ASSERT_NE(first, nullptr);
  auto buffer = Allocate(first.get());
  ASSERT_NE(buffer, nullptr);
  const auto* identity =
      iree_hal_buffer_memory_view(buffer->hal_buffer).backing;
  const uint64_t backing_bytes = BackingStats().bytes_committed;
  first.reset();
  WriteAndCheck(buffer.get(), 0x6B);
  buffer.reset();
  JoinMaintenance();
  EXPECT_EQ(device_->mem_pool_backing.retention_head, nullptr);
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);

  auto second = CreatePool();
  ASSERT_NE(second, nullptr);
  buffer = Allocate(second.get());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_memory_view(buffer->hal_buffer).backing, identity);
  WriteAndCheck(buffer.get(), 0x97);
  buffer.reset();
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

TEST_F(MemPoolBackingTest, LogicalLimitsRemainIndependentOfSharedBacking) {
  auto first = CreatePool(1024);
  auto second = CreatePool(2048);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  hrx_buffer_t rejected = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        hrx_status_to_iree(hrx_mem_pool_allocate_buffer(
                            first.get(), 1025, &rejected)));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_EQ(device_->mem_pool_backing.cache, nullptr);

  auto first_buffer = Allocate(first.get());
  auto second_buffer = Allocate(second.get(), 2048);
  ASSERT_NE(first_buffer, nullptr);
  ASSERT_NE(second_buffer, nullptr);
  WriteAndCheck(first_buffer.get(), 0x2A);
  WriteAndCheck(second_buffer.get(), 0xE3);
  const uint64_t backing_bytes = BackingStats().bytes_committed;
  EXPECT_EQ(ReservedBytes(first.get()) + ReservedBytes(second.get()),
            backing_bytes);
  for (auto* pool : {first.get(), second.get()}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        hrx_status_to_iree(hrx_mem_pool_allocate_buffer(pool, 1, &rejected)));
    EXPECT_EQ(rejected, nullptr);
  }
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);
  first_buffer.reset();
  JoinMaintenance();
  first_buffer = Allocate(first.get());
  ASSERT_NE(first_buffer, nullptr);
  WriteAndCheck(first_buffer.get(), 0x58);
  Check(second_buffer.get(), 0xE3);
  EXPECT_EQ(BackingStats().bytes_committed, backing_bytes);
  first_buffer.reset();
  second_buffer.reset();
  Trim(first.get());
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

TEST_F(MemPoolBackingTest, SharedRetentionFloorsSurviveSiblingTrimAndTeardown) {
  auto first = CreatePool();
  auto second = CreatePool();
  auto third = CreatePool();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(third, nullptr);
  auto first_buffer = Allocate(first.get());
  auto second_buffer = Allocate(second.get());
  auto third_buffer = Allocate(third.get());
  ASSERT_NE(first_buffer, nullptr);
  ASSERT_NE(second_buffer, nullptr);
  ASSERT_NE(third_buffer, nullptr);
  const uint64_t slab_bytes = ReservedBytes(first.get());
  ASSERT_GT(slab_bytes, 0u);
  EXPECT_EQ(BackingStats().bytes_committed, 3 * slab_bytes);
  SetThreshold(first.get(), slab_bytes);
  SetThreshold(second.get(), 2 * slab_bytes);
  SetThreshold(third.get(), 3 * slab_bytes);
  first_buffer.reset();
  second_buffer.reset();
  third_buffer.reset();
  JoinMaintenance();

  Trim(first.get());
  EXPECT_EQ(BackingStats().bytes_committed, 3 * slab_bytes);
  third.reset();
  Trim(first.get());
  EXPECT_EQ(BackingStats().bytes_committed, 2 * slab_bytes);
  SetThreshold(first.get(), slab_bytes);
  SetThreshold(second.get(), slab_bytes);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(first.get())));
  JoinMaintenance();
  EXPECT_EQ(BackingStats().bytes_committed, slab_bytes);

  // A larger new floor preserves available storage without creating more.
  SetThreshold(first.get(), 2 * slab_bytes);
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, slab_bytes);
  first.reset();
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

TEST_F(MemPoolBackingTest, ExplicitFloorLastsUntilAutomaticRelease) {
  auto first = CreatePool();
  auto second = CreatePool();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  auto buffer = Allocate(first.get());
  ASSERT_NE(buffer, nullptr);
  const uint64_t slab_bytes = BackingStats().bytes_committed;
  buffer.reset();
  Trim(first.get(), slab_bytes);
  Trim(second.get());
  EXPECT_EQ(BackingStats().bytes_committed, slab_bytes);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_release_unused(first.get())));
  JoinMaintenance();
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

TEST_F(MemPoolBackingTest, ConcurrentPoolCreationAllocationAndTrim) {
  std::array<std::thread, 4> threads;
  for (size_t thread_index = 0; thread_index < threads.size(); ++thread_index) {
    threads[thread_index] = std::thread([&, thread_index] {
      for (size_t iteration = 0; iteration < 16; ++iteration) {
        auto pool = CreatePool(1024);
        ASSERT_NE(pool, nullptr);
        auto buffer = Allocate(pool.get());
        ASSERT_NE(buffer, nullptr);
        WriteAndCheck(buffer.get(),
                      static_cast<uint8_t>(thread_index * 16 + iteration + 1));
        IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_trim(pool.get(), 0)));
        buffer.reset();
        IREE_ASSERT_OK(
            hrx_status_to_iree(hrx_mem_pool_release_unused(pool.get())));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(device_->mem_pool_backing.retention_head, nullptr);
  auto pool = CreatePool();
  ASSERT_NE(pool, nullptr);
  Trim(pool.get());
  EXPECT_EQ(BackingStats().bytes_committed, 0u);
}

}  // namespace
