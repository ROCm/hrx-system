// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "mem_pool_backing.h"

#include <memory>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

  // Initialized runtime device owning all pools in each test.
  hrx_device_t device_ = nullptr;
};

TEST_F(MemPoolBackingTest, SignalsShareQualifiedBackingAndReuseIdleStorage) {
  const auto* family = iree_hal_queue_family(device_->dispatch_queue);
  const auto& operations = iree_hal_queue_family_spec(family)
                               ->zero_compute_atomic_capabilities.operations;
  iree_hal_pool_t* raw_pool = nullptr;
  const bool supports_signals =
      (operations.system_scope_32 & IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT) &&
      (operations.system_scope_64 & IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT);
  if (!supports_signals) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        hrx_mem_pool_backing_retain_signal_pool(device_, &raw_pool));
    EXPECT_EQ(raw_pool, nullptr);
    return;
  }
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_signal_pool(device_, &raw_pool));
  std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)> pool(
      raw_pool, iree_hal_pool_release);
  EXPECT_EQ(pool.get(), device_->mem_pool_backing.signals);
  iree_hal_memory_scope_t scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      device_->hal_device_group, {IREE_HAL_MEMORY_SITE_QUEUE, family}, &scope));
  const auto native =
      iree_hal_pool_query_zero_compute_atomic_operations(pool.get(), scope);
  const auto required = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  EXPECT_EQ(native.system_scope_32 & required, required);
  EXPECT_EQ(native.system_scope_64 & required, required);
  using HalBuffer =
      std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>;
  std::vector<HalBuffer> signals;
  for (size_t i = 0; i < 128; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool.get(), iree_hal_make_pool_reservation_request(8, 8),
        iree_infinite_timeout(), &buffer));
    signals.emplace_back(buffer, iree_hal_buffer_release);
    EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 8u);
    EXPECT_EQ(iree_hal_buffer_memory_view(buffer).backing,
              iree_hal_buffer_memory_view(signals.front().get()).backing);
  }
  const auto* backing =
      iree_hal_buffer_memory_view(signals.front().get()).backing;
  iree_hal_pool_stats_t stats = {};
  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_EQ(stats.bytes_reserved, 128u * 8u);
  signals.clear();
  iree_hal_pool_trim(pool.get(), IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  iree_hal_buffer_t* reused = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool.get(), iree_hal_make_pool_reservation_request(8, 8),
      iree_infinite_timeout(), &reused));
  HalBuffer signal(reused, iree_hal_buffer_release);
  EXPECT_EQ(iree_hal_buffer_memory_view(signal.get()).backing, backing);
}

}  // namespace
