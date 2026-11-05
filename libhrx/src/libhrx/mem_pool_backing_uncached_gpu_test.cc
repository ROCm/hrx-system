// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "mem_pool_backing.h"

namespace {

class MemPoolBackingUncachedGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hrx_status_t status = hrx_gpu_initialize(0);
    initialized_ = hrx_status_is_ok(status);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &device_)));
  }

  void TearDown() override {
    if (initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
    }
  }

  // Whether this test owns the process-global GPU runtime.
  bool initialized_ = false;
  // Borrowed first GPU device, valid while the runtime is initialized.
  hrx_device_t device_ = nullptr;
};

TEST_F(MemPoolBackingUncachedGpuTest, HostMappingSourceKeepsBothRequirements) {
  const hrx_memory_type_t memory_type =
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
      HRX_MEMORY_TYPE_DEVICE_VISIBLE | HRX_MEMORY_TYPE_DEVICE_UNCACHED;
  iree_hal_pool_t* raw_pool = nullptr;
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_allocation_source(
      device_, memory_type, HRX_BUFFER_USAGE_MAPPING_SOURCE, &raw_pool));
  std::unique_ptr<iree_hal_pool_t, decltype(&iree_hal_pool_release)> pool(
      raw_pool, iree_hal_pool_release);
  EXPECT_EQ(pool.get(), device_->mem_pool_backing.uncached_host.mapping);
  EXPECT_EQ(device_->mem_pool_backing.host.mapping, nullptr);
  EXPECT_EQ(device_->mem_pool_backing.uncached_device.mapping, nullptr);

  const auto host = iree_hal_pool_query_host_access(pool.get());
  EXPECT_EQ(host.access, IREE_HAL_MEMORY_ACCESS_ALL);
  EXPECT_EQ(host.modes,
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool.get(), iree_hal_make_pool_reservation_request(4096, 0),
      iree_infinite_timeout(), &buffer));
  iree_hal_buffer_release(buffer);
}

}  // namespace
