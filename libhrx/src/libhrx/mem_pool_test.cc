// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class MemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(/*flags=*/0)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_cpu_device_get(/*index=*/0, &device_)));

    const hrx_mem_pool_props_t properties = {
        /*.alloc_handle_type=*/0,
        /*.location_type=*/0,
        /*.location_id=*/device_->ordinal,
        /*.max_size=*/0,
    };
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_mem_pool_create(device_, &properties, &pool_)));
  }

  void TearDown() override {
    hrx_mem_pool_release(pool_);
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  hrx_device_t device_ = nullptr;
  hrx_mem_pool_t pool_ = nullptr;
};

class GpuMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_status_t status = hrx_status_to_iree(hrx_gpu_initialize(/*flags=*/0));
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "GPU runtime is unavailable";
    }
    initialized_ = true;

    int device_count = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_count(&device_count)));
    if (device_count < 2) {
      GTEST_SKIP() << "peer access requires at least two GPU devices";
    }
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(0, &owner_device_)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_gpu_device_get(1, &peer_device_)));
    const hrx_mem_pool_props_t properties = {
        /*.alloc_handle_type=*/0,
        /*.location_type=*/0,
        /*.location_id=*/owner_device_->ordinal,
        /*.max_size=*/0,
    };
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_mem_pool_create(owner_device_, &properties, &pool_)));
  }

  void TearDown() override {
    hrx_mem_pool_release(pool_);
    if (initialized_) {
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_gpu_shutdown()));
    }
  }

  bool initialized_ = false;
  hrx_device_t owner_device_ = nullptr;
  hrx_device_t peer_device_ = nullptr;
  hrx_mem_pool_t pool_ = nullptr;
};

TEST_F(MemoryPoolTest, TracksOwnerDeviceAccess) {
  hrx_memory_access_t access = HRX_MEMORY_ACCESS_NONE;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_get_device_access(pool_, device_, &access)));
  EXPECT_EQ(access, HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE);
}

TEST_F(MemoryPoolTest, RejectsAccessThatCannotBeEnforced) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_mem_pool_set_device_access(
                            pool_, device_, HRX_MEMORY_ACCESS_READ)));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_mem_pool_set_device_access(
                            pool_, device_, HRX_MEMORY_ACCESS_NONE)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_mem_pool_set_device_access(
                            pool_, device_, HRX_MEMORY_ACCESS_DISCARD)));
}

TEST_F(GpuMemoryPoolTest, TracksPeerDeviceAccess) {
  hrx_memory_access_t access = HRX_MEMORY_ACCESS_NONE;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_get_device_access(pool_, peer_device_, &access)));
  EXPECT_EQ(access, HRX_MEMORY_ACCESS_NONE);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_device_access(
      pool_, peer_device_, HRX_MEMORY_ACCESS_READ)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_get_device_access(pool_, peer_device_, &access)));
  EXPECT_EQ(access, HRX_MEMORY_ACCESS_READ);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_mem_pool_set_device_access(
      pool_, peer_device_, HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_mem_pool_get_device_access(pool_, peer_device_, &access)));
  EXPECT_EQ(access, HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE);
}

}  // namespace
