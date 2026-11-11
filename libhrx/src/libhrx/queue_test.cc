// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuQueueTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  inline static hrx_device_group_t group_ = nullptr;

  static void SetUpTestSuite() {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    device_ = hrx_device_group_get_device(group_, 0);
  }

  static void TearDownTestSuite() {
    hrx_device_group_release(group_);
    group_ = nullptr;
    device_ = nullptr;
  }

  // Shared runtime-owned CPU device; each test releases acquired queues.
  static hrx_device_t device_;
};

hrx_device_t CpuQueueTest::device_ = nullptr;

TEST_F(CpuQueueTest, ProvisionedInventoryHasStableBorrowedIdentity) {
  ASSERT_GT(hrx_device_queue_family_count(device_), 0u);
  for (uint32_t family = 0; family < hrx_device_queue_family_count(device_);
       ++family) {
    hrx_queue_family_info_t info;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_device_query_queue_family(device_, family, &info)));
    for (uint32_t ordinal = 0; ordinal < info.provisioned_queue_count;
         ++ordinal) {
      hrx_queue_t queue = hrx_device_get_queue(device_, family, ordinal);
      ASSERT_NE(queue, nullptr);
      EXPECT_EQ(hrx_queue_get_device(queue), device_);
      EXPECT_EQ(hrx_queue_get_family_ordinal(queue), family);
      hrx_queue_retain(queue);
      hrx_queue_release(queue);
      EXPECT_EQ(hrx_device_get_queue(device_, family, ordinal), queue);
    }
    EXPECT_EQ(
        hrx_device_get_queue(device_, family, info.provisioned_queue_count),
        nullptr);
  }
  EXPECT_EQ(hrx_device_get_queue(device_, UINT32_MAX, 0), nullptr);
  EXPECT_EQ(hrx_device_get_queue(nullptr, 0, 0), nullptr);
}

TEST_F(CpuQueueTest, AcquisitionIsDistinctAndDoesNotChangeInventory) {
  hrx_queue_family_info_t info;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_device_query_queue_family(device_, 0, &info)));
  ASSERT_TRUE(info.supports_acquisition);
  hrx::runtime::queue_ptr first;
  hrx::runtime::queue_ptr second;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_queue_acquire(device_, 0, nullptr, first.for_output())));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_queue_acquire(device_, 0, nullptr, second.for_output())));
  EXPECT_NE(first.get(), second.get());
  EXPECT_NE(first.get()->hal_queue, second.get()->hal_queue);
  EXPECT_EQ(hrx_queue_get_device(first.get()), device_);
  hrx_queue_family_info_t after;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_device_query_queue_family(device_, 0, &after)));
  EXPECT_EQ(after.provisioned_queue_count, info.provisioned_queue_count);
  for (uint32_t i = 0; i < info.provisioned_queue_count; ++i) {
    EXPECT_NE(first.get()->hal_queue,
              hrx_device_get_queue(device_, 0, i)->hal_queue);
  }
  hrx::runtime::queue_ptr retained = first;
  first.reset();
  hrx_queue_params_t achieved;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_queue_query_params(retained.get(), &achieved)));
  EXPECT_EQ(achieved.priority, 0);
  EXPECT_EQ(achieved.features, HRX_QUEUE_FEATURE_NONE);
  EXPECT_EQ(achieved.execution_resources.count, 0u);
}

TEST_F(CpuQueueTest, InvalidAcquisitionDoesNotSubstituteProvisionedQueue) {
  hrx_queue_t queue = hrx_device_get_queue(device_, 0, 0);
  hrx_queue_params_t params = {
      .placement = static_cast<hrx_queue_placement_t>(UINT32_MAX),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_queue_acquire(device_, 0, &params, &queue)));
  EXPECT_EQ(queue, nullptr);
  params = {.execution_resources = {.count = 1}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_queue_acquire(device_, 0, &params, &queue)));
  EXPECT_EQ(queue, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_queue_acquire(
                            device_, UINT32_MAX, nullptr, &queue)));
  EXPECT_EQ(queue, nullptr);
  hrx_queue_family_info_t family;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_device_query_queue_family(
                            device_, UINT32_MAX, &family)));
}

}  // namespace
