// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/device.h"

#include "common/hrx_bridge.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

TEST(StreamingDeviceQueryTest, UninitializedRegistryPreservesQueryFailures) {
  ASSERT_EQ(iree_hal_streaming_device_registry(), nullptr);
  EXPECT_EQ(iree_hal_streaming_device_entry(0), nullptr);

  iree_device_size_t free_memory = 123;
  iree_device_size_t total_memory = 456;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_streaming_device_memory_info(0, &free_memory, &total_memory));
  EXPECT_EQ(free_memory, 0u);
  EXPECT_EQ(total_memory, 0u);

  char name[] = "unchanged";
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_device_name(0, name, sizeof(name)));
  EXPECT_STREQ(name, "unchanged");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_streaming_device_get_string_property(
                            0, "hal.device", "name", name, sizeof(name)));
  EXPECT_STREQ(name, "unchanged");
}

TEST(StreamingQueuePlacementTest, SelectsOnlyAdvertisedHostLocalPlacement) {
  const uint32_t cases[] = {
      0, IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL,
      IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL,
      IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
          IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL};
  for (uint32_t capabilities : cases) {
    const iree_hal_queue_family_spec_t spec = {
        .placement_capabilities = capabilities,
    };
    const iree_hal_queue_family_t family = {.spec = &spec};
    EXPECT_EQ(iree_hal_streaming_queue_family_select_placement(&family),
              capabilities & IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL
                  ? IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL
                  : IREE_HAL_QUEUE_PLACEMENT_DEFAULT);
  }
}

// Changes only which real CPU queue the dependency provisions. Queue creation,
// properties, execution, and lifetime remain the task HAL's implementation.
struct ProvisionedQueueView {
  // Stack-owned device interface, borrowed only during the selection call.
  iree_hal_resource_t resource;
  // Real CPU device supplying the immutable family specification.
  iree_hal_device_t* device;
  // Real queue presented as the dependency's provisioned primary queue.
  iree_hal_queue_t* queue;

  iree_hal_device_t* Initialize() {
    static const iree_hal_device_vtable_t vtable = {
        .destroy = +[](iree_hal_device_t*) {},
        .device_spec =
            +[](iree_hal_device_t* base) {
              return iree_hal_device_spec(
                  reinterpret_cast<ProvisionedQueueView*>(base)->device);
            },
        .queue =
            +[](iree_hal_device_t* base, iree_hal_queue_family_ordinal_t family,
                iree_hal_queue_ordinal_t ordinal) {
              auto* self = reinterpret_cast<ProvisionedQueueView*>(base);
              EXPECT_EQ(family, iree_hal_queue_family(self->queue)->ordinal);
              EXPECT_EQ(ordinal, 0u);
              return self->queue;
            },
    };
    iree_hal_resource_initialize(&vtable, &resource);
    return reinterpret_cast<iree_hal_device_t*>(this);
  }
};

class StreamingDeviceQueueTest : public ::testing::Test {
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
    hrx_device_t device = hrx_device_group_get_device(group_, 0);
    entry_.hal_device = hrx_device_hal(device);
  }

  void TearDown() override {
    iree_hal_queue_release(acquired_);
    iree_hal_queue_release(device_local_);
    hrx_device_group_release(group_);
    group_ = nullptr;
  }

  // Real streaming registry entry's HAL ownership boundary.
  iree_hal_streaming_device_t entry_ = {};
  // Caller-owned result of the production streaming selection.
  iree_hal_queue_t* acquired_ = nullptr;
  // Caller-owned alternate provisioned queue used by the device view.
  iree_hal_queue_t* device_local_ = nullptr;
};

TEST_F(StreamingDeviceQueueTest, ReusesProvisionedHostLocalQueue) {
  iree_hal_queue_t* provisioned = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_select_primary_queue(&entry_, &provisioned));
  IREE_ASSERT_OK(
      iree_hal_streaming_device_acquire_primary_queue(&entry_, &acquired_));
  EXPECT_EQ(acquired_, provisioned);
  EXPECT_EQ(iree_hal_queue_placement(acquired_),
            IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL);
  iree_hal_queue_release(acquired_);
  acquired_ = nullptr;
  // Releasing the caller's reference leaves the provisioned owner intact.
  IREE_ASSERT_OK(
      iree_hal_streaming_device_acquire_primary_queue(&entry_, &acquired_));
  EXPECT_EQ(acquired_, provisioned);
}

TEST_F(StreamingDeviceQueueTest, AcquiresHostLocalFromDeviceLocalProvisioning) {
  iree_hal_queue_t* provisioned = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_select_primary_queue(&entry_, &provisioned));
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.placement = IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL;
  IREE_ASSERT_OK(iree_hal_queue_acquire(iree_hal_queue_family(provisioned),
                                        &params, &device_local_));
  ProvisionedQueueView view = {
      .device = entry_.hal_device,
      .queue = device_local_,
  };
  iree_hal_streaming_device_t viewed_entry = {.hal_device = view.Initialize()};
  IREE_ASSERT_OK(iree_hal_streaming_device_acquire_primary_queue(&viewed_entry,
                                                                 &acquired_));
  EXPECT_NE(acquired_, device_local_);
  EXPECT_EQ(iree_hal_queue_family(acquired_),
            iree_hal_queue_family(device_local_));
  EXPECT_EQ(iree_hal_queue_placement(acquired_),
            IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL);
  EXPECT_EQ(iree_hal_queue_placement(device_local_),
            IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL);
  iree_hal_device_release(viewed_entry.hal_device);
}

TEST_F(StreamingDeviceQueueTest, TransferQueuePreservesPrimaryProperties) {
  iree_hal_queue_t* primary = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_select_primary_queue(&entry_, &primary));
  IREE_ASSERT_OK(iree_hal_streaming_device_acquire_transfer_queue(
      &entry_, primary, &acquired_));
  EXPECT_NE(primary, acquired_);
  EXPECT_EQ(iree_hal_queue_family(primary), iree_hal_queue_family(acquired_));
  EXPECT_EQ(iree_hal_queue_priority(primary),
            iree_hal_queue_priority(acquired_));
  EXPECT_EQ(iree_hal_queue_placement(primary),
            iree_hal_queue_placement(acquired_));
  EXPECT_EQ(iree_hal_queue_features(primary),
            iree_hal_queue_features(acquired_));
  auto expected = iree_hal_queue_execution_resources(primary);
  auto actual = iree_hal_queue_execution_resources(acquired_);
  ASSERT_EQ(expected.count, actual.count);
  for (iree_host_size_t i = 0; i < expected.count; ++i) {
    EXPECT_EQ(expected.ordinals[i], actual.ordinals[i]);
  }
}

TEST_F(StreamingDeviceQueueTest, TransferQueueReusesCompatibleProvisioning) {
  iree_hal_queue_t* primary = nullptr;
  IREE_ASSERT_OK(
      iree_hal_streaming_device_select_primary_queue(&entry_, &primary));
  IREE_ASSERT_OK(iree_hal_streaming_device_acquire_transfer_queue(
      &entry_, primary, &acquired_));
  ProvisionedQueueView view = {
      .device = entry_.hal_device,
      .queue = acquired_,
  };
  iree_hal_streaming_device_t viewed_entry = {.hal_device = view.Initialize()};
  iree_hal_queue_t* reused = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_device_acquire_transfer_queue(
      &viewed_entry, primary, &reused));
  EXPECT_EQ(acquired_, reused);
  iree_hal_queue_release(reused);
  iree_hal_device_release(viewed_entry.hal_device);
}

}  // namespace
