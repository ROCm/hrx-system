// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/driver/task/registration/driver_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DeviceUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_driver_registry_allocate(iree_allocator_system(),
                                                     &driver_registry_));
    IREE_ASSERT_OK(iree_hal_task_driver_module_register(driver_registry_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_));
  }

  void TearDown() override {
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_driver_registry_free(driver_registry_);
  }

  iree_hal_device_create_params_t CreateParams() const {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    return create_params;
  }

  iree_hal_driver_registry_t* driver_registry_ = nullptr;
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
};

TEST_F(DeviceUtilTest, CreatesDefaultDevice) {
  ASSERT_EQ(iree_hal_device_flag_list().count, 0u);
  const iree_hal_device_create_params_t create_params = CreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_create_device_from_flags(
      driver_registry_, IREE_SV("task"), &create_params,
      iree_allocator_system(), &device));
  ASSERT_NE(device, nullptr);
  iree_hal_device_release(device);
}

TEST_F(DeviceUtilTest, CreatesFlagDevice) {
  ASSERT_EQ(iree_hal_device_flag_list().count, 1u);
  const iree_hal_device_create_params_t create_params = CreateParams();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_create_device_from_flags(
      driver_registry_, iree_string_view_empty(), &create_params,
      iree_allocator_system(), &device));
  ASSERT_NE(device, nullptr);
  iree_hal_device_release(device);
}

TEST_F(DeviceUtilTest, RejectsRepeatedDeviceFlags) {
  ASSERT_EQ(iree_hal_device_flag_list().count, 2u);
  const iree_hal_device_create_params_t create_params = CreateParams();
  iree_hal_device_t* device = reinterpret_cast<iree_hal_device_t*>(1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_create_device_from_flags(
                            driver_registry_, iree_string_view_empty(),
                            &create_params, iree_allocator_system(), &device));
  EXPECT_EQ(device, nullptr);
}

}  // namespace
