// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>

#include "api.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);

// Owns the RTLD_LOCAL handle and exact public entry points exercised by this
// test.
struct HipRuntimeApi {
  // Initializes the loaded HIP runtime instance.
  HipInitFn init = nullptr;
  // Deinitializes the loaded HRX runtime instance before unloading its DSO.
  HipHalDeinitFn hal_deinit = nullptr;
  // Reports the calling thread's current device.
  HipGetDeviceFn get_device = nullptr;
  // Selects the calling thread's current device.
  HipSetDeviceFn set_device = nullptr;
  // Reports every device visible to the loaded runtime instance.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Reports one public device attribute for one visible device.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  // Returns and clears the calling thread's last HIP error.
  HipGetLastErrorFn get_last_error = nullptr;
  // Returns without clearing the calling thread's last HIP error.
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
};

class HipUnifiedAddressingAttributeApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_device = dso_.Resolve<HipGetDeviceFn>("hipGetDevice");
      api_.set_device = dso_.Resolve<HipSetDeviceFn>("hipSetDevice");
      api_.get_device_count =
          dso_.Resolve<HipGetDeviceCountFn>("hipGetDeviceCount");
      api_.device_get_attribute =
          dso_.Resolve<HipDeviceGetAttributeFn>("hipDeviceGetAttribute");
      api_.get_last_error = dso_.Resolve<HipGetLastErrorFn>("hipGetLastError");
      api_.peek_at_last_error =
          dso_.Resolve<HipPeekAtLastErrorFn>("hipPeekAtLastError");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.set_device);
    ASSERT_NE(nullptr, api_.get_device_count);
    ASSERT_NE(nullptr, api_.device_get_attribute);
    ASSERT_NE(nullptr, api_.get_last_error);
    ASSERT_NE(nullptr, api_.peek_at_last_error);

    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count_));
    ASSERT_GT(device_count_, 0);
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    current_device_ = device_count_ - 1;
    ASSERT_EQ(hipSuccess, api_.set_device(current_device_));
    ExpectCurrentDevice();
    ASSERT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) return;
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  void ExpectCurrentDevice() {
    int actual_device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&actual_device));
    EXPECT_EQ(current_device_, actual_device);
  }

  template <typename Callable>
  void ExpectErrorState(hipError_t expected, Callable&& callable) {
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    EXPECT_EQ(expected, callable());
    EXPECT_EQ(expected, api_.peek_at_last_error());
    ExpectCurrentDevice();
    EXPECT_EQ(expected, api_.peek_at_last_error());
    EXPECT_EQ(expected, api_.get_last_error());
    EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  // Process-lifetime runtime instance loaded from the exact built DSO.
  static HipRuntimeApi api_;
  // Exact DSO owner shared by every test in this fixture.
  static hrx::hip::testing::HipDso dso_;
  // Number of compatible devices visible to the loaded runtime.
  int device_count_ = 0;
  // Device selected in calling-thread TLS before each attribute query.
  int current_device_ = -1;
};

HipRuntimeApi HipUnifiedAddressingAttributeApiTest::api_;
hrx::hip::testing::HipDso HipUnifiedAddressingAttributeApiTest::dso_;

TEST_F(HipUnifiedAddressingAttributeApiTest, ReturnsOneOnEveryVisibleDevice) {
  for (int device = 0; device < device_count_; ++device) {
    SCOPED_TRACE(device);
    int attribute_value = -1;
    ASSERT_EQ(hipSuccess, api_.device_get_attribute(
                              &attribute_value,
                              hipDeviceAttributeUnifiedAddressing, device));
    EXPECT_EQ(1, attribute_value);
    ExpectCurrentDevice();
    EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
    EXPECT_EQ(hipSuccess, api_.get_last_error());
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest,
       NullOutputTakesPrecedenceAndPublishesLastError) {
  const int device_ids[] = {0, -1, device_count_, INT_MIN, INT_MAX};
  for (int device : device_ids) {
    SCOPED_TRACE(device);
    ExpectErrorState(hipErrorInvalidValue, [&] {
      return api_.device_get_attribute(
          nullptr, hipDeviceAttributeUnifiedAddressing, device);
    });
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest,
       InvalidDevicesPreserveOutputAndPublishLastError) {
  const int invalid_devices[] = {-1, device_count_, INT_MIN, INT_MAX};
  for (int device : invalid_devices) {
    SCOPED_TRACE(device);
    int attribute_value = 0x5a5a5a5a;
    ExpectErrorState(hipErrorInvalidDevice, [&] {
      return api_.device_get_attribute(
          &attribute_value, hipDeviceAttributeUnifiedAddressing, device);
    });
    EXPECT_EQ(0x5a5a5a5a, attribute_value);
  }
}

TEST_F(HipUnifiedAddressingAttributeApiTest, SuccessPreservesPendingLastError) {
  ASSERT_EQ(hipSuccess, api_.get_last_error());
  ASSERT_EQ(hipErrorInvalidValue,
            api_.device_get_attribute(
                nullptr, hipDeviceAttributeUnifiedAddressing, /*device=*/0));
  int attribute_value = -1;
  EXPECT_EQ(hipSuccess,
            api_.device_get_attribute(&attribute_value,
                                      hipDeviceAttributeUnifiedAddressing,
                                      /*device=*/0));
  EXPECT_EQ(1, attribute_value);
  ExpectCurrentDevice();
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
}

}  // namespace
