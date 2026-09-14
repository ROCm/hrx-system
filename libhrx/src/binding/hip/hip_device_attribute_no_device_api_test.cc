// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <climits>
#include <cstdlib>

#include "api.h"
#include "hip_device_attribute_validation_test_shim.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

constexpr int kOutputSentinel = 0x5a5a5a5a;

using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);
using HipHalDeinitFn = hipError_t (*)(void);

TEST(HipDeviceAttributeNoDeviceApiTest,
     InitializationPrecedesArgumentValidation) {
  ASSERT_STREQ("-1", std::getenv("ROCR_VISIBLE_DEVICES"));
  hrx::hip::testing::HipDso dso;
  ASSERT_TRUE(dso.Open()) << dso.error();

  HipDeviceGetAttributeFn device_get_attribute =
      dso.Resolve<HipDeviceGetAttributeFn>("hipDeviceGetAttribute");
  HipGetLastErrorFn get_last_error =
      dso.Resolve<HipGetLastErrorFn>("hipGetLastError");
  HipPeekAtLastErrorFn peek_at_last_error =
      dso.Resolve<HipPeekAtLastErrorFn>("hipPeekAtLastError");
  HipHalDeinitFn hal_deinit = dso.Resolve<HipHalDeinitFn>("hipHALDeinit");
  ASSERT_NE(nullptr, device_get_attribute);
  ASSERT_NE(nullptr, get_last_error);
  ASSERT_NE(nullptr, peek_at_last_error);
  ASSERT_NE(nullptr, hal_deinit);

  EXPECT_EQ(hipErrorNoDevice,
            hrx_test_hip_device_get_attribute(
                device_get_attribute, nullptr,
                hipDeviceAttributeMaxThreadsPerBlock, /*device=*/0));
  EXPECT_EQ(hipErrorNoDevice, peek_at_last_error());
  EXPECT_EQ(hipErrorNoDevice, get_last_error());
  EXPECT_EQ(hipSuccess, peek_at_last_error());

  int attribute_value = kOutputSentinel;
  EXPECT_EQ(hipErrorNoDevice,
            hrx_test_hip_device_get_attribute(
                device_get_attribute, &attribute_value, INT_MAX, INT_MAX));
  EXPECT_EQ(kOutputSentinel, attribute_value);
  EXPECT_EQ(hipErrorNoDevice, peek_at_last_error());
  EXPECT_EQ(hipErrorNoDevice, get_last_error());
  EXPECT_EQ(hipSuccess, peek_at_last_error());

  EXPECT_EQ(hipSuccess, hal_deinit());
  EXPECT_TRUE(dso.Close()) << dso.error();
}

}  // namespace
