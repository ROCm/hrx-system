// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_properties.h"

#include <climits>

#include "iree/testing/gtest.h"

namespace {

TEST(HipDevicePropertiesTest, ParsesThreeAndFourDigitTargets) {
  int architecture = -1;
  EXPECT_TRUE(iree_hip_parse_gcn_arch_name("gfx942", &architecture));
  EXPECT_EQ(942, architecture);
  EXPECT_TRUE(
      iree_hip_parse_gcn_arch_name("gfx1100:sramecc+:xnack-", &architecture));
  EXPECT_EQ(1100, architecture);
}

TEST(HipDevicePropertiesTest, RejectsMalformedAndOutOfRangeTargets) {
  int architecture = -1;
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name(nullptr, &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx11x0", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx2147483648", &architecture));
  EXPECT_EQ(-1, architecture);
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx942", nullptr));
}

}  // namespace
