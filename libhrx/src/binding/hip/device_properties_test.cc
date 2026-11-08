// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_properties.h"

#include <climits>
#include <cstring>
#include <string>

#include "iree/testing/gtest.h"

namespace {

TEST(HipDevicePropertiesTest, ParsesThreeAndFourDigitTargets) {
  int architecture = -1;
  EXPECT_TRUE(iree_hip_parse_gcn_arch_name("gfx942", &architecture));
  EXPECT_EQ(942, architecture);
  EXPECT_TRUE(
      iree_hip_parse_gcn_arch_name("gfx1100:sramecc+:xnack-", &architecture));
  EXPECT_EQ(1100, architecture);
  EXPECT_TRUE(iree_hip_parse_gcn_arch_name("gfx90a:xnack+", &architecture));
  EXPECT_EQ(910, architecture);
}

TEST(HipDevicePropertiesTest, RejectsMalformedAndOutOfRangeTargets) {
  int architecture = -1;
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name(nullptr, &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx11x0", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx2147483648", &architecture));
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx942:unknown+", &architecture));
  EXPECT_EQ(-1, architecture);
  EXPECT_FALSE(iree_hip_parse_gcn_arch_name("gfx942", nullptr));
}

TEST(HipDevicePropertiesTest, AmdUuidMatchesRocrHexCharacters) {
  const iree_hal_uuid_t source = {
      {0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10}};
  uint8_t bytes[16] = {};
  iree_hip_format_device_uuid(IREE_SV("amd"), &source, bytes);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes), sizeof(bytes)),
            "fedcba9876543210");
  const iree_hal_uuid_t leading_zero = {{0x01, 0x23}};
  iree_hip_format_device_uuid(IREE_SV("amd"), &leading_zero, bytes);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes), sizeof(bytes)),
            "0123000000000000");
}

TEST(HipDevicePropertiesTest, OtherProvidersPreserveOpaqueUuidBytes) {
  const iree_hal_uuid_t source = {{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd,
                                   0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                                   0x32, 0x10}};
  uint8_t bytes[16] = {};
  iree_hip_format_device_uuid(IREE_SV("vulkan"), &source, bytes);
  EXPECT_EQ(std::memcmp(bytes, source.bytes, sizeof(bytes)), 0);
}

}  // namespace
