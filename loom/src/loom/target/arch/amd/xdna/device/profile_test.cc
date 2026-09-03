// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/device/profile.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(XdnaDeviceProfileTest, ResolvesExactStrixHaloPciRevision) {
  const loom_xdna_device_profile_t* profile = nullptr;
  IREE_ASSERT_OK(
      loom_xdna_device_profile_resolve_pci(0x1022, 0x17F0, 0x11, &profile));
  ASSERT_NE(profile, nullptr);
  EXPECT_STREQ(profile->key, "amd.xdna.strix_halo.17f0_11");
  EXPECT_STREQ(profile->display_name, "NPU Strix Halo");
  EXPECT_EQ(profile->identity, UINT64_C(0x535848414C4F0001));
  EXPECT_EQ(profile->firmware_abi_identity, UINT64_C(0x4E5055320006000C));
  EXPECT_EQ(profile->firmware_protocol_major, 6u);
  EXPECT_EQ(profile->firmware_protocol_minor, 12u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_xdna_device_profile_resolve_pci(0x1022, 0x17F0, 0x10, &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(XdnaDeviceProfileTest, ResolvesSerializedIdentityWithoutFallback) {
  const loom_xdna_device_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_xdna_device_profile_resolve_identity(
      UINT64_C(0x535848414C4F0001), 1, UINT64_C(0x4E5055320006000C), &profile));
  EXPECT_EQ(loom_xdna_device_profile_array_family(profile),
            loom_xdna_npu2_array_family());

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_xdna_device_profile_resolve_identity(
                            UINT64_C(0x535848414C4F0001), 2,
                            UINT64_C(0x4E5055320006000C), &profile));
}

TEST(XdnaDeviceProfileTest, AcceptsEveryContiguousPhysicalPartitionFit) {
  const loom_xdna_device_profile_t* profile = nullptr;
  IREE_ASSERT_OK(
      loom_xdna_device_profile_resolve_pci(0x1022, 0x17F0, 0x11, &profile));
  for (uint16_t width = 1; width <= 8; ++width) {
    for (uint16_t origin = 0; origin + width <= 8; ++origin) {
      IREE_EXPECT_OK(
          loom_xdna_device_profile_validate_partition(profile, origin, width));
    }
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_device_profile_validate_partition(profile, 7, 2));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_xdna_device_profile_validate_partition(profile, 0, 0));
}

}  // namespace
