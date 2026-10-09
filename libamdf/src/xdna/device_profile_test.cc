// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/device_profile.h"

#include "gtest/gtest.h"

namespace {

amdf_endpoint_info_t MakeXdnaEndpointInfo(uint32_t device_id,
                                          uint32_t revision_id) {
  amdf_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
  info.engine_kind = AMDF_ENGINE_KIND_XDNA;
  info.pci.vendor_id = 0x1022u;
  info.pci.device_id = device_id;
  info.pci.revision_id = revision_id;
  return info;
}

TEST(XdnaDeviceProfileTest, SeparatesIdentityFromNativeGeometry) {
  struct TargetCase {
    // PCI revision selecting an AIE2P firmware implementation.
    uint32_t revision;
    // Canonical compiler target identity, independent of live geometry.
    const char* target_id;
    // Nominal accounting required by this target's native admission protocol.
    uint32_t operations_per_cycle;
  };
  constexpr TargetCase kCases[] = {
      {0x10u, "amd.xdna.strix.17f0_10", 16384},
      {0x11u, "amd.xdna.strix_halo.17f0_11", 2048},
      {0x20u, "amd.xdna.krackan.17f0_20", 16384},
  };
  for (const auto& target : kCases) {
    SCOPED_TRACE(target.target_id);
    const auto endpoint = MakeXdnaEndpointInfo(0x17F0u, target.revision);
    amdf_xdna_endpoint_info_t identity = {};
    ASSERT_TRUE(amdf_xdna_query_endpoint_info(&endpoint, &identity));
    EXPECT_EQ(identity.architecture, AMDF_XDNA_ARCHITECTURE_AIE2P);
    EXPECT_STREQ(identity.target_id, target.target_id);
    amdf_xdna_device_info_t info = {};
    amdf_xdna_device_profile_t profile = {};
    ASSERT_TRUE(
        amdf_xdna_device_profile_initialize(&endpoint, &info, &profile));
    EXPECT_EQ(profile.info, &info);
    // Construction has no native metadata yet and cannot invent array size.
    EXPECT_EQ(info.array.column_count, 0u);
    EXPECT_EQ(info.array.row_count, 0u);
    EXPECT_EQ(info.context.maximum_column_count, 0u);
    EXPECT_EQ(profile.rows.core_count, 0u);
    EXPECT_NE(profile.execution_capabilities, 0u);
    ASSERT_NE(profile.bootstrap, nullptr);
    EXPECT_EQ(profile.bootstrap->context.operations_per_cycle,
              target.operations_per_cycle);
    EXPECT_GT(info.instruction.maximum_byte_length, 0u);
    EXPECT_GT(info.instruction.address_alignment, 0u);
  }
}

TEST(XdnaDeviceProfileTest, IdentityDoesNotPromiseExecution) {
  const auto endpoint = MakeXdnaEndpointInfo(0x1502u, 0x00u);
  amdf_xdna_endpoint_info_t identity = {};
  ASSERT_TRUE(amdf_xdna_query_endpoint_info(&endpoint, &identity));
  EXPECT_EQ(identity.architecture, AMDF_XDNA_ARCHITECTURE_AIE2);
  EXPECT_STREQ(identity.target_id, "amd.xdna.phoenix.1502_00");
  amdf_xdna_device_info_t info = {};
  amdf_xdna_device_profile_t profile = {};
  ASSERT_TRUE(amdf_xdna_device_profile_initialize(&endpoint, &info, &profile));
  EXPECT_EQ(profile.execution_capabilities, 0u);
  EXPECT_EQ(profile.bootstrap, nullptr);
  EXPECT_EQ(info.instruction.maximum_byte_length, 0u);
}

TEST(XdnaDeviceProfileTest, UnknownIdentityLeavesOutputsUnchanged) {
  for (uint32_t device_id : {0x17F0u, 0x17F1u}) {
    const auto endpoint = MakeXdnaEndpointInfo(device_id, 0x12u);
    amdf_xdna_endpoint_info_t identity = {.architecture = UINT32_MAX};
    EXPECT_FALSE(amdf_xdna_query_endpoint_info(&endpoint, &identity));
    EXPECT_EQ(identity.architecture, UINT32_MAX);
    amdf_xdna_device_info_t info = {.reset_epoch = UINT64_MAX};
    amdf_xdna_device_profile_t profile = {.execution_capabilities = UINT64_MAX};
    EXPECT_FALSE(
        amdf_xdna_device_profile_initialize(&endpoint, &info, &profile));
    EXPECT_EQ(info.reset_epoch, UINT64_MAX);
    EXPECT_EQ(profile.execution_capabilities, UINT64_MAX);
  }
}

TEST(XdnaDeviceProfileTest, RejectsWrongEngineAndVendor) {
  auto endpoint = MakeXdnaEndpointInfo(0x17F0u, 0x11u);
  amdf_xdna_endpoint_info_t identity = {};
  endpoint.engine_kind = AMDF_ENGINE_KIND_GPU;
  EXPECT_FALSE(amdf_xdna_query_endpoint_info(&endpoint, &identity));
  endpoint.engine_kind = AMDF_ENGINE_KIND_XDNA;
  endpoint.pci.vendor_id = 0x1002u;
  EXPECT_FALSE(amdf_xdna_query_endpoint_info(&endpoint, &identity));
}

}  // namespace
