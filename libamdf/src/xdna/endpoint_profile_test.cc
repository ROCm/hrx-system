// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/endpoint_profile.h"

#include <cstring>
#include <iomanip>

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

TEST(XdnaEndpointProfileTest, SelectsPublishedStaticProfiles) {
  struct ProfileCase {
    // PCI device identifier.
    uint32_t device_id;
    // PCI revision identifier.
    uint32_t revision_id;
    // Expected AIE architecture.
    amdf_xdna_architecture_t architecture;
    // First physical column owned by the endpoint.
    uint32_t column_origin;
    // Number of addressable columns.
    uint32_t column_count;
    // Supported context scheduling modes.
    amdf_xdna_scheduling_modes_t scheduling_modes;
    // Maximum concurrently live contexts.
    uint32_t maximum_live_context_count;
    // Maximum hardware-resident contexts.
    uint32_t maximum_hardware_context_count;
    // Whether an instruction submission contract is qualified.
    bool supports_execution;
    // Stable compiler target identifier.
    const char* target_id;
  };
  constexpr ProfileCase kCases[] = {
      {0x1502u, 0x00u, AMDF_XDNA_ARCHITECTURE_AIE2, 1u, 4u,
       AMDF_XDNA_SCHEDULING_MODE_SPATIAL |
           AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
       6u, 6u, false, "amd.xdna.phoenix.1502_00"},
      {0x17F0u, 0x10u, AMDF_XDNA_ARCHITECTURE_AIE2P, 0u, 8u,
       AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED, 32u, 16u, true,
       "amd.xdna.strix.17f0_10"},
      {0x17F0u, 0x11u, AMDF_XDNA_ARCHITECTURE_AIE2P, 0u, 8u,
       AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED, 32u, 16u, true,
       "amd.xdna.strix_halo.17f0_11"},
      {0x17F0u, 0x20u, AMDF_XDNA_ARCHITECTURE_AIE2P, 0u, 8u,
       AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED, 32u, 16u, false,
       "amd.xdna.krackan.17f0_20"},
  };

  for (const ProfileCase& test_case : kCases) {
    SCOPED_TRACE(testing::Message()
                 << "device=" << std::hex << test_case.device_id
                 << " revision=" << test_case.revision_id);
    const amdf_endpoint_info_t info =
        MakeXdnaEndpointInfo(test_case.device_id, test_case.revision_id);
    const amdf_xdna_endpoint_profile_t* profile =
        amdf_xdna_endpoint_profile_select(&info);

    ASSERT_NE(profile, nullptr);
    const amdf_xdna_endpoint_info_t* profile_info =
        amdf_xdna_endpoint_profile_get_info(profile);
    EXPECT_EQ(profile_info->type, AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO);
    EXPECT_EQ(profile_info->structure_size, sizeof(*profile_info));
    EXPECT_EQ(profile_info->next, nullptr);
    EXPECT_EQ(profile_info->architecture, test_case.architecture);
    EXPECT_EQ(profile_info->array.column_origin, test_case.column_origin);
    EXPECT_EQ(profile_info->array.column_count, test_case.column_count);
    EXPECT_EQ(profile_info->array.row_count, 6u);
    EXPECT_EQ(profile_info->array.column_stride, UINT64_C(1) << 25);
    EXPECT_EQ(profile_info->context.scheduling_modes,
              test_case.scheduling_modes);
    EXPECT_EQ(profile_info->context.minimum_column_count, 1u);
    EXPECT_EQ(profile_info->context.maximum_column_count,
              test_case.column_count);
    EXPECT_EQ(profile_info->context.column_count_granularity, 1u);
    EXPECT_EQ(profile_info->context.maximum_live_context_count,
              test_case.maximum_live_context_count);
    EXPECT_EQ(profile_info->context.maximum_hardware_context_count,
              test_case.maximum_hardware_context_count);
    if (!test_case.supports_execution) {
      EXPECT_EQ(profile->execution_capabilities, 0u);
      EXPECT_EQ(profile->bootstrap, nullptr);
      EXPECT_EQ(profile->firmware_heap_byte_length, 0u);
      EXPECT_EQ(profile_info->instruction.maximum_byte_length, 0u);
      EXPECT_EQ(profile_info->instruction.format.format,
                AMDF_XDNA_BINARY_FORMAT_UNKNOWN);
    } else {
      EXPECT_NE(profile->execution_capabilities, 0u);
      ASSERT_NE(profile->bootstrap, nullptr);
      EXPECT_NE(profile->bootstrap->pdi_bytes, nullptr);
      EXPECT_GT(profile->bootstrap->pdi_byte_length, 0u);
      EXPECT_EQ(profile->firmware_heap_byte_length, 64u * 1024u * 1024u);
      EXPECT_EQ(profile->transaction.device_generation, 4u);
      EXPECT_EQ(profile->rows.shim_origin, 0u);
      EXPECT_EQ(profile->rows.shim_count, 1u);
      EXPECT_EQ(profile->rows.memory_origin, 1u);
      EXPECT_EQ(profile->rows.memory_count, 1u);
      EXPECT_EQ(profile->rows.core_origin, 2u);
      EXPECT_EQ(profile->rows.core_count, 4u);
      EXPECT_EQ(profile_info->instruction.maximum_byte_length,
                UINT32_MAX & ~uint64_t{3});
      EXPECT_EQ(profile_info->instruction.address_alignment, 32u * 1024u);
      EXPECT_EQ(profile_info->instruction.byte_length_granularity, 4u);
      EXPECT_EQ(profile_info->instruction.format.format,
                AMDF_XDNA_BINARY_FORMAT_TRANSACTION);
      EXPECT_EQ(profile_info->instruction.format.version,
                AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1);
    }
    EXPECT_STREQ(profile_info->target_id, test_case.target_id);
  }
}

TEST(XdnaEndpointProfileTest, RejectsXdnaIdentityWithoutStaticProfile) {
  const amdf_endpoint_info_t info = MakeXdnaEndpointInfo(0x17F1u, 0x10u);

  EXPECT_EQ(amdf_xdna_endpoint_profile_select(&info), nullptr);
}

TEST(XdnaEndpointProfileTest, RejectsUnknownPciIdentity) {
  const amdf_endpoint_info_t info = MakeXdnaEndpointInfo(0x17F0u, 0x12u);

  EXPECT_EQ(amdf_xdna_endpoint_profile_select(&info), nullptr);
}

TEST(XdnaEndpointProfileTest, RejectsWrongEngineKind) {
  amdf_endpoint_info_t info = MakeXdnaEndpointInfo(0x17F0u, 0x11u);
  info.engine_kind = AMDF_ENGINE_KIND_GPU;

  EXPECT_EQ(amdf_xdna_endpoint_profile_select(&info), nullptr);
}

TEST(XdnaEndpointProfileTest, RejectsWrongVendor) {
  amdf_endpoint_info_t info = MakeXdnaEndpointInfo(0x17F0u, 0x11u);
  info.pci.vendor_id = 0x1002u;

  EXPECT_EQ(amdf_xdna_endpoint_profile_select(&info), nullptr);
}

}  // namespace
