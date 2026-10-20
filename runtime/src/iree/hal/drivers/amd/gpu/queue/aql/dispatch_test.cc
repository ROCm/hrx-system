// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/dispatch.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class AqlDispatchTest : public ::testing::Test {
 protected:
  iree_status_t Prepare(
      iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags = 0,
      iree_const_byte_span_t constants = iree_const_byte_span_empty(),
      iree_host_size_t binding_count = 0) {
    return iree_hal_amd_gpu_aql_dispatch_prepare(
        &function_, 65536, config, constants, binding_count, flags, &dispatch_);
  }
  // Image-owned public constraints.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Prepared native argument representation.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Descriptor-based function with no PM4 register state.
  iree_hal_amd_gpu_aql_function_t function_ = {
      .base = {.metadata = &metadata_, .kernarg = &layout_},
      .kernel_object = UINT64_C(0x123456780040),
  };
  // Public admission output, shared with hidden argument packing.
  iree_hal_amd_gpu_dispatch_geometry_t dispatch_ = {};
};

TEST_F(AqlDispatchTest, ExactPacketCarriesNativeDescriptorAndResources) {
  static_assert(sizeof(iree_hal_amd_gpu_aql_function_t) == 32);
  metadata_.private_segment_fixed_size = 36;
  metadata_.group_segment_fixed_size = 128;
  // Native firmware supplies private storage, dispatch pointer, and queue ABI.
  function_.code_properties = 1u | 2u | 4u | 8u;
  auto config = iree_hal_make_static_dispatch_config(3, 2, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.dynamic_workgroup_local_memory = 384;
  IREE_ASSERT_OK(Prepare(config));
  auto packet = iree_hal_amd_gpu_aql_dispatch_packet(
      &function_, &dispatch_,
      IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
          IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
          IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT,
      0x567812340080, 0x987654320040, 0);
  const std::array<uint32_t, 16> expected = {
      0x00020d02, 0x00040020, 1,          96,     8, 1, 36,         512,
      0x56780040, 0x1234,     0x12340080, 0x5678, 0, 0, 0x54320040, 0x9876};
  EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);

  config.workitem_count[0] = UINT32_MAX;
  config.workitem_count[1] = 7;
  config.workitem_count[2] = 1;
  const auto exact = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, exact));
  packet = iree_hal_amd_gpu_aql_dispatch_packet(&function_, &dispatch_, 0,
                                                0x4000, 0, exact);
  EXPECT_EQ(packet.words[3], UINT32_MAX);
  EXPECT_EQ(packet.words[4], 7u);
  EXPECT_EQ(packet.words[5], 1u);
  EXPECT_EQ(packet.words[0], 0x00020002u);
}

TEST_F(AqlDispatchTest, PacketGridIsAlwaysBoundedInWorkitems) {
  auto config = iree_hal_make_static_dispatch_config(UINT32_MAX / 32, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  const auto saved = dispatch_;
  ++config.workgroup_count[0];
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  EXPECT_EQ(std::memcmp(&saved, &dispatch_, sizeof(saved)), 0);
}

TEST_F(AqlDispatchTest, FunctionRequirementsPreserveStrongerCallerScopes) {
  function_.header = IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                     IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT |
                     IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT;
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = config.workgroup_size[1] =
      config.workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  // Native scope encodings are 0/1/2; combining agent and system must never
  // produce the reserved encoding 3 in either fence field.
  const std::array<std::array<uint32_t, 2>, 5> headers = {{
      {0, 0x00010b02},
      {0x0a00, 0x00010b02},
      {0x1400, 0x00011502},
      {0x0c00, 0x00010d02},
      {0x1200, 0x00011302},
  }};
  for (const auto& header : headers) {
    const auto packet = iree_hal_amd_gpu_aql_dispatch_packet(
        &function_, &dispatch_, header[0], 0x4000, 0x8000, 0);
    const std::array<uint32_t, 16> expected = {
        header[1],  0x00010001, 1,      1, 1, 1, 0,      0,
        0x56780040, 0x1234,     0x4000, 0, 0, 0, 0x8000, 0};
    EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);
  }
}

TEST_F(AqlDispatchTest,
       EveryIndirectModeLeavesGeometryForDeviceMaterialization) {
  auto config =
      iree_hal_make_static_dispatch_config(UINT32_MAX, UINT32_MAX, UINT32_MAX);
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  layout_.byte_length = 32;
  layout_.native_byte_length = 8;
  layout_.constant_byte_length = 8;
  layout_.binding_count = 2;
  std::array<uint8_t, 32> constants = {};
  for (auto parameters : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                          IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    for (auto arguments : std::array<iree_hal_dispatch_flags_t, 4>{
             0, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
             IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_ARGUMENTS,
             IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_ARGUMENTS}) {
      const auto flags = parameters | arguments;
      const bool external = iree_hal_dispatch_uses_indirect_arguments(flags);
      const bool native =
          arguments == IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
      IREE_ASSERT_OK(Prepare(config, flags,
                             {constants.data(), external ? 0u : 8u},
                             external ? 1u
                             : native ? 0u
                                      : 2u));
      const auto packet = iree_hal_amd_gpu_aql_dispatch_packet(
          &function_, &dispatch_, 0, 0x4000, 0x8000, flags);
      const std::array<uint32_t, 16> expected = {
          0x00030002, 0x00010040, 1,      0, 0, 0, 0,      0,
          0x56780040, 0x1234,     0x4000, 0, 0, 0, 0x8000, 0};
      EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);
    }
  }
}

TEST_F(AqlDispatchTest, UnprovisionedServicesFailBeforePublication) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  for (uint32_t requirement = 1;
       requirement <= IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_QUEUE_POINTER;
       requirement <<= 1) {
    layout_.requirements = requirement;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  }
  layout_.requirements = 0;
  function_.code_properties = 1u << 11;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  function_.code_properties = 0;
  metadata_.has_workgroup_cluster_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
}

}  // namespace
