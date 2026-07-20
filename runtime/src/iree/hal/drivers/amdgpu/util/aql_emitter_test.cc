// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hsa_signal_t MakeSignal(uint64_t handle) {
  iree_hsa_signal_t signal = {};
  signal.handle = handle;
  return signal;
}

static iree_hal_amdgpu_aql_dispatch_params_t MakeExtendedDispatchParams() {
  iree_hal_amdgpu_aql_dispatch_params_t params = {};
  params.kernel_object = 0x0102030405060708ull;
  params.kernarg_address = reinterpret_cast<const void*>(
      static_cast<uintptr_t>(0x1112131415161718ull));
  params.workgroup_size[0] = 64;
  params.workgroup_size[1] = 2;
  params.workgroup_size[2] = 1;
  params.workgroup_count[0] = 10;
  params.workgroup_count[1] = 21;
  params.workgroup_count[2] = 36;
  params.workgroup_cluster_size[0] = 2;
  params.workgroup_cluster_size[1] = 3;
  params.workgroup_cluster_size[2] = 4;
  params.private_segment_size = 0x11223344u;
  params.group_segment_size = 0x55667788u;
  params.packet_control = iree_hal_amdgpu_aql_packet_control_barrier(
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_SYSTEM);
  params.completion_signal = MakeSignal(0x2122232425262728ull);
  return params;
}

TEST(AQLEmitterTest, OrdinaryDispatchGoldenBytesRemainStable) {
  iree_hsa_kernel_dispatch_packet_t packet;
  std::memset(&packet, 0xCC, sizeof(packet));

  const uint16_t workgroup_size[3] = {64, 2, 1};
  const uint32_t grid_size[3] = {640, 14, 9};
  const iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_SYSTEM);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_dispatch(
      &packet, /*kernel_object=*/0x0102030405060708ull,
      reinterpret_cast<const void*>(
          static_cast<uintptr_t>(0x1112131415161718ull)),
      workgroup_size, grid_size, /*private_segment_size=*/0x11223344u,
      /*group_segment_size=*/0x55667788u, packet_control,
      MakeSignal(0x2122232425262728ull), &setup);

  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH, packet_control));
  EXPECT_EQ(setup, 3u);
  const std::array<uint8_t, 64> expected = {
      0xCC, 0xCC, 0xCC, 0xCC, 0x40, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00,
      0x00, 0x80, 0x02, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x09, 0x00,
      0x00, 0x00, 0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0x08,
      0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15,
      0x14, 0x13, 0x12, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,
  };
  EXPECT_EQ(std::memcmp(&packet, expected.data(), expected.size()), 0);
}

TEST(AQLEmitterTest, EmitsExtendedDispatchPacketBody) {
  iree_hsa_amd_ext_kernel_dispatch_packet_t packet;
  std::memset(&packet, 0xCC, sizeof(packet));
  const iree_hal_amdgpu_aql_dispatch_params_t params =
      MakeExtendedDispatchParams();

  uint16_t header = 0;
  uint16_t setup = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_emit_extended_dispatch(&packet, &params,
                                                            &header, &setup));

  EXPECT_EQ(header,
            iree_hal_amdgpu_aql_make_header(
                IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, params.packet_control));
  EXPECT_EQ(setup, 0x0303u);
  uint32_t first_dword = 0;
  std::memcpy(&first_dword, &packet, sizeof(first_dword));
  EXPECT_EQ(first_dword, 0xCCCCCCCCu);
  EXPECT_EQ(packet.workgroup_size[0], 64u);
  EXPECT_EQ(packet.workgroup_size[1], 2u);
  EXPECT_EQ(packet.workgroup_size[2], 1u);
  EXPECT_EQ(packet.reserved0, 0u);
  EXPECT_EQ(packet.cluster_count_x, 5u);
  EXPECT_EQ(packet.cluster_count_y, 7u);
  EXPECT_EQ(packet.cluster_count_z, 9u);
  EXPECT_EQ(packet.cluster_size[0], 2u);
  EXPECT_EQ(packet.cluster_size[1], 3u);
  EXPECT_EQ(packet.cluster_size[2], 4u);
  EXPECT_EQ(packet.perf_hint, 0u);
  EXPECT_EQ(packet.private_segment_size, 0x11223344u);
  EXPECT_EQ(packet.group_segment_size, 0x55667788u);
  EXPECT_EQ(packet.kernel_object, 0x0102030405060708ull);
  EXPECT_EQ(packet.kernarg_address, params.kernarg_address);
  EXPECT_EQ(packet.dep_signal.handle, 0u);
  EXPECT_EQ(packet.completion_signal.handle, 0x2122232425262728ull);

  const std::array<uint8_t, 64> expected = {
      0xCC, 0xCC, 0xCC, 0xCC, 0x40, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00,
      0x00, 0x05, 0x00, 0x00, 0x00, 0x07, 0x00, 0x09, 0x00, 0x02, 0x03,
      0x04, 0x00, 0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0x08,
      0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15,
      0x14, 0x13, 0x12, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21,
  };
  EXPECT_EQ(std::memcmp(&packet, expected.data(), expected.size()), 0);
}

TEST(AQLEmitterTest, SelectsOrdinaryDispatchWithoutChangingPacketBytes) {
  iree_hal_amdgpu_aql_dispatch_params_t params = MakeExtendedDispatchParams();
  params.workgroup_cluster_size[0] = 0;
  params.workgroup_cluster_size[1] = 0;
  params.workgroup_cluster_size[2] = 0;

  union {
    iree_hsa_kernel_dispatch_packet_t ordinary;
    iree_hsa_amd_ext_kernel_dispatch_packet_t extended;
  } selected_packet;
  std::memset(&selected_packet, 0xCC, sizeof(selected_packet));
  uint16_t selected_header = 0;
  uint16_t selected_setup = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_emit_dispatch_packet(
      &selected_packet.ordinary, &selected_packet.extended, &params,
      &selected_header, &selected_setup));

  iree_hsa_kernel_dispatch_packet_t direct_packet;
  std::memset(&direct_packet, 0xCC, sizeof(direct_packet));
  const uint32_t grid_size[3] = {
      params.workgroup_count[0] * params.workgroup_size[0],
      params.workgroup_count[1] * params.workgroup_size[1],
      params.workgroup_count[2] * params.workgroup_size[2],
  };
  uint16_t direct_setup = 0;
  const uint16_t direct_header = iree_hal_amdgpu_aql_emit_dispatch(
      &direct_packet, params.kernel_object, params.kernarg_address,
      params.workgroup_size, grid_size, params.private_segment_size,
      params.group_segment_size, params.packet_control,
      params.completion_signal, &direct_setup);

  EXPECT_EQ(selected_header, direct_header);
  EXPECT_EQ(selected_setup, direct_setup);
  EXPECT_EQ(std::memcmp(&selected_packet.ordinary, &direct_packet,
                        sizeof(direct_packet)),
            0);
}

TEST(AQLEmitterTest, SelectsExtendedDispatchWithoutChangingPacketBytes) {
  const iree_hal_amdgpu_aql_dispatch_params_t params =
      MakeExtendedDispatchParams();
  union {
    iree_hsa_kernel_dispatch_packet_t ordinary;
    iree_hsa_amd_ext_kernel_dispatch_packet_t extended;
  } selected_packet;
  std::memset(&selected_packet, 0xCC, sizeof(selected_packet));
  uint16_t selected_header = 0;
  uint16_t selected_setup = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_emit_dispatch_packet(
      &selected_packet.ordinary, &selected_packet.extended, &params,
      &selected_header, &selected_setup));

  iree_hsa_amd_ext_kernel_dispatch_packet_t direct_packet;
  std::memset(&direct_packet, 0xCC, sizeof(direct_packet));
  uint16_t direct_header = 0;
  uint16_t direct_setup = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_emit_extended_dispatch(
      &direct_packet, &params, &direct_header, &direct_setup));

  EXPECT_EQ(selected_header, direct_header);
  EXPECT_EQ(selected_setup, direct_setup);
  EXPECT_EQ(std::memcmp(&selected_packet.extended, &direct_packet,
                        sizeof(direct_packet)),
            0);
}

TEST(AQLEmitterTest, OrdinaryNoopDispatchRemainsValid) {
  iree_hal_amdgpu_aql_dispatch_params_t params = MakeExtendedDispatchParams();
  params.workgroup_count[0] = 0;
  params.workgroup_count[1] = 0;
  params.workgroup_count[2] = 0;
  params.workgroup_cluster_size[0] = 0;
  params.workgroup_cluster_size[1] = 0;
  params.workgroup_cluster_size[2] = 0;
  IREE_EXPECT_OK(
      iree_hal_amdgpu_aql_validate_dispatch_params(&params, nullptr));
}

enum class InvalidExtendedDispatchVariant {
  kZeroWorkgroupSize,
  kZeroWorkgroupCount,
  kZeroClusterSize,
  kWideClusterSize,
  kTrivialClusterSize,
  kNondivisibleWorkgroupCount,
  kWideYClusterCount,
  kWideZClusterCount,
  kWideWorkItemGrid,
};

TEST(AQLEmitterTest, RejectsInvalidExtendedDispatchGeometryBeforeMutation) {
  const InvalidExtendedDispatchVariant variants[] = {
      InvalidExtendedDispatchVariant::kZeroWorkgroupSize,
      InvalidExtendedDispatchVariant::kZeroWorkgroupCount,
      InvalidExtendedDispatchVariant::kZeroClusterSize,
      InvalidExtendedDispatchVariant::kWideClusterSize,
      InvalidExtendedDispatchVariant::kTrivialClusterSize,
      InvalidExtendedDispatchVariant::kNondivisibleWorkgroupCount,
      InvalidExtendedDispatchVariant::kWideYClusterCount,
      InvalidExtendedDispatchVariant::kWideZClusterCount,
      InvalidExtendedDispatchVariant::kWideWorkItemGrid,
  };
  for (const InvalidExtendedDispatchVariant variant : variants) {
    iree_hal_amdgpu_aql_dispatch_params_t params = MakeExtendedDispatchParams();
    iree_status_code_t expected_code = IREE_STATUS_INVALID_ARGUMENT;
    switch (variant) {
      case InvalidExtendedDispatchVariant::kZeroWorkgroupSize:
        params.workgroup_size[0] = 0;
        break;
      case InvalidExtendedDispatchVariant::kZeroWorkgroupCount:
        params.workgroup_count[0] = 0;
        break;
      case InvalidExtendedDispatchVariant::kZeroClusterSize:
        params.workgroup_cluster_size[0] = 0;
        break;
      case InvalidExtendedDispatchVariant::kWideClusterSize:
        params.workgroup_cluster_size[0] = 256;
        params.workgroup_count[0] = 512;
        break;
      case InvalidExtendedDispatchVariant::kTrivialClusterSize:
        params.workgroup_cluster_size[0] = 1;
        params.workgroup_cluster_size[1] = 1;
        params.workgroup_cluster_size[2] = 1;
        break;
      case InvalidExtendedDispatchVariant::kNondivisibleWorkgroupCount:
        params.workgroup_count[0] = 11;
        break;
      case InvalidExtendedDispatchVariant::kWideYClusterCount:
        params.workgroup_count[1] = (UINT16_MAX + 1u) * 3u;
        expected_code = IREE_STATUS_OUT_OF_RANGE;
        break;
      case InvalidExtendedDispatchVariant::kWideZClusterCount:
        params.workgroup_count[2] = (UINT16_MAX + 1u) * 4u;
        expected_code = IREE_STATUS_OUT_OF_RANGE;
        break;
      case InvalidExtendedDispatchVariant::kWideWorkItemGrid:
        params.workgroup_size[0] = UINT16_MAX;
        params.workgroup_count[0] = UINT32_MAX - 1u;
        expected_code = IREE_STATUS_OUT_OF_RANGE;
        break;
    }

    iree_hsa_amd_ext_kernel_dispatch_packet_t packet;
    std::memset(&packet, 0xCC, sizeof(packet));
    const iree_hsa_amd_ext_kernel_dispatch_packet_t original_packet = packet;
    uint16_t header = UINT16_MAX;
    uint16_t setup = UINT16_MAX;
    IREE_EXPECT_STATUS_IS(expected_code,
                          iree_hal_amdgpu_aql_emit_extended_dispatch(
                              &packet, &params, &header, &setup));
    EXPECT_EQ(header, 0u);
    EXPECT_EQ(setup, 0u);
    EXPECT_EQ(std::memcmp(&packet, &original_packet, sizeof(packet)), 0);
  }
}

TEST(AQLEmitterTest, EmitsBarrierValuePacketBody) {
  iree_hsa_amd_barrier_value_packet_t packet;
  std::memset(&packet, 0xCC, sizeof(packet));

  const iree_hal_amdgpu_aql_packet_control_t packet_control =
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_SYSTEM);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_value(
      &packet, MakeSignal(0x123456789ABCDEF0ull), IREE_HSA_SIGNAL_CONDITION_LT,
      /*compare_value=*/0x1122334455667788ll,
      /*mask=*/0x7FFFFFFFFFFFFFFFll, packet_control,
      MakeSignal(0x0FEDCBA987654320ull), &setup);

  EXPECT_EQ(header, iree_hal_amdgpu_aql_make_header(
                        IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC, packet_control));
  EXPECT_EQ(setup, IREE_HSA_AMD_AQL_FORMAT_BARRIER_VALUE);

  uint32_t first_dword = 0;
  std::memcpy(&first_dword, &packet, sizeof(first_dword));
  EXPECT_EQ(first_dword, 0xCCCCCCCCu);
  EXPECT_EQ(packet.reserved0, 0u);
  EXPECT_EQ(packet.signal.handle, 0x123456789ABCDEF0ull);
  EXPECT_EQ(packet.value, 0x1122334455667788ll);
  EXPECT_EQ(packet.mask, 0x7FFFFFFFFFFFFFFFll);
  EXPECT_EQ(packet.cond, static_cast<iree_hsa_signal_condition32_t>(
                             IREE_HSA_SIGNAL_CONDITION_LT));
  EXPECT_EQ(packet.reserved1, 0u);
  EXPECT_EQ(packet.reserved2, 0u);
  EXPECT_EQ(packet.reserved3, 0u);
  EXPECT_EQ(packet.completion_signal.handle, 0x0FEDCBA987654320ull);
}

}  // namespace
