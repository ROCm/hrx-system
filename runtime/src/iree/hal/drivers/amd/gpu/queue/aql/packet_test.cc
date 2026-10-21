// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"

namespace {

TEST(PacketTest, DispatchPreservesNativeWidthsAndZerosReservedFields) {
  const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
      .header = 2u | (1u << 8) | (2u << 9) | (2u << 11) | (3u << 16),
      .workgroup_size = {256, 2, 1},
      .grid_size = {0x80000000, 67, 3},
      .private_segment_size = 96,
      .group_segment_size = 32768,
      .kernel_object = 0x1234fedcba40,
      .kernarg_address = 0x5678abcdef00,
      .completion_signal = 0x9abc12345600,
  };
  const auto packet = iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
  const std::array<uint32_t, 16> expected = {
      0x00031502, 0x00020100, 0x00000001, 0x80000000, 0x00000043, 0x00000003,
      0x00000060, 0x00008000, 0xfedcba40, 0x00001234, 0xabcdef00, 0x00005678,
      0x00000000, 0x00000000, 0x12345600, 0x00009abc,
  };
  EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);
}

TEST(PacketTest, FiveDependenciesAndCompleteFenceHeader) {
  const std::array<uint64_t, 5> dependencies = {0x100000000080, 0x2000000000c0,
                                                0x300000000100, 0x400000000140,
                                                0x500000000180};
  const auto packet = iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
          IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
          IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
          IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT,
      dependencies.size(), dependencies.data(), 0x6000000001c0);
  const std::array<uint32_t, 16> expected = {
      0x00000d03, 0,          0x00000080, 0x00001000, 0x000000c0, 0x00002000,
      0x00000100, 0x00003000, 0x00000140, 0x00004000, 0x00000180, 0x00005000,
      0,          0,          0x000001c0, 0x00006000,
  };
  EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);
}

TEST(PacketTest, EmptyBarrierHasNoImplicitScopesOrDependencies) {
  for (uint32_t header : {IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND,
                          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_OR}) {
    const auto packet =
        iree_hal_amd_gpu_aql_packet_barrier(header, 0, nullptr, 0);
    EXPECT_EQ(packet.words[0], header);
    for (size_t i = 1; i < 16; ++i) {
      EXPECT_EQ(packet.words[i], 0u);
    }
  }
}

TEST(PacketTest, SignalMatchesFirmwareAbiWithoutHostPayload) {
  EXPECT_EQ(sizeof(iree_hal_amd_gpu_aql_signal_t), 64u);
  EXPECT_EQ(alignof(iree_hal_amd_gpu_aql_signal_t), 64u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, value), 8u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, event_mailbox), 16u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, event_id), 24u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, start_timestamp), 32u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, end_timestamp), 40u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, queue_address), 48u);
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_aql_signal_t, reserved1), 56u);
}

TEST(PacketTest, VendorIndirectBufferCarriesNativeCompletion) {
  const auto packet = iree_hal_amd_gpu_aql_packet_indirect_buffer(
      UINT64_C(0x123456789a40), 7, UINT64_C(0x456789abc000));
  const std::array<uint32_t, 16> expected = {
      0x00010000, 0xc0023f00, 0x56789a40, 0x00001234, 0x00800007, 0x0000000a,
      0,          0,          0,          0,          0,          0,
      0,          0,          0x89abc000, 0x00004567,
  };
  EXPECT_EQ(std::memcmp(packet.words, expected.data(), sizeof(packet)), 0);
}

}  // namespace
