// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint64_t kernel_object, uint16_t kernarg_size, uint16_t kernarg_alignment) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = kernel_object;
  kernel_args.setup = 3;
  kernel_args.workgroup_size[0] = 4;
  kernel_args.workgroup_size[1] = 5;
  kernel_args.workgroup_size[2] = 6;
  kernel_args.private_segment_size = 7;
  kernel_args.group_segment_size = 8;
  kernel_args.kernarg_size = kernarg_size;
  kernel_args.kernarg_alignment = kernarg_alignment;
  return kernel_args;
}

TEST(DispatchTest, EmplacePacketPreservesZeroWorkgroupCounts) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeKernelArgs(/*kernel_object=*/0xBEEFu, /*kernarg_size=*/0,
                     /*kernarg_alignment=*/16);
  iree_hsa_kernel_dispatch_packet_t packet = {};
  packet.header = 0xFFFFu;
  alignas(16) std::array<uint8_t, 64> kernargs = {};
  const uint32_t workgroup_count[3] = {0, 2, 0};

  iree_hal_amdgpu_device_dispatch_emplace_packet(
      &kernel_args, workgroup_count,
      /*dynamic_workgroup_local_memory=*/9, &packet, kernargs.data());

  EXPECT_EQ(packet.header, 0xFFFFu);
  EXPECT_EQ(packet.setup, 3u);
  EXPECT_EQ(packet.workgroup_size[0], 4u);
  EXPECT_EQ(packet.workgroup_size[1], 5u);
  EXPECT_EQ(packet.workgroup_size[2], 6u);
  EXPECT_EQ(packet.reserved0, 0u);
  EXPECT_EQ(packet.grid_size[0], 0u);
  EXPECT_EQ(packet.grid_size[1], 10u);
  EXPECT_EQ(packet.grid_size[2], 0u);
  EXPECT_EQ(packet.private_segment_size, 7u);
  EXPECT_EQ(packet.group_segment_size, 17u);
  EXPECT_EQ(packet.kernel_object, 0xBEEFu);
  EXPECT_EQ(packet.kernarg_address, kernargs.data());
  EXPECT_EQ(packet.reserved2, 0u);
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);
}

TEST(DispatchTest, EmplaceImplicitArgsWritesSuffix) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = MakeKernelArgs(
      /*kernel_object=*/0x1234u,
      /*kernarg_size=*/32 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      /*kernarg_alignment=*/16);
  iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {
      /*.explicit_kernarg_size=*/32,
      /*.implicit_args_offset=*/32,
      /*.total_kernarg_size=*/32 + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
      /*.has_implicit_args=*/true,
  };
  const uint32_t workgroup_count[3] = {7, 8, 9};
  alignas(16) std::array<uint8_t, 256> kernargs = {};
  kernargs.fill(0xFD);

  iree_hal_amdgpu_device_dispatch_emplace_implicit_args(
      &kernel_args, workgroup_count, /*dynamic_workgroup_local_memory=*/13,
      &layout, kernargs.data());

  EXPECT_EQ(kernargs[31], 0xFDu);

  const auto* implicit_args =
      reinterpret_cast<const iree_amdgpu_kernel_implicit_args_t*>(
          kernargs.data() + layout.implicit_args_offset);
  EXPECT_EQ(implicit_args->block_count[0], 7u);
  EXPECT_EQ(implicit_args->block_count[1], 8u);
  EXPECT_EQ(implicit_args->block_count[2], 9u);
  EXPECT_EQ(implicit_args->group_size[0], 4u);
  EXPECT_EQ(implicit_args->group_size[1], 5u);
  EXPECT_EQ(implicit_args->group_size[2], 6u);
  EXPECT_EQ(implicit_args->remainder[0], 0u);
  EXPECT_EQ(implicit_args->remainder[1], 0u);
  EXPECT_EQ(implicit_args->remainder[2], 0u);
  EXPECT_EQ(implicit_args->reserved0, 0u);
  EXPECT_EQ(implicit_args->reserved1, 0u);
  EXPECT_EQ(implicit_args->global_offset[0], 0u);
  EXPECT_EQ(implicit_args->global_offset[1], 0u);
  EXPECT_EQ(implicit_args->global_offset[2], 0u);
  EXPECT_EQ(implicit_args->grid_dims, 3u);
  EXPECT_EQ(implicit_args->printf_buffer, nullptr);
  EXPECT_EQ(implicit_args->hostcall_buffer, nullptr);
  EXPECT_EQ(implicit_args->deprecated_multigrid_sync_arg, 0u);
  EXPECT_EQ(implicit_args->unused_heap_v1, 0u);
  EXPECT_EQ(implicit_args->unused_default_queue, 0u);
  EXPECT_EQ(implicit_args->unused_completion_action, 0u);
  EXPECT_EQ(implicit_args->dynamic_lds_size, 13u);
}

TEST(DispatchTest, EmplaceCustomKernargsCopiesRawBlob) {
  iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout = {};
  const std::array<uint8_t, 20> custom_kernargs = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
      0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13,
  };
  alignas(16) std::array<uint8_t, 32> kernargs = {};
  kernargs.fill(0xFD);

  iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
      &layout, custom_kernargs.data(), custom_kernargs.size(), kernargs.data());

  EXPECT_EQ(std::memcmp(kernargs.data(), custom_kernargs.data(),
                        custom_kernargs.size()),
            0);
  for (size_t i = custom_kernargs.size(); i < kernargs.size(); ++i) {
    EXPECT_EQ(kernargs[i], 0xFD);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
