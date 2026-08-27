// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/tsan.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_device_kernel_args_t MakeQueueInitializeKernelArgs() {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = 0x12345678ull;
  kernel_args.setup = 1;
  kernel_args.workgroup_size[0] = 32;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.kernarg_alignment = 16;
  return kernel_args;
}

static iree_hal_amdgpu_tsan_queue_initialize_args_t MakeQueueInitializeArgs(
    uint64_t shadow_size) {
  iree_hal_amdgpu_tsan_queue_initialize_args_t initialize_args = {};
  initialize_args.queue_state =
      reinterpret_cast<iree_hal_amdgpu_tsan_queue_state_t*>(0x1000);
  initialize_args.shadow_base = reinterpret_cast<void*>(0x2000);
  initialize_args.shadow_size = shadow_size;
  return initialize_args;
}

TEST(TsanTest, QueueInitializeUsesMinimumClearGrid) {
  const iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeQueueInitializeKernelArgs();
  const iree_hal_amdgpu_tsan_queue_initialize_args_t initialize_args =
      MakeQueueInitializeArgs(/*shadow_size=*/65);
  iree_hsa_kernel_dispatch_packet_t packet = {};
  alignas(16)
      std::array<uint8_t, sizeof(iree_hal_amdgpu_tsan_queue_initialize_args_t)>
          kernargs = {};

  iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
      &kernel_args, &initialize_args, /*max_workgroup_count=*/384, &packet,
      kernargs.data());

  const auto* args =
      reinterpret_cast<const iree_hal_amdgpu_tsan_queue_initialize_args_t*>(
          kernargs.data());
  EXPECT_EQ(packet.grid_size[0], 96u);
  EXPECT_EQ(args->clear_workgroup_size, 32u);
  EXPECT_EQ(args->clear_byte_stride, 96u);
}

TEST(TsanTest, QueueInitializeCapsLargeClearToResidentWork) {
  const iree_hal_amdgpu_device_kernel_args_t kernel_args =
      MakeQueueInitializeKernelArgs();
  const iree_hal_amdgpu_tsan_queue_initialize_args_t initialize_args =
      MakeQueueInitializeArgs(/*shadow_size=*/0x20008000ull);
  iree_hsa_kernel_dispatch_packet_t packet = {};
  alignas(16)
      std::array<uint8_t, sizeof(iree_hal_amdgpu_tsan_queue_initialize_args_t)>
          kernargs = {};

  iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
      &kernel_args, &initialize_args, /*max_workgroup_count=*/384, &packet,
      kernargs.data());

  const auto* args =
      reinterpret_cast<const iree_hal_amdgpu_tsan_queue_initialize_args_t*>(
          kernargs.data());
  EXPECT_EQ(packet.grid_size[0], 12288u);
  EXPECT_EQ(args->clear_workgroup_size, 32u);
  EXPECT_EQ(args->clear_byte_stride, 12288u);
  EXPECT_LT(args->clear_byte_stride, args->shadow_size);
}

}  // namespace
}  // namespace iree::hal::amdgpu
