// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/atomic.h"

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

constexpr uint64_t kWaitX32KernelObject = 0xA032u;
constexpr uint64_t kWaitX64KernelObject = 0xA064u;
constexpr uint64_t kStoreX32KernelObject = 0xB032u;
constexpr uint64_t kStoreX64KernelObject = 0xB064u;
constexpr uint64_t kRmwX32KernelObject = 0xC032u;
constexpr uint64_t kRmwX64KernelObject = 0xC064u;

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint64_t kernel_object, uint16_t setup, uint16_t kernarg_size,
    uint16_t kernarg_alignment) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = kernel_object;
  kernel_args.setup = setup;
  kernel_args.workgroup_size[0] = 1;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.private_segment_size = 4;
  kernel_args.group_segment_size = 8;
  kernel_args.kernarg_size = kernarg_size;
  kernel_args.kernarg_alignment = kernarg_alignment;
  return kernel_args;
}

static iree_hal_amdgpu_device_kernels_t MakeKernels() {
  iree_hal_amdgpu_device_kernels_t kernels = {};
  kernels.iree_hal_amdgpu_device_atomic_wait_x32 = MakeKernelArgs(
      kWaitX32KernelObject, 1, IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_ALIGNMENT);
  kernels.iree_hal_amdgpu_device_atomic_wait_x64 = MakeKernelArgs(
      kWaitX64KernelObject, 2, IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_ALIGNMENT);
  kernels.iree_hal_amdgpu_device_atomic_store_x32 =
      MakeKernelArgs(kStoreX32KernelObject, 3,
                     IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_SIZE,
                     IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_ALIGNMENT);
  kernels.iree_hal_amdgpu_device_atomic_store_x64 =
      MakeKernelArgs(kStoreX64KernelObject, 4,
                     IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_SIZE,
                     IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_ALIGNMENT);
  kernels.iree_hal_amdgpu_device_atomic_rmw_x32 = MakeKernelArgs(
      kRmwX32KernelObject, 5, IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_ALIGNMENT);
  kernels.iree_hal_amdgpu_device_atomic_rmw_x64 = MakeKernelArgs(
      kRmwX64KernelObject, 6, IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_SIZE,
      IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_ALIGNMENT);
  return kernels;
}

static void ExpectOneWorkitemDispatch(
    const iree_hsa_kernel_dispatch_packet_t& packet, uint16_t setup,
    uint64_t kernel_object, const void* kernarg_ptr) {
  EXPECT_EQ(packet.setup, setup);
  EXPECT_EQ(packet.workgroup_size[0], 1);
  EXPECT_EQ(packet.workgroup_size[1], 1);
  EXPECT_EQ(packet.workgroup_size[2], 1);
  EXPECT_EQ(packet.grid_size[0], 1);
  EXPECT_EQ(packet.grid_size[1], 1);
  EXPECT_EQ(packet.grid_size[2], 1);
  EXPECT_EQ(packet.private_segment_size, 4u);
  EXPECT_EQ(packet.group_segment_size, 8u);
  EXPECT_EQ(packet.kernel_object, kernel_object);
  EXPECT_EQ(packet.kernarg_address, kernarg_ptr);
  EXPECT_EQ(packet.completion_signal.handle, iree_hsa_signal_null().handle);
}

TEST(AtomicTest, WaitX32NormalizesModeAndEmplacesDispatch) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_wait_kernargs_t kernargs = {};
  const iree_hal_atomic_wait_params_t params = {
      /*.value=*/0x12345678u,
      /*.mask=*/0xFFFF00FFu,
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
  };

  iree_hal_amdgpu_device_atomic_wait_emplace(
      &kernels, &packet, reinterpret_cast<const void*>(0x1000), params,
      &kernargs);

  ExpectOneWorkitemDispatch(packet, 1, kWaitX32KernelObject, &kernargs);
  EXPECT_EQ(kernargs.target_ptr, reinterpret_cast<const void*>(0x1000));
  EXPECT_EQ(kernargs.value, params.value);
  EXPECT_EQ(kernargs.mask, params.mask);
  EXPECT_EQ(kernargs.condition,
            IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_NOT_EQUAL);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE |
                               IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE);
}

TEST(AtomicTest, WaitX64SelectsWidth) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_wait_kernargs_t kernargs = {};
  const iree_hal_atomic_wait_params_t params = {
      /*.value=*/0x123456789ABCDEF0ull,
      /*.mask=*/UINT64_MAX,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
  };

  iree_hal_amdgpu_device_atomic_wait_emplace(
      &kernels, &packet, reinterpret_cast<const void*>(0x2000), params,
      &kernargs);

  ExpectOneWorkitemDispatch(packet, 2, kWaitX64KernelObject, &kernargs);
  EXPECT_EQ(
      kernargs.condition,
      IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_NONE);
}

TEST(AtomicTest, StoreX32NormalizesModeAndClearsPadding) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_store_kernargs_t kernargs = {};
  kernargs.reserved = UINT32_MAX;
  const iree_hal_atomic_store_params_t params = {
      /*.value=*/0x89ABCDEFu,
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };

  iree_hal_amdgpu_device_atomic_store_emplace(
      &kernels, &packet, reinterpret_cast<void*>(0x3000), params, &kernargs);

  ExpectOneWorkitemDispatch(packet, 3, kStoreX32KernelObject, &kernargs);
  EXPECT_EQ(kernargs.target_ptr, reinterpret_cast<void*>(0x3000));
  EXPECT_EQ(kernargs.value, params.value);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE |
                               IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE);
  EXPECT_EQ(kernargs.reserved, 0u);
}

TEST(AtomicTest, StoreX64SelectsWidth) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_store_kernargs_t kernargs = {};
  const iree_hal_atomic_store_params_t params = {
      /*.value=*/0xFEDCBA9876543210ull,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
  };

  iree_hal_amdgpu_device_atomic_store_emplace(
      &kernels, &packet, reinterpret_cast<void*>(0x4000), params, &kernargs);

  ExpectOneWorkitemDispatch(packet, 4, kStoreX64KernelObject, &kernargs);
  EXPECT_EQ(kernargs.value, params.value);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_NONE);
}

TEST(AtomicTest, RmwX32PreservesCompleteModeAndOperation) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_rmw_kernargs_t kernargs = {};
  const iree_hal_atomic_rmw_params_t params = {
      /*.operand=*/0x55AA55AAu,
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
  };

  iree_hal_amdgpu_device_atomic_rmw_emplace(
      &kernels, &packet, reinterpret_cast<void*>(0x5000), params, &kernargs);

  ExpectOneWorkitemDispatch(packet, 5, kRmwX32KernelObject, &kernargs);
  EXPECT_EQ(kernargs.target_ptr, reinterpret_cast<void*>(0x5000));
  EXPECT_EQ(kernargs.operand, params.operand);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_ACQUIRE |
                               IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_RELEASE |
                               IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_SYSTEM_SCOPE);
  EXPECT_EQ(kernargs.operation,
            IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_XOR);
}

TEST(AtomicTest, RmwX64SelectsWidth) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeKernels();
  iree_hsa_kernel_dispatch_packet_t packet = {};
  iree_hal_amdgpu_device_atomic_rmw_kernargs_t kernargs = {};
  const iree_hal_atomic_rmw_params_t params = {
      /*.operand=*/0x0123456789ABCDEFull,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
  };

  iree_hal_amdgpu_device_atomic_rmw_emplace(
      &kernels, &packet, reinterpret_cast<void*>(0x6000), params, &kernargs);

  ExpectOneWorkitemDispatch(packet, 6, kRmwX64KernelObject, &kernargs);
  EXPECT_EQ(kernargs.operation,
            IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_OPERATION_SUBTRACT);
  EXPECT_EQ(kernargs.mode, IREE_HAL_AMDGPU_DEVICE_ATOMIC_MODE_NONE);
}

}  // namespace
}  // namespace iree::hal::amdgpu
