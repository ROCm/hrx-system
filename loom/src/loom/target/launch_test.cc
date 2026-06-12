// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/launch.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_target_snapshot_t TestSnapshot() {
  return loom_target_snapshot_t{
      /*.name=*/IREE_SV("test-target"),
      /*.codegen_format=*/{},
      /*.artifact_format=*/{},
      /*.default_pointer_bitwidth=*/{},
      /*.index_bitwidth=*/{},
      /*.offset_bitwidth=*/{},
      /*.max_workgroup_size=*/{/*.x=*/1024, /*.y=*/1024, /*.z=*/1024},
      /*.max_flat_workgroup_size=*/1024,
      /*.max_workgroup_storage_bytes=*/{},
      /*.subgroup_size=*/{},
      /*.max_grid_size=*/{/*.x=*/4096, /*.y=*/2048, /*.z=*/1024},
      /*.max_flat_grid_size=*/8388608,
      /*.max_workgroup_count=*/{/*.x=*/4096, /*.y=*/2048, /*.z=*/1024},
  };
}

loom_target_hal_kernel_abi_t TestHalKernelAbi(
    loom_target_workgroup_size_t required_workgroup_size) {
  return loom_target_hal_kernel_abi_t{
      /*.required_workgroup_size=*/required_workgroup_size,
  };
}

TEST(TargetLaunchTest, AllowsDynamicRequiredWorkgroupSize) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/0, /*.y=*/0, /*.z=*/0});
  IREE_EXPECT_OK(
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsPartialRequiredWorkgroupSize) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/64, /*.y=*/1, /*.z=*/0});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsRequiredWorkgroupSizeDimensionAboveLimit) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/1025, /*.y=*/1, /*.z=*/1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsRequiredFlatWorkgroupSizeAboveLimit) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/33, /*.y=*/33, /*.z=*/1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RejectsRequiredFlatWorkgroupSizeOverflow) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  snapshot.max_workgroup_size = {/*.x=*/0, /*.y=*/0, /*.z=*/0};
  loom_target_hal_kernel_abi_t hal_kernel = TestHalKernelAbi(
      {/*.x=*/UINT32_MAX, /*.y=*/UINT32_MAX, /*.z=*/UINT32_MAX});

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, ComputesFlatWorkgroupSize) {
  uint32_t flat_size = 0;
  loom_target_workgroup_size_t concrete = {/*.x=*/64, /*.y=*/2, /*.z=*/1};
  EXPECT_TRUE(
      loom_target_workgroup_size_flat_product_u32(&concrete, &flat_size));
  EXPECT_EQ(flat_size, 128u);

  loom_target_workgroup_size_t unresolved = {/*.x=*/0, /*.y=*/0, /*.z=*/0};
  EXPECT_TRUE(
      loom_target_workgroup_size_flat_product_u32(&unresolved, &flat_size));
  EXPECT_EQ(flat_size, 0u);

  loom_target_workgroup_size_t overflowing = {
      /*.x=*/UINT32_MAX, /*.y=*/UINT32_MAX, /*.z=*/UINT32_MAX};
  EXPECT_FALSE(
      loom_target_workgroup_size_flat_product_u32(&overflowing, &flat_size));
}

TEST(TargetLaunchTest, ValidatesSelectedFlatWorkgroupRange) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/64, /*.y=*/2, /*.z=*/1});
  hal_kernel.flat_workgroup_size_min = 64;
  hal_kernel.flat_workgroup_size_max = 256;
  IREE_EXPECT_OK(
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));

  hal_kernel.flat_workgroup_size_min = 256;
  hal_kernel.flat_workgroup_size_max = 512;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_validate_hal_kernel_launch(&snapshot, &hal_kernel));
}

TEST(TargetLaunchTest, RequiresConcreteLaunchForConcreteConsumers) {
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/0, /*.y=*/0, /*.z=*/0});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_target_require_concrete_hal_kernel_launch(
                            &hal_kernel, IREE_SV("test consumer")));

  hal_kernel.required_workgroup_size =
      loom_target_workgroup_size_t{/*.x=*/64, /*.y=*/1, /*.z=*/0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_require_concrete_hal_kernel_launch(
                            &hal_kernel, IREE_SV("test consumer")));

  hal_kernel.required_workgroup_size =
      loom_target_workgroup_size_t{/*.x=*/64, /*.y=*/1, /*.z=*/1};
  IREE_EXPECT_OK(loom_target_require_concrete_hal_kernel_launch(
      &hal_kernel, IREE_SV("test consumer")));
}

TEST(TargetLaunchTest, ValidatesDispatchCountAgainstCountLimits) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/0, /*.y=*/0, /*.z=*/0});
  loom_target_dispatch_workgroup_count_t valid = {
      /*.x=*/4096,
      /*.y=*/2048,
      /*.z=*/1,
  };
  loom_target_dispatch_workgroup_count_t exceeds_x = {
      /*.x=*/4097,
      /*.y=*/1,
      /*.z=*/1,
  };
  loom_target_dispatch_workgroup_count_t partial = {
      /*.x=*/0,
      /*.y=*/1,
      /*.z=*/1,
  };

  IREE_EXPECT_OK(loom_target_validate_hal_dispatch_workgroup_count(
      &snapshot, &hal_kernel, &valid));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_target_validate_hal_dispatch_workgroup_count(
                            &snapshot, &hal_kernel, &exceeds_x));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_validate_hal_dispatch_workgroup_count(
                            &snapshot, &hal_kernel, &partial));
}

TEST(TargetLaunchTest, DynamicLocalWorkgroupChecksGridLowerBoundOnly) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  snapshot.max_workgroup_count = {/*.x=*/4096, /*.y=*/5000, /*.z=*/5000};
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/0, /*.y=*/0, /*.z=*/0});
  loom_target_dispatch_workgroup_count_t valid = {
      /*.x=*/4096,
      /*.y=*/1,
      /*.z=*/1,
  };

  IREE_EXPECT_OK(loom_target_validate_hal_dispatch_workgroup_count(
      &snapshot, &hal_kernel, &valid));
}

TEST(TargetLaunchTest, ConcreteLocalWorkgroupValidatesGridLimits) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_hal_kernel_abi_t hal_kernel =
      TestHalKernelAbi({/*.x=*/4, /*.y=*/2, /*.z=*/1});
  loom_target_dispatch_workgroup_count_t valid = {
      /*.x=*/1024,
      /*.y=*/1024,
      /*.z=*/1,
  };
  loom_target_dispatch_workgroup_count_t exceeds_grid_x = {
      /*.x=*/1025,
      /*.y=*/1,
      /*.z=*/1,
  };
  loom_target_dispatch_workgroup_count_t exceeds_flat_grid = {
      /*.x=*/1024,
      /*.y=*/1024,
      /*.z=*/2,
  };

  IREE_EXPECT_OK(loom_target_validate_hal_dispatch_workgroup_count(
      &snapshot, &hal_kernel, &valid));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_target_validate_hal_dispatch_workgroup_count(
                            &snapshot, &hal_kernel, &exceeds_grid_x));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_target_validate_hal_dispatch_workgroup_count(
                            &snapshot, &hal_kernel, &exceeds_flat_grid));
}

TEST(TargetLaunchTest, RejectsFlatGridOverflow) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  snapshot.max_workgroup_size = {/*.x=*/0, /*.y=*/0, /*.z=*/0};
  snapshot.max_flat_workgroup_size = 0;
  snapshot.max_grid_size = {/*.x=*/0, /*.y=*/0, /*.z=*/0};
  snapshot.max_flat_grid_size = 0;
  snapshot.max_workgroup_count = {/*.x=*/0, /*.y=*/0, /*.z=*/0};
  loom_target_hal_kernel_abi_t hal_kernel = TestHalKernelAbi(
      {/*.x=*/UINT32_MAX, /*.y=*/UINT32_MAX, /*.z=*/UINT32_MAX});
  loom_target_dispatch_workgroup_count_t workgroup_count = {
      /*.x=*/UINT32_MAX,
      /*.y=*/UINT32_MAX,
      /*.z=*/UINT32_MAX,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_target_validate_hal_dispatch_workgroup_count(
                            &snapshot, &hal_kernel, &workgroup_count));
}

}  // namespace
}  // namespace loom
