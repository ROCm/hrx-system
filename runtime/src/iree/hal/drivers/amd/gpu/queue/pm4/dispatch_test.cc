// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    function_.metadata = &metadata_;
    function_.kernarg = &layout_;
  }
  iree_status_t Prepare(
      iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags = 0,
      iree_const_byte_span_t constants = iree_const_byte_span_empty(),
      iree_host_size_t binding_count = 0) {
    return iree_hal_amd_gpu_pm4_dispatch_prepare(
        &function_, 65536, config, constants, binding_count, flags, &dispatch_);
  }
  // Normalized image metadata; launch inputs remain public caller values.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Argument representation used by launch admission.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Prepared executable function borrowing this fixture's metadata.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Admitted native launch geometry.
  iree_hal_amd_gpu_pm4_dispatch_t dispatch_ = {};
};

TEST_F(DispatchTest, DefaultGeometryUsesRequiredMetadata) {
  auto config = iree_hal_make_static_dispatch_config(7, 3, 2);
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 1u);
  metadata_.has_required_workgroup_size = true;
  metadata_.required_workgroup_size[0] = 32;
  metadata_.required_workgroup_size[1] = 2;
  metadata_.required_workgroup_size[2] = 1;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.workgroup_size[0], 32u);
  EXPECT_EQ(dispatch_.workgroup_size[1], 2u);
  EXPECT_EQ(dispatch_.workgroup_count[0], 7u);
  config.workgroup_size[0] = 16;
  config.workgroup_size[1] = 2;
  config.workgroup_size[2] = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, ExactItemsAdmitPartialGroupsAndUniformRequirements) {
  iree_hal_dispatch_config_t config = {};
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = 4;
  config.workgroup_size[2] = 1;
  config.workitem_count[0] = 65;
  config.workitem_count[1] = 8;
  config.workitem_count[2] = 1;
  const auto flags = IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.workgroup_count[0], 3u);
  EXPECT_EQ(dispatch_.workgroup_count[1], 2u);
  EXPECT_EQ(dispatch_.partial_size[0], 1u);
  EXPECT_EQ(dispatch_.partial_size[1], 4u);
  EXPECT_EQ(dispatch_.partial_size[2], 1u);
  metadata_.uniform_workgroup_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config, flags));
  config.workitem_count[0] = 64;
  IREE_ASSERT_OK(Prepare(config, flags));
  EXPECT_EQ(dispatch_.partial_size[0], 0u);
  EXPECT_EQ(dispatch_.partial_size[1], 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Prepare(config));
}

TEST_F(DispatchTest, GeometryAndLdsRespectExecutableLimits) {
  auto config = iree_hal_make_static_dispatch_config(5, 1, 1);
  config.workgroup_size[0] = 32;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  metadata_.max_flat_workgroup_size = 16;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_flat_workgroup_size = 32;
  metadata_.max_workgroup_count[0] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Prepare(config));
  metadata_.max_workgroup_count[0] = 5;
  metadata_.group_segment_fixed_size = 32768;
  config.dynamic_workgroup_local_memory = 32768;
  IREE_ASSERT_OK(Prepare(config));
  EXPECT_EQ(dispatch_.group_segment_byte_length, 65536u);
  ++config.dynamic_workgroup_local_memory;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
  config.dynamic_workgroup_local_memory = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Prepare(config));
}

TEST_F(DispatchTest, UnprovisionedRuntimeServicesFailBeforeCapture) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  function_.launch.requirements =
      IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  function_.launch.requirements = 0;
  layout_.requirements = IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_HOSTCALL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
  layout_.requirements = 0;
  metadata_.has_workgroup_cluster_size = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Prepare(config));
}

TEST_F(DispatchTest, DirectArgumentsRespectDenseAndNativeExtents) {
  const auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  std::array<uint8_t, 24> constants = {};
  layout_.byte_length = 24;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 8;
  IREE_ASSERT_OK(Prepare(config, 0, {constants.data(), 8}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, 0, {constants.data(), 12}));
  const auto native = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 12}));
  IREE_ASSERT_OK(Prepare(config, native, {constants.data(), 24}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Prepare(config, native, {constants.data(), 11}));
}

}  // namespace
