// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-remote-check/artifact.h"

#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::tools::remote_check {
namespace {

static iree_hal_device_spec_t* CreateDeviceSpec(
    const iree_hal_executable_target_t* targets,
    iree_host_size_t target_count) {
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    IREE_CHECK_OK(iree_hal_device_spec_builder_add_executable_target(
        &builder, &targets[i]));
  }
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_CHECK_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  iree_hal_device_spec_builder_deinitialize(&builder);
  return device_spec;
}

#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
TEST(ArtifactTest, SelectsGenericCodeObjectForExactDevice) {
  const iree_hal_executable_target_t targets[] = {
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/IREE_SV("gfx1100:sramecc-:xnack-"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/100,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/IREE_SV("gfx11-generic:sramecc-:xnack-"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          /*.priority=*/50,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(targets, IREE_ARRAYSIZE(targets));

  iree_remote_check_artifact_t artifact;
  IREE_ASSERT_OK(iree_remote_check_select_artifact(device_spec, &artifact));
  ASSERT_NE(artifact.executable_target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(artifact.artifact_target_key,
                                     IREE_SV("gfx11-generic")));
  EXPECT_TRUE(iree_string_view_equal(artifact.executable_target->target_key,
                                     IREE_SV("gfx11-generic:sramecc-:xnack-")));
  EXPECT_GT(artifact.executable_data.data_length, 0u);
  EXPECT_EQ(artifact.dispatch_constants.data_length, sizeof(int32_t));

  iree_hal_device_spec_release(device_spec);
}

TEST(ArtifactTest, SelectsExactCodeObjectWhenAvailable) {
  const iree_hal_executable_target_t target = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/IREE_SV("gfx90a:sramecc+:xnack-"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  iree_hal_device_spec_t* device_spec = CreateDeviceSpec(&target, 1);

  iree_remote_check_artifact_t artifact;
  IREE_ASSERT_OK(iree_remote_check_select_artifact(device_spec, &artifact));
  EXPECT_TRUE(
      iree_string_view_equal(artifact.artifact_target_key, IREE_SV("gfx90a")));
  EXPECT_EQ(artifact.executable_target->kind,
            IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT);

  iree_hal_device_spec_release(device_spec);
}

#endif  // IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT

#if defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
TEST(ArtifactTest, SelectsVulkanBdaSpirv) {
  const iree_hal_executable_target_t target = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/50,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  iree_hal_device_spec_t* device_spec = CreateDeviceSpec(&target, 1);

  iree_remote_check_artifact_t artifact;
  IREE_ASSERT_OK(iree_remote_check_select_artifact(device_spec, &artifact));
  ASSERT_NE(artifact.executable_target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(artifact.executable_target->family,
                                     IREE_SV("spirv")));
  EXPECT_TRUE(iree_string_view_equal(artifact.artifact_target_key,
                                     IREE_SV("vulkan1.3+bda")));
  EXPECT_TRUE(iree_string_view_equal(artifact.entry_point, IREE_SV("main")));
  EXPECT_GT(artifact.executable_data.data_length, 0u);
  EXPECT_EQ(artifact.dispatch_constants.data_length, 0u);
  EXPECT_EQ(artifact.dispatch_config.workgroup_count[0], 4u);
  EXPECT_EQ(artifact.dispatch_config.workgroup_count[1], 1u);
  EXPECT_EQ(artifact.dispatch_config.workgroup_count[2], 1u);

  iree_hal_device_spec_release(device_spec);
}
#endif  // IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT

TEST(ArtifactTest, RejectsDeviceWithoutCompatibleArtifact) {
  const iree_hal_executable_target_t target = {
      /*.family=*/IREE_SV("cuda"),
      /*.target_key=*/IREE_SV("sm_80"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/50,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  iree_hal_device_spec_t* device_spec = CreateDeviceSpec(&target, 1);

  iree_remote_check_artifact_t artifact;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_remote_check_select_artifact(device_spec, &artifact));

  iree_hal_device_spec_release(device_spec);
}

}  // namespace
}  // namespace iree::tools::remote_check
