// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable/amdgpu/executable_target.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static void CreateDeviceSpecWithFeatureQualifiedTargets(
    iree_hal_device_spec_t** out_device_spec) {
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);

  const iree_hal_executable_target_t targets[] = {
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/IREE_SV("gfx942:sramecc+:xnack-"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/100,
          /*.physical_device_affinity=*/1ull,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/IREE_SV("gfx9-4-generic:sramecc+:xnack-"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          /*.priority=*/50,
          /*.physical_device_affinity=*/1ull,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  for (const auto& target : targets) {
    IREE_ASSERT_OK(
        iree_hal_device_spec_builder_add_executable_target(&builder, &target));
  }

  IREE_ASSERT_OK(
      iree_hal_device_spec_builder_finalize(&builder, out_device_spec));
  iree_hal_device_spec_builder_deinitialize(&builder);
}

TEST(ExecutableTargetTest, SelectsFeatureQualifiedExactTarget) {
  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecWithFeatureQualifiedTargets(&device_spec);

  iree_hal_executable_target_selection_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx942"), /*physical_device_affinity=*/1ull,
      &result));

  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(result.target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(result.target->target_key,
                                     IREE_SV("gfx942:sramecc+:xnack-")));
  iree_hal_device_spec_release(device_spec);
}

TEST(ExecutableTargetTest, AdvertisesResolvedPhysicalTarget) {
  iree_hal_amdgpu_target_identity_t identity;
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      IREE_SV("gfx1250"), &identity));
  IREE_ASSERT_OK(
      iree_hal_amdgpu_target_identity_resolve_physical_target(0, &identity));

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_builder_add_executable_targets(
      &builder, &identity, /*physical_device_affinity=*/1ull));
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  iree_hal_device_spec_builder_deinitialize(&builder);

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_EQ(executable_spec->target_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(executable_spec->targets[0].target_key,
                                     IREE_SV("gfx1250-a0")));
  EXPECT_TRUE(iree_string_view_equal(executable_spec->targets[1].target_key,
                                     IREE_SV("gfx12-5-generic")));

  iree_hal_executable_target_selection_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx1250-a0"),
      /*physical_device_affinity=*/1ull, &result));
  EXPECT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx1250"),
      /*physical_device_affinity=*/1ull, &result));
  EXPECT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
  iree_hal_device_spec_release(device_spec);
}

TEST(ExecutableTargetTest, SelectsFeatureQualifiedGenericTarget) {
  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecWithFeatureQualifiedTargets(&device_spec);

  iree_hal_executable_target_selection_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx9-4-generic"),
      /*physical_device_affinity=*/1ull, &result));

  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  ASSERT_NE(result.target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      result.target->target_key, IREE_SV("gfx9-4-generic:sramecc+:xnack-")));
  iree_hal_device_spec_release(device_spec);
}

TEST(ExecutableTargetTest, RejectsIncompatibleFeatureAndAffinity) {
  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecWithFeatureQualifiedTargets(&device_spec);

  iree_hal_executable_target_selection_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx9-4-generic:xnack+"),
      /*physical_device_affinity=*/1ull, &result));
  EXPECT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);

  IREE_ASSERT_OK(iree_hal_amdgpu_device_spec_select_executable_target(
      device_spec, IREE_SV("gfx9-4-generic"),
      /*physical_device_affinity=*/2ull, &result));
  EXPECT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
  iree_hal_device_spec_release(device_spec);
}

TEST(ExecutableTargetTest, RejectsMalformedArtifactTarget) {
  iree_hal_device_spec_t* device_spec = NULL;
  CreateDeviceSpecWithFeatureQualifiedTargets(&device_spec);

  iree_hal_executable_target_selection_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_device_spec_select_executable_target(
                            device_spec, IREE_SV("definitely-not-amdgpu"),
                            /*physical_device_affinity=*/0, &result));
  iree_hal_device_spec_release(device_spec);
}

}  // namespace
}  // namespace iree::hal::amdgpu
