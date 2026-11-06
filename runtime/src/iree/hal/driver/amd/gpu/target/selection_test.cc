// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/target/selection.h"

#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/status_matchers.h"

namespace {

class SelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder_);
  }

  void TearDown() override {
    iree_hal_device_spec_release(spec_);
    iree_hal_device_spec_builder_deinitialize(&builder_);
  }

  void Add(iree_string_view_t key, uint64_t affinity = 1,
           uint32_t priority = 0) {
    const iree_hal_executable_target_t target = {
        .family = IREE_SV("amdgpu"),
        .target_key = key,
        .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
        .priority = priority,
        .physical_device_affinity = affinity,
    };
    IREE_ASSERT_OK(
        iree_hal_device_spec_builder_add_executable_target(&builder_, &target));
  }

  void Finalize() {
    IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder_, &spec_));
  }

  iree_hal_executable_target_selection_result_t Select(const char* key,
                                                       uint64_t affinity = 0) {
    iree_hal_executable_target_selection_result_t result;
    IREE_CHECK_OK(iree_hal_amd_gpu_device_spec_select_executable_target(
        spec_, iree_make_cstring_view(key), affinity, &result));
    return result;
  }

  // Production builder for the same exact-target records native devices emit.
  iree_hal_device_spec_builder_t builder_;
  // Immutable target inventory retained through selection.
  iree_hal_device_spec_t* spec_ = nullptr;
};

TEST_F(SelectionTest, CanonicalTargetsAcceptExactAndProjectedArtifacts) {
  static const char* const targets[] = {
#define IREE_AMDGPU_TARGET_MAPPING(target, ...) target,
#include "iree/hal/driver/amd/gpu/target/identity_catalog.inl"
#undef IREE_AMDGPU_TARGET_MAPPING
  };
  for (const char* target : targets) {
    SCOPED_TRACE(target);
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        iree_make_cstring_view(target), &identity));
    if (identity.kind != IREE_HAL_AMD_GPU_TARGET_KIND_EXACT) {
      continue;
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_device_spec_builder_add_executable_targets(
        &builder_, &identity, 1));
  }
  Finalize();
  for (const char* target : targets) {
    SCOPED_TRACE(target);
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        iree_make_cstring_view(target), &identity));
    if (identity.kind != IREE_HAL_AMD_GPU_TARGET_KIND_EXACT) {
      continue;
    }
    EXPECT_EQ(Select(target).outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
    iree_hal_amd_gpu_target_identity_t projected;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_project_code_object(
        &identity, &projected));
    char key[128];
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_format_artifact_key(
        &projected, sizeof(key), key, nullptr));
    const auto result = Select(key);
    // A generic family may match several exact devices at equal priority.
    EXPECT_NE(result.outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
    const iree_hal_executable_target_selection_t selection = {
        .family = IREE_SV("amdgpu"),
        .target_key = iree_make_cstring_view(key),
    };
    EXPECT_EQ(iree_hal_device_spec_select_executable_target(spec_, &selection)
                  .outcome,
              IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  }
}

TEST_F(SelectionTest, StrictPhysicalTargetHasNoGenericFallback) {
  iree_hal_amd_gpu_target_identity_t identity;
  IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      IREE_SV("gfx1250"), &identity));
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_target_identity_resolve_physical_target(0, &identity));
  IREE_ASSERT_OK(iree_hal_amd_gpu_device_spec_builder_add_executable_targets(
      &builder_, &identity, 1));
  Finalize();

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(spec_);
  ASSERT_EQ(executable_spec->target_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(executable_spec->targets[0].target_key,
                                     IREE_SV("gfx1250-strict")));
  EXPECT_EQ(Select("gfx1250-strict").outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  EXPECT_EQ(Select("gfx12-5-generic").outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
}

TEST_F(SelectionTest, FeatureRequirementsFilterBeforeAmbiguityResolution) {
  Add(IREE_SV("gfx942:sramecc+:xnack-"));
  Add(IREE_SV("gfx942:sramecc+:xnack+"));
  Finalize();
  EXPECT_EQ(Select("gfx9-4-generic").outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS);
  const auto result = Select("gfx9-4-generic:xnack-");
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  EXPECT_EQ(result.target_ordinal, 0u);
  EXPECT_EQ(Select("gfx942:sramecc-").outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
}

TEST_F(SelectionTest, AffinityFiltersCompatiblePhysicalDevices) {
  Add(IREE_SV("gfx1100"), 1);
  Add(IREE_SV("gfx1101"), 2);
  Finalize();
  EXPECT_EQ(Select("gfx11-generic").outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS);
  const auto result = Select("gfx11-generic", 2);
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  EXPECT_EQ(result.target->physical_device_affinity, 2u);
  EXPECT_EQ(Select("gfx1100", 2).outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
}

TEST_F(SelectionTest, HighestPriorityCompatibleTargetWins) {
  Add(IREE_SV("gfx1100"), 1, 0);
  Add(IREE_SV("gfx1101"), 2, 10);
  Add(IREE_SV("gfx1201"), 4, 20);
  Finalize();
  const auto result = Select("gfx11-generic");
  ASSERT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  EXPECT_EQ(result.target_ordinal, 1u);
}

TEST_F(SelectionTest, MalformedArtifactIsAnInputError) {
  Add(IREE_SV("gfx1100"));
  Finalize();
  iree_hal_executable_target_selection_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_gpu_device_spec_select_executable_target(
                            spec_, IREE_SV("malformed"), 0, &result));
  EXPECT_EQ(result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH);
  EXPECT_EQ(result.target, nullptr);
}

}  // namespace
