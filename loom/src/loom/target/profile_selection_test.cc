// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile_selection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/provider.h"

namespace loom {
namespace {

static const loom_target_snapshot_t kSnapshot = {};
static const loom_target_export_plan_t kExportPlan = {};
static const loom_target_config_t kConfig = {};
static const loom_target_bundle_t kBundle = {
    /*.name=*/IREE_SVL("test"),
    /*.snapshot=*/&kSnapshot,
    /*.export_plan=*/&kExportPlan,
    /*.config=*/&kConfig,
};
static const loom_target_profile_type_t kProfileType = {
    /*.name=*/IREE_SVL("test"),
};
static const loom_target_profile_t kProfile = {
    /*.type=*/&kProfileType,
    /*.target_bundle=*/&kBundle,
};

static iree_status_t SelectProfile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  (void)provider;
  if (!iree_string_view_equal(selector, IREE_SV("alias"))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown test target selector");
  }
  uint8_t* storage = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *out_selection = (loom_target_profile_selection_t){
      /*.provider=*/nullptr,
      /*.profile=*/&kProfile,
      /*.selector=*/IREE_SV("canonical"),
      /*.storage=*/storage,
      /*.allocator=*/iree_allocator_null(),
  };
  return iree_ok_status();
}

static void ReleaseProfileSelection(
    const loom_target_provider_t* provider,
    loom_target_profile_selection_t* selection) {
  (void)provider;
  iree_allocator_free(selection->allocator, selection->storage);
}

static const loom_target_provider_t kProvider = {
    /*.profile_type=*/&kProfileType,
    /*.materialize_definition=*/{},
    /*.register_context=*/{},
    /*.initialize_low_descriptor_registry=*/{},
    /*.initialize_low_lower_policy_registry=*/{},
    /*.initialize_math_policy_registry=*/{},
    /*.low_legality_provider_list=*/{},
    /*.legalizer_provider_list=*/{},
    /*.low_packet_diagnostic_provider_list=*/{},
    /*.low_asm_diagnostic_provider_list=*/{},
    /*.low_verify_provider_list=*/{},
    /*.emitter_list=*/{},
    /*.pass_registry=*/{},
    /*.contribute_pipeline=*/{},
    /*.select_profile=*/SelectProfile,
    /*.release_profile_selection=*/ReleaseProfileSelection,
};
static const loom_target_provider_t* const kProviders[] = {
    &kProvider,
};
static const loom_target_provider_set_t kProviderSet = {
    /*.providers=*/kProviders,
    /*.provider_count=*/IREE_ARRAYSIZE(kProviders),
};

TEST(TargetProfileSpecificationTest, SplitsOnlyTheFamilyDelimiter) {
  loom_target_profile_specification_t specification = {};
  IREE_ASSERT_OK(loom_target_profile_specification_parse(
      IREE_SV("  amdgpu : gfx942:sramecc+:xnack-  "), &specification));

  EXPECT_TRUE(iree_string_view_equal(specification.family, IREE_SV("amdgpu")));
  EXPECT_TRUE(iree_string_view_equal(specification.selector,
                                     IREE_SV("gfx942:sramecc+:xnack-")));
}

TEST(TargetProfileSpecificationTest, RejectsMissingComponents) {
  loom_target_profile_specification_t specification = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_profile_specification_parse(
                            IREE_SV("amdgpu"), &specification));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_profile_specification_parse(
                            IREE_SV(":gfx1151"), &specification));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_profile_specification_parse(
                            IREE_SV("amdgpu:"), &specification));
}

TEST(TargetProfileSelectionTest, ResolvesAndOwnsCanonicalSelection) {
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&kProviderSet, &environment));

  loom_target_profile_selection_t selection = {};
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &environment, IREE_SV("test:alias"), iree_allocator_system(),
      &selection));
  EXPECT_EQ(selection.provider, &kProvider);
  EXPECT_EQ(selection.profile, &kProfile);
  EXPECT_TRUE(iree_string_view_equal(selection.selector, IREE_SV("canonical")));
  EXPECT_NE(selection.storage, nullptr);

  loom_target_profile_selection_deinitialize(&selection);
  EXPECT_EQ(selection.provider, nullptr);
  EXPECT_EQ(selection.profile, nullptr);
  EXPECT_EQ(selection.storage, nullptr);
  loom_target_environment_deinitialize(&environment);
}

TEST(TargetProfileSelectionTest, RejectsUnknownAndUnselectableFamilies) {
  static const loom_target_profile_type_t kUnselectableProfileType = {
      /*.name=*/IREE_SVL("unselectable"),
  };
  static const loom_target_provider_t kUnselectableProvider = {
      /*.profile_type=*/&kUnselectableProfileType,
  };
  static const loom_target_provider_t* const kMixedProviders[] = {
      &kProvider,
      &kUnselectableProvider,
  };
  const loom_target_provider_set_t provider_set = {
      /*.providers=*/kMixedProviders,
      /*.provider_count=*/IREE_ARRAYSIZE(kMixedProviders),
  };
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  loom_target_profile_selection_t selection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_target_environment_select_profile(
                            &environment, IREE_SV("missing:value"),
                            iree_allocator_system(), &selection));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_target_environment_select_profile(
                            &environment, IREE_SV("unselectable:value"),
                            iree_allocator_system(), &selection));

  loom_target_environment_deinitialize(&environment);
}

TEST(TargetProfileSelectionTest, RejectsDuplicateFamilyNames) {
  static const loom_target_profile_type_t kDuplicateProfileType = {
      /*.name=*/IREE_SVL("test"),
  };
  static const loom_target_provider_t kDuplicateProvider = {
      /*.profile_type=*/&kDuplicateProfileType,
  };
  static const loom_target_provider_t* const kDuplicateProviders[] = {
      &kProvider,
      &kDuplicateProvider,
  };
  const loom_target_provider_set_t provider_set = {
      /*.providers=*/kDuplicateProviders,
      /*.provider_count=*/IREE_ARRAYSIZE(kDuplicateProviders),
  };
  loom_target_environment_t environment = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_target_environment_initialize(&provider_set, &environment));
}

}  // namespace
}  // namespace loom
