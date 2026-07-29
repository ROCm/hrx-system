// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/selection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/profile.h"

namespace loom {
namespace {

static const loom_target_profile_type_t kTestProfileType = {
    /*.name=*/IREE_SVL("test"),
};

TEST(TargetSelectionTest, EmptyCapabilityAccessorsReturnEmptyValues) {
  EXPECT_TRUE(loom_target_selection_is_empty(
      loom_target_pass_capability_target_selection(nullptr)));
  EXPECT_FALSE(loom_symbol_ref_is_valid(
      loom_target_pass_capability_target_ref(nullptr)));
  EXPECT_FALSE(loom_symbol_ref_is_valid(loom_target_effective_target_ref(
      loom_symbol_ref_null(), /*capability=*/nullptr)));

  const loom_symbol_ref_t authored_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/3,
  };
  const loom_symbol_ref_t effective_target_ref =
      loom_target_effective_target_ref(authored_target_ref,
                                       /*capability=*/nullptr);
  EXPECT_EQ(effective_target_ref.module_id, authored_target_ref.module_id);
  EXPECT_EQ(effective_target_ref.symbol_id, authored_target_ref.symbol_id);
}

TEST(TargetSelectionTest, EnvironmentCarriesInvocationTargetSelection) {
  static const loom_target_bundle_t kBundle = {
      /*.name=*/IREE_SVL("selected-target"),
  };
  const loom_target_profile_t target_profile = {
      /*.type=*/&kTestProfileType,
      /*.target_bundle=*/&kBundle,
  };
  const loom_target_selection_t target_selection = {
      /*.profile=*/&target_profile,
  };
  const loom_symbol_ref_t invocation_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/7,
  };
  const loom_symbol_ref_t authored_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/5,
  };
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make(
          /*target_environment=*/nullptr, target_selection,
          invocation_target_ref);
  const loom_pass_environment_capability_t* capabilities[] = {
      &target_capability.base,
  };
  const loom_pass_environment_t environment =
      loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));

  IREE_ASSERT_OK(loom_pass_environment_verify(&environment));
  const loom_target_pass_capability_t* found_capability =
      loom_target_pass_capability_from_environment(&environment);
  ASSERT_EQ(found_capability, &target_capability);
  const loom_target_selection_t found_selection =
      loom_target_pass_capability_target_selection(found_capability);
  EXPECT_EQ(loom_target_selection_profile(found_selection), &target_profile);
  EXPECT_EQ(loom_target_selection_bundle(found_selection), &kBundle);

  loom_symbol_ref_t effective_target_ref = loom_target_effective_target_ref(
      loom_symbol_ref_null(), found_capability);
  EXPECT_EQ(effective_target_ref.module_id, invocation_target_ref.module_id);
  EXPECT_EQ(effective_target_ref.symbol_id, invocation_target_ref.symbol_id);

  effective_target_ref =
      loom_target_effective_target_ref(authored_target_ref, found_capability);
  EXPECT_EQ(effective_target_ref.module_id, authored_target_ref.module_id);
  EXPECT_EQ(effective_target_ref.symbol_id, authored_target_ref.symbol_id);
}

TEST(TargetSelectionTest, InvocationRefinesOnlyCompatibleTargetFamily) {
  static const loom_target_snapshot_t kSnapshot = {
      /*.name=*/IREE_SVL("snapshot"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
  };
  static const loom_target_config_t kSelectedConfig = {
      /*.name=*/IREE_SVL("selected-config"),
      /*.contract_set_key=*/IREE_SVL("target.family.a"),
  };
  static const loom_target_config_t kCompatibleConfig = {
      /*.name=*/IREE_SVL("compatible-config"),
      /*.contract_set_key=*/IREE_SVL("target.family.a"),
  };
  static const loom_target_config_t kIncompatibleConfig = {
      /*.name=*/IREE_SVL("incompatible-config"),
      /*.contract_set_key=*/IREE_SVL("target.family.b"),
  };
  static const loom_target_bundle_t kSelectedBundle = {
      /*.name=*/IREE_SVL("selected-target"),
      /*.snapshot=*/&kSnapshot,
      /*.export_plan=*/nullptr,
      /*.config=*/&kSelectedConfig,
  };
  static const loom_target_bundle_t kCompatibleBundle = {
      /*.name=*/IREE_SVL("compatible-target"),
      /*.snapshot=*/&kSnapshot,
      /*.export_plan=*/nullptr,
      /*.config=*/&kCompatibleConfig,
  };
  static const loom_target_bundle_t kIncompatibleBundle = {
      /*.name=*/IREE_SVL("incompatible-target"),
      /*.snapshot=*/&kSnapshot,
      /*.export_plan=*/nullptr,
      /*.config=*/&kIncompatibleConfig,
  };
  static const loom_target_profile_t kSelectedProfile = {
      /*.type=*/&kTestProfileType,
      /*.target_bundle=*/&kSelectedBundle,
  };
  const loom_symbol_ref_t invocation_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/7,
  };
  const loom_symbol_ref_t authored_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/5,
  };
  const loom_target_pass_capability_t target_capability =
      loom_target_pass_capability_make(/*target_environment=*/nullptr,
                                       {/*.profile=*/&kSelectedProfile},
                                       invocation_target_ref);

  EXPECT_TRUE(loom_target_pass_capability_can_refine_target_bundle(
      &target_capability, invocation_target_ref, &kCompatibleBundle));
  EXPECT_FALSE(loom_target_pass_capability_can_refine_target_bundle(
      &target_capability, invocation_target_ref, &kIncompatibleBundle));
  EXPECT_FALSE(loom_target_pass_capability_can_refine_target_bundle(
      &target_capability, authored_target_ref, &kCompatibleBundle));
  EXPECT_FALSE(loom_target_pass_capability_can_refine_target_bundle(
      nullptr, invocation_target_ref, &kCompatibleBundle));
}

}  // namespace
}  // namespace loom
