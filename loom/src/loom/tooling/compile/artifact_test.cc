// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/artifact.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const loom_target_snapshot_t kTargetSnapshot = {
    /*.name=*/IREE_SVL("snapshot-123"),
};
static const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SVL("export-123"),
};
static const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SVL("config-123"),
};
static const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SVL("bundle-123"),
    /*.snapshot=*/&kTargetSnapshot,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};
static const loom_target_fact_type_t kTargetFactType = {
    /*.name=*/IREE_SVL("target-family-123"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};

static iree_status_t ProjectTargetFacts(const loom_target_profile_t* profile,
                                        iree_arena_allocator_t* arena,
                                        loom_target_facts_t* out_facts) {
  (void)profile;
  (void)arena;
  (void)out_facts;
  return iree_ok_status();
}

static const loom_target_profile_type_t kTargetProfileType = {
    /*.name=*/IREE_SVL("target-family-123"),
    /*.fact_type=*/&kTargetFactType,
    /*.project_facts=*/ProjectTargetFacts,
};
static const loom_target_profile_t kTargetProfile = {
    /*.type=*/&kTargetProfileType,
    /*.target_bundle=*/&kTargetBundle,
};
static const loom_target_fact_type_t kOtherTargetFactType = {
    /*.name=*/IREE_SVL("target-family-456"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};
static const loom_target_profile_type_t kOtherTargetProfileType = {
    /*.name=*/IREE_SVL("target-family-456"),
    /*.fact_type=*/&kOtherTargetFactType,
    /*.project_facts=*/ProjectTargetFacts,
};

static iree_status_t SelectTargetProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (!iree_string_view_equal(selector, IREE_SV("selector-123"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown target selector");
  }
  *out_profile = &kTargetProfile;
  return iree_ok_status();
}

class ArtifactTargetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    target_provider_.profile_type = &kTargetProfileType;
    target_provider_.select_profile = SelectTargetProfile;
    target_providers_[0] = &target_provider_;
    provider_set_ = loom_target_provider_set_make(
        target_providers_, IREE_ARRAYSIZE(target_providers_));
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&provider_set_, &environment_));

    artifact_provider_.name = IREE_SVL("format-provider-123");
    artifact_provider_.target_profile_type = &kTargetProfileType;
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&environment_);
  }

  loom_target_provider_t target_provider_ = {};
  const loom_target_provider_t* target_providers_[1] = {};
  loom_target_provider_set_t provider_set_ = {};
  loom_target_environment_t environment_ = {};
  loom_artifact_provider_t artifact_provider_ = {};
};

TEST_F(ArtifactTargetTest, SelectsBorrowedCompatibleTarget) {
  const char target_specification[] = "target-family-123:selector-123";
  loom_artifact_target_t target = {};
  IREE_ASSERT_OK(loom_artifact_target_select(
      &artifact_provider_, &environment_,
      iree_make_cstring_view(target_specification), &target));

  EXPECT_EQ(target.target_profile, &kTargetProfile);
  EXPECT_EQ(target.target_key.data,
            target_specification + IREE_ARRAYSIZE("target-family-123:") - 1);
  EXPECT_TRUE(
      iree_string_view_equal(target.target_key, IREE_SV("selector-123")));
}

TEST_F(ArtifactTargetTest, RejectsMalformedTarget) {
  loom_artifact_target_t target = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_artifact_target_select(&artifact_provider_, &environment_,
                                  IREE_SV("selector-123"), &target));
  EXPECT_EQ(target.target_profile, nullptr);
}

TEST_F(ArtifactTargetTest, RejectsIncompatibleTargetFamily) {
  artifact_provider_.target_profile_type = &kOtherTargetProfileType;
  loom_artifact_target_t target = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_artifact_target_select(&artifact_provider_, &environment_,
                                  IREE_SV("target-family-123:selector-123"),
                                  &target));
  EXPECT_EQ(target.target_profile, nullptr);
}

}  // namespace
}  // namespace loom
