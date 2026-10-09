// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/selection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const loom_target_snapshot_t kTargetSnapshot = {
    /*.name=*/IREE_SVL("fake.snapshot"),
};
static const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SVL("fake.export"),
};
static const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SVL("fake.config"),
};
static const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SVL("fake.bundle"),
    /*.snapshot=*/&kTargetSnapshot,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};
static const loom_target_fact_type_t kTargetFactType = {
    /*.name=*/IREE_SVL("fake"),
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
    /*.name=*/IREE_SVL("fake"),
    /*.fact_type=*/&kTargetFactType,
    /*.project_facts=*/ProjectTargetFacts,
};
static const loom_target_profile_t kTargetProfile = {
    .type = &kTargetProfileType,
    .target_bundle = &kTargetBundle,
};
static const loom_target_fact_type_t kOtherTargetFactType = {
    /*.name=*/IREE_SVL("other"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};
static const loom_target_profile_type_t kOtherTargetProfileType = {
    /*.name=*/IREE_SVL("other"),
    /*.fact_type=*/&kOtherTargetFactType,
    /*.project_facts=*/ProjectTargetFacts,
};
static const loom_target_profile_t kOtherTargetProfile = {
    .type = &kOtherTargetProfileType,
    .target_bundle = &kTargetBundle,
};

static iree_status_t SelectFakeProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = nullptr;
  if (!iree_string_view_equal(selector, IREE_SV("target-123"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown fake target selector");
  }
  *out_profile = &kTargetProfile;
  return iree_ok_status();
}

static iree_status_t SelectOtherProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  (void)selector;
  *out_profile = &kOtherTargetProfile;
  return iree_ok_status();
}

static int gDestroyedCpuProfileCount = 0;

static void DestroyCpuProfile(loom_target_profile_t* profile,
                              iree_allocator_t allocator) {
  ++gDestroyedCpuProfileCount;
  iree_allocator_free(allocator, profile);
}

static iree_status_t SelectOwnedCpuProfile(
    const loom_target_profile_type_t* profile_type,
    loom_target_profile_selection_t* out_selection,
    iree_allocator_t allocator) {
  loom_target_profile_t* profile = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*profile), (void**)&profile));
  *profile = {
      .type = profile_type,
      .target_bundle = &kTargetBundle,
  };
  *out_selection = {
      .profile = profile,
      .destroy = DestroyCpuProfile,
  };
  return iree_ok_status();
}

static iree_status_t SelectOwnedFakeCpuProfile(
    const iree_cpu_data_t* cpu_data, const loom_target_facts_t* requirement,
    const loom_target_profile_t* profile,
    loom_target_profile_selection_t* out_selection,
    iree_allocator_t allocator) {
  (void)cpu_data;
  (void)requirement;
  (void)profile;
  return SelectOwnedCpuProfile(&kTargetProfileType, out_selection, allocator);
}

static iree_status_t SelectOwnedOtherCpuProfile(
    const iree_cpu_data_t* cpu_data, const loom_target_facts_t* requirement,
    const loom_target_profile_t* profile,
    loom_target_profile_selection_t* out_selection,
    iree_allocator_t allocator) {
  (void)cpu_data;
  (void)requirement;
  (void)profile;
  return SelectOwnedCpuProfile(&kOtherTargetProfileType, out_selection,
                               allocator);
}

static iree_status_t FailCpuProfileSelection(
    const iree_cpu_data_t* cpu_data, const loom_target_facts_t* requirement,
    const loom_target_profile_t* profile,
    loom_target_profile_selection_t* out_selection,
    iree_allocator_t allocator) {
  (void)cpu_data;
  (void)requirement;
  (void)profile;
  (void)out_selection;
  (void)allocator;
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "injected provider failure");
}

TEST(TargetSpecificationTest, ParsesBorrowedFamilyAndSelector) {
  loom_target_specification_t specification = {};
  IREE_ASSERT_OK(loom_target_specification_parse(
      IREE_SV("  fake : target-123:feature+  "), &specification));

  EXPECT_TRUE(iree_string_view_equal(specification.family, IREE_SV("fake")));
  EXPECT_TRUE(iree_string_view_equal(specification.selector,
                                     IREE_SV("target-123:feature+")));
}

TEST(TargetSpecificationTest, RejectsMalformedSpecifications) {
  loom_target_specification_t specification = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_specification_parse(IREE_SV("target-123"), &specification));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_specification_parse(IREE_SV(":target-123"), &specification));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_specification_parse(IREE_SV("fake:"), &specification));
}

TEST(TargetSelectionTest, SelectsBorrowedProfile) {
  loom_target_provider_t provider = {};
  provider.profile_type = &kTargetProfileType;
  provider.select_profile = SelectFakeProfile;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  const loom_target_specification_t specification = {
      /*.family=*/IREE_SVL("fake"),
      /*.selector=*/IREE_SVL("target-123"),
  };
  const loom_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &environment, &specification, &profile));
  EXPECT_EQ(profile, &kTargetProfile);

  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, RejectsUnknownFamily) {
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(nullptr, 0);
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  const loom_target_specification_t specification = {
      /*.family=*/IREE_SVL("missing"),
      /*.selector=*/IREE_SVL("target-123"),
  };
  const loom_target_profile_t* profile = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_select_profile(
                            &environment, &specification, &profile));
  EXPECT_EQ(profile, nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, RejectsFamilyWithoutNamedProfiles) {
  loom_target_provider_t provider = {};
  provider.profile_type = &kTargetProfileType;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  const loom_target_specification_t specification = {
      /*.family=*/IREE_SVL("fake"),
      /*.selector=*/IREE_SVL("target-123"),
  };
  const loom_target_profile_t* profile = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_target_environment_select_profile(
                            &environment, &specification, &profile));
  EXPECT_EQ(profile, nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, RejectsAmbiguousFamilyProvidersAtInitialization) {
  loom_target_provider_t first_provider = {};
  first_provider.profile_type = &kTargetProfileType;
  first_provider.select_profile = SelectFakeProfile;
  loom_target_provider_t second_provider = {};
  second_provider.profile_type = &kTargetProfileType;
  second_provider.select_profile = SelectFakeProfile;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST(TargetSelectionTest, RejectsProfileFromAnotherFamily) {
  loom_target_provider_t provider = {};
  provider.profile_type = &kTargetProfileType;
  provider.select_profile = SelectOtherProfile;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  const loom_target_specification_t specification = {
      /*.family=*/IREE_SVL("fake"),
      /*.selector=*/IREE_SVL("target-123"),
  };
  const loom_target_profile_t* profile = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        loom_target_environment_select_profile(
                            &environment, &specification, &profile));
  EXPECT_EQ(profile, nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, TransfersOwnedCpuProfile) {
  loom_target_provider_t provider = {};
  provider.profile_type = &kTargetProfileType;
  provider.select_cpu_profile = SelectOwnedFakeCpuProfile;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  gDestroyedCpuProfileCount = 0;
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
  };
  loom_target_profile_selection_t selection = {};
  IREE_ASSERT_OK(loom_target_environment_select_cpu_profile(
      &environment, &cpu_data, nullptr, nullptr, &selection,
      iree_allocator_system()));
  ASSERT_NE(selection.profile, nullptr);
  EXPECT_EQ(selection.profile->type, &kTargetProfileType);
  EXPECT_EQ(gDestroyedCpuProfileCount, 0);

  loom_target_profile_selection_release(&selection, iree_allocator_system());
  EXPECT_EQ(selection.profile, nullptr);
  EXPECT_EQ(gDestroyedCpuProfileCount, 1);
  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, ReleasesOwnedCpuProfilesOnAmbiguity) {
  loom_target_provider_t first_provider = {};
  first_provider.profile_type = &kTargetProfileType;
  first_provider.select_cpu_profile = SelectOwnedFakeCpuProfile;
  loom_target_provider_t second_provider = {};
  second_provider.profile_type = &kOtherTargetProfileType;
  second_provider.select_cpu_profile = SelectOwnedOtherCpuProfile;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  gDestroyedCpuProfileCount = 0;
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
  };
  loom_target_profile_selection_t selection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_select_cpu_profile(
                            &environment, &cpu_data, nullptr, nullptr,
                            &selection, iree_allocator_system()));
  EXPECT_EQ(selection.profile, nullptr);
  EXPECT_EQ(gDestroyedCpuProfileCount, 2);
  loom_target_environment_deinitialize(&environment);
}

TEST(TargetSelectionTest, ReleasesOwnedCpuProfileOnLaterProviderFailure) {
  loom_target_provider_t first_provider = {};
  first_provider.profile_type = &kTargetProfileType;
  first_provider.select_cpu_profile = SelectOwnedFakeCpuProfile;
  loom_target_provider_t second_provider = {};
  second_provider.profile_type = &kOtherTargetProfileType;
  second_provider.select_cpu_profile = FailCpuProfileSelection;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  gDestroyedCpuProfileCount = 0;
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
  };
  loom_target_profile_selection_t selection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_target_environment_select_cpu_profile(
                            &environment, &cpu_data, nullptr, nullptr,
                            &selection, iree_allocator_system()));
  EXPECT_EQ(selection.profile, nullptr);
  EXPECT_EQ(gDestroyedCpuProfileCount, 1);
  loom_target_environment_deinitialize(&environment);
}

}  // namespace
}  // namespace loom
