// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library_target.h"

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

static iree_hal_amdgpu_target_identity_t ParseIdentity(
    const char* target_name) {
  iree_hal_amdgpu_target_identity_t identity;
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
      iree_make_cstring_view(target_name), &identity));
  return identity;
}

static iree_hal_amdgpu_target_identity_t QualifiedIdentity(
    const char* target_name, uint32_t asic_revision) {
  iree_hal_amdgpu_target_identity_t identity = ParseIdentity(target_name);
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_apply_asic_revision(
      asic_revision, &identity));
  return identity;
}

static std::vector<std::string> CandidateValues(
    const iree_hal_amdgpu_target_identity_t& physical_identity,
    const iree_hal_amdgpu_target_identity_t& isa_identity) {
  iree_hal_amdgpu_device_library_target_candidate_list_t candidates = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_device_library_target_candidates_from_agent_isa(
      &physical_identity, &isa_identity, &candidates));
  std::vector<std::string> values;
  values.reserve(candidates.count);
  for (iree_host_size_t i = 0; i < candidates.count; ++i) {
    values.emplace_back(candidates.values[i].value.data,
                        candidates.values[i].value.size);
  }
  return values;
}

static std::vector<std::string> CandidateValues(const char* target_name,
                                                uint32_t asic_revision = 0) {
  const auto identity = QualifiedIdentity(target_name, asic_revision);
  return CandidateValues(identity, identity);
}

TEST(DeviceLibraryTargetTest,
     PreservesFeatureBearingCandidatesBeforeFallbacks) {
  const auto values =
      CandidateValues("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-");

  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(values[0], "gfx942:sramecc+:xnack-");
  EXPECT_EQ(values[1], "gfx942");
  EXPECT_EQ(values[2], "gfx9-4-generic:sramecc+:xnack-");
  EXPECT_EQ(values[3], "gfx9-4-generic");
}

TEST(DeviceLibraryTargetTest, Gfx1250A0SelectsOnlyExactVariant) {
  const auto values = CandidateValues("amdgcn-amd-amdhsa--gfx1250");

  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0], "gfx1250-a0");
}

TEST(DeviceLibraryTargetTest, Gfx1250A0RejectsGenericAlternateIsa) {
  const auto physical_identity =
      QualifiedIdentity("amdgcn-amd-amdhsa--gfx1250", 0);
  const auto generic_isa_identity =
      QualifiedIdentity("amdgcn-amd-amdhsa--gfx12-5-generic", 0);

  EXPECT_TRUE(CandidateValues(physical_identity, generic_isa_identity).empty());
}

TEST(DeviceLibraryTargetTest, Gfx1250B0IncludesGenericFallback) {
  const auto values = CandidateValues("amdgcn-amd-amdhsa--gfx1250", 1);

  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], "gfx1250:asic-revision=b0");
  EXPECT_EQ(values[1], "gfx1250");
  EXPECT_EQ(values[2], "gfx12-5-generic");
}

TEST(DeviceLibraryTargetTest, RejectsUnqualifiedGfx1250PhysicalIdentity) {
  const auto identity = ParseIdentity("amdgcn-amd-amdhsa--gfx1250");
  iree_hal_amdgpu_device_library_target_candidate_list_t candidates = {0};

  EXPECT_THAT(
      Status(iree_hal_amdgpu_device_library_target_candidates_from_agent_isa(
          &identity, &identity, &candidates)),
      StatusIs(StatusCode::kFailedPrecondition));
}

TEST(DeviceLibraryTargetTest, MatchesOnlyWholeFileArchSegments) {
  EXPECT_TRUE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx1250-a0.so"), IREE_SV("gfx1250-a0")));
  EXPECT_TRUE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx9-4-generic.so"), IREE_SV("gfx9-4-generic")));
  EXPECT_TRUE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942.debug.so"), IREE_SV("gfx942")));

  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942x.so"), IREE_SV("gfx942")));
  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx1250-a0.so"), IREE_SV("gfx1250")));
  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx9-4-generic.so"), IREE_SV("gfx9-4-generic:sramecc+:xnack-")));
  EXPECT_FALSE(iree_hal_amdgpu_device_library_target_matches_file_arch(
      IREE_SV("gfx942.so"), IREE_SV("")));
}

}  // namespace
}  // namespace iree::hal::amdgpu
