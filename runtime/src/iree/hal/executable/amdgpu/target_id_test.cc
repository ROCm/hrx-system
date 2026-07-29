// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable/amdgpu/target_id.h"

#include <cstring>
#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static iree_hal_amdgpu_target_identity_t ParseArtifactIdentity(
    const char* value) {
  iree_hal_amdgpu_target_identity_t identity;
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_parse_artifact_key(
      iree_make_cstring_view(value), &identity));
  return identity;
}

static iree_hal_amdgpu_target_identity_t ParseHsaIdentity(const char* value) {
  iree_hal_amdgpu_target_identity_t identity;
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
      iree_make_cstring_view(value), &identity));
  return identity;
}

static std::string FormatIdentity(
    const iree_hal_amdgpu_target_identity_t* identity) {
  char buffer[128] = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_format_artifact_key(
      identity, sizeof(buffer), buffer, /*out_buffer_length=*/nullptr));
  return std::string(buffer);
}

TEST(TargetIdentityTest, ParsesExactAndGenericProcessors) {
  auto identity = ParseArtifactIdentity("gfx90a");
  EXPECT_EQ(identity.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_EQ(identity.version.major, 9u);
  EXPECT_EQ(identity.version.minor, 0u);
  EXPECT_EQ(identity.version.stepping, 10u);
  EXPECT_TRUE(iree_string_view_equal(identity.processor, IREE_SV("gfx90a")));
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);

  identity = ParseArtifactIdentity("gfx11-generic");
  EXPECT_EQ(identity.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_EQ(identity.version.major, 11u);
  EXPECT_EQ(identity.version.minor, 0u);
  EXPECT_EQ(identity.version.stepping, 0u);
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
}

TEST(TargetIdentityTest, ProcessorParserDoesNotAcceptOtherCoordinates) {
  iree_hal_amdgpu_target_identity_t identity = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_parse_processor(
      IREE_SV("gfx1250"), &identity));
  EXPECT_FALSE(identity.asic_revision.specified);
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_target_identity_parse_processor(
                            IREE_SV("gfx1250:asic-revision=a0"), &identity));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_target_identity_parse_processor(
                            IREE_SV("amdgcn-amd-amdhsa--gfx1250"), &identity));
}

TEST(TargetIdentityTest, SeparatesArtifactAndAmdhsaCoordinates) {
  auto identity = ParseArtifactIdentity("gfx942:xnack-:sramecc+");
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
  EXPECT_FALSE(identity.asic_revision.specified);
  EXPECT_EQ(FormatIdentity(&identity), "gfx942:sramecc+:xnack-");

  identity = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  EXPECT_TRUE(identity.asic_revision.specified);
  EXPECT_EQ(identity.asic_revision.value, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(identity.asic_revision.name, IREE_SV("a0")));
  EXPECT_EQ(FormatIdentity(&identity), "gfx1250:asic-revision=a0");

  identity = ParseHsaIdentity("amdgcn-amd-amdhsa--gfx942:xnack-:sramecc+");
  EXPECT_FALSE(identity.asic_revision.specified);
  EXPECT_EQ(FormatIdentity(&identity), "gfx942:sramecc+:xnack-");

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
          IREE_SV("amdgcn-amd-amdhsa--gfx1250:asic-revision=a0"), &identity));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_target_identity_parse_artifact_key(
                            IREE_SV("amdgcn-amd-amdhsa--gfx1250"), &identity));
}

TEST(TargetIdentityTest, ResolvesFinitePhysicalRevisions) {
  struct Case {
    uint32_t value;
    const char* name;
  };
  static const Case cases[] = {
      {0, "a0"},
      {1, "b0"},
  };
  for (const Case& test_case : cases) {
    auto identity = ParseHsaIdentity("amdgcn-amd-amdhsa--gfx1250");
    EXPECT_TRUE(
        iree_hal_amdgpu_target_identity_requires_asic_revision(&identity));
    IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_apply_asic_revision(
        test_case.value, &identity));
    EXPECT_TRUE(identity.asic_revision.specified);
    EXPECT_EQ(identity.asic_revision.value, test_case.value);
    EXPECT_TRUE(iree_string_view_equal(identity.asic_revision.name,
                                       iree_make_cstring_view(test_case.name)));
    EXPECT_EQ(FormatIdentity(&identity),
              std::string("gfx1250:asic-revision=") + test_case.name);
  }

  auto identity = ParseHsaIdentity("amdgcn-amd-amdhsa--gfx1250");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_target_identity_apply_asic_revision(2, &identity));

  identity = ParseHsaIdentity("amdgcn-amd-amdhsa--gfx1100");
  EXPECT_FALSE(
      iree_hal_amdgpu_target_identity_requires_asic_revision(&identity));
  IREE_ASSERT_OK(
      iree_hal_amdgpu_target_identity_apply_asic_revision(99, &identity));
  EXPECT_FALSE(identity.asic_revision.specified);
}

TEST(TargetIdentityTest, RejectsContradictoryPhysicalRevision) {
  auto identity = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_amdgpu_target_identity_apply_asic_revision(1, &identity));
}

TEST(TargetIdentityTest, EqualityComparesStructuredIdentityByValue) {
  std::string first_storage = "gfx942:sramecc+:xnack-";
  const auto first = ParseArtifactIdentity(first_storage.c_str());
  std::string second_storage = "gfx942:sramecc+:xnack-";
  const auto second = ParseArtifactIdentity(second_storage.c_str());
  EXPECT_TRUE(iree_hal_amdgpu_target_identity_equal(&first, &second));

  const auto different_feature =
      ParseArtifactIdentity("gfx942:sramecc+:xnack+");
  EXPECT_FALSE(
      iree_hal_amdgpu_target_identity_equal(&first, &different_feature));

  const auto a0 = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  const auto b0 = ParseArtifactIdentity("gfx1250:asic-revision=b0");
  EXPECT_FALSE(iree_hal_amdgpu_target_identity_equal(&a0, &b0));
}

TEST(TargetIdentityTest, RejectsMalformedOrUnsupportedCoordinates) {
  static const iree_string_view_t values[] = {
      IREE_SVL("gfx942foo"),
      IREE_SVL("gfx942:xnack+:xnack-"),
      IREE_SVL("gfx942:wavefrontsize64+"),
      IREE_SVL("gfx1100:xnack+"),
      IREE_SVL("gfx942:"),
      IREE_SVL("gfx942:xnack+:"),
      IREE_SVL("gfx1250:asic-revision=c0"),
      IREE_SVL("gfx1251:asic-revision=a0"),
      IREE_SVL("gfx1250:asic-revision=a0:asic-revision=b0"),
  };
  for (iree_string_view_t value : values) {
    iree_hal_amdgpu_target_identity_t identity;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_target_identity_parse_artifact_key(value, &identity));
  }
}

TEST(TargetIdentityTest, FormatsIntoQueriedBufferLength) {
  const auto identity = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  iree_host_size_t required_length = 0;
  IREE_EXPECT_OK(iree_hal_amdgpu_target_identity_format_artifact_key(
      &identity, /*buffer_capacity=*/0, /*buffer=*/nullptr, &required_length));
  EXPECT_EQ(required_length, strlen("gfx1250:asic-revision=a0"));

  char buffer[8] = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_target_identity_format_artifact_key(
          &identity, sizeof(buffer), buffer, &required_length));
  EXPECT_EQ(required_length, strlen("gfx1250:asic-revision=a0"));
}

TEST(TargetIdentityTest, ProjectsCodeObjectIdentityWithoutPhysicalRevision) {
  const auto exact = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  iree_hal_amdgpu_target_identity_t code_object;
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_project_code_object(
      &exact, &code_object));
  EXPECT_EQ(code_object.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_TRUE(iree_string_view_equal(code_object.processor,
                                     IREE_SV("gfx12-5-generic")));
  EXPECT_FALSE(code_object.asic_revision.specified);

  const auto feature_exact = ParseArtifactIdentity("gfx942:sramecc+:xnack-");
  IREE_ASSERT_OK(iree_hal_amdgpu_target_identity_project_code_object(
      &feature_exact, &code_object));
  EXPECT_TRUE(
      iree_string_view_equal(code_object.processor, IREE_SV("gfx9-4-generic")));
  EXPECT_EQ(code_object.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON);
  EXPECT_EQ(code_object.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
}

TEST(TargetIdentityTest, LooksUpGeneratedWavefrontSupport) {
  struct Case {
    const char* processor;
    uint32_t default_size;
    iree_hal_amdgpu_wavefront_size_flags_t explicit_sizes;
  };
  static const Case cases[] = {
      {"gfx90a", 64, IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_NONE},
      {"gfx942", 64, IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64},
      {"gfx1100", 32,
       IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32 |
           IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_64},
      {"gfx1250", 32, IREE_HAL_AMDGPU_WAVEFRONT_SIZE_FLAG_32},
  };
  for (const Case& test_case : cases) {
    const auto identity = ParseArtifactIdentity(test_case.processor);
    iree_hal_amdgpu_wavefront_size_support_t support;
    EXPECT_TRUE(iree_hal_amdgpu_target_identity_lookup_wavefront_size_support(
        &identity, &support));
    EXPECT_EQ(support.default_size, test_case.default_size);
    EXPECT_EQ(support.explicit_supported_sizes, test_case.explicit_sizes);
  }

  const auto unknown = ParseArtifactIdentity("gfx1300");
  iree_hal_amdgpu_wavefront_size_support_t support;
  EXPECT_FALSE(iree_hal_amdgpu_target_identity_lookup_wavefront_size_support(
      &unknown, &support));
}

TEST(TargetIdentityTest, ChecksProcessorFeatureAndRevisionCompatibility) {
  auto artifact = ParseArtifactIdentity("gfx1100");
  auto agent = ParseArtifactIdentity("gfx1101");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_identity_check_compatible(&artifact, &agent),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_PROCESSOR));

  artifact = ParseArtifactIdentity("gfx11-generic");
  artifact.generic_version = 1;
  agent = ParseArtifactIdentity("gfx1151");
  EXPECT_EQ(iree_hal_amdgpu_target_identity_check_compatible(&artifact, &agent),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);

  artifact = ParseArtifactIdentity("gfx942:xnack+");
  agent = ParseArtifactIdentity("gfx942:xnack-");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_identity_check_compatible(&artifact, &agent),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK));

  artifact = ParseArtifactIdentity("gfx1250:asic-revision=a0");
  agent = ParseArtifactIdentity("gfx1250:asic-revision=b0");
  EXPECT_TRUE(iree_any_bit_set(
      iree_hal_amdgpu_target_identity_check_compatible(&artifact, &agent),
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_ASIC_REVISION));

  artifact = ParseArtifactIdentity("gfx1250");
  EXPECT_EQ(iree_hal_amdgpu_target_identity_check_compatible(&artifact, &agent),
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE);
}

TEST(TargetIdentityTest, FormatsCompatibilityReasons) {
  char buffer[64] = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_target_compatibility_format(
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY |
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC |
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK |
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_ASIC_REVISION,
      sizeof(buffer), buffer, /*out_buffer_length=*/nullptr));
  EXPECT_STREQ(buffer, "generic family, sramecc, xnack, ASIC revision");
}

}  // namespace
}  // namespace iree::hal::amdgpu
