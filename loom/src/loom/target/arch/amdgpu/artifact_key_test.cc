// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/artifact_key.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace loom {
namespace {

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
};

static const loom_amdgpu_target_info_t* LookupTarget(const char* name) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_CHECK_OK(loom_amdgpu_target_info_lookup_target(
      iree_make_cstring_view(name), &target));
  return target;
}

static std::string FormatArtifactKey(
    const loom_amdgpu_target_identity_t& identity) {
  char buffer[128] = {};
  iree_string_view_t artifact_key = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_artifact_key_format(&identity, sizeof(buffer),
                                                buffer, &artifact_key));
  return std::string(artifact_key.data, artifact_key.size);
}

static std::string FormatArenaArtifactKey(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t artifact_key = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_artifact_key_format_arena(&identity, arena.arena(),
                                                      &artifact_key));
  return std::string(artifact_key.data, artifact_key.size);
}

TEST(AmdgpuArtifactKeyTest, ParsesCanonicalTargetClasses) {
  loom_amdgpu_target_identity_t identity = {};
  IREE_ASSERT_OK(loom_amdgpu_artifact_key_parse(
      IREE_SV("gfx942:sramecc+:xnack-"), &identity));
  ASSERT_NE(identity.target, nullptr);
  EXPECT_EQ(identity.target, LookupTarget("gfx942"));
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOM_AMDGPU_TARGET_FEATURE_OFF);

  IREE_ASSERT_OK(
      loom_amdgpu_artifact_key_parse(IREE_SV("gfx11-generic"), &identity));
  EXPECT_EQ(identity.target, LookupTarget("gfx11-generic"));

  IREE_ASSERT_OK(
      loom_amdgpu_artifact_key_parse(IREE_SV("gfx1250-a0"), &identity));
  EXPECT_EQ(identity.target, LookupTarget("gfx1250-a0"));
}

TEST(AmdgpuArtifactKeyTest, FormatsCanonicalFeatureOrder) {
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx942"), &identity);
  identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_ON;
  identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  EXPECT_EQ(FormatArtifactKey(identity), "gfx942:sramecc+:xnack-");
  EXPECT_EQ(FormatArenaArtifactKey(identity), "gfx942:sramecc+:xnack-");

  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250-a0"), &identity);
  EXPECT_EQ(FormatArtifactKey(identity), "gfx1250-a0");
  EXPECT_EQ(FormatArenaArtifactKey(identity), "gfx1250-a0");
}

TEST(AmdgpuArtifactKeyTest, RoundTripsEveryCompilerIdentityCombination) {
  static constexpr loom_amdgpu_target_id_feature_support_bit_t kFeatures[] = {
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
  };
  static constexpr loom_amdgpu_target_feature_state_t kStates[] = {
      LOOM_AMDGPU_TARGET_FEATURE_ANY,
      LOOM_AMDGPU_TARGET_FEATURE_OFF,
      LOOM_AMDGPU_TARGET_FEATURE_ON,
  };

  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t target_ordinal = 0; target_ordinal < target_count;
       ++target_ordinal) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(target_ordinal);
    ASSERT_NE(target, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr);

    loom_amdgpu_target_id_feature_support_bit_t
        supported_features[IREE_ARRAYSIZE(kFeatures)] = {};
    iree_host_size_t supported_feature_count = 0;
    for (const auto feature : kFeatures) {
      if (loom_amdgpu_processor_supports_target_id_features(processor,
                                                            feature)) {
        supported_features[supported_feature_count++] = feature;
      }
    }
    iree_host_size_t combination_count = 1;
    for (iree_host_size_t i = 0; i < supported_feature_count; ++i) {
      combination_count *= IREE_ARRAYSIZE(kStates);
    }

    for (iree_host_size_t combination = 0; combination < combination_count;
         ++combination) {
      loom_amdgpu_target_identity_t identity = {};
      loom_amdgpu_target_identity_initialize(target, &identity);
      iree_host_size_t state_digits = combination;
      for (iree_host_size_t feature_ordinal = 0;
           feature_ordinal < supported_feature_count; ++feature_ordinal) {
        loom_amdgpu_target_feature_state_t* state =
            loom_amdgpu_amdhsa_feature_state_select(
                &identity.amdhsa_features, supported_features[feature_ordinal]);
        ASSERT_NE(state, nullptr);
        *state = kStates[state_digits % IREE_ARRAYSIZE(kStates)];
        state_digits /= IREE_ARRAYSIZE(kStates);
      }

      const std::string artifact_key = FormatArtifactKey(identity);
      EXPECT_EQ(FormatArenaArtifactKey(identity), artifact_key);
      loom_amdgpu_target_identity_t parsed = {};
      IREE_ASSERT_OK(loom_amdgpu_artifact_key_parse(
          iree_make_string_view(artifact_key.data(), artifact_key.size()),
          &parsed));
      EXPECT_TRUE(loom_amdgpu_target_identity_equal(&identity, &parsed))
          << artifact_key;
    }
  }
}

TEST(AmdgpuArtifactKeyTest, RejectsMalformedOrUnsupportedKeys) {
  static const iree_string_view_t kInvalidKeys[] = {
      IREE_SV(""),
      IREE_SV("gfx9999"),
      IREE_SV("gfx942:"),
      IREE_SV("gfx942::xnack+"),
      IREE_SV("gfx942:wavefrontsize32+"),
      IREE_SV("gfx942:xnack+:xnack-"),
      IREE_SV("gfx1250:xnack+"),
  };
  for (const iree_string_view_t key : kInvalidKeys) {
    loom_amdgpu_target_identity_t identity = {};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_amdgpu_artifact_key_parse(key, &identity));
    EXPECT_EQ(identity.target, nullptr);
  }
}

}  // namespace
}  // namespace loom
