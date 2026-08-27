// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_id/target_id.h"

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

static std::string FormatArtifactTargetKey(
    const loom_amdgpu_target_identity_t& identity) {
  char buffer[128] = {};
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_artifact_target_key_format(
      &identity, sizeof(buffer), buffer, &target_key));
  return std::string(target_key.data, target_key.size);
}

static std::string FormatArenaArtifactTargetKey(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_artifact_target_key_format_arena(
      &identity, arena.arena(), &target_key));
  return std::string(target_key.data, target_key.size);
}

static std::string FormatCodeObjectTargetId(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t target_id = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_amdhsa_code_object_target_id_format(
      &identity, arena.arena(), &target_id));
  return std::string(target_id.data, target_id.size);
}

TEST(AmdgpuTargetIdTest, SeparatesArtifactAndCodeObjectIdentity) {
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx942"), &identity);
  identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_ON;
  identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  EXPECT_EQ(FormatArtifactTargetKey(identity), "gfx942:sramecc+:xnack-");
  EXPECT_EQ(FormatArenaArtifactTargetKey(identity), "gfx942:sramecc+:xnack-");
  EXPECT_EQ(FormatCodeObjectTargetId(identity),
            "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-");

  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250"), &identity);
  EXPECT_EQ(FormatArtifactTargetKey(identity), "gfx1250");
  EXPECT_EQ(FormatCodeObjectTargetId(identity), "amdgcn-amd-amdhsa--gfx1250");

  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250-a0"), &identity);
  EXPECT_EQ(FormatArtifactTargetKey(identity), "gfx1250-a0");
  EXPECT_EQ(FormatCodeObjectTargetId(identity), "amdgcn-amd-amdhsa--gfx1250");
}

TEST(AmdgpuTargetIdTest, ProjectsEveryGeneratedIdentityCombination) {
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
    iree_host_size_t feature_combination_count = 1;
    for (iree_host_size_t i = 0; i < supported_feature_count; ++i) {
      feature_combination_count *= IREE_ARRAYSIZE(kStates);
    }

    for (iree_host_size_t combination = 0;
         combination < feature_combination_count; ++combination) {
      loom_amdgpu_target_identity_t identity = {};
      loom_amdgpu_target_identity_initialize(target, &identity);
      iree_host_size_t state_digits = combination;
      for (iree_host_size_t feature_ordinal = 0;
           feature_ordinal < supported_feature_count; ++feature_ordinal) {
        loom_amdgpu_target_feature_state_t* feature_state =
            loom_amdgpu_amdhsa_feature_state_select(
                &identity.amdhsa_features, supported_features[feature_ordinal]);
        ASSERT_NE(feature_state, nullptr);
        *feature_state = kStates[state_digits % IREE_ARRAYSIZE(kStates)];
        state_digits /= IREE_ARRAYSIZE(kStates);
      }

      const std::string code_object_target = FormatCodeObjectTargetId(identity);
      loom_amdgpu_amdhsa_target_id_t parsed = {};
      IREE_ASSERT_OK(loom_amdgpu_target_info_parse_amdhsa_target_id(
          iree_make_string_view(code_object_target.data(),
                                code_object_target.size()),
          &parsed));
      EXPECT_EQ(parsed.processor, processor) << code_object_target;
      for (const auto feature : kFeatures) {
        EXPECT_EQ(
            loom_amdgpu_amdhsa_feature_state_query(&parsed.features, feature),
            loom_amdgpu_amdhsa_feature_state_query(&identity.amdhsa_features,
                                                   feature))
            << code_object_target;
      }

      std::string expected_artifact_target(target->name.data,
                                           target->name.size);
      expected_artifact_target.append(code_object_target.substr(
          loom_amdgpu_target_info_amdhsa_target_id_prefix.size +
          processor->name.size));
      EXPECT_EQ(FormatArtifactTargetKey(identity), expected_artifact_target);
      EXPECT_EQ(FormatArenaArtifactTargetKey(identity),
                expected_artifact_target);
    }
  }
}

}  // namespace
}  // namespace loom
