// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/amdhsa_target_id.h"

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

static std::string FormatTargetId(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t target_id = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_amdhsa_target_id_format(&identity, arena.arena(),
                                                    &target_id));
  return std::string(target_id.data, target_id.size);
}

TEST(AmdhsaTargetIdTest, ExposesCanonicalPrefix) {
  EXPECT_TRUE(iree_string_view_equal(loom_amdgpu_amdhsa_target_id_prefix,
                                     IREE_SV("amdgcn-amd-amdhsa--")));
}

TEST(AmdhsaTargetIdTest, ParsesFeatureSuffix) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_ASSERT_OK(loom_amdgpu_amdhsa_target_id_parse(
      IREE_SV("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"), &target_id));
  ASSERT_NE(target_id.processor, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(target_id.processor->name, IREE_SV("gfx942")));
  EXPECT_TRUE(iree_string_view_equal(target_id.feature_suffix,
                                     IREE_SV("sramecc+:xnack-")));
  EXPECT_EQ(target_id.features.sramecc, LOOM_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(target_id.features.xnack, LOOM_AMDGPU_TARGET_FEATURE_OFF);
}

TEST(AmdhsaTargetIdTest, DistinguishesUnconstrainedAndUnsupportedFeatures) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_ASSERT_OK(loom_amdgpu_amdhsa_target_id_parse(
      IREE_SV("amdgcn-amd-amdhsa--gfx942"), &target_id));
  EXPECT_EQ(target_id.features.sramecc, LOOM_AMDGPU_TARGET_FEATURE_ANY);
  EXPECT_EQ(target_id.features.xnack, LOOM_AMDGPU_TARGET_FEATURE_ANY);

  IREE_ASSERT_OK(loom_amdgpu_amdhsa_target_id_parse(
      IREE_SV("amdgcn-amd-amdhsa--gfx1151"), &target_id));
  EXPECT_EQ(target_id.features.sramecc, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
  EXPECT_EQ(target_id.features.xnack, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
}

TEST(AmdhsaTargetIdTest, ProjectsCompilerTargetsToBackendProcessors) {
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(LookupTarget("gfx942"), &identity);
  identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_ON;
  identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_OFF;
  EXPECT_EQ(FormatTargetId(identity),
            "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-");

  loom_amdgpu_target_identity_initialize(LookupTarget("gfx1250-a0"), &identity);
  EXPECT_EQ(FormatTargetId(identity), "amdgcn-amd-amdhsa--gfx1250");
}

TEST(AmdhsaTargetIdTest, RoundTripsEveryProjectedCompilerIdentity) {
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

      const std::string formatted = FormatTargetId(identity);
      loom_amdgpu_amdhsa_target_id_t parsed = {};
      IREE_ASSERT_OK(loom_amdgpu_amdhsa_target_id_parse(
          iree_make_string_view(formatted.data(), formatted.size()), &parsed));
      EXPECT_EQ(parsed.processor, processor) << formatted;
      for (const auto feature : kFeatures) {
        EXPECT_EQ(
            loom_amdgpu_amdhsa_feature_state_query(&parsed.features, feature),
            loom_amdgpu_amdhsa_feature_state_query(&identity.amdhsa_features,
                                                   feature))
            << formatted;
      }
    }
  }
}

TEST(AmdhsaTargetIdTest, EncodesFeatureSuffixInElfFlags) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_ASSERT_OK(loom_amdgpu_amdhsa_target_id_parse(
      IREE_SV("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"), &target_id));
  uint32_t elf_flags = 0;
  IREE_ASSERT_OK(
      loom_amdgpu_amdhsa_target_id_elf_flags(&target_id, &elf_flags));
  EXPECT_EQ(elf_flags, target_id.processor->properties.elf.machine_flags |
                           LOOM_AMDGPU_ELF_FEATURE_SRAMECC_ON_V4 |
                           LOOM_AMDGPU_ELF_FEATURE_XNACK_OFF_V4);
}

TEST(AmdhsaTargetIdTest, RejectsMalformedOrUnsupportedTargetIds) {
  static const iree_string_view_t kInvalidTargetIds[] = {
      IREE_SV(""),
      IREE_SV("amdgcn-amd-amdpal--gfx1100"),
      IREE_SV("amdgcn-amd-amdhsa--gfx1100:"),
      IREE_SV("amdgcn-amd-amdhsa--gfx1100:wavefrontsize32+"),
      IREE_SV("amdgcn-amd-amdhsa--gfx1100:sramecc+"),
      IREE_SV("amdgcn-amd-amdhsa--gfx942:xnack+:xnack-"),
      IREE_SV("amdgcn-amd-amdhsa--gfx1100\\bad"),
  };
  for (const iree_string_view_t value : kInvalidTargetIds) {
    loom_amdgpu_amdhsa_target_id_t target_id = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_amdgpu_amdhsa_target_id_parse(value, &target_id));
    EXPECT_EQ(target_id.processor, nullptr);
  }
}

}  // namespace
}  // namespace loom
