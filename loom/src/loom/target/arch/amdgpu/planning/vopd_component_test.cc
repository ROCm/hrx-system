// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/vopd_component.h"

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/planning/vopd_data.h"

namespace loom {
namespace {

enum RuleIndex { kFmamk, kFmaak, kFmac, kSubtract, kAdd, kMove, kRuleCount };

class AmdgpuVopdComponentTest : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    loom_target_low_descriptor_registry_t registry = {};
    loom_amdgpu_low_descriptor_registry_initialize(&registry);
    const auto* descriptors = loom_low_descriptor_registry_lookup(
        &registry.registry, iree_make_cstring_view(GetParam()));
    if (descriptors == nullptr) {
      GTEST_SKIP() << "Descriptor set disabled by the build configuration";
    }
    const char* names[] = {"amdgpu.v_fmamk_f32", "amdgpu.v_fmaak_f32",
                           "amdgpu.v_fmac_f32",  "amdgpu.v_sub_f32",
                           "amdgpu.v_add_f32",   "amdgpu.v_mov_b32_copy"};
    const auto range = loom_amdgpu_vopd_component_descriptor_lookup_ranges
        [descriptors->descriptor_set_ordinal];
    for (uint32_t i = 0; i < kRuleCount; ++i) {
      SCOPED_TRACE(names[i]);
      const uint32_t ordinal = loom_low_descriptor_set_lookup_descriptor(
          descriptors, iree_make_cstring_view(names[i]));
      ASSERT_LT(ordinal, range.descriptor_lookup_count);
      const uint8_t index = loom_amdgpu_vopd_component_descriptor_lookups
          [range.first_descriptor_lookup + ordinal];
      ASSERT_NE(index, 0);
      rules_[i] = &loom_amdgpu_vopd_component_rules[index - 1];
    }
  }

  loom_amdgpu_vopd_candidate_component_t Component(RuleIndex rule,
                                                   uint16_t destination,
                                                   uint16_t source0,
                                                   uint16_t source1 = 0) const {
    const loom_amdgpu_vopd_pair_flags_t flags =
        (rule == kFmamk || rule == kFmaak) ? LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL
                                           : LOOM_AMDGPU_VOPD_PAIR_FLAG_NONE;
    return {
        .rule = rules_[rule],
        .vdst = destination,
        .src0 = source0,
        .vsrc1 = source1,
        .flags = flags,
        .literal_u32 = (rule == kFmamk || rule == kFmaak) ? 0x40000000u : 0u,
    };
  }

  void Check(
      loom_amdgpu_vopd_candidate_pair_t pair,
      loom_amdgpu_vopd_register_constraint_flags_t expected_constraints) {
    ASSERT_TRUE(loom_amdgpu_vopd_component_infos_pair_reason(
        pair.x.rule->info, pair.y.rule->info, &pair.reason));
    loom_amdgpu_vopd_register_constraint_flags_t constraints = 0;
    EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
              expected_constraints
                  ? LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS
                  : LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN);
    EXPECT_EQ(constraints, expected_constraints);
  }

  // Production descriptor rules for this target's component forms.
  const loom_amdgpu_vopd_component_rule_t* rules_[kRuleCount] = {};
};

TEST_P(AmdgpuVopdComponentTest, LiteralAddendsUseThirdCacheParity) {
  Check({.x = Component(kFmamk, 8, 0, 4), .y = Component(kFmamk, 9, 1, 6)},
        LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC2_BANK);
  Check({.x = Component(kFmamk, 8, 0, 4), .y = Component(kFmamk, 9, 1, 7)}, 0);
  // FMAMK's addend and FMAAK's second source use distinct cache ports.
  Check({.x = Component(kFmamk, 8, 0, 4), .y = Component(kFmaak, 9, 1, 4)}, 0);
}

TEST_P(AmdgpuVopdComponentTest, AccumulatorAndLiteralAddendShareThirdCache) {
  Check({.x = Component(kFmamk, 8, 0, 5), .y = Component(kFmac, 9, 1, 6)},
        LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC2_BANK);
  Check({.x = Component(kFmamk, 8, 0, 4), .y = Component(kFmac, 9, 1, 6)}, 0);
}

TEST_P(AmdgpuVopdComponentTest, BinarySourceInterlockUsesTargetBankMask) {
  const bool source_parity = iree_string_view_starts_with(
      iree_make_cstring_view(GetParam()), IREE_SV("amdgpu.rdna3"));
  Check(
      {.x = Component(kSubtract, 8, 0, 1), .y = Component(kSubtract, 9, 2, 3)},
      source_parity ? LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK |
                          LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK
                    : 0);
}

TEST_P(AmdgpuVopdComponentTest, SharedLiteralMustMatch) {
  for (uint32_t literal : {0u, 0x40000000u, UINT32_MAX}) {
    auto first = Component(kFmamk, 8, 0, 4);
    auto second = Component(kFmaak, 9, 1, 4);
    first.literal_u32 = second.literal_u32 = literal;
    loom_amdgpu_vopd_candidate_pair_t pair = {.x = first, .y = second};
    loom_amdgpu_vopd_register_constraint_flags_t constraints = 0;
    EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
              LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN);
    EXPECT_EQ(pair.flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
    EXPECT_EQ(pair.literal_u32, literal);
    pair.y.literal_u32 ^= 1;
    EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
              LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH);
    EXPECT_EQ(constraints, 0);
  }
}

TEST_P(AmdgpuVopdComponentTest, DestinationsMustHaveOppositeParity) {
  Check(
      {.x = Component(kSubtract, 8, 0, 1), .y = Component(kSubtract, 10, 3, 2)},
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY);
}

TEST_P(AmdgpuVopdComponentTest, CrossComponentAliasesRemainForbidden) {
  // Distinct bank residues avoid conflating aliases with cache conflicts.
  Check(
      {.x = Component(kSubtract, 8, 1, 2), .y = Component(kSubtract, 9, 8, 3)},
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0);
  Check(
      {.x = Component(kSubtract, 8, 0, 1), .y = Component(kSubtract, 9, 3, 8)},
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1);
  Check(
      {.x = Component(kSubtract, 8, 9, 2), .y = Component(kSubtract, 9, 0, 3)},
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0);
  Check(
      {.x = Component(kSubtract, 8, 0, 9), .y = Component(kSubtract, 9, 3, 2)},
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1);
}

TEST_P(AmdgpuVopdComponentTest, SourceCommutationRepairsBankConflicts) {
  loom_amdgpu_vopd_candidate_pair_t pair = {
      .x = Component(kAdd, 8, 0, 1),
      .y = Component(kAdd, 9, 4, 5),
  };
  loom_amdgpu_vopd_register_constraint_flags_t constraints = 0;
  EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
            LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN);
  EXPECT_EQ(pair.flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED);
  EXPECT_EQ(pair.x.src0, 1);
  EXPECT_EQ(pair.x.vsrc1, 0);
  EXPECT_EQ(pair.y.src0, 4);
  EXPECT_EQ(pair.y.vsrc1, 5);

  pair = {.x = Component(kSubtract, 8, 0, 1), .y = Component(kAdd, 9, 4, 5)};
  EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
            LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN);
  EXPECT_EQ(pair.flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED);
  EXPECT_EQ(pair.y.src0, 5);
  EXPECT_EQ(pair.y.vsrc1, 4);

  pair = {.x = Component(kAdd, 8, 0, 1), .y = Component(kAdd, 10, 4, 5)};
  EXPECT_EQ(loom_amdgpu_vopd_candidate_pair_resolve(&pair, &constraints),
            LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS);
  EXPECT_EQ(pair.flags, 0);
  EXPECT_EQ(pair.x.src0, 0);
  EXPECT_EQ(pair.x.vsrc1, 1);
  EXPECT_EQ(pair.y.src0, 4);
  EXPECT_EQ(pair.y.vsrc1, 5);
}

TEST_P(AmdgpuVopdComponentTest, DualMovesUseTargetCacheRouting) {
  const bool separate_caches = !iree_string_view_starts_with(
      iree_make_cstring_view(GetParam()), IREE_SV("amdgpu.rdna3"));
  Check({.x = Component(kMove, 2, 5), .y = Component(kMove, 3, 1)},
        separate_caches ? 0
                        : LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK);
}

INSTANTIATE_TEST_SUITE_P(Targets, AmdgpuVopdComponentTest,
                         ::testing::Values("amdgpu.rdna3.core",
                                           "amdgpu.rdna3_5.core",
                                           "amdgpu.rdna4.core",
                                           "amdgpu.rdna4.gfx125x.core"));

}  // namespace
}  // namespace loom
