// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/math_policy.h"

#include "iree/testing/gtest.h"
#include "loom/ir/types.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/math_policy.h"
#include "loom/target/types.h"

namespace {

struct Profile {
  // Registered contract selecting the production math policy.
  const char* contract;
  // Stable identity of |contract|.
  uint64_t contract_stable_id;
  // Largest native F32/F64 vector, in bits.
  int vector_bits;
};
constexpr Profile kProfiles[] = {
    {"x86.avx2.core", UINT64_C(0x34e8f11d46a8a495), 256},
    {"x86.avx2_features.core", UINT64_C(0x4a4769fa07a2ddeb), 256},
    {"x86.packed_dot.core", UINT64_C(0x6ebbd729aef5811e), 256},
    {"x86.avx512.core", UINT64_C(0x06092b7cdc8ed7f3), 512},
    {"x86.avx512_features.core", UINT64_C(0x7ddea86e8df4a0d5), 512},
};

TEST(MathPolicyTest, ApproximateElementaryFamilies) {
  loom_target_math_policy_registry_t registry = {};
  loom_x86_math_policy_registry_initialize(&registry);
  const loom_target_math_op_t operations[] = {
      LOOM_TARGET_MATH_OP_LOGF,   LOOM_TARGET_MATH_OP_LOG2F,
      LOOM_TARGET_MATH_OP_LOG10F, LOOM_TARGET_MATH_OP_LOG1PF,
      LOOM_TARGET_MATH_OP_EXPF,   LOOM_TARGET_MATH_OP_EXP2F,
      LOOM_TARGET_MATH_OP_EXPM1F,
  };
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  for (const Profile& profile : kProfiles) {
    SCOPED_TRACE(profile.contract);
    const auto* policy = loom_target_math_policy_registry_lookup(
        &registry, profile.contract_stable_id);
    ASSERT_NE(policy, nullptr);
    for (loom_target_math_op_t operation : operations) {
      SCOPED_TRACE(operation);
      for (loom_scalar_type_t element : elements) {
        SCOPED_TRACE(element);
        // Zero denotes a scalar; every other shape exercises vector policy,
        // including tails and values wider than any physical packet.
        for (int lanes : {0, 1, 2, 3, 4, 8, 16, 32, 65}) {
          SCOPED_TRACE(lanes);
          for (uint8_t flags :
               {uint8_t(0), uint8_t(LOOM_TARGET_MATH_FASTMATH_FLAG_AFN),
                uint8_t(LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)}) {
            SCOPED_TRACE(flags);
            const loom_target_math_query_t query = {
                .math_op = operation,
                .lane_domain = lanes ? LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR
                                     : LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
                .value_type =
                    lanes ? loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                                loom_dim_pack_static(lanes), 0)
                          : loom_type_scalar(element),
                .element_type = element,
                .fastmath_flags = flags,
            };
            loom_target_math_policy_decision_t decision = {};
            loom_target_math_policy_query(policy, &query, &decision);
            if (flags == 0) {
              EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
            } else {
              EXPECT_EQ(decision.action,
                        LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
              loom_target_math_recipe_t recipe =
                  LOOM_TARGET_MATH_RECIPE_APPROXIMATE_LOGARITHM;
              if (operation == LOOM_TARGET_MATH_OP_EXPM1F ||
                  operation == LOOM_TARGET_MATH_OP_EXPF ||
                  operation == LOOM_TARGET_MATH_OP_EXP2F) {
                recipe = LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXPONENTIAL;
                if (element == LOOM_SCALAR_TYPE_F32) {
                  if (operation == LOOM_TARGET_MATH_OP_EXPF) {
                    recipe = LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32;
                  } else if (operation == LOOM_TARGET_MATH_OP_EXP2F) {
                    recipe = LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32;
                  }
                }
              }
              EXPECT_EQ(decision.recipe,
                        loom_scalar_type_bitwidth(element) < 32
                            ? LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND
                            : recipe);
            }
          }
        }
      }
    }
  }
}

TEST(MathPolicyTest, ExactRoundingFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_x86_math_policy_registry_initialize(&registry);
  const loom_target_math_op_t operations[] = {
      LOOM_TARGET_MATH_OP_CEILF,      LOOM_TARGET_MATH_OP_FLOORF,
      LOOM_TARGET_MATH_OP_ROUNDEVENF, LOOM_TARGET_MATH_OP_TRUNCF,
      LOOM_TARGET_MATH_OP_ROUNDF,
  };
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  for (const Profile& profile : kProfiles) {
    SCOPED_TRACE(profile.contract);
    const auto* policy = loom_target_math_policy_registry_lookup(
        &registry, profile.contract_stable_id);
    ASSERT_NE(policy, nullptr);
    const uint64_t feature_sets[] = {
        0,
        LOOM_X86_FEATURE_AVX512_VL,
        LOOM_X86_FEATURE_AVX512_FP16,
        LOOM_X86_FEATURE_AVX512_FP16 | LOOM_X86_FEATURE_AVX512_VL,
    };
    for (uint64_t features : feature_sets) {
      SCOPED_TRACE(features);
      const loom_target_config_t config = {.contract_feature_bits = features};
      const loom_target_bundle_t bundle = {.config = &config};
      for (loom_target_math_op_t operation : operations) {
        SCOPED_TRACE(operation);
        for (loom_scalar_type_t element : elements) {
          SCOPED_TRACE(element);
          for (int lanes : {0, 1, 2, 3, 4, 8, 16, 32, 65}) {
            SCOPED_TRACE(lanes);
            for (uint8_t flags :
                 {uint8_t(0), uint8_t(LOOM_TARGET_MATH_FASTMATH_FLAG_FAST)}) {
              const loom_target_math_query_t query = {
                  .target_bundle = &bundle,
                  .math_op = operation,
                  .lane_domain = lanes ? LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR
                                       : LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
                  .value_type = lanes ? loom_type_shaped_1d(
                                            LOOM_TYPE_VECTOR, element,
                                            loom_dim_pack_static(lanes), 0)
                                      : loom_type_scalar(element),
                  .element_type = element,
                  .fastmath_flags = flags,
              };
              const bool away = operation == LOOM_TARGET_MATH_OP_ROUNDF;
              const bool native_half =
                  !away && element == LOOM_SCALAR_TYPE_F16 &&
                  profile.vector_bits == 512 &&
                  (features & LOOM_X86_FEATURE_AVX512_FP16);
              const bool widen =
                  loom_scalar_type_bitwidth(element) < 32 && !native_half;
              loom_target_math_policy_decision_t decision = {};
              loom_target_math_policy_query(policy, &query, &decision);
              EXPECT_EQ(decision.action,
                        widen || away ? LOOM_TARGET_MATH_POLICY_ACTION_REWRITE
                                      : LOOM_TARGET_MATH_POLICY_ACTION_KEEP);
              if (widen || away) {
                EXPECT_EQ(decision.recipe,
                          widen ? LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND
                                : LOOM_TARGET_MATH_RECIPE_ROUND_AWAY);
              }
            }
          }
        }
      }
    }
  }
}

TEST(MathPolicyTest, NativeFmaFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_x86_math_policy_registry_initialize(&registry);
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I32,
  };
  for (const Profile& profile : kProfiles) {
    SCOPED_TRACE(profile.contract);
    const auto* policy = loom_target_math_policy_registry_lookup(
        &registry, profile.contract_stable_id);
    ASSERT_NE(policy, nullptr);
    const uint64_t feature_sets[] = {
        0,
        LOOM_X86_FEATURE_AVX512_VL,
        LOOM_X86_FEATURE_AVX512_FP16,
        LOOM_X86_FEATURE_AVX512_FP16 | LOOM_X86_FEATURE_AVX512_VL,
    };
    for (uint64_t features : feature_sets) {
      SCOPED_TRACE(features);
      const bool fp16 = (features & LOOM_X86_FEATURE_AVX512_FP16) != 0;
      const bool vector_length = (features & LOOM_X86_FEATURE_AVX512_VL) != 0;
      const loom_target_config_t config = {
          .contract_feature_bits = features,
      };
      const loom_target_bundle_t bundle = {.config = &config};
      for (loom_scalar_type_t element : elements) {
        SCOPED_TRACE(element);
        const bool native_float =
            element == LOOM_SCALAR_TYPE_F32 || element == LOOM_SCALAR_TYPE_F64;
        const bool native_half = element == LOOM_SCALAR_TYPE_F16 && fp16 &&
                                 profile.vector_bits == 512;
        EXPECT_EQ(loom_target_math_policy_prefers_fma(
                      policy, &bundle, loom_type_scalar(element), 0),
                  native_float || native_half);
        for (int lanes = 1; lanes <= 33; ++lanes) {
          SCOPED_TRACE(lanes);
          const int bits = lanes * loom_scalar_type_bitwidth(element);
          const bool power_of_two = (lanes & (lanes - 1)) == 0;
          const bool expected =
              power_of_two &&
              ((native_float && bits >= 64 && bits <= profile.vector_bits) ||
               (native_half && bits >= 32 && bits <= 512 &&
                (bits == 512 || vector_length)));
          const loom_type_t type = loom_type_shaped_1d(
              LOOM_TYPE_VECTOR, element, loom_dim_pack_static(lanes), 0);
          EXPECT_EQ(
              loom_target_math_policy_prefers_fma(policy, &bundle, type, 0),
              expected);
        }
        const loom_type_t dynamic = loom_type_shaped_1d(
            LOOM_TYPE_VECTOR, element, loom_dim_pack_dynamic(1), 0);
        const loom_type_t rank_two = loom_type_shaped_2d(
            LOOM_TYPE_VECTOR, element, loom_dim_pack_static(2),
            loom_dim_pack_static(2), 0);
        EXPECT_FALSE(
            loom_target_math_policy_prefers_fma(policy, &bundle, dynamic, 0));
        EXPECT_FALSE(
            loom_target_math_policy_prefers_fma(policy, &bundle, rank_two, 0));
      }
    }
  }
}

}  // namespace
