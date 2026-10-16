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

TEST(MathPolicyTest, NativeFmaFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_x86_math_policy_registry_initialize(&registry);
  struct Profile {
    // Registered contract selecting the production math policy.
    const char* contract;
    // Largest native F32/F64 vector, in bits.
    int vector_bits;
  };
  const Profile profiles[] = {
      {"x86.avx2.core", 256},
      {"x86.avx2_features.core", 256},
      {"x86.packed_dot.core", 256},
      {"x86.avx512.core", 512},
      {"x86.avx512_features.core", 512},
  };
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16, LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64, LOOM_SCALAR_TYPE_I32,
  };
  for (const Profile& profile : profiles) {
    SCOPED_TRACE(profile.contract);
    const auto* policy = loom_target_math_policy_registry_lookup(
        &registry, iree_make_cstring_view(profile.contract));
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
