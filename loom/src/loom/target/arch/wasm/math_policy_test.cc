// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/math_policy.h"

#include "iree/testing/gtest.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/types.h"

namespace {

TEST(MathPolicyTest, ExponentialFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_wasm_math_policy_registry_initialize(&registry);
  const auto* policy = loom_target_math_policy_registry_lookup(
      &registry, UINT64_C(0x49e70aa103499a7e));
  ASSERT_NE(policy, nullptr);
  const loom_target_config_t config = {};
  const loom_target_bundle_t bundle = {.config = &config};
  for (auto operation : {LOOM_TARGET_MATH_OP_EXPF, LOOM_TARGET_MATH_OP_EXP2F,
                         LOOM_TARGET_MATH_OP_EXPM1F}) {
    SCOPED_TRACE(operation);
    for (auto element : {LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2,
                         LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16,
                         LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_F64}) {
      SCOPED_TRACE(element);
      for (int lanes : {0, 1, 2, 3, 4, 8, 16, 32, 65}) {
        SCOPED_TRACE(lanes);
        for (auto flags : {LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                           LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                           LOOM_TARGET_MATH_FASTMATH_FLAG_FAST}) {
          SCOPED_TRACE(flags);
          const loom_target_math_query_t query = {
              .target_bundle = &bundle,
              .math_op = static_cast<loom_target_math_op_t>(operation),
              .lane_domain = lanes ? LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR
                                   : LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
              .value_type =
                  lanes ? loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                              loom_dim_pack_static(lanes), 0)
                        : loom_type_scalar(element),
              .element_type = static_cast<loom_scalar_type_t>(element),
              .fastmath_flags =
                  static_cast<loom_target_math_fastmath_flags_t>(flags),
          };
          loom_target_math_policy_decision_t decision = {};
          loom_target_math_policy_query(policy, &query, &decision);
          if (!(flags & LOOM_TARGET_MATH_FASTMATH_FLAG_AFN)) {
            EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
            continue;
          }
          EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
          EXPECT_EQ(
              decision.recipe,
              element == LOOM_SCALAR_TYPE_F32 || element == LOOM_SCALAR_TYPE_F64
                  ? LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXPONENTIAL
                  : LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND);
          EXPECT_EQ(decision.recipe_fastmath_flags, 0);
        }
      }
    }
  }
}

}  // namespace
