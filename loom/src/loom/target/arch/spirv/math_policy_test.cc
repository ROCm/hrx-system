// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/math_policy.h"

#include "iree/testing/gtest.h"
#include "loom/ir/types.h"
#include "loom/target/math_policy.h"
#include "loom/target/types.h"

namespace {

TEST(MathPolicyTest, TrigonometryFamily) {
  loom_target_math_policy_registry_t registry = {};
  loom_spirv_math_policy_registry_initialize(&registry);
  const auto* policy = loom_target_math_policy_registry_lookup(
      &registry, IREE_SV("spirv.logical.core"));
  ASSERT_NE(policy, nullptr);
  const loom_target_config_t config = {};
  const loom_target_bundle_t bundle = {.config = &config};
  const loom_scalar_type_t elements[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  const loom_target_math_op_t operations[] = {
      LOOM_TARGET_MATH_OP_SINF,
      LOOM_TARGET_MATH_OP_COSF,
      LOOM_TARGET_MATH_OP_SINTURNSF,
      LOOM_TARGET_MATH_OP_COSTURNSF,
  };
  for (auto operation : operations) {
    SCOPED_TRACE(operation);
    for (auto element : elements) {
      SCOPED_TRACE(element);
      for (int lanes : {0, 1, 2, 3, 4, 8, 16, 32}) {
        SCOPED_TRACE(lanes);
        for (bool approximate : {false, true}) {
          SCOPED_TRACE(approximate);
          const loom_target_math_query_t query = {
              .target_bundle = &bundle,
              .math_op = operation,
              .lane_domain = lanes ? LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR
                                   : LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
              .value_type =
                  lanes ? loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                              loom_dim_pack_static(lanes), 0)
                        : loom_type_scalar(element),
              .element_type = element,
              .fastmath_flags = static_cast<loom_target_math_fastmath_flags_t>(
                  approximate ? LOOM_TARGET_MATH_FASTMATH_FLAG_AFN
                              : LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
          };
          loom_target_math_policy_decision_t decision = {};
          loom_target_math_policy_query(policy, &query, &decision);
          if (!approximate || element == LOOM_SCALAR_TYPE_F64) {
            EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
            continue;
          }
          const bool radians = operation == LOOM_TARGET_MATH_OP_SINF ||
                               operation == LOOM_TARGET_MATH_OP_COSF;
          const bool native = radians && (element == LOOM_SCALAR_TYPE_F16 ||
                                          element == LOOM_SCALAR_TYPE_F32);
          EXPECT_EQ(decision.action,
                    native ? LOOM_TARGET_MATH_POLICY_ACTION_KEEP
                           : LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
          if (!native) {
            EXPECT_EQ(decision.recipe,
                      LOOM_TARGET_MATH_RECIPE_TRIGONOMETRIC_F32);
            EXPECT_EQ(decision.evaluation.accumulator_element_type,
                      LOOM_SCALAR_TYPE_F32);
            EXPECT_EQ(decision.evaluation.kind,
                      element == LOOM_SCALAR_TYPE_F32
                          ? LOOM_TARGET_MATH_EVALUATION_SOURCE
                          : LOOM_TARGET_MATH_EVALUATION_WIDENED);
          }
          EXPECT_EQ(decision.recipe_fastmath_flags, 0);
        }
      }
    }
  }
}

}  // namespace
