// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"

#include <string>

#include "iree/testing/gtest.h"

namespace {

struct GeluVariant {
  loom_target_math_op_t math_op;
  loom_target_math_recipe_t recipe;
  const char* form_constraint;
  const char* permission_constraint;
};

constexpr loom_scalar_type_t kGeluScalarElementTypes[] = {
    LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
    LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,
};

constexpr loom_scalar_type_t kGeluVectorElementTypes[] = {
    LOOM_SCALAR_TYPE_F8E4M3,
    LOOM_SCALAR_TYPE_F8E5M2,
    LOOM_SCALAR_TYPE_BF16,
    LOOM_SCALAR_TYPE_F32,
};

constexpr GeluVariant kGeluVariants[] = {
    {
        LOOM_TARGET_MATH_OP_GELUF_ERF,
        LOOM_TARGET_MATH_RECIPE_GELU_TANH_BF16_PACKET,
        "math.recipe.gelu_erf_tanh_bf16_packet",
        "math.gelu.erf.exact",
    },
    {
        LOOM_TARGET_MATH_OP_GELUF_TANH,
        LOOM_TARGET_MATH_RECIPE_GELU_TANH_BF16_PACKET,
        "math.recipe.gelu_tanh_bf16_packet",
        "math.gelu.tanh.exact",
    },
    {
        LOOM_TARGET_MATH_OP_GELUF_LOGISTIC,
        LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_BF16_PACKET,
        "math.recipe.gelu_logistic_bf16_packet",
        "math.gelu.logistic.exact",
    },
};

static std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class MathPolicyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_aie2p_math_policy_registry_initialize(&registry_);
    policy_ = loom_target_math_policy_registry_lookup(
        &registry_, IREE_SV("amd.xdna.aie2p.core"));
    ASSERT_NE(policy_, nullptr);
  }

  loom_target_math_policy_decision_t Query(
      loom_target_math_op_t math_op,
      loom_target_math_fastmath_flags_t fastmath_flags,
      loom_type_t value_type) const {
    const loom_target_math_lane_domain_t lane_domain =
        loom_type_is_scalar(value_type) ? LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR
                                        : LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR;
    const loom_target_math_query_t query = {
        /*.math_op=*/math_op,
        /*.lane_domain=*/lane_domain,
        /*.value_type=*/value_type,
        /*.element_type=*/loom_type_element_type(value_type),
        /*.fastmath_flags=*/fastmath_flags,
    };
    loom_target_math_policy_decision_t decision;
    loom_target_math_policy_query(policy_, &query, &decision);
    return decision;
  }

  loom_target_math_policy_registry_t registry_;
  const loom_target_math_policy_t* policy_ = nullptr;
};

TEST_F(MathPolicyTest, AdmitsCompleteScalarGeluFamily) {
  for (const GeluVariant& variant : kGeluVariants) {
    for (loom_scalar_type_t element_type : kGeluScalarElementTypes) {
      SCOPED_TRACE(::testing::Message()
                   << "math_op=" << variant.math_op
                   << " element_type=" << static_cast<int>(element_type));
      const loom_target_math_policy_decision_t decision =
          Query(variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                loom_type_scalar(element_type));
      EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
      EXPECT_EQ(decision.recipe, variant.recipe);
      EXPECT_EQ(StringViewToString(decision.constraint_key),
                variant.form_constraint);
    }
  }
}

TEST_F(MathPolicyTest, AdmitsCompleteVectorGeluFamily) {
  for (const GeluVariant& variant : kGeluVariants) {
    for (loom_scalar_type_t element_type : kGeluVectorElementTypes) {
      for (uint64_t element_count = 1; element_count <= 16; ++element_count) {
        const loom_type_t type = loom_type_shaped_1d(
            LOOM_TYPE_VECTOR, element_type, element_count, 0);
        SCOPED_TRACE(::testing::Message()
                     << "math_op=" << variant.math_op
                     << " element_type=" << static_cast<int>(element_type)
                     << " element_count=" << element_count);
        const loom_target_math_policy_decision_t decision =
            Query(variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, type);
        EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
        EXPECT_EQ(decision.recipe, variant.recipe);
        EXPECT_EQ(StringViewToString(decision.constraint_key),
                  variant.form_constraint);
      }

      const loom_type_t matrix_type =
          loom_type_shaped_2d(LOOM_TYPE_VECTOR, element_type, 4, 4, 0);
      const loom_target_math_policy_decision_t matrix_decision = Query(
          variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, matrix_type);
      EXPECT_EQ(matrix_decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
      EXPECT_EQ(matrix_decision.recipe, variant.recipe);
      EXPECT_EQ(StringViewToString(matrix_decision.constraint_key),
                variant.form_constraint);
    }
  }
}

TEST_F(MathPolicyTest, RejectsMissingApproximationPermission) {
  for (const GeluVariant& variant : kGeluVariants) {
    SCOPED_TRACE(::testing::Message() << "math_op=" << variant.math_op);
    const loom_target_math_policy_decision_t decision = Query(
        variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 16, 0));
    EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
    EXPECT_EQ(StringViewToString(decision.constraint_key),
              variant.permission_constraint);
  }
}

TEST_F(MathPolicyTest, RejectsUnsupportedPrecision) {
  for (const GeluVariant& variant : kGeluVariants) {
    SCOPED_TRACE(::testing::Message() << "math_op=" << variant.math_op);
    const loom_target_math_policy_decision_t scalar_decision =
        Query(variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
              loom_type_scalar(LOOM_SCALAR_TYPE_F64));
    EXPECT_EQ(scalar_decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
    EXPECT_EQ(StringViewToString(scalar_decision.constraint_key),
              "math.element.aie2p_gelu_bf16_packet_scalar");

    const loom_type_t vector_types[] = {
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F16, 16, 0),
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F64, 16, 0),
    };
    for (loom_type_t vector_type : vector_types) {
      const loom_target_math_policy_decision_t vector_decision = Query(
          variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, vector_type);
      EXPECT_EQ(vector_decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
      EXPECT_EQ(StringViewToString(vector_decision.constraint_key),
                "math.element.aie2p_gelu_bf16_packet_vector");
    }
  }
}

TEST_F(MathPolicyTest, RejectsVectorsLargerThanOnePacket) {
  const loom_type_t types[] = {
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 17, 0),
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 4, 5, 0),
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(0), 0),
  };
  for (const GeluVariant& variant : kGeluVariants) {
    for (loom_type_t type : types) {
      SCOPED_TRACE(::testing::Message()
                   << "math_op=" << variant.math_op
                   << " rank=" << static_cast<int>(loom_type_rank(type)));
      const loom_target_math_policy_decision_t decision =
          Query(variant.math_op, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, type);
      EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
      EXPECT_EQ(StringViewToString(decision.constraint_key),
                "math.shape.aie2p_gelu_bf16_packet");
    }
  }
}

}  // namespace
