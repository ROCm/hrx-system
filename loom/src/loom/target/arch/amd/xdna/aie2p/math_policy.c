// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"

static loom_target_math_policy_decision_t loom_aie2p_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_aie2p_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_aie2p_math_rewrite(
    loom_target_math_recipe_t recipe, iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = recipe,
      .constraint_key = constraint_key,
  };
}

typedef struct loom_aie2p_math_form_t {
  // Semantic math operation implemented by the lowering form.
  loom_target_math_op_t math_op;
  // Scalar element type accepted by the lowering form.
  loom_scalar_type_t element_type;
  // Source lane domain accepted by the lowering form.
  loom_target_math_lane_domain_t lane_domain;
  // Inclusive source lane-count bounds accepted by the row.
  int64_t minimum_lane_count;
  int64_t maximum_lane_count;
  // Stable diagnostic constraint describing the required shape.
  iree_string_view_t shape_constraint_key;
  // Stable diagnostic constraint naming the selected lowering form.
  iree_string_view_t form_constraint_key;
  // Fast-math permissions required by the selected lowering form.
  loom_target_math_fastmath_flags_t required_fastmath_flags;
  // Stable diagnostic constraint reported when permissions are absent.
  iree_string_view_t permission_constraint_key;
  // Target-independent rewrite recipe, or UNKNOWN when the op stays intact.
  loom_target_math_recipe_t recipe;
} loom_aie2p_math_form_t;

static const loom_aie2p_math_form_t kAie2pMathForms[] = {
    {
        .math_op = LOOM_TARGET_MATH_OP_SINF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_COSF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SINTURNSF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_COSTURNSF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_EXPF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_exp"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_exp"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.exp.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_EXPF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_exp"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_exp"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.exp.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TANHF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_tanh"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_tanh"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.tanh.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TANHF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_tanh"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_tanh"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.tanh.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_type = LOOM_SCALAR_TYPE_BF16,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 32,
        .maximum_lane_count = 32,
        .shape_constraint_key = IREE_SVL("math.shape.vector_bf16x32"),
        .form_constraint_key = IREE_SVL("math.op.native_vector_bf16x32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_type = LOOM_SCALAR_TYPE_F16,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f16_multiply"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary16"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.recipe.round_away_f32"),
        .recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.recipe.round_away_f32"),
        .recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDEVENF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_roundeven"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDEVENF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_roundeven"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TRUNCF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_trunc"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TRUNCF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_trunc"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_multiply"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_multiply"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_add"),
        .form_constraint_key = IREE_SVL("math.op.promote_vector_f32x64"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_add"),
        .form_constraint_key = IREE_SVL("math.op.promote_vector_f32x64"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_domain = LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
        .minimum_lane_count = 64,
        .maximum_lane_count = 64,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_add"),
        .form_constraint_key = IREE_SVL("math.op.native_vector_f32x64"),
    },
};

static bool loom_aie2p_math_query_matches_form(
    const loom_target_math_query_t* query, const loom_aie2p_math_form_t* form) {
  if (query->lane_domain != form->lane_domain ||
      query->element_type != form->element_type) {
    return false;
  }
  if (query->lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR) {
    return loom_type_is_scalar(query->value_type) &&
           form->minimum_lane_count <= 1 && form->maximum_lane_count >= 1;
  }
  if (!loom_type_is_vector(query->value_type) ||
      loom_type_rank(query->value_type) != 1 ||
      !loom_type_is_all_static(query->value_type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(query->value_type, 0);
  return lane_count >= form->minimum_lane_count &&
         lane_count <= form->maximum_lane_count;
}

static void loom_aie2p_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  iree_string_view_t constraint_key = IREE_SV("math.op.supported");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAie2pMathForms); ++i) {
    const loom_aie2p_math_form_t* form = &kAie2pMathForms[i];
    if (query->math_op != form->math_op) continue;
    constraint_key = form->shape_constraint_key;
    if (loom_aie2p_math_query_matches_form(query, form)) {
      if (!iree_all_bits_set(query->fastmath_flags,
                             form->required_fastmath_flags)) {
        *out_decision = loom_aie2p_math_reject(form->permission_constraint_key);
        return;
      }
      *out_decision = form->recipe == LOOM_TARGET_MATH_RECIPE_UNKNOWN
                          ? loom_aie2p_math_keep(form->form_constraint_key)
                          : loom_aie2p_math_rewrite(form->recipe,
                                                    form->form_constraint_key);
      return;
    }
  }
  *out_decision = loom_aie2p_math_reject(constraint_key);
}

static const loom_target_math_policy_t kAie2pMathPolicy = {
    .name = IREE_SVL("aie2p-math"),
    .query = loom_aie2p_math_policy_query,
};

static const loom_target_math_policy_registry_entry_t
    kAie2pMathPolicyEntries[] = {
        {
            .contract_set_key = IREE_SVL("amd.xdna.aie2p.core"),
            .policy = &kAie2pMathPolicy,
        },
};

void loom_aie2p_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kAie2pMathPolicyEntries,
      IREE_ARRAYSIZE(kAie2pMathPolicyEntries));
}
