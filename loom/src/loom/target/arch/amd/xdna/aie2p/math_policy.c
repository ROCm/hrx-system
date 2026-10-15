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
    const loom_target_math_query_t* query, loom_target_math_recipe_t recipe,
    const loom_target_math_evaluation_t* evaluation,
    iree_string_view_t constraint_key) {
  loom_target_math_policy_decision_t decision = {
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = recipe,
      .constraint_key = constraint_key,
  };
  if (evaluation != NULL) {
    decision.evaluation = *evaluation;
    if (query->element_type == LOOM_SCALAR_TYPE_BF16) {
      decision.evaluation.conversion.input_flags =
          LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ;
      decision.evaluation.conversion.result_flags =
          LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ;
    }
  }
  return decision;
}

typedef enum loom_aie2p_math_lane_domain_set_bits_e {
  LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR =
      1u << LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
  LOOM_AIE2P_MATH_LANE_DOMAIN_SET_VECTOR =
      1u << LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
  LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR |
                                        LOOM_AIE2P_MATH_LANE_DOMAIN_SET_VECTOR,
} loom_aie2p_math_lane_domain_set_bits_t;
typedef uint8_t loom_aie2p_math_lane_domain_set_t;

typedef enum loom_aie2p_math_shape_e {
  // Match the authored rank-one vector lane count.
  LOOM_AIE2P_MATH_SHAPE_RANK_ONE = 0,
  // Match the total element count of any static vector shape.
  LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS = 1,
} loom_aie2p_math_shape_t;

typedef struct loom_aie2p_math_form_t {
  // Semantic math operation implemented by the lowering form.
  loom_target_math_op_t math_op;
  // Scalar element types accepted by the lowering form.
  loom_scalar_type_set_t element_types;
  // Source lane domains accepted by the lowering form.
  loom_aie2p_math_lane_domain_set_t lane_domains;
  // Interpretation of vector lane-count bounds.
  loom_aie2p_math_shape_t shape;
  // Inclusive source lane-count bounds accepted by the row.
  int64_t minimum_lane_count;
  int64_t maximum_lane_count;
  // Stable diagnostic constraint describing the accepted element types.
  iree_string_view_t element_constraint_key;
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
  // Optional representation used to evaluate the selected recipe.
  const loom_target_math_evaluation_t* evaluation;
} loom_aie2p_math_form_t;

#define LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES        \
  (LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 | \
   LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_BF16 |      \
   LOOM_SCALAR_TYPE_SET_F32)

static const loom_target_math_evaluation_t
    kAie2pApproximateActivationEvaluation = {
        .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
        .product_element_type = LOOM_SCALAR_TYPE_BF16,
        .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        .packet_lane_count = 16,
        .conversion =
            {
                .product_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
            },
};

static const loom_aie2p_math_form_t kAie2pMathForms[] = {
    {
        .math_op = LOOM_TARGET_MATH_OP_EXP2F,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = INT64_MAX,
        .element_constraint_key = IREE_SVL("math.element.f32"),
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_static_f32_exp2"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_exp2"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.exp2.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SQRTF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .element_constraint_key = IREE_SVL("math.element.f32"),
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_scalar_f32_root"),
        .form_constraint_key = IREE_SVL("math.op.native_approximate_sqrt_f32"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.sqrt.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_RSQRTF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .element_constraint_key = IREE_SVL("math.element.f32"),
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_scalar_f32_root"),
        .form_constraint_key = IREE_SVL("math.op.native_approximate_rsqrt_f32"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.rsqrt.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_GELUF_ERF,
        .element_types = LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .element_constraint_key =
            IREE_SVL("math.element.aie2p_approximate_activation"),
        .shape_constraint_key =
            IREE_SVL("math.shape.aie2p_approximate_activation_packet"),
        .form_constraint_key = IREE_SVL("math.recipe.gelu_erf_grouped_product"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.gelu.erf.exact"),
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
        .evaluation = &kAie2pApproximateActivationEvaluation,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_GELUF_TANH,
        .element_types = LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .element_constraint_key =
            IREE_SVL("math.element.aie2p_approximate_activation"),
        .shape_constraint_key =
            IREE_SVL("math.shape.aie2p_approximate_activation_packet"),
        .form_constraint_key =
            IREE_SVL("math.recipe.gelu_tanh_grouped_product"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.gelu.tanh.exact"),
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
        .evaluation = &kAie2pApproximateActivationEvaluation,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_GELUF_LOGISTIC,
        .element_types = LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .element_constraint_key =
            IREE_SVL("math.element.aie2p_approximate_activation"),
        .shape_constraint_key =
            IREE_SVL("math.shape.aie2p_approximate_activation_packet"),
        .form_constraint_key =
            IREE_SVL("math.recipe.gelu_logistic_grouped_product"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.gelu.logistic.exact"),
        .recipe = LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
        .evaluation = &kAie2pApproximateActivationEvaluation,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_LOGISTICF,
        .element_types = LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .element_constraint_key =
            IREE_SVL("math.element.aie2p_approximate_activation"),
        .shape_constraint_key =
            IREE_SVL("math.shape.aie2p_approximate_activation_packet"),
        .form_constraint_key =
            IREE_SVL("math.recipe.logistic_tanh_grouped_product"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.logistic.exact"),
        .recipe = LOOM_TARGET_MATH_RECIPE_LOGISTIC_TANH_F32,
        .evaluation = &kAie2pApproximateActivationEvaluation,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SILUF,
        .element_types = LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .element_constraint_key =
            IREE_SVL("math.element.aie2p_approximate_activation"),
        .shape_constraint_key =
            IREE_SVL("math.shape.aie2p_approximate_activation_packet"),
        .form_constraint_key =
            IREE_SVL("math.recipe.silu_logistic_grouped_product"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.silu.exact"),
        .recipe = LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
        .evaluation = &kAie2pApproximateActivationEvaluation,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SINF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_COSF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SINTURNSF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_COSTURNSF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_trig"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_mac_trig"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.trig.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_EXPF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_exp"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_exp"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.exp.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TANHF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_tanh"),
        .form_constraint_key = IREE_SVL("math.op.native_bf16_tanh"),
        .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
        .permission_constraint_key = IREE_SVL("math.tanh.exact_f32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_types = LOOM_SCALAR_TYPE_SET_F16,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_SCALAR,
        .minimum_lane_count = 1,
        .maximum_lane_count = 1,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f16_multiply"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary16"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.recipe.round_away_f32"),
        .recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY,
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ROUNDEVENF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_roundeven"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_TRUNCF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 16,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_round"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32_trunc"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 32,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_multiply"),
        .form_constraint_key = IREE_SVL("math.op.exact_binary32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 32,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_add"),
        .form_constraint_key = IREE_SVL("math.op.promote_vector_f32x64"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_VECTOR,
        .minimum_lane_count = 64,
        .maximum_lane_count = 64,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_add"),
        .form_constraint_key = IREE_SVL("math.op.native_vector_f32x64"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SUBF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_ALL,
        .minimum_lane_count = 1,
        .maximum_lane_count = 32,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_sub"),
        .form_constraint_key = IREE_SVL("math.op.promote_vector_f32x64"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_SUBF,
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_AIE2P_MATH_LANE_DOMAIN_SET_VECTOR,
        .minimum_lane_count = 64,
        .maximum_lane_count = 64,
        .shape_constraint_key = IREE_SVL("math.shape.aie2p_f32_sub"),
        .form_constraint_key = IREE_SVL("math.op.native_vector_f32x64"),
    },
};

#undef LOOM_AIE2P_APPROXIMATE_ACTIVATION_ELEMENT_TYPES

static bool loom_aie2p_math_query_matches_shape(
    const loom_target_math_query_t* query, const loom_aie2p_math_form_t* form) {
  if (query->lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR) {
    return loom_type_is_scalar(query->value_type) &&
           form->minimum_lane_count <= 1 && form->maximum_lane_count >= 1;
  }
  if (!loom_type_is_vector(query->value_type) ||
      !loom_type_is_all_static(query->value_type)) {
    return false;
  }
  int64_t lane_count = 0;
  if (form->shape == LOOM_AIE2P_MATH_SHAPE_STATIC_ELEMENTS) {
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(query->value_type, &element_count) ||
        element_count > INT64_MAX) {
      return false;
    }
    lane_count = (int64_t)element_count;
  } else {
    if (loom_type_rank(query->value_type) != 1) {
      return false;
    }
    lane_count = loom_type_dim_static_size_at(query->value_type, 0);
  }
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
    if (query->math_op != form->math_op) {
      continue;
    }
    const loom_aie2p_math_lane_domain_set_t query_lane_domain =
        (loom_aie2p_math_lane_domain_set_t)(1u << query->lane_domain);
    if (!iree_any_bit_set(form->lane_domains, query_lane_domain)) {
      continue;
    }
    if (!loom_scalar_type_set_contains(form->element_types,
                                       query->element_type)) {
      constraint_key = iree_string_view_is_empty(form->element_constraint_key)
                           ? form->shape_constraint_key
                           : form->element_constraint_key;
      continue;
    }
    constraint_key = form->shape_constraint_key;
    if (loom_aie2p_math_query_matches_shape(query, form)) {
      if (!iree_all_bits_set(query->fastmath_flags,
                             form->required_fastmath_flags)) {
        *out_decision = loom_aie2p_math_reject(form->permission_constraint_key);
        return;
      }
      *out_decision =
          form->recipe == LOOM_TARGET_MATH_RECIPE_UNKNOWN
              ? loom_aie2p_math_keep(form->form_constraint_key)
              : loom_aie2p_math_rewrite(query, form->recipe, form->evaluation,
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
