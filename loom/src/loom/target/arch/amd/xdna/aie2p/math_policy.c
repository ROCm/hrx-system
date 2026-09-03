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

typedef struct loom_aie2p_native_math_shape_t {
  // Semantic math operation implemented by the native form.
  loom_target_math_op_t math_op;
  // Scalar element type accepted by the native form.
  loom_scalar_type_t element_type;
  // Exact number of vector lanes accepted by the native form.
  int64_t lane_count;
  // Stable diagnostic constraint describing the required shape.
  iree_string_view_t shape_constraint_key;
  // Stable diagnostic constraint naming the selected native operation.
  iree_string_view_t native_constraint_key;
} loom_aie2p_native_math_shape_t;

static const loom_aie2p_native_math_shape_t kAie2pNativeMathShapes[] = {
    {
        .math_op = LOOM_TARGET_MATH_OP_MULF,
        .element_type = LOOM_SCALAR_TYPE_BF16,
        .lane_count = 32,
        .shape_constraint_key = IREE_SVL("math.shape.vector_bf16x32"),
        .native_constraint_key = IREE_SVL("math.op.native_vector_bf16x32"),
    },
    {
        .math_op = LOOM_TARGET_MATH_OP_ADDF,
        .element_type = LOOM_SCALAR_TYPE_F32,
        .lane_count = 64,
        .shape_constraint_key = IREE_SVL("math.shape.vector_f32x64"),
        .native_constraint_key = IREE_SVL("math.op.native_vector_f32x64"),
    },
};

static bool loom_aie2p_math_query_matches_native_shape(
    const loom_target_math_query_t* query,
    const loom_aie2p_native_math_shape_t* shape) {
  return query->lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR &&
         query->element_type == shape->element_type &&
         loom_type_is_vector(query->value_type) &&
         loom_type_rank(query->value_type) == 1 &&
         loom_type_is_all_static(query->value_type) &&
         loom_type_dim_static_size_at(query->value_type, 0) ==
             shape->lane_count;
}

static void loom_aie2p_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  iree_string_view_t constraint_key = IREE_SV("math.op.supported");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAie2pNativeMathShapes);
       ++i) {
    const loom_aie2p_native_math_shape_t* shape = &kAie2pNativeMathShapes[i];
    if (query->math_op != shape->math_op) continue;
    constraint_key = shape->shape_constraint_key;
    if (loom_aie2p_math_query_matches_native_shape(query, shape)) {
      *out_decision = loom_aie2p_math_keep(shape->native_constraint_key);
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
