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

static bool loom_aie2p_math_query_is_bf16x32_vector(
    const loom_target_math_query_t* query) {
  return query->lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR &&
         query->element_type == LOOM_SCALAR_TYPE_BF16 &&
         loom_type_is_vector(query->value_type) &&
         loom_type_rank(query->value_type) == 1 &&
         loom_type_is_all_static(query->value_type) &&
         loom_type_dim_static_size_at(query->value_type, 0) == 32;
}

static void loom_aie2p_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  if (query->math_op != LOOM_TARGET_MATH_OP_MULF) {
    *out_decision = loom_aie2p_math_reject(IREE_SV("math.op.supported"));
    return;
  }
  if (!loom_aie2p_math_query_is_bf16x32_vector(query)) {
    *out_decision =
        loom_aie2p_math_reject(IREE_SV("math.shape.vector_bf16x32"));
    return;
  }
  *out_decision =
      loom_aie2p_math_keep(IREE_SV("math.op.native_vector_bf16x32"));
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
