// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/math_policy.h"

static loom_target_math_policy_decision_t loom_ireevm_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_ireevm_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static bool loom_ireevm_math_op_is_native_scalar(
    loom_target_math_op_t math_op) {
  switch (math_op) {
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_MULF:
    case LOOM_TARGET_MATH_OP_EXPF:
    case LOOM_TARGET_MATH_OP_LOGF:
    case LOOM_TARGET_MATH_OP_LOG2F:
    case LOOM_TARGET_MATH_OP_SINF:
    case LOOM_TARGET_MATH_OP_COSF:
    case LOOM_TARGET_MATH_OP_TANHF:
    case LOOM_TARGET_MATH_OP_POWF:
    case LOOM_TARGET_MATH_OP_ERFF:
    case LOOM_TARGET_MATH_OP_CEILF:
    case LOOM_TARGET_MATH_OP_FLOORF:
    case LOOM_TARGET_MATH_OP_ROUNDF:
    case LOOM_TARGET_MATH_OP_ROUNDEVENF:
      return true;
    case LOOM_TARGET_MATH_OP_LOGISTICF:
    case LOOM_TARGET_MATH_OP_SILUF:
    case LOOM_TARGET_MATH_OP_SOFTPLUSF:
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
    case LOOM_TARGET_MATH_OP_SINTURNSF:
    case LOOM_TARGET_MATH_OP_COSTURNSF:
    case LOOM_TARGET_MATH_OP_TRUNCF:
    case LOOM_TARGET_MATH_OP_UNKNOWN:
      return false;
  }
  return false;
}

static void loom_ireevm_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  if (query->lane_domain != LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR) {
    *out_decision = loom_ireevm_math_reject(IREE_SV("math.scalar"));
    return;
  }
  if (query->element_type != LOOM_SCALAR_TYPE_F32 &&
      query->element_type != LOOM_SCALAR_TYPE_F64) {
    *out_decision = loom_ireevm_math_reject(IREE_SV("math.element.f32_f64"));
    return;
  }
  if (!loom_ireevm_math_op_is_native_scalar(query->math_op)) {
    *out_decision = loom_ireevm_math_reject(IREE_SV("math.op.supported"));
    return;
  }

  *out_decision = loom_ireevm_math_keep(IREE_SV("math.op.native_scalar"));
}

static const loom_target_math_policy_t kIreeVmMathPolicy = {
    .name = IREE_SVL("ireevm-math"),
    .query = loom_ireevm_math_policy_query,
};

static const loom_target_math_policy_registry_entry_t
    kIreeVmMathPolicyEntries[] = {
        {/*.contract_set_key=*/IREE_SVL("ireevm.core"),
         /*.policy=*/&kIreeVmMathPolicy},
};

void loom_ireevm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kIreeVmMathPolicyEntries,
      IREE_ARRAYSIZE(kIreeVmMathPolicyEntries));
}
