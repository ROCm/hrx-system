// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/math_policy.h"

static loom_target_math_policy_decision_t loom_vm_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_vm_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static void loom_vm_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  if (query->lane_domain != LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR &&
      query->lane_domain != LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR) {
    *out_decision = loom_vm_math_reject(IREE_SV("math.lane.scalar_vector"));
    return;
  }
  if (query->element_type != LOOM_SCALAR_TYPE_F32 &&
      query->element_type != LOOM_SCALAR_TYPE_F64) {
    *out_decision = loom_vm_math_reject(IREE_SV("math.element.f32_f64"));
    return;
  }
  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_UNKNOWN:
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
      *out_decision = loom_vm_math_reject(IREE_SV("math.op.supported"));
      return;
    case LOOM_TARGET_MATH_OP_LOGISTICF:
    case LOOM_TARGET_MATH_OP_SILUF:
    case LOOM_TARGET_MATH_OP_SOFTPLUSF:
    case LOOM_TARGET_MATH_OP_EXPF:
    case LOOM_TARGET_MATH_OP_ERFF:
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_MULF:
    case LOOM_TARGET_MATH_OP_LOGF:
    case LOOM_TARGET_MATH_OP_LOG2F:
    case LOOM_TARGET_MATH_OP_SINF:
    case LOOM_TARGET_MATH_OP_COSF:
    case LOOM_TARGET_MATH_OP_SINTURNSF:
    case LOOM_TARGET_MATH_OP_COSTURNSF:
    case LOOM_TARGET_MATH_OP_TANHF:
    case LOOM_TARGET_MATH_OP_POWF:
    case LOOM_TARGET_MATH_OP_CEILF:
    case LOOM_TARGET_MATH_OP_FLOORF:
    case LOOM_TARGET_MATH_OP_ROUNDF:
    case LOOM_TARGET_MATH_OP_ROUNDEVENF:
    case LOOM_TARGET_MATH_OP_TRUNCF:
      break;
  }

  *out_decision = loom_vm_math_keep(IREE_SV("math.op.scalarizable"));
}

static const loom_target_math_policy_t kVmMathPolicy = {
    .name = IREE_SVL("vm-math"),
    .query = loom_vm_math_policy_query,
};

static const loom_target_math_policy_registry_entry_t kVmMathPolicyEntries[] = {
    {
        .contract_set_key = IREE_SVL("vm.core"),
        .policy = &kVmMathPolicy,
    },
};

void loom_vm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kVmMathPolicyEntries, IREE_ARRAYSIZE(kVmMathPolicyEntries));
}
