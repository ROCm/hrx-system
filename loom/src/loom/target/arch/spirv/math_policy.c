// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/math_policy.h"

static loom_target_math_policy_decision_t loom_spirv_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_spirv_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static bool loom_spirv_math_op_is_native_scalar_arithmetic(
    loom_target_math_op_t math_op) {
  return math_op == LOOM_TARGET_MATH_OP_ADDF ||
         math_op == LOOM_TARGET_MATH_OP_MULF;
}

static bool loom_spirv_math_element_type_is_native_arithmetic(
    loom_scalar_type_t element_type) {
  return element_type == LOOM_SCALAR_TYPE_F16 ||
         element_type == LOOM_SCALAR_TYPE_F32 ||
         element_type == LOOM_SCALAR_TYPE_F64;
}

static void loom_spirv_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  if (query->lane_domain != LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR) {
    *out_decision = loom_spirv_math_reject(IREE_SV("math.scalar"));
    return;
  }
  if (!loom_spirv_math_op_is_native_scalar_arithmetic(query->math_op)) {
    *out_decision =
        loom_spirv_math_reject(IREE_SV("math.op.native_arithmetic"));
    return;
  }
  if (!loom_spirv_math_element_type_is_native_arithmetic(query->element_type)) {
    *out_decision = loom_spirv_math_reject(IREE_SV("math.element.f16_f32_f64"));
    return;
  }

  *out_decision = loom_spirv_math_keep(IREE_SV("math.op.native_scalar"));
}

static const loom_target_math_policy_t kSpirvMathPolicy = {
    .name = IREE_SVL("spirv-math"),
    .query = loom_spirv_math_policy_query,
};

static const loom_target_math_policy_registry_entry_t
    kSpirvMathPolicyEntries[] = {
        {/*.contract_set_key=*/IREE_SVL("spirv.logical.core"),
         /*.policy=*/&kSpirvMathPolicy},
};

void loom_spirv_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kSpirvMathPolicyEntries,
      IREE_ARRAYSIZE(kSpirvMathPolicyEntries));
}
