// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/math_policy.h"

static loom_target_math_policy_decision_t loom_wasm_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_wasm_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static bool loom_wasm_math_op_is_native_arithmetic(
    loom_target_math_op_t math_op) {
  return math_op == LOOM_TARGET_MATH_OP_ADDF ||
         math_op == LOOM_TARGET_MATH_OP_MULF;
}

static bool loom_wasm_math_lane_domain_is_supported(
    loom_target_math_lane_domain_t lane_domain) {
  return lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR ||
         lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR;
}

static void loom_wasm_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  if (!loom_wasm_math_op_is_native_arithmetic(query->math_op)) {
    *out_decision = loom_wasm_math_reject(IREE_SV("math.op.supported"));
    return;
  }
  if (query->element_type != LOOM_SCALAR_TYPE_F32) {
    *out_decision = loom_wasm_math_reject(IREE_SV("math.element.f32"));
    return;
  }
  if (!loom_wasm_math_lane_domain_is_supported(query->lane_domain)) {
    *out_decision = loom_wasm_math_reject(IREE_SV("math.lane.scalar_vector"));
    return;
  }

  *out_decision = loom_wasm_math_keep(IREE_SV("math.op.native_f32"));
}

static const loom_target_math_policy_t kWasmMathPolicy = {
    .name = IREE_SVL("wasm-math"),
    .query = loom_wasm_math_policy_query,
};

static const loom_target_math_policy_registry_entry_t kWasmMathPolicyEntries[] =
    {
        {/*.contract_set_key=*/IREE_SVL("wasm.core.simd128"),
         /*.policy=*/&kWasmMathPolicy},
};

void loom_wasm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kWasmMathPolicyEntries,
      IREE_ARRAYSIZE(kWasmMathPolicyEntries));
}
