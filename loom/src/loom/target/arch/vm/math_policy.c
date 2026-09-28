// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/math_policy.h"

static void loom_vm_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  loom_target_math_recipe_t recipe = LOOM_TARGET_MATH_RECIPE_UNKNOWN;
  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_SUBF:
    case LOOM_TARGET_MATH_OP_MULF:
      if (query->element_type == LOOM_SCALAR_TYPE_F16 ||
          query->element_type == LOOM_SCALAR_TYPE_BF16) {
        *out_decision = (loom_target_math_policy_decision_t){
            .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
            .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
            .constraint_key = IREE_SVL("math.recipe.widen_f32_round"),
        };
        return;
      }
      break;
    case LOOM_TARGET_MATH_OP_CEILF:
    case LOOM_TARGET_MATH_OP_FLOORF:
    case LOOM_TARGET_MATH_OP_ROUNDF:
    case LOOM_TARGET_MATH_OP_ROUNDEVENF:
    case LOOM_TARGET_MATH_OP_TRUNCF:
    case LOOM_TARGET_MATH_OP_LOG2F:
      break;
    case LOOM_TARGET_MATH_OP_EXPF:
      recipe = LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32;
      break;
    case LOOM_TARGET_MATH_OP_LOGF:
      recipe = LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32;
      break;
    case LOOM_TARGET_MATH_OP_SINF:
      recipe = LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32;
      break;
    case LOOM_TARGET_MATH_OP_COSF:
      recipe = LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32;
      break;
    case LOOM_TARGET_MATH_OP_SINTURNSF:
    case LOOM_TARGET_MATH_OP_COSTURNSF:
      if (query->element_type != LOOM_SCALAR_TYPE_F32) {
        *out_decision = (loom_target_math_policy_decision_t){
            .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
            .constraint_key = IREE_SVL("math.turns_trig.f32"),
        };
        return;
      }
      break;
    default:
      *out_decision = (loom_target_math_policy_decision_t){
          .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
          .constraint_key = IREE_SVL("math.op.supported"),
      };
      return;
  }
  if (recipe != LOOM_TARGET_MATH_RECIPE_UNKNOWN) {
    // Rounded base/angle conversions require approximate source semantics.
    *out_decision = (loom_target_math_policy_decision_t){
        .action = query->element_type == LOOM_SCALAR_TYPE_F32 &&
                          iree_any_bit_set(query->fastmath_flags,
                                           LOOM_TARGET_MATH_FASTMATH_FLAG_AFN)
                      ? LOOM_TARGET_MATH_POLICY_ACTION_REWRITE
                      : LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
        .recipe = recipe,
        .constraint_key = IREE_SVL("math.recipe.afn_f32"),
    };
    return;
  }
  if (query->element_type != LOOM_SCALAR_TYPE_F32 &&
      query->element_type != LOOM_SCALAR_TYPE_F64) {
    *out_decision = (loom_target_math_policy_decision_t){
        .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
        .constraint_key = IREE_SVL("math.element.f32_f64"),
    };
    return;
  }
  if (query->math_op == LOOM_TARGET_MATH_OP_ROUNDF) {
    *out_decision = (loom_target_math_policy_decision_t){
        .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
        .recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY,
        .constraint_key = IREE_SVL("math.recipe.round_away"),
    };
    return;
  }
  *out_decision = (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = IREE_SVL("math.op.selected_width"),
  };
}

static bool loom_vm_math_prefer_fma(
    const loom_target_math_policy_t* policy, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  (void)policy;
  (void)fastmath_flags;
  const loom_scalar_type_t element_type = loom_type_element_type(value_type);
  return loom_type_is_scalar(value_type) &&
         (element_type == LOOM_SCALAR_TYPE_F32 ||
          element_type == LOOM_SCALAR_TYPE_F64);
}

static const loom_target_math_policy_t loom_vm_math_policy = {
    .name = IREE_SVL("vm-math"),
    .query = loom_vm_math_policy_query,
    .prefer_fma = loom_vm_math_prefer_fma,
};

void loom_vm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  static const loom_target_math_policy_registry_entry_t kEntries[] = {
      {.contract_set_key = IREE_SVL("vm.core"), .policy = &loom_vm_math_policy},
  };
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
