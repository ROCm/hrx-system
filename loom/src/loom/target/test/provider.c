// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/provider.h"

#include "loom/pass/test/registry.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"

static void loom_test_widen_f32_round_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  if (query->math_op == LOOM_TARGET_MATH_OP_MULF &&
      (query->element_type == LOOM_SCALAR_TYPE_F16 ||
       query->element_type == LOOM_SCALAR_TYPE_BF16)) {
    *out_decision = (loom_target_math_policy_decision_t){
        .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
        .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
        .constraint_key = IREE_SVL("test.math.recipe.widen_f32_round"),
    };
    return;
  }
  *out_decision = (loom_target_math_policy_decision_t){
      .action = query->math_op == LOOM_TARGET_MATH_OP_MULF &&
                        query->element_type == LOOM_SCALAR_TYPE_F32
                    ? LOOM_TARGET_MATH_POLICY_ACTION_KEEP
                    : LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = IREE_SVL("test.math.widen_f32_round.fixture"),
  };
}

static void loom_test_gelu_bf16_packet_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  (void)policy;
  loom_target_math_recipe_t recipe = LOOM_TARGET_MATH_RECIPE_UNKNOWN;
  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_TANHF:
      *out_decision = (loom_target_math_policy_decision_t){
          .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
          .constraint_key = IREE_SV("test.math.gelu_bf16_packet.tanh"),
      };
      return;
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
      recipe = LOOM_TARGET_MATH_RECIPE_GELU_TANH_BF16_PACKET;
      break;
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
      recipe = LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_BF16_PACKET;
      break;
    default:
      break;
  }
  if (recipe != LOOM_TARGET_MATH_RECIPE_UNKNOWN) {
    *out_decision = (loom_target_math_policy_decision_t){
        .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
        .recipe = recipe,
        .constraint_key = IREE_SV("test.math.gelu_bf16_packet"),
    };
    return;
  }
  *out_decision = (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = IREE_SV("test.math.gelu_bf16_packet"),
  };
}

static void loom_test_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  static const loom_target_math_policy_t kWidenF32RoundPolicy = {
      .name = IREE_SVL("test-widen-f32-round"),
      .query = loom_test_widen_f32_round_math_policy_query,
  };
  static const loom_target_math_policy_t kGeluBf16PacketPolicy = {
      .name = IREE_SVL("test-gelu-bf16-packet"),
      .query = loom_test_gelu_bf16_packet_math_policy_query,
  };
  static const loom_target_math_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.math.widen_f32_round"),
          .policy = &kWidenF32RoundPolicy,
      },
      {
          .contract_set_key = IREE_SVL("test.math.gelu_bf16_packet"),
          .policy = &kGeluBf16PacketPolicy,
      },
  };
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}

const loom_target_provider_t loom_test_target_provider = {
    .initialize_low_descriptor_registry =
        loom_target_core_test_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_test_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_test_math_policy_registry_initialize,
    .pass_registry = &loom_test_pass_registry_storage,
    .view_boundary_carrier = LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET,
    .loop_predicate_carrier = LOOM_TARGET_LOOP_PREDICATE_CARRIER_I32,
    .loop_predicate_max_vector_element_count = 32,
    .target_fact_type = &loom_test_target_fact_type,
};
