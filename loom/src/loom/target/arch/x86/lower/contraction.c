// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/contraction.h"

#include "loom/target/arch/x86/packed_dot_contract.h"
#include "loom/target/arch/x86/packed_dot_contract_projection.h"

static bool loom_x86_packed_dot_mma_shape_matches(
    const loom_contract_request_t* request) {
  if (request->kind != LOOM_CONTRACT_KIND_MATRIX_MULTIPLY ||
      request->shape.m != 1 || request->shape.n != 1 || request->shape.k <= 0 ||
      request->shape.block_count <= 0 ||
      request->shape.k != request->k_group_size ||
      request->shape.block_count != request->fragment.result_lane_count) {
    return false;
  }
  const uint64_t source_element_count =
      (uint64_t)request->shape.block_count * (uint64_t)request->shape.k;
  return source_element_count == request->lhs.payload_element_count &&
         source_element_count == request->rhs.payload_element_count &&
         request->shape.block_count ==
             request->accumulator.payload_element_count &&
         request->shape.block_count == request->result.payload_element_count;
}

static iree_string_view_t loom_x86_packed_dot_rejection_constraint(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FAMILY)) {
    return IREE_SV("x86.packed_dot.family");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("x86.packed_dot.shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV("x86.packed_dot.payload");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FLAGS)) {
    return IREE_SV("x86.packed_dot.flags");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV("x86.packed_dot.features");
  }
  return IREE_SV("x86.packed_dot.descriptor");
}

iree_status_t loom_x86_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)environment;
  (void)rule;
  *out_options = (loom_contract_vector_mma_options_t){
      .fragment_projection =
          LOOM_CONTRACT_VECTOR_MMA_FRAGMENT_PROJECTION_PACKED_VECTOR,
      .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT,
      .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
  return iree_ok_status();
}

iree_status_t loom_x86_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();

  if (!loom_x86_packed_dot_mma_shape_matches(request)) {
    return loom_low_lower_descriptor_matrix_reject(
        environment, rule, source_op,
        IREE_SV("x86.packed_dot.scalar_dot_shape"), 0,
        LOOM_X86_PACKED_DOT_REJECTION_SHAPE, 0, out_result);
  }

  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  IREE_ASSERT(bundle != NULL && bundle->config != NULL,
              "x86 contraction lowering requires a target config");
  loom_x86_packed_dot_match_request_t match_request = {0};
  loom_contract_diagnostic_t contract_diagnostic = {0};
  if (!loom_x86_packed_dot_match_request_from_contract(
          request, bundle->config->contract_feature_bits, &match_request,
          &contract_diagnostic)) {
    return loom_low_lower_descriptor_matrix_reject(
        environment, rule, source_op, IREE_SV("x86.packed_dot.contract"),
        contract_diagnostic.rejection_bits, 0, 0, out_result);
  }

  loom_x86_packed_dot_match_diagnostic_t match_diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* semantic_descriptor =
      loom_x86_packed_dot_select(&match_request, &match_diagnostic);
  if (semantic_descriptor == NULL) {
    return loom_low_lower_descriptor_matrix_reject(
        environment, rule, source_op,
        loom_x86_packed_dot_rejection_constraint(
            match_diagnostic.rejection_bits),
        0, match_diagnostic.rejection_bits, 0, out_result);
  }

  const loom_low_descriptor_t* low_descriptor =
      loom_x86_packed_dot_low_descriptor(environment->descriptor_set,
                                         semantic_descriptor);
  if (low_descriptor == NULL) {
    return loom_low_lower_descriptor_matrix_reject(
        environment, rule, source_op, IREE_SV("x86.packed_dot.descriptor_set"),
        0, LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST, 0, out_result);
  }

  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor = low_descriptor;
  out_result->selected_native_contraction_facts =
      semantic_descriptor->native_contraction_facts;
  return iree_ok_status();
}
