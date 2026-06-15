// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_contract_projection.h"

static bool loom_x86_packed_dot_numeric_from_contract(
    loom_contract_numeric_type_t numeric_type,
    loom_x86_packed_dot_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN;
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I8:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_I8;
      return true;
    case LOOM_CONTRACT_NUMERIC_U8:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_U8;
      return true;
    case LOOM_CONTRACT_NUMERIC_I16:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_I16;
      return true;
    case LOOM_CONTRACT_NUMERIC_U16:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_U16;
      return true;
    case LOOM_CONTRACT_NUMERIC_F16:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_F16;
      return true;
    case LOOM_CONTRACT_NUMERIC_BF16:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_BF16;
      return true;
    case LOOM_CONTRACT_NUMERIC_I32:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_I32;
      return true;
    case LOOM_CONTRACT_NUMERIC_F32:
      *out_numeric_type = LOOM_X86_PACKED_DOT_NUMERIC_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_x86_packed_dot_contract_fragment_is_complete(
    const loom_contract_fragment_t* fragment) {
  return iree_any_bit_set(fragment->atom_bits,
                          LOOM_CONTRACT_FRAGMENT_VECTOR_LANE) &&
         fragment->vector_bit_width != 0 && fragment->source_lane_count != 0 &&
         fragment->result_lane_count != 0;
}

static bool loom_x86_packed_dot_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

bool loom_x86_packed_dot_match_request_from_contract(
    const loom_contract_request_t* contract_request,
    loom_x86_packed_dot_feature_bits_t feature_bits,
    loom_x86_packed_dot_match_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  *out_request = (loom_x86_packed_dot_match_request_t){0};

  loom_contract_diagnostic_t diagnostic = {0};
  if (!loom_contract_request_validate(contract_request, &diagnostic)) {
    return loom_x86_packed_dot_fail(diagnostic.rejection_bits, out_diagnostic);
  }
  if (contract_request->capability_class !=
      LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT) {
    return loom_x86_packed_dot_fail(LOOM_CONTRACT_REJECTION_CAPABILITY,
                                    out_diagnostic);
  }
  if (!loom_x86_packed_dot_contract_fragment_is_complete(
          &contract_request->fragment)) {
    return loom_x86_packed_dot_fail(LOOM_CONTRACT_REJECTION_FRAGMENT,
                                    out_diagnostic);
  }
  if (contract_request->k_group_size == 0) {
    return loom_x86_packed_dot_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                    out_diagnostic);
  }

  loom_x86_packed_dot_match_request_t request = {0};
  request.family = LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN;
  request.shape.vector_bit_width = contract_request->fragment.vector_bit_width;
  request.shape.input_lane_count = contract_request->fragment.source_lane_count;
  request.shape.result_lane_count =
      contract_request->fragment.result_lane_count;
  request.shape.reduction_group_size = contract_request->k_group_size;
  request.feature_bits = feature_bits;
  if (!loom_x86_packed_dot_numeric_from_contract(
          contract_request->lhs.numeric_type, &request.lhs_numeric_type) ||
      !loom_x86_packed_dot_numeric_from_contract(
          contract_request->rhs.numeric_type, &request.rhs_numeric_type) ||
      !loom_x86_packed_dot_numeric_from_contract(
          contract_request->accumulator.numeric_type,
          &request.accumulator_numeric_type) ||
      !loom_x86_packed_dot_numeric_from_contract(
          contract_request->result.numeric_type,
          &request.result_numeric_type)) {
    return loom_x86_packed_dot_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                    out_diagnostic);
  }

  *out_request = request;
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }
  return true;
}
