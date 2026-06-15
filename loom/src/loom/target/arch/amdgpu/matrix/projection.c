// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix/projection.h"

#include "loom/analysis/contract_storage.h"

static bool loom_amdgpu_matrix_numeric_from_contract(
    loom_contract_numeric_type_t numeric_type,
    loom_amdgpu_matrix_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN;
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I4:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_I4;
      return true;
    case LOOM_CONTRACT_NUMERIC_U4:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_IU4;
      return true;
    case LOOM_CONTRACT_NUMERIC_I8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_I8;
      return true;
    case LOOM_CONTRACT_NUMERIC_U8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_IU8;
      return true;
    case LOOM_CONTRACT_NUMERIC_I32:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_I32;
      return true;
    case LOOM_CONTRACT_NUMERIC_F16:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_F16;
      return true;
    case LOOM_CONTRACT_NUMERIC_BF16:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF16;
      return true;
    case LOOM_CONTRACT_NUMERIC_F32:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_F32;
      return true;
    case LOOM_CONTRACT_NUMERIC_F64:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_F64;
      return true;
    case LOOM_CONTRACT_NUMERIC_FP8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP8;
      return true;
    case LOOM_CONTRACT_NUMERIC_BF8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF8;
      return true;
    case LOOM_CONTRACT_NUMERIC_FP6:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP6;
      return true;
    case LOOM_CONTRACT_NUMERIC_BF6:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF6;
      return true;
    case LOOM_CONTRACT_NUMERIC_FP4:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP4;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_matrix_payload_from_contract(
    loom_contract_operand_t operand,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  if (!loom_amdgpu_matrix_numeric_from_contract(operand.numeric_type,
                                                &out_payload->numeric_type)) {
    return false;
  }
  out_payload->register_count = operand.payload_register_count;
  out_payload->element_count = operand.payload_element_count;
  return true;
}

static bool loom_amdgpu_matrix_contract_fragment_is_complete(
    const loom_contract_fragment_t* fragment, uint32_t wave_size) {
  if (!iree_any_bit_set(fragment->atom_bits,
                        LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE)) {
    return false;
  }
  return fragment->subgroup_size == 0 || fragment->subgroup_size == wave_size;
}

static bool loom_amdgpu_matrix_contract_shape_is_native(
    const loom_contract_request_t* request) {
  return request->shape.m > 0 && request->shape.m <= UINT16_MAX &&
         request->shape.n > 0 && request->shape.n <= UINT16_MAX &&
         request->shape.k > 0 && request->shape.k <= UINT16_MAX &&
         request->k_group_size != 0;
}

static loom_amdgpu_matrix_contract_flags_t
loom_amdgpu_matrix_flags_from_contract(
    loom_contract_capability_flags_t capability_flags) {
  loom_amdgpu_matrix_contract_flags_t flags = 0;
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  }
  if (iree_any_bit_set(capability_flags, LOOM_CONTRACT_CAPABILITY_REUSE)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE;
  }
  if (iree_any_bit_set(capability_flags, LOOM_CONTRACT_CAPABILITY_CLAMP)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_SIGN_SELECT)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_OPERAND_MODIFIERS)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_ACCUMULATOR_MODIFIER)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER;
  }
  if (iree_any_bit_set(capability_flags, LOOM_CONTRACT_CAPABILITY_OPSEL)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  }
  if (iree_any_bit_set(capability_flags,
                       LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK;
  }
  return flags;
}

static bool loom_amdgpu_matrix_scale_from_generic(
    loom_contract_scale_kind_t scale_kind,
    loom_amdgpu_matrix_scale_kind_t* out_scale_kind) {
  *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_NONE;
  switch (scale_kind) {
    case LOOM_CONTRACT_SCALE_NONE:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_NONE;
      return true;
    case LOOM_CONTRACT_SCALE_32:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_32;
      return true;
    case LOOM_CONTRACT_SCALE_16:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_16;
      return true;
    case LOOM_CONTRACT_SCALE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_matrix_scale_kind_from_contract_operand(
    loom_contract_operand_t operand,
    loom_contract_scale_kind_t* out_scale_kind) {
  return loom_contract_scale_kind_from_storage_schema(
      operand.encoded.target_schema, out_scale_kind);
}

static bool loom_amdgpu_matrix_scale_from_contract(
    const loom_contract_request_t* contract_request,
    loom_amdgpu_matrix_scale_kind_t* out_scale_kind) {
  loom_contract_scale_kind_t lhs_scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  loom_contract_scale_kind_t rhs_scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  if (!loom_amdgpu_matrix_scale_kind_from_contract_operand(
          contract_request->lhs, &lhs_scale_kind) ||
      !loom_amdgpu_matrix_scale_kind_from_contract_operand(
          contract_request->rhs, &rhs_scale_kind) ||
      lhs_scale_kind != rhs_scale_kind) {
    return false;
  }
  return loom_amdgpu_matrix_scale_from_generic(lhs_scale_kind, out_scale_kind);
}

static bool loom_amdgpu_matrix_contract_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

bool loom_amdgpu_matrix_contract_match_request_from_contract(
    const loom_contract_request_t* contract_request,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size,
    loom_amdgpu_matrix_contract_match_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  *out_request = (loom_amdgpu_matrix_contract_match_request_t){0};

  loom_contract_diagnostic_t diagnostic = {0};
  if (!loom_contract_request_validate(contract_request, &diagnostic)) {
    return loom_amdgpu_matrix_contract_fail(diagnostic.rejection_bits,
                                            out_diagnostic);
  }
  if (contract_request->capability_class !=
      LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX) {
    return loom_amdgpu_matrix_contract_fail(LOOM_CONTRACT_REJECTION_CAPABILITY,
                                            out_diagnostic);
  }
  if (!loom_amdgpu_matrix_contract_fragment_is_complete(
          &contract_request->fragment, wave_size)) {
    return loom_amdgpu_matrix_contract_fail(LOOM_CONTRACT_REJECTION_FRAGMENT,
                                            out_diagnostic);
  }
  if (!loom_amdgpu_matrix_contract_shape_is_native(contract_request)) {
    return loom_amdgpu_matrix_contract_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                            out_diagnostic);
  }

  loom_amdgpu_matrix_contract_match_request_t request = {0};
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN;
  request.tile_shape.result_row_count = (uint16_t)contract_request->shape.m;
  request.tile_shape.result_column_count = (uint16_t)contract_request->shape.n;
  request.tile_shape.reduction_count = (uint16_t)contract_request->shape.k;
  request.feature_bits = feature_bits;
  request.wave_size = wave_size;
  request.available_flags = loom_amdgpu_matrix_flags_from_contract(
      loom_contract_request_available_capability_flags(contract_request));
  request.required_flags = loom_amdgpu_matrix_flags_from_contract(
      loom_contract_request_required_capability_flags(contract_request));
  if (!loom_amdgpu_matrix_payload_from_contract(contract_request->lhs,
                                                &request.lhs_payload) ||
      !loom_amdgpu_matrix_payload_from_contract(contract_request->rhs,
                                                &request.rhs_payload) ||
      !loom_amdgpu_matrix_payload_from_contract(contract_request->accumulator,
                                                &request.accumulator_payload) ||
      !loom_amdgpu_matrix_payload_from_contract(contract_request->result,
                                                &request.result_payload)) {
    return loom_amdgpu_matrix_contract_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                            out_diagnostic);
  }
  if (!loom_amdgpu_matrix_scale_from_contract(contract_request,
                                              &request.scale_kind)) {
    return loom_amdgpu_matrix_contract_fail(LOOM_CONTRACT_REJECTION_CAPABILITY,
                                            out_diagnostic);
  }

  *out_request = request;
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }
  return true;
}
