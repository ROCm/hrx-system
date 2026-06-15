// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_contract_operand_role_matches(
    loom_contract_operand_t operand, loom_contract_operand_role_t role) {
  return operand.role == role;
}

static bool loom_contract_operand_numeric_is_known(
    loom_contract_operand_t operand) {
  return operand.numeric_type != LOOM_CONTRACT_NUMERIC_UNKNOWN;
}

static bool loom_contract_shape_dimension_is_valid(
    int64_t exact_value, loom_contract_value_ref_t value_ref) {
  if (exact_value < 0) {
    return false;
  }
  return exact_value > 0 || loom_contract_value_ref_is_present(value_ref);
}

static bool loom_contract_shape_is_valid(
    loom_contract_shape_t shape, loom_contract_shape_value_refs_t value_refs) {
  return loom_contract_shape_dimension_is_valid(shape.m, value_refs.m) &&
         loom_contract_shape_dimension_is_valid(shape.n, value_refs.n) &&
         loom_contract_shape_dimension_is_valid(shape.k, value_refs.k);
}

static bool loom_contract_k_group_size_is_valid(
    uint16_t exact_value, loom_contract_value_ref_t value_ref) {
  return exact_value != 0 || loom_contract_value_ref_is_present(value_ref);
}

static bool loom_contract_storage_schema_is_known(
    loom_value_fact_storage_schema_t schema) {
  return schema.static_spec_encoding_id != 0 ||
         !loom_value_fact_encoded_operand_schema_is_unknown(
             schema.encoded_operand);
}

static bool loom_contract_encoded_operand_has_auxiliary_value_refs(
    const loom_contract_encoded_operand_t* encoded) {
  static const loom_contract_value_ref_t
      zero_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_] = {0};
  return memcmp(encoded->auxiliary_value_refs, zero_value_refs,
                sizeof(encoded->auxiliary_value_refs)) != 0;
}

static loom_contract_auxiliary_operand_flags_t
loom_contract_auxiliary_operand_all_flags(void) {
  return UINT32_MAX >> (32 - LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_);
}

static bool loom_contract_encoded_operand_has_facts(
    const loom_contract_encoded_operand_t* encoded) {
  return loom_contract_storage_schema_is_known(encoded->source_schema) ||
         loom_contract_storage_schema_is_known(encoded->target_schema) ||
         encoded->available_auxiliary_operands != 0 ||
         loom_contract_encoded_operand_has_auxiliary_value_refs(encoded) ||
         encoded->required_auxiliary_operands != 0 ||
         encoded->available_capability_flags != 0 ||
         encoded->required_capability_flags != 0;
}

static bool loom_contract_encoded_operand_auxiliary_values_are_consistent(
    const loom_contract_encoded_operand_t* encoded) {
  for (uint8_t i = 0; i < LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_; ++i) {
    loom_contract_auxiliary_operand_key_t key =
        (loom_contract_auxiliary_operand_key_t)i;
    const bool has_available_flag =
        iree_any_bit_set(encoded->available_auxiliary_operands,
                         loom_contract_auxiliary_operand_key_flag(key));
    const bool has_value_ref =
        loom_contract_value_ref_is_present(encoded->auxiliary_value_refs[key]);
    if (has_available_flag != has_value_ref) {
      return false;
    }
  }
  return true;
}

static loom_contract_rejection_bits_t
loom_contract_encoded_operand_rejection_bits(
    const loom_contract_encoded_operand_t* encoded) {
  if (!loom_contract_encoded_operand_has_facts(encoded)) {
    return LOOM_CONTRACT_REJECTION_NONE;
  }

  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  if (loom_value_fact_encoded_operand_schema_is_unknown(
          encoded->target_schema.encoded_operand) ||
      !loom_value_fact_encoded_operand_schema_scale_is_complete(
          encoded->target_schema.encoded_operand)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SCHEMA;
  }
  if (!iree_all_bits_set(encoded->available_auxiliary_operands,
                         encoded->required_auxiliary_operands)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  if (iree_any_bit_set(encoded->available_auxiliary_operands |
                           encoded->required_auxiliary_operands,
                       ~loom_contract_auxiliary_operand_all_flags())) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  if (!loom_contract_encoded_operand_auxiliary_values_are_consistent(encoded)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  if (!iree_all_bits_set(encoded->available_capability_flags,
                         encoded->required_capability_flags)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_CAPABILITY;
  }
  return rejection_bits;
}

static loom_contract_capability_flags_t
loom_contract_operand_available_capability_flags(
    loom_contract_operand_t operand) {
  return operand.encoded.available_capability_flags;
}

static loom_contract_capability_flags_t
loom_contract_operand_required_capability_flags(
    loom_contract_operand_t operand) {
  return operand.encoded.required_capability_flags;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void loom_contract_request_initialize(loom_contract_request_t* out_request) {
  *out_request = (loom_contract_request_t){
      .policy = LOOM_LOWERING_POLICY_REFERENCE_ALLOWED,
  };
}

bool loom_contract_request_validate(
    const loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  loom_contract_rejection_bits_t rejection_bits = 0;
  if (request->kind == LOOM_CONTRACT_KIND_UNKNOWN ||
      request->arithmetic == LOOM_CONTRACT_ARITHMETIC_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  if (!loom_contract_shape_is_valid(request->shape,
                                    request->shape_value_refs) ||
      !loom_contract_k_group_size_is_valid(
          request->k_group_size, request->shape_value_refs.k_group_size)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SHAPE;
  }
  if (!loom_contract_operand_role_matches(request->lhs,
                                          LOOM_CONTRACT_OPERAND_ROLE_LHS) ||
      !loom_contract_operand_role_matches(request->rhs,
                                          LOOM_CONTRACT_OPERAND_ROLE_RHS) ||
      !loom_contract_operand_role_matches(
          request->accumulator, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR) ||
      !loom_contract_operand_role_matches(request->result,
                                          LOOM_CONTRACT_OPERAND_ROLE_RESULT)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_ROLE;
  }
  if (!loom_contract_operand_numeric_is_known(request->lhs) ||
      !loom_contract_operand_numeric_is_known(request->rhs) ||
      !loom_contract_operand_numeric_is_known(request->accumulator) ||
      !loom_contract_operand_numeric_is_known(request->result)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  if (request->capability_class == LOOM_CONTRACT_CAPABILITY_CLASS_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_CAPABILITY;
  }
  rejection_bits |=
      loom_contract_encoded_operand_rejection_bits(&request->lhs.encoded);
  rejection_bits |=
      loom_contract_encoded_operand_rejection_bits(&request->rhs.encoded);
  rejection_bits |= loom_contract_encoded_operand_rejection_bits(
      &request->accumulator.encoded);
  rejection_bits |=
      loom_contract_encoded_operand_rejection_bits(&request->result.encoded);
  if (request->policy == LOOM_LOWERING_POLICY_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_POLICY;
  }

  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return rejection_bits == 0;
}

loom_contract_capability_flags_t
loom_contract_request_available_capability_flags(
    const loom_contract_request_t* request) {
  return loom_contract_operand_available_capability_flags(request->lhs) |
         loom_contract_operand_available_capability_flags(request->rhs) |
         loom_contract_operand_available_capability_flags(
             request->accumulator) |
         loom_contract_operand_available_capability_flags(request->result);
}

loom_contract_capability_flags_t
loom_contract_request_required_capability_flags(
    const loom_contract_request_t* request) {
  return loom_contract_operand_required_capability_flags(request->lhs) |
         loom_contract_operand_required_capability_flags(request->rhs) |
         loom_contract_operand_required_capability_flags(request->accumulator) |
         loom_contract_operand_required_capability_flags(request->result);
}

bool loom_contract_numeric_type_from_scalar(
    loom_scalar_type_t scalar_type, bool unsigned_integer,
    loom_contract_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U8
                                           : LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U16
                                           : LOOM_CONTRACT_NUMERIC_I16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U32
                                           : LOOM_CONTRACT_NUMERIC_I32;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F64;
      return true;
    default:
      return false;
  }
}

iree_string_view_t loom_contract_numeric_type_name(
    loom_contract_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I4:
      return IREE_SV("i4");
    case LOOM_CONTRACT_NUMERIC_U4:
      return IREE_SV("u4");
    case LOOM_CONTRACT_NUMERIC_I8:
      return IREE_SV("i8");
    case LOOM_CONTRACT_NUMERIC_U8:
      return IREE_SV("u8");
    case LOOM_CONTRACT_NUMERIC_I16:
      return IREE_SV("i16");
    case LOOM_CONTRACT_NUMERIC_U16:
      return IREE_SV("u16");
    case LOOM_CONTRACT_NUMERIC_I32:
      return IREE_SV("i32");
    case LOOM_CONTRACT_NUMERIC_U32:
      return IREE_SV("u32");
    case LOOM_CONTRACT_NUMERIC_F16:
      return IREE_SV("f16");
    case LOOM_CONTRACT_NUMERIC_BF16:
      return IREE_SV("bf16");
    case LOOM_CONTRACT_NUMERIC_F32:
      return IREE_SV("f32");
    case LOOM_CONTRACT_NUMERIC_F64:
      return IREE_SV("f64");
    case LOOM_CONTRACT_NUMERIC_FP8:
      return IREE_SV("fp8");
    case LOOM_CONTRACT_NUMERIC_BF8:
      return IREE_SV("bf8");
    case LOOM_CONTRACT_NUMERIC_FP6:
      return IREE_SV("fp6");
    case LOOM_CONTRACT_NUMERIC_BF6:
      return IREE_SV("bf6");
    case LOOM_CONTRACT_NUMERIC_FP4:
      return IREE_SV("fp4");
    case LOOM_CONTRACT_NUMERIC_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_contract_rejection_detail(
    loom_contract_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("contract request is incomplete or invalid");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("contract shape or K grouping is not statically proven");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV("contract operand roles are missing or inconsistent");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("contract numeric payload facts are missing or unsupported");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV("contract fragment facts are insufficient");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
    return IREE_SV("contract encoded schema facts are missing or inconsistent");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND)) {
    return IREE_SV("contract auxiliary data operands are missing");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("contract capability class is unsupported");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_POLICY)) {
    return IREE_SV("contract lowering policy forbids the selected path");
  }
  return IREE_SV("contract request is not representable");
}
