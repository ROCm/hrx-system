// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_preparation.h"

static bool loom_contract_preparation_fail(
    loom_contract_preparation_rejection_bits_t rejection_bits,
    loom_contract_preparation_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

static bool loom_contract_preparation_policy_requires_optimized(
    loom_lowering_policy_t policy) {
  return policy == LOOM_LOWERING_POLICY_OPTIMIZED_REQUIRED ||
         policy == LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
}

static bool loom_contract_preparation_fallback_is_allowed(
    loom_lowering_policy_t policy) {
  return policy == LOOM_LOWERING_POLICY_REFERENCE_ALLOWED ||
         policy == LOOM_LOWERING_POLICY_VECTOR_PREFERRED;
}

static bool loom_contract_view_payload_has_storage_schema(
    const loom_contract_view_payload_t* payload) {
  const loom_value_fact_storage_schema_t source_schema =
      payload->operand.encoded.source_schema;
  return source_schema.static_spec_encoding_id != 0 ||
         !loom_value_fact_encoded_operand_schema_is_unknown(
             source_schema.encoded_operand);
}

static void loom_contract_operand_preparation_initialize_none(
    const loom_contract_operand_preparation_options_t* options,
    loom_contract_operand_preparation_t* out_preparation) {
  *out_preparation = (loom_contract_operand_preparation_t){
      .role = options->role,
      .family = LOOM_CONTRACT_PREPARATION_FAMILY_NONE,
      .numeric_transform = LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE,
      .source_payload = options->source_payload,
  };
  if (loom_contract_view_payload_has_storage_schema(&options->source_payload)) {
    out_preparation->flags |=
        LOOM_CONTRACT_PREPARATION_PRESERVES_STORAGE_SCHEMA;
    out_preparation->storage_schema =
        options->source_payload.operand.encoded.source_schema;
  }
}

static bool loom_contract_preparation_requires_address_layout(
    loom_contract_preparation_family_t family) {
  return family == LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED;
}

static bool loom_contract_preparation_family_role_matches(
    loom_contract_preparation_family_t family,
    loom_contract_operand_role_t role) {
  if (family == LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED) {
    return role == LOOM_CONTRACT_OPERAND_ROLE_RHS;
  }
  return role != LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
}

static bool loom_contract_preparation_payload_role_matches(
    const loom_contract_view_payload_t* payload,
    loom_contract_operand_role_t role) {
  return payload->operand.role == LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN ||
         payload->operand.role == role;
}

static bool loom_contract_preparation_logical_flags_are_valid(
    loom_contract_preparation_flags_t flags) {
  const loom_contract_preparation_flags_t known_logical_flags =
      LOOM_CONTRACT_PREPARATION_LOGICAL_RHS_TRANSPOSE;
  return (flags & ~known_logical_flags) == 0;
}

static bool loom_contract_preparation_has_strided_layout(
    loom_value_fact_address_layout_t layout) {
  return layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED &&
         layout.rank > 0 && layout.strides != NULL;
}

static loom_contract_preparation_flags_t
loom_contract_preparation_family_implied_flags(
    loom_contract_preparation_family_t family) {
  switch (family) {
    case LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED:
      return LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING |
             LOOM_CONTRACT_PREPARATION_PHYSICAL_N_MAJOR_BLOCKED;
    case LOOM_CONTRACT_PREPARATION_FAMILY_SUBGROUP_MATRIX_FRAGMENT:
      return LOOM_CONTRACT_PREPARATION_FRAGMENT_OWNERSHIP;
    case LOOM_CONTRACT_PREPARATION_FAMILY_NONE:
    case LOOM_CONTRACT_PREPARATION_FAMILY_NUMERIC_TRANSFORM:
    case LOOM_CONTRACT_PREPARATION_FAMILY_UNKNOWN:
    default:
      return 0;
  }
}

bool loom_contract_operand_preparation_select(
    const loom_contract_operand_preparation_options_t* options,
    loom_contract_operand_preparation_t* out_preparation,
    loom_contract_preparation_diagnostic_t* out_diagnostic) {
  *out_preparation = (loom_contract_operand_preparation_t){0};
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_preparation_diagnostic_t){0};
  }

  if (options->role == LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN ||
      options->family == LOOM_CONTRACT_PREPARATION_FAMILY_UNKNOWN ||
      options->policy == LOOM_LOWERING_POLICY_UNKNOWN ||
      options->source_payload.kind == LOOM_CONTRACT_VIEW_PAYLOAD_UNKNOWN ||
      !loom_contract_preparation_logical_flags_are_valid(
          options->logical_flags) ||
      (options->family != LOOM_CONTRACT_PREPARATION_FAMILY_NONE &&
       options->availability ==
           LOOM_CONTRACT_PREPARATION_AVAILABILITY_UNKNOWN)) {
    return loom_contract_preparation_fail(
        LOOM_CONTRACT_PREPARATION_REJECTION_INVALID_REQUEST, out_diagnostic);
  }

  if (!loom_contract_preparation_family_role_matches(options->family,
                                                     options->role)) {
    return loom_contract_preparation_fail(
        LOOM_CONTRACT_PREPARATION_REJECTION_ROLE, out_diagnostic);
  }
  if (!loom_contract_preparation_payload_role_matches(&options->source_payload,
                                                      options->role)) {
    return loom_contract_preparation_fail(
        LOOM_CONTRACT_PREPARATION_REJECTION_ROLE, out_diagnostic);
  }

  if (options->family == LOOM_CONTRACT_PREPARATION_FAMILY_NONE) {
    loom_contract_operand_preparation_initialize_none(options, out_preparation);
    out_preparation->flags |= options->logical_flags;
    return true;
  }

  if (options->availability !=
      LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE) {
    loom_contract_preparation_rejection_bits_t rejection_bits =
        options->availability == LOOM_CONTRACT_PREPARATION_AVAILABILITY_TOO_LATE
            ? LOOM_CONTRACT_PREPARATION_REJECTION_TOO_LATE
            : LOOM_CONTRACT_PREPARATION_REJECTION_UNAVAILABLE;
    if (loom_contract_preparation_policy_requires_optimized(options->policy)) {
      rejection_bits |= LOOM_CONTRACT_PREPARATION_REJECTION_POLICY;
      return loom_contract_preparation_fail(rejection_bits, out_diagnostic);
    }
    if (loom_contract_preparation_fallback_is_allowed(options->policy)) {
      loom_contract_operand_preparation_initialize_none(options,
                                                        out_preparation);
      out_preparation->flags |= options->logical_flags;
      return true;
    }
    return loom_contract_preparation_fail(rejection_bits, out_diagnostic);
  }

  if (loom_contract_preparation_requires_address_layout(options->family) &&
      !loom_contract_preparation_has_strided_layout(options->address_layout)) {
    return loom_contract_preparation_fail(
        LOOM_CONTRACT_PREPARATION_REJECTION_ADDRESS_LAYOUT, out_diagnostic);
  }

  loom_contract_numeric_transform_t numeric_transform =
      options->numeric_transform;
  if (options->family != LOOM_CONTRACT_PREPARATION_FAMILY_NUMERIC_TRANSFORM) {
    if (numeric_transform == LOOM_CONTRACT_NUMERIC_TRANSFORM_UNKNOWN) {
      numeric_transform = LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE;
    }
    if (numeric_transform != LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE) {
      return loom_contract_preparation_fail(
          LOOM_CONTRACT_PREPARATION_REJECTION_NUMERIC_TRANSFORM,
          out_diagnostic);
    }
  } else if (numeric_transform == LOOM_CONTRACT_NUMERIC_TRANSFORM_UNKNOWN ||
             numeric_transform == LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE) {
    return loom_contract_preparation_fail(
        LOOM_CONTRACT_PREPARATION_REJECTION_NUMERIC_TRANSFORM, out_diagnostic);
  }

  *out_preparation = (loom_contract_operand_preparation_t){
      .role = options->role,
      .family = options->family,
      .flags = options->logical_flags |
               loom_contract_preparation_family_implied_flags(options->family),
      .numeric_transform = numeric_transform,
      .source_payload = options->source_payload,
      .address_layout = options->address_layout,
      .storage_schema = options->source_payload.operand.encoded.source_schema,
  };
  if (options->address_layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN) {
    out_preparation->flags |= LOOM_CONTRACT_PREPARATION_HAS_ADDRESS_LAYOUT;
  }
  if (loom_contract_view_payload_has_storage_schema(&options->source_payload)) {
    out_preparation->flags |=
        LOOM_CONTRACT_PREPARATION_PRESERVES_STORAGE_SCHEMA;
  }
  return true;
}
