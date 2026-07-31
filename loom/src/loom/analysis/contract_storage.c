// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_storage.h"

#include "loom/ops/encoding/storage.h"

static bool loom_contract_storage_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

bool loom_contract_numeric_type_from_encoded_format(
    loom_value_fact_numeric_format_flags_t format,
    loom_contract_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  const loom_numeric_format_info_t* info = NULL;
  if (!loom_numeric_format_info(format, &info)) {
    return false;
  }
  switch (info->kind) {
    case LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER:
      switch (info->storage_bit_count) {
        case 4:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_I4;
          return true;
        case 8:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_I8;
          return true;
        case 16:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_I16;
          return true;
        case 32:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
          return true;
        default:
          return false;
      }
    case LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER:
      switch (info->storage_bit_count) {
        case 4:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_U4;
          return true;
        case 8:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_U8;
          return true;
        case 16:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_U16;
          return true;
        case 32:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_U32;
          return true;
        default:
          return false;
      }
    case LOOM_NUMERIC_FORMAT_KIND_QUANTIZED_SIGNED_INTEGER:
      if (info->storage_bit_count == 8) {
        *out_numeric_type = LOOM_CONTRACT_NUMERIC_I8;
        return true;
      }
      return false;
    case LOOM_NUMERIC_FORMAT_KIND_FLOAT:
      switch (info->float_family) {
        case LOOM_NUMERIC_FLOAT_FAMILY_IEEE:
          switch (info->storage_bit_count) {
            case 16:
              if (info->exponent_bit_count == 5 &&
                  info->mantissa_bit_count == 10) {
                *out_numeric_type = LOOM_CONTRACT_NUMERIC_F16;
                return true;
              }
              return false;
            case 32:
              if (info->exponent_bit_count == 8 &&
                  info->mantissa_bit_count == 23) {
                *out_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
                return true;
              }
              if (info->exponent_bit_count == 8 &&
                  info->mantissa_bit_count == 10) {
                *out_numeric_type = LOOM_CONTRACT_NUMERIC_TF32;
                return true;
              }
              return false;
            case 64:
              if (info->exponent_bit_count == 11 &&
                  info->mantissa_bit_count == 52) {
                *out_numeric_type = LOOM_CONTRACT_NUMERIC_F64;
                return true;
              }
              return false;
            default:
              return false;
          }
        case LOOM_NUMERIC_FLOAT_FAMILY_BFLOAT:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF16;
          return true;
        case LOOM_NUMERIC_FLOAT_FAMILY_FP8:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP8;
          return true;
        case LOOM_NUMERIC_FLOAT_FAMILY_BF8:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF8;
          return true;
        case LOOM_NUMERIC_FLOAT_FAMILY_FP6:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP6;
          return true;
        case LOOM_NUMERIC_FLOAT_FAMILY_BF6:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF6;
          return true;
        case LOOM_NUMERIC_FLOAT_FAMILY_FP4:
          *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP4;
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

bool loom_contract_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_contract_scale_kind_t* out_scale_kind) {
  *out_scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!loom_value_fact_encoded_operand_schema_has_scale(operand)) {
    *out_scale_kind = LOOM_CONTRACT_SCALE_NONE;
    return true;
  }
  if (operand.scale_topology == 0 || operand.scale_operand_count == 0) {
    return false;
  }
  switch (operand.scale_group.element_count) {
    case 32:
      *out_scale_kind = LOOM_CONTRACT_SCALE_32;
      return true;
    case 16:
      *out_scale_kind = LOOM_CONTRACT_SCALE_16;
      return true;
    default:
      return false;
  }
}

static bool loom_contract_encoded_format_requires_selector(
    loom_value_fact_numeric_format_flags_t format) {
  return loom_numeric_format_needs_encoded_payload_selector(format);
}

loom_contract_auxiliary_operand_flags_t
loom_contract_required_auxiliary_operands_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_contract_auxiliary_operand_flags_t flags = 0;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (operand.scale_operand_count != 0) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE;
  }
  if (operand.scale_operand_count > 1 ||
      operand.secondary_scale_format !=
          LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN ||
      iree_any_bit_set(
          operand.scale_topology,
          LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL |
              LOOM_VALUE_FACT_SCALE_TOPOLOGY_SUBBLOCK_IN_SUPERBLOCK) ||
      iree_any_bit_set(
          operand.affine_policy,
          LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_SECONDARY_SCALE;
  }
  if (iree_any_bit_set(operand.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_ZERO_POINT;
  }
  if (iree_any_bit_set(operand.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_MIN;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_SPARSE_METADATA;
  }
  if (iree_any_bit_set(
          operand.codebook_policy,
          LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_SYMBOL_TABLE |
              LOOM_VALUE_FACT_CODEBOOK_POLICY_GLOBAL_DATA_TABLE |
              LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND |
              LOOM_VALUE_FACT_CODEBOOK_POLICY_PER_SUPERBLOCK_TABLE)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_CODEBOOK_TABLE;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_OUTLIER_SIDE_STREAM)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_RESIDUAL;
  }
  if (iree_any_bit_set(operand.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED)) {
    flags |= LOOM_CONTRACT_AUXILIARY_OPERAND_RUNTIME_AMAX;
  }
  return flags;
}

loom_contract_capability_flags_t
loom_contract_available_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_contract_capability_flags_t flags = 0;
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (loom_contract_numeric_type_from_encoded_format(operand.element_format,
                                                     &numeric_type)) {
    flags |= LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;
    if (loom_contract_numeric_type_is_signed_integer(numeric_type) ||
        loom_contract_numeric_type_is_unsigned_integer(numeric_type)) {
      flags |= LOOM_CONTRACT_CAPABILITY_SIGN_SELECT;
    }
  }
  if (loom_value_fact_encoded_operand_schema_has_scale(operand) &&
      operand.scale_operand_count != 0) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  }
  if (operand.scale_format != 0 || operand.secondary_scale_format != 0) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS;
  }
  if (iree_any_bit_set(
          operand.flags,
          LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK)) {
    flags |= LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    flags |= LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA;
  }
  return flags;
}

loom_contract_capability_flags_t
loom_contract_required_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_contract_capability_flags_t flags = 0;
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  const loom_contract_auxiliary_operand_flags_t auxiliary_operands =
      loom_contract_required_auxiliary_operands_from_storage_schema(schema);
  if (iree_any_bit_set(auxiliary_operands,
                       LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE)) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    flags |= LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA;
  }
  return flags;
}

bool loom_contract_operand_from_storage_schema(
    loom_contract_operand_role_t role, loom_value_fact_storage_schema_t schema,
    loom_contract_operand_t* out_operand) {
  *out_operand = (loom_contract_operand_t){
      .role = role,
  };
  out_operand->encoded.source_schema = schema;
  out_operand->encoded.target_schema = schema;
  out_operand->encoded.required_auxiliary_operands =
      loom_contract_required_auxiliary_operands_from_storage_schema(schema);
  out_operand->encoded.available_capability_flags =
      loom_contract_available_capability_flags_from_storage_schema(schema) |
      loom_contract_plain_fragment_available_capability_flags(role);
  out_operand->encoded.required_capability_flags =
      loom_contract_required_capability_flags_from_storage_schema(schema);
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count == 0 ||
      operand.payload_element_count == 0) {
    return false;
  }
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_encoded_format(operand.element_format,
                                                      &numeric_type)) {
    return false;
  }
  out_operand->numeric_type = numeric_type;
  out_operand->payload_register_count = operand.payload_register_count;
  out_operand->payload_element_count = operand.payload_element_count;
  return true;
}

bool loom_contract_view_payload_from_type(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_contract_operand_role_t role,
    bool plain_integer_is_unsigned, loom_contract_view_payload_t* out_payload) {
  *out_payload = (loom_contract_view_payload_t){
      .kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNKNOWN,
      .operand =
          (loom_contract_operand_t){
              .role = role,
          },
  };
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_fact_storage_schema_t storage_schema = {0};
  if (loom_encoding_query_type_storage_schema(context, module, view_type,
                                              &storage_schema)) {
    out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA;
    if (!loom_contract_operand_from_storage_schema(role, storage_schema,
                                                   &out_payload->operand)) {
      return true;
    }
    out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA;
    return true;
  }

  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_scalar(loom_type_element_type(view_type),
                                              plain_integer_is_unsigned,
                                              &numeric_type)) {
    return false;
  }
  out_payload->kind = LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT;
  out_payload->operand.numeric_type = numeric_type;
  out_payload->operand.encoded.available_capability_flags =
      loom_contract_plain_fragment_available_capability_flags(role);
  return true;
}

static bool loom_contract_payload_is_supported(
    const loom_contract_view_payload_t* payload) {
  return payload->kind == LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT ||
         payload->kind == LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA;
}

bool loom_contract_request_from_matrix_payloads(
    const loom_contract_matrix_request_options_t* options,
    loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  loom_contract_request_initialize(out_request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  if (!loom_contract_payload_is_supported(&options->lhs) ||
      !loom_contract_payload_is_supported(&options->rhs)) {
    return loom_contract_storage_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                      out_diagnostic);
  }

  out_request->kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  out_request->arithmetic = options->arithmetic;
  out_request->shape = options->shape;
  out_request->shape_value_refs = options->shape_value_refs;
  if (out_request->shape.block_count == 0 &&
      !loom_contract_value_ref_is_present(
          out_request->shape_value_refs.block_count)) {
    out_request->shape.block_count = 1;
  }
  out_request->k_group_size = options->k_group_size;
  out_request->lhs = options->lhs.operand;
  out_request->lhs.role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
  if (options->lhs.kind == LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT) {
    out_request->lhs.encoded.available_capability_flags =
        loom_contract_plain_fragment_available_capability_flags(
            LOOM_CONTRACT_OPERAND_ROLE_LHS);
  }
  out_request->rhs = options->rhs.operand;
  out_request->rhs.role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
  if (options->rhs.kind == LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT) {
    out_request->rhs.encoded.available_capability_flags =
        loom_contract_plain_fragment_available_capability_flags(
            LOOM_CONTRACT_OPERAND_ROLE_RHS);
  }
  out_request->accumulator = (loom_contract_operand_t){
      .role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
      .numeric_type = options->accumulator_numeric_type,
      .encoded =
          {
              .available_capability_flags =
                  loom_contract_plain_fragment_available_capability_flags(
                      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR),
          },
  };
  out_request->result = (loom_contract_operand_t){
      .role = LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      .numeric_type = options->result_numeric_type,
  };
  out_request->fragment = options->fragment;
  out_request->capability_class = options->capability_class;
  out_request->policy = options->policy;
  return loom_contract_request_validate(out_request, out_diagnostic);
}
