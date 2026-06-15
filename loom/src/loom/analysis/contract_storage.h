// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-schema fact adapters for generic contract requests.

#ifndef LOOM_ANALYSIS_CONTRACT_STORAGE_H_
#define LOOM_ANALYSIS_CONTRACT_STORAGE_H_

#include "loom/analysis/contract.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps an encoded operand format to a generic contract numeric type.
bool loom_contract_numeric_type_from_encoded_format(
    loom_value_fact_numeric_format_flags_t format,
    loom_contract_numeric_type_t* out_numeric_type);

// Maps a matrix storage-schema scale kind to a generic contract scale kind.
bool loom_contract_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_contract_scale_kind_t* out_scale_kind);

// Returns auxiliary SSA data roles required by an encoded storage schema.
loom_contract_auxiliary_operand_flags_t
loom_contract_required_auxiliary_operands_from_storage_schema(
    loom_value_fact_storage_schema_t schema);

// Returns generic contract capability facts proven by a storage schema.
loom_contract_capability_flags_t
loom_contract_available_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema);

// Returns generic contract capability facts required to preserve an encoded
// storage schema without reference decode.
loom_contract_capability_flags_t
loom_contract_required_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema);

// Maps an encoded storage-schema payload to a generic contract operand.
bool loom_contract_operand_from_storage_schema(
    loom_contract_operand_role_t role, loom_value_fact_storage_schema_t schema,
    loom_contract_operand_t* out_operand);

typedef enum loom_contract_view_payload_kind_e {
  // No usable payload interpretation is known.
  LOOM_CONTRACT_VIEW_PAYLOAD_UNKNOWN = 0,
  // The view payload is the shaped type's element type.
  LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT = 1,
  // The view payload is an encoded storage schema with native-fragment facts.
  LOOM_CONTRACT_VIEW_PAYLOAD_ENCODED_OPERAND_SCHEMA = 2,
  // The view has a storage schema, but not one representable as a native
  // matrix contract payload.
  LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA = 3,
} loom_contract_view_payload_kind_t;

typedef struct loom_contract_view_payload_t {
  // Payload interpretation category.
  loom_contract_view_payload_kind_t kind;

  // Generic operand facts for this payload.
  loom_contract_operand_t operand;
} loom_contract_view_payload_t;

// Queries a view type's contract payload interpretation.
//
// Physical storage schemas are preserved as schema-backed payloads. Views
// without storage schemas fall back to their shaped element type. Integer
// element views use |plain_integer_is_unsigned| to select signedness because
// Loom integer element types do not carry signedness.
bool loom_contract_view_payload_from_type(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_contract_operand_role_t role,
    bool plain_integer_is_unsigned, loom_contract_view_payload_t* out_payload);

typedef struct loom_contract_matrix_request_options_t {
  // Exact matrix/vector contraction shape facts, with 0 for dynamic dimensions.
  loom_contract_shape_t shape;

  // Optional SSA witnesses for dynamic shape and K-group dimensions.
  loom_contract_shape_value_refs_t shape_value_refs;

  // Exact K-group element count when proven, or 0 when dynamic.
  uint16_t k_group_size;

  // Left-hand payload interpretation.
  loom_contract_view_payload_t lhs;

  // Right-hand payload interpretation.
  loom_contract_view_payload_t rhs;

  // Accumulator lane numeric type.
  loom_contract_numeric_type_t accumulator_numeric_type;

  // Result lane numeric type.
  loom_contract_numeric_type_t result_numeric_type;

  // Target-independent arithmetic family.
  loom_contract_arithmetic_t arithmetic;

  // Fragment ownership facts that must survive target selection.
  loom_contract_fragment_t fragment;

  // Requested target primitive capability class.
  loom_contract_capability_class_t capability_class;

  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_contract_matrix_request_options_t;

// Builds a target-independent matrix/packed-dot contract from view payloads.
//
// The resulting request does not name target descriptors, processor feature
// bits, opcodes, packets, intrinsics, or target-specific enum values. Target
// adapters project this generic request into their own descriptor tables.
bool loom_contract_request_from_matrix_payloads(
    const loom_contract_matrix_request_options_t* options,
    loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_STORAGE_H_
