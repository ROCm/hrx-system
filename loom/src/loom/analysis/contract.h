// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent contraction request records.
//
// Contract requests preserve algebra, operand roles, payload types, fragment
// summaries, and lowering policy before a target adapter projects the request
// into a target-owned descriptor table. The generic layer must not name target
// descriptors, opcodes, native packets, or LLVM intrinsics.

#ifndef LOOM_ANALYSIS_CONTRACT_H_
#define LOOM_ANALYSIS_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_roles.h"
#include "loom/analysis/policy.h"
#include "loom/ir/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_contract_kind_e {
  // Unknown or uninitialized contract kind.
  LOOM_CONTRACT_KIND_UNKNOWN = 0,
  // Matrix multiply-like M/N/K contraction.
  LOOM_CONTRACT_KIND_MATRIX_MULTIPLY = 1,
  // Vector dot-product contraction.
  LOOM_CONTRACT_KIND_VECTOR_DOT = 2,
  // Outer-product contraction.
  LOOM_CONTRACT_KIND_OUTER_PRODUCT = 3,
} loom_contract_kind_t;

typedef enum loom_contract_arithmetic_e {
  // Unknown or uninitialized arithmetic family.
  LOOM_CONTRACT_ARITHMETIC_UNKNOWN = 0,
  // Floating-point multiply-add style contraction.
  LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT = 1,
  // Integer multiply-add style contraction.
  LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT = 2,
  // Mixed payload/accumulator contraction.
  LOOM_CONTRACT_ARITHMETIC_MIXED_DOT = 3,
  // Contraction depends on an explicit numeric transform.
  LOOM_CONTRACT_ARITHMETIC_TRANSFORM_BACKED = 4,
} loom_contract_arithmetic_t;

typedef enum loom_contract_numeric_type_e {
  // Unknown or uninitialized numeric payload.
  LOOM_CONTRACT_NUMERIC_UNKNOWN = 0,
  // The payload is interpreted as signed i4 elements.
  LOOM_CONTRACT_NUMERIC_I4 = 1,
  // The payload is interpreted as unsigned i4 elements.
  LOOM_CONTRACT_NUMERIC_U4 = 2,
  // The payload is interpreted as signed i8 elements.
  LOOM_CONTRACT_NUMERIC_I8 = 3,
  // The payload is interpreted as unsigned i8 elements.
  LOOM_CONTRACT_NUMERIC_U8 = 4,
  // The payload is interpreted as signed i16 elements.
  LOOM_CONTRACT_NUMERIC_I16 = 5,
  // The payload is interpreted as unsigned i16 elements.
  LOOM_CONTRACT_NUMERIC_U16 = 6,
  // The payload is interpreted as signed i32 elements.
  LOOM_CONTRACT_NUMERIC_I32 = 7,
  // The payload is interpreted as unsigned i32 elements.
  LOOM_CONTRACT_NUMERIC_U32 = 8,
  // The payload is interpreted as IEEE f16 elements.
  LOOM_CONTRACT_NUMERIC_F16 = 9,
  // The payload is interpreted as BF16 elements.
  LOOM_CONTRACT_NUMERIC_BF16 = 10,
  // The payload is interpreted as IEEE f32 elements.
  LOOM_CONTRACT_NUMERIC_F32 = 11,
  // The payload is interpreted as IEEE f64 elements.
  LOOM_CONTRACT_NUMERIC_F64 = 12,
  // The payload is interpreted as FP8 elements.
  LOOM_CONTRACT_NUMERIC_FP8 = 13,
  // The payload is interpreted as BF8 elements.
  LOOM_CONTRACT_NUMERIC_BF8 = 14,
  // The payload is interpreted as FP6 elements.
  LOOM_CONTRACT_NUMERIC_FP6 = 15,
  // The payload is interpreted as BF6 elements.
  LOOM_CONTRACT_NUMERIC_BF6 = 16,
  // The payload is interpreted as FP4 elements.
  LOOM_CONTRACT_NUMERIC_FP4 = 17,
} loom_contract_numeric_type_t;

typedef enum loom_contract_scale_kind_e {
  // Unknown or uninitialized scale requirement.
  LOOM_CONTRACT_SCALE_UNKNOWN = 0,
  // No explicit scale operands are required.
  LOOM_CONTRACT_SCALE_NONE = 1,
  // 32-bit scale exponent operands are required.
  LOOM_CONTRACT_SCALE_32 = 2,
  // 16-bit scale exponent operands are required.
  LOOM_CONTRACT_SCALE_16 = 3,
} loom_contract_scale_kind_t;

typedef enum loom_contract_capability_class_e {
  // Unknown or no target primitive capability class.
  LOOM_CONTRACT_CAPABILITY_CLASS_UNKNOWN = 0,
  // CPU packed-dot/vector-dot target primitive class.
  LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT = 1,
  // GPU matrix contract target primitive class.
  LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX = 2,
  // Target tensor-memory movement class.
  LOOM_CONTRACT_CAPABILITY_CLASS_TENSOR_MOVEMENT = 3,
  // Target or runtime ukernel contract class.
  LOOM_CONTRACT_CAPABILITY_CLASS_UKERNEL = 4,
} loom_contract_capability_class_t;

typedef enum loom_contract_fragment_atom_bits_e {
  // The fragment carries a logical algebra axis.
  LOOM_CONTRACT_FRAGMENT_LOGICAL = 1u << 0,
  // The fragment carries a physical storage-record axis.
  LOOM_CONTRACT_FRAGMENT_STORAGE_RECORD = 1u << 1,
  // The fragment carries payload-internal grouping.
  LOOM_CONTRACT_FRAGMENT_INTERNAL = 1u << 2,
  // The fragment carries CPU/vector lane ownership.
  LOOM_CONTRACT_FRAGMENT_VECTOR_LANE = 1u << 3,
  // The fragment groups multiple target instructions for one logical contract.
  LOOM_CONTRACT_FRAGMENT_INSTRUCTION_GROUP = 1u << 4,
  // The fragment carries subgroup or wave lane ownership.
  LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE = 1u << 5,
  // The fragment carries workgroup-level ownership.
  LOOM_CONTRACT_FRAGMENT_WORKGROUP = 1u << 6,
  // The fragment carries bank or swizzle-sensitive ownership.
  LOOM_CONTRACT_FRAGMENT_MEMORY_BANK = 1u << 7,
} loom_contract_fragment_atom_bits_t;

// Bitset of loom_contract_fragment_atom_bits_t values.
typedef uint32_t loom_contract_fragment_atom_flags_t;

typedef enum loom_contract_capability_flag_bits_e {
  // The request has sparse metadata operands or facts available.
  LOOM_CONTRACT_CAPABILITY_SPARSE_METADATA = 1u << 0,
  // The request has explicit scale operands or facts available.
  LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS = 1u << 1,
  // The request has matrix or payload format-selector operands or facts
  // available.
  LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS = 1u << 2,
  // The request has target reuse operands or facts available.
  LOOM_CONTRACT_CAPABILITY_REUSE = 1u << 3,
  // The request has clamp operands or facts available.
  LOOM_CONTRACT_CAPABILITY_CLAMP = 1u << 4,
  // The request has sign-selection operands or facts available.
  LOOM_CONTRACT_CAPABILITY_SIGN_SELECT = 1u << 5,
  // The request has operand modifier operands or facts available.
  LOOM_CONTRACT_CAPABILITY_OPERAND_MODIFIERS = 1u << 6,
  // The request has accumulator modifier operands or facts available.
  LOOM_CONTRACT_CAPABILITY_ACCUMULATOR_MODIFIER = 1u << 7,
  // The request has op-select operands or facts available.
  LOOM_CONTRACT_CAPABILITY_OPSEL = 1u << 8,
  // The request has scale-format selector operands or facts available.
  LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS = 1u << 9,
  // The request permits zero-scale fallback to an unscaled primitive.
  LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK = 1u << 10,
} loom_contract_capability_flag_bits_t;

// Bitset of loom_contract_capability_flag_bits_t values.
typedef uint32_t loom_contract_capability_flags_t;

// Dense auxiliary operand key slots. Keep this enum in the same order as the
// bit enum below so key -> bit conversion is a shift, not a lookup.
typedef enum loom_contract_auxiliary_operand_key_e {
  // Primary/local scale value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE = 0,
  // Secondary/global/super-scale value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SECONDARY_SCALE = 1,
  // Affine zero-point value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_ZERO_POINT = 2,
  // Affine minimum/base value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_MIN = 3,
  // Sparse mask, index, or structured metadata value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA = 4,
  // Codebook or table lookup value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_CODEBOOK_TABLE = 5,
  // Residual/outlier correction value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RESIDUAL = 6,
  // Sign-side-stream value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SIGN = 7,
  // Runtime amax/statistics value.
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RUNTIME_AMAX = 8,
  LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_ = 9,
} loom_contract_auxiliary_operand_key_t;

typedef enum loom_contract_auxiliary_operand_flag_bits_e {
  // Primary/local scale values are provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE,
  // Secondary/global/super-scale values are provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_SECONDARY_SCALE =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SECONDARY_SCALE,
  // Affine zero-point values are provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_ZERO_POINT =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_ZERO_POINT,
  // Affine minimum/base values are provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_MIN =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_MIN,
  // Sparse mask, index, or structured metadata is provided as an auxiliary
  // operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_SPARSE_METADATA =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA,
  // Codebook or table lookup data is provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_CODEBOOK_TABLE =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_CODEBOOK_TABLE,
  // Residual/outlier correction data is provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_RESIDUAL =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RESIDUAL,
  // Sign-side-stream data is provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_SIGN =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SIGN,
  // Runtime amax/statistics data is provided as an auxiliary operand.
  LOOM_CONTRACT_AUXILIARY_OPERAND_RUNTIME_AMAX =
      1u << LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_RUNTIME_AMAX,
} loom_contract_auxiliary_operand_flag_bits_t;

// Bitset of loom_contract_auxiliary_operand_flag_bits_t values.
typedef uint32_t loom_contract_auxiliary_operand_flags_t;

static_assert(LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_ <= 32,
              "auxiliary operand flags must fit in one 32-bit word");

// Optional SSA value reference. Zero is absent; present entries store the
// referenced loom_value_id_t plus one so all-zero initialization means absent.
typedef uint32_t loom_contract_value_ref_t;

// Returns an absent optional value reference.
static inline loom_contract_value_ref_t loom_contract_value_ref_absent(void) {
  return 0;
}

// Returns true when |ref| names an SSA value.
static inline bool loom_contract_value_ref_is_present(
    loom_contract_value_ref_t ref) {
  return ref != 0;
}

// Packs |value_id| into an optional value reference.
static inline loom_contract_value_ref_t loom_contract_value_ref_from_value_id(
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return loom_contract_value_ref_absent();
  }
  return value_id + 1u;
}

// Unpacks |ref| into an SSA value ID, or LOOM_VALUE_ID_INVALID when absent.
static inline loom_value_id_t loom_contract_value_ref_value_id(
    loom_contract_value_ref_t ref) {
  return loom_contract_value_ref_is_present(ref) ? ref - 1u
                                                 : LOOM_VALUE_ID_INVALID;
}

// Returns the flag bit corresponding to |key|, or zero for invalid values.
static inline loom_contract_auxiliary_operand_flags_t
loom_contract_auxiliary_operand_key_flag(
    loom_contract_auxiliary_operand_key_t key) {
  if (key >= LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_) {
    return 0;
  }
  return 1u << key;
}

typedef struct loom_contract_shape_t {
  // Exact M/result-row extent when proven, or 0 when dynamic.
  int64_t m;

  // Exact N/result-column extent when proven, or 0 when dynamic.
  int64_t n;

  // Exact K/reduction extent when proven, or 0 when dynamic.
  int64_t k;
} loom_contract_shape_t;

typedef struct loom_contract_shape_value_refs_t {
  // Optional SSA value carrying the M/result-row extent.
  loom_contract_value_ref_t m;

  // Optional SSA value carrying the N/result-column extent.
  loom_contract_value_ref_t n;

  // Optional SSA value carrying the K/reduction extent.
  loom_contract_value_ref_t k;

  // Optional SSA value carrying the K-group element count.
  loom_contract_value_ref_t k_group_size;
} loom_contract_shape_value_refs_t;

typedef struct loom_contract_encoded_operand_t {
  // Schema facts proven for the source operand before preparation or
  // reencoding.
  loom_value_fact_storage_schema_t source_schema;

  // Schema facts the selected target primitive or reference path consumes.
  loom_value_fact_storage_schema_t target_schema;

  // Bitset of auxiliary SSA operands currently attached to this role.
  loom_contract_auxiliary_operand_flags_t available_auxiliary_operands;

  // Optional SSA value references indexed by contract auxiliary key.
  // Entries are present iff the corresponding available bit is set.
  loom_contract_value_ref_t
      auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_];

  // Bitset of auxiliary SSA operands required by target_schema.
  loom_contract_auxiliary_operand_flags_t required_auxiliary_operands;

  // Bitset of capability facts or operands currently attached to this role.
  loom_contract_capability_flags_t available_capability_flags;

  // Bitset of capability facts the selected target primitive must preserve for
  // this role.
  loom_contract_capability_flags_t required_capability_flags;
} loom_contract_encoded_operand_t;

typedef struct loom_contract_operand_t {
  // Role this operand plays in the contraction.
  loom_contract_operand_role_t role;

  // Logical numeric interpretation of this operand payload.
  loom_contract_numeric_type_t numeric_type;

  // Target-fragment payload register count when already known, or 0.
  uint16_t payload_register_count;

  // Logical scalar element count represented by the payload, or 0.
  uint16_t payload_element_count;

  // Role-local encoded payload, scale, sparsity, and auxiliary-data facts.
  loom_contract_encoded_operand_t encoded;
} loom_contract_operand_t;

typedef struct loom_contract_fragment_t {
  // Bitset of loom_contract_fragment_atom_bits_t values.
  loom_contract_fragment_atom_flags_t atom_bits;

  // Native vector width in bits for CPU/vector fragments, or 0.
  uint16_t vector_bit_width;

  // Source lanes consumed from each input vector, or 0.
  uint16_t source_lane_count;

  // Accumulator/result lanes produced by one vector primitive, or 0.
  uint16_t result_lane_count;

  // Subgroup or wave size for GPU fragments, or 0 when target-owned.
  uint16_t subgroup_size;
} loom_contract_fragment_t;

typedef struct loom_contract_request_t {
  // High-level contraction kind.
  loom_contract_kind_t kind;

  // Arithmetic family for the contraction.
  loom_contract_arithmetic_t arithmetic;

  // Exact logical contract shape facts, with 0 for dynamic dimensions.
  loom_contract_shape_t shape;

  // Optional SSA witnesses for dynamic logical shape and K-group dimensions.
  loom_contract_shape_value_refs_t shape_value_refs;

  // Exact K-group element count when proven, or 0 when dynamic.
  uint16_t k_group_size;

  // Left-hand source operand facts.
  loom_contract_operand_t lhs;

  // Right-hand source operand facts.
  loom_contract_operand_t rhs;

  // Accumulator operand facts.
  loom_contract_operand_t accumulator;

  // Result operand facts.
  loom_contract_operand_t result;

  // Fragment ownership facts required to preserve target-shaped structure.
  loom_contract_fragment_t fragment;

  // Requested target primitive capability class.
  loom_contract_capability_class_t capability_class;

  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_contract_request_t;

enum loom_contract_rejection_bits_e {
  // No generic rejection reason was recorded.
  LOOM_CONTRACT_REJECTION_NONE = 0u,
  // The request itself was invalid or absent.
  LOOM_CONTRACT_REJECTION_INVALID_REQUEST = 1u << 0,
  // The logical M/N/K shape or K grouping was missing or invalid.
  LOOM_CONTRACT_REJECTION_SHAPE = 1u << 1,
  // One or more operand roles were missing or inconsistent.
  LOOM_CONTRACT_REJECTION_ROLE = 1u << 2,
  // One or more numeric payload facts were missing or unsupported.
  LOOM_CONTRACT_REJECTION_NUMERIC = 1u << 3,
  // Fragment facts required by the requested capability class were missing.
  LOOM_CONTRACT_REJECTION_FRAGMENT = 1u << 4,
  // The requested capability class or capability flags were unsupported.
  LOOM_CONTRACT_REJECTION_CAPABILITY = 1u << 5,
  // Lowering policy does not permit the only available lowering family.
  LOOM_CONTRACT_REJECTION_POLICY = 1u << 6,
  // One or more encoded schema facts were missing or internally inconsistent.
  LOOM_CONTRACT_REJECTION_SCHEMA = 1u << 7,
  // One or more required auxiliary data operands were missing.
  LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND = 1u << 8,
};

// Bitset of loom_contract_rejection_bits_e values.
typedef uint32_t loom_contract_rejection_bits_t;

typedef enum loom_contract_rejection_detail_e {
  // No finer-grained source rejection detail was recorded.
  LOOM_CONTRACT_REJECTION_DETAIL_NONE = 0,
  // Matrix LHS payload is subgroup-distributed without reference ownership
  // semantics.
  LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_LHS_FRAGMENT_OWNERSHIP = 1,
  // Matrix RHS payload is subgroup-distributed without reference ownership
  // semantics.
  LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RHS_FRAGMENT_OWNERSHIP = 2,
  // Matrix init payload is subgroup-distributed without reference ownership
  // semantics.
  LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_INIT_FRAGMENT_OWNERSHIP = 3,
  // Matrix result payload is not representable by the scalar reference
  // aggregate shape selected for the op.
  LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RESULT_FRAGMENT_PAYLOAD = 4,
} loom_contract_rejection_detail_t;

typedef struct loom_contract_diagnostic_t {
  // Bitset of loom_contract_rejection_bits_t values.
  loom_contract_rejection_bits_t rejection_bits;
} loom_contract_diagnostic_t;

// Initializes a request with explicit unknown values and reference-allowed
// policy. Callers then fill the fields they have proven.
void loom_contract_request_initialize(loom_contract_request_t* out_request);

// Validates the target-independent fields required before target projection.
bool loom_contract_request_validate(const loom_contract_request_t* request,
                                    loom_contract_diagnostic_t* out_diagnostic);

// Returns the union of role-local capability facts available to target
// adapters.
loom_contract_capability_flags_t
loom_contract_request_available_capability_flags(
    const loom_contract_request_t* request);

// Returns the union of role-local capability facts the selected target
// primitive must preserve.
loom_contract_capability_flags_t
loom_contract_request_required_capability_flags(
    const loom_contract_request_t* request);

// Maps a Loom scalar type to a generic contract numeric type.
bool loom_contract_numeric_type_from_scalar(
    loom_scalar_type_t scalar_type, bool unsigned_integer,
    loom_contract_numeric_type_t* out_numeric_type);

// Returns the stable diagnostic spelling for a generic numeric type.
iree_string_view_t loom_contract_numeric_type_name(
    loom_contract_numeric_type_t numeric_type);

// Returns a diagnostic detail string for generic contract rejection flags.
iree_string_view_t loom_contract_rejection_detail(
    loom_contract_rejection_bits_t rejection_bits);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_H_
