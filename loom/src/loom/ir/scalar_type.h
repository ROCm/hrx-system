// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar element type vocabulary for Loom IR.
//
// These ordinals are serialized directly by the current bytecode format. They
// may change only when LOOM_BYTECODE_FORMAT_VERSION advances with them.

#ifndef LOOM_IR_SCALAR_TYPE_H_
#define LOOM_IR_SCALAR_TYPE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar element type kind.
//
// Ordered: address types, integers by width, floats by width.
enum loom_scalar_type_e {
  // No scalar type. Zero so absent and zero-initialized fields remain invalid.
  LOOM_SCALAR_TYPE_NONE = 0,
  // Signed target-width integer for loop bounds, dimension sizes, and general
  // indexing arithmetic. Arithmetic follows signed semantics.
  LOOM_SCALAR_TYPE_INDEX = 1,
  // Unsigned target-selected-width integer for buffer byte offsets and
  // addressing. Arithmetic follows unsigned semantics. Its target carrier may
  // be wider than the index carrier.
  LOOM_SCALAR_TYPE_OFFSET = 2,
  // 1-bit integer. Boolean results (comparisons, predicates).
  LOOM_SCALAR_TYPE_I1 = 3,
  // 8-bit signed integer. Quantized weights, byte-level data.
  LOOM_SCALAR_TYPE_I8 = 4,
  // 16-bit signed integer. Intermediate quantized computations.
  LOOM_SCALAR_TYPE_I16 = 5,
  // 32-bit signed integer. General-purpose integer arithmetic.
  LOOM_SCALAR_TYPE_I32 = 6,
  // 64-bit signed integer. Large counts, hash values.
  LOOM_SCALAR_TYPE_I64 = 7,
  // 8-bit float, E4M3 variant.
  LOOM_SCALAR_TYPE_F8E4M3 = 8,
  // 8-bit float, E5M2 variant.
  LOOM_SCALAR_TYPE_F8E5M2 = 9,
  // IEEE 754 binary16 half-precision.
  LOOM_SCALAR_TYPE_F16 = 10,
  // bfloat16.
  LOOM_SCALAR_TYPE_BF16 = 11,
  // IEEE 754 binary32 single-precision.
  LOOM_SCALAR_TYPE_F32 = 12,
  // IEEE 754 binary64 double-precision.
  LOOM_SCALAR_TYPE_F64 = 13,
  // One past the final scalar type ordinal.
  LOOM_SCALAR_TYPE_COUNT_,
};
// Raw scalar type storage. Parsed and bytecode-loaded types may carry invalid
// ordinals until validation reports them.
typedef uint8_t loom_scalar_type_t;

// Returns true if |type| identifies a concrete scalar type.
static inline bool loom_scalar_type_is_valid(loom_scalar_type_t type) {
  return type > LOOM_SCALAR_TYPE_NONE && type < LOOM_SCALAR_TYPE_COUNT_;
}

typedef enum loom_scalar_type_fp8_special_policy_e {
  // Top exponent encodes infinities and NaNs.
  LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE = 0,
  // Top exponent remains finite except for the all-ones mantissa NaN.
  LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN = 1,
  // All payloads are finite except the negative-zero encoding, which is NaN.
  LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO = 2,
} loom_scalar_type_fp8_special_policy_t;

// Compact set of scalar types.
typedef uint16_t loom_scalar_type_set_t;
static_assert(LOOM_SCALAR_TYPE_COUNT_ <= 16,
              "scalar type sets must fit in uint16_t");

enum loom_scalar_type_set_bits_e {
  LOOM_SCALAR_TYPE_SET_NONE = 0,
  LOOM_SCALAR_TYPE_SET_INDEX = 1u << LOOM_SCALAR_TYPE_INDEX,
  LOOM_SCALAR_TYPE_SET_OFFSET = 1u << LOOM_SCALAR_TYPE_OFFSET,
  LOOM_SCALAR_TYPE_SET_I1 = 1u << LOOM_SCALAR_TYPE_I1,
  LOOM_SCALAR_TYPE_SET_I8 = 1u << LOOM_SCALAR_TYPE_I8,
  LOOM_SCALAR_TYPE_SET_I16 = 1u << LOOM_SCALAR_TYPE_I16,
  LOOM_SCALAR_TYPE_SET_I32 = 1u << LOOM_SCALAR_TYPE_I32,
  LOOM_SCALAR_TYPE_SET_I64 = 1u << LOOM_SCALAR_TYPE_I64,
  LOOM_SCALAR_TYPE_SET_F8E4M3 = 1u << LOOM_SCALAR_TYPE_F8E4M3,
  LOOM_SCALAR_TYPE_SET_F8E5M2 = 1u << LOOM_SCALAR_TYPE_F8E5M2,
  LOOM_SCALAR_TYPE_SET_F16 = 1u << LOOM_SCALAR_TYPE_F16,
  LOOM_SCALAR_TYPE_SET_BF16 = 1u << LOOM_SCALAR_TYPE_BF16,
  LOOM_SCALAR_TYPE_SET_F32 = 1u << LOOM_SCALAR_TYPE_F32,
  LOOM_SCALAR_TYPE_SET_F64 = 1u << LOOM_SCALAR_TYPE_F64,
  LOOM_SCALAR_TYPE_SET_ADDRESS =
      LOOM_SCALAR_TYPE_SET_INDEX | LOOM_SCALAR_TYPE_SET_OFFSET,
  LOOM_SCALAR_TYPE_SET_INTEGER =
      LOOM_SCALAR_TYPE_SET_I1 | LOOM_SCALAR_TYPE_SET_I8 |
      LOOM_SCALAR_TYPE_SET_I16 | LOOM_SCALAR_TYPE_SET_I32 |
      LOOM_SCALAR_TYPE_SET_I64,
  LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD =
      LOOM_SCALAR_TYPE_SET_I8 | LOOM_SCALAR_TYPE_SET_I16 |
      LOOM_SCALAR_TYPE_SET_I32 | LOOM_SCALAR_TYPE_SET_I64,
  LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD_LE32 = LOOM_SCALAR_TYPE_SET_I8 |
                                              LOOM_SCALAR_TYPE_SET_I16 |
                                              LOOM_SCALAR_TYPE_SET_I32,
  LOOM_SCALAR_TYPE_SET_FLOAT =
      LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 |
      LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_BF16 |
      LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
  LOOM_SCALAR_TYPE_SET_16BIT_FLOAT =
      LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_BF16,
};

static inline bool loom_scalar_type_set_contains(loom_scalar_type_set_t set,
                                                 loom_scalar_type_t type) {
  if (!loom_scalar_type_is_valid(type)) {
    return false;
  }
  return iree_all_bits_set(set, (loom_scalar_type_set_t)(1u << type));
}

typedef struct loom_scalar_type_fp8_format_t {
  // Number of encoded exponent bits.
  uint8_t exponent_bits;
  // Number of encoded mantissa bits.
  uint8_t mantissa_bits;
  // Exponent bias used by the encoded type.
  uint8_t exponent_bias;
  // Top-exponent handling policy.
  loom_scalar_type_fp8_special_policy_t special_policy;
} loom_scalar_type_fp8_format_t;

// Returns the name string for a scalar type (e.g., "f32", "index").
// Returns NULL if |type| is not a concrete scalar type.
const char* loom_scalar_type_name(loom_scalar_type_t type);

// Returns the bitwidth of a scalar type, or 0 if |type| is not concrete. Index
// and offset return 64.
int32_t loom_scalar_type_bitwidth(loom_scalar_type_t type);

// Returns the signed i64 value domain used to reason about integer-like scalar
// values of |type|. Offset is represented as the non-negative address domain
// because IR integer literals are signed i64 payloads.
bool loom_scalar_type_integer_domain(loom_scalar_type_t type, int64_t* out_lo,
                                     int64_t* out_hi);

// Returns the FP8 binary format for |type|, if |type| is an FP8 scalar.
bool loom_scalar_type_fp8_format(loom_scalar_type_t type,
                                 loom_scalar_type_fp8_format_t* out_format);

// Parses a scalar type name. Returns true on success.
bool loom_scalar_type_parse(iree_string_view_t name,
                            loom_scalar_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_SCALAR_TYPE_H_
