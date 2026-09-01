// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent numeric format facts used by encoded operand schemas.
//
// These describe the semantic value represented by payload, scale, table, or
// sparse metadata fields. They are not promises that a physical vector or
// buffer value uses unpacked lanes of this type.

#ifndef LOOM_UTIL_NUMERIC_FORMAT_H_
#define LOOM_UTIL_NUMERIC_FORMAT_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Numeric format bit values use macros because C enums cannot portably
// represent values wider than int.
typedef uint64_t loom_value_fact_numeric_format_bits_t;

// Zero: unknown for required fields, none for optional fields.
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN UINT64_C(0)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE UINT64_C(0)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F64 (UINT64_C(1) << 0)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F32 (UINT64_C(1) << 1)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_TF32 (UINT64_C(1) << 2)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 (UINT64_C(1) << 3)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16 (UINT64_C(1) << 4)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I32 (UINT64_C(1) << 5)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U32 (UINT64_C(1) << 6)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I16 (UINT64_C(1) << 7)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U16 (UINT64_C(1) << 8)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I8 (UINT64_C(1) << 9)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U8 (UINT64_C(1) << 10)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I6 (UINT64_C(1) << 11)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U6 (UINT64_C(1) << 12)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I5 (UINT64_C(1) << 13)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U5 (UINT64_C(1) << 14)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I4 (UINT64_C(1) << 15)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U4 (UINT64_C(1) << 16)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I3 (UINT64_C(1) << 17)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U3 (UINT64_C(1) << 18)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I2 (UINT64_C(1) << 19)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U2 (UINT64_C(1) << 20)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_I1 (UINT64_C(1) << 21)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_U1 (UINT64_C(1) << 22)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3 (UINT64_C(1) << 23)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2 (UINT64_C(1) << 24)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN (UINT64_C(1) << 25)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ (UINT64_C(1) << 26)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ (UINT64_C(1) << 27)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0 (UINT64_C(1) << 28)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8 (UINT64_C(1) << 29)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2 (UINT64_C(1) << 30)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3 (UINT64_C(1) << 31)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6 (UINT64_C(1) << 32)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1 (UINT64_C(1) << 33)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_TERNARY (UINT64_C(1) << 34)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_SIGN_BIT (UINT64_C(1) << 35)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX (UINT64_C(1) << 36)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8 (UINT64_C(1) << 37)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6 (UINT64_C(1) << 38)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4 (UINT64_C(1) << 39)
#define LOOM_VALUE_FACT_NUMERIC_FORMAT_ALL                                    \
  (LOOM_VALUE_FACT_NUMERIC_FORMAT_F64 | LOOM_VALUE_FACT_NUMERIC_FORMAT_F32 |  \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_TF32 | LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 | \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I32 | \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U32 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I16 |  \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U16 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I8 |   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U8 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I6 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U6 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I5 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U5 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I4 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U4 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I3 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U3 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I2 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U2 | LOOM_VALUE_FACT_NUMERIC_FORMAT_I1 |    \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_U1 |                                        \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN |                                 \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ |                               \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ |                               \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8 |                                       \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6 |                                       \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1 |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_TERNARY |                                   \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_SIGN_BIT |                                  \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX |                            \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8 |                                  \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6 |                                  \
   LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4)

typedef uint64_t loom_value_fact_numeric_format_flags_t;

typedef enum loom_numeric_format_kind_e {
  // Unknown or uninitialized numeric format.
  LOOM_NUMERIC_FORMAT_KIND_UNKNOWN = 0,
  // Signed integer payload.
  LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER = 1,
  // Unsigned integer payload.
  LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER = 2,
  // Floating-point payload.
  LOOM_NUMERIC_FORMAT_KIND_FLOAT = 3,
  // Quantized signed integer payload.
  LOOM_NUMERIC_FORMAT_KIND_QUANTIZED_SIGNED_INTEGER = 4,
  // Codebook lookup index payload.
  LOOM_NUMERIC_FORMAT_KIND_CODEBOOK_INDEX = 5,
  // Ternary payload.
  LOOM_NUMERIC_FORMAT_KIND_TERNARY = 6,
  // Sign-only payload.
  LOOM_NUMERIC_FORMAT_KIND_SIGN_BIT = 7,
} loom_numeric_format_kind_t;

typedef enum loom_numeric_float_family_e {
  // Not a floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_NONE = 0,
  // IEEE-like binary floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_IEEE = 1,
  // BF16/bfloat-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_BFLOAT = 2,
  // FP8 E4M3-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_FP8 = 3,
  // BF8/E5M2-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_BF8 = 4,
  // FP8 E8M0 exponent-only floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_F8_E8M0 = 5,
  // FP6-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_FP6 = 6,
  // BF6-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_BF6 = 7,
  // FP4-style floating-point format.
  LOOM_NUMERIC_FLOAT_FAMILY_FP4 = 8,
} loom_numeric_float_family_t;

typedef enum loom_numeric_format_flag_bits_e {
  // The format encodes a sign bit.
  LOOM_NUMERIC_FORMAT_FLAG_SIGNED = 1u << 0,
  // The format has at least one NaN encoding.
  LOOM_NUMERIC_FORMAT_FLAG_HAS_NAN = 1u << 1,
  // The format has at least one infinity encoding.
  LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY = 1u << 2,
  // The format has no infinity encodings.
  LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY = 1u << 3,
  // The format does not encode negative zero.
  LOOM_NUMERIC_FORMAT_FLAG_UNSIGNED_ZERO = 1u << 4,
  // Matrix/packed-dot contracts need payload format selector facts for this
  // encoded element format.
  LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR = 1u << 5,
} loom_numeric_format_flag_bits_t;

// Bitset of loom_numeric_format_flag_bits_t values.
typedef uint32_t loom_numeric_format_flags_t;

typedef struct loom_numeric_format_info_t {
  // Single-bit numeric-format fact represented by this row.
  loom_value_fact_numeric_format_flags_t format;

  // Broad semantic payload category.
  loom_numeric_format_kind_t kind;

  // Floating-point family, or NONE for non-floating-point formats.
  loom_numeric_float_family_t float_family;

  // Encoded payload bit count.
  uint8_t storage_bit_count;

  // Encoded exponent bit count for floating-point formats.
  uint8_t exponent_bit_count;

  // Encoded mantissa bit count for floating-point formats.
  uint8_t mantissa_bit_count;

  // Direct Loom scalar carrier, or LOOM_SCALAR_TYPE_NONE when none exists.
  loom_scalar_type_t direct_scalar_type;

  // Special-value and contract behavior flags.
  loom_numeric_format_flags_t flags;
} loom_numeric_format_info_t;

// Returns metadata for a single numeric-format fact bit.
bool loom_numeric_format_info(loom_value_fact_numeric_format_flags_t format,
                              const loom_numeric_format_info_t** out_info);

// Returns true when the format is a single known finite-only format.
bool loom_numeric_format_is_finite_only(
    loom_value_fact_numeric_format_flags_t format);

// Returns true when encoded matrix/dot payloads need selector facts for this
// format.
bool loom_numeric_format_needs_encoded_payload_selector(
    loom_value_fact_numeric_format_flags_t format);

// Returns the directly represented Loom scalar type for |format|. Returns
// false for packed or otherwise schema-only formats without a scalar carrier.
bool loom_numeric_format_direct_scalar_type(
    loom_value_fact_numeric_format_flags_t format,
    loom_scalar_type_t* out_type);

// Returns true when integer arithmetic interprets |format| as unsigned.
bool loom_numeric_format_uses_unsigned_integer_semantics(
    loom_value_fact_numeric_format_flags_t format);

// Returns the numeric-format fact corresponding to a directly represented Loom
// scalar type, or NONE when no single numeric-format fact exists.
loom_value_fact_numeric_format_flags_t loom_numeric_format_from_scalar_type(
    loom_scalar_type_t type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_NUMERIC_FORMAT_H_
