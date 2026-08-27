// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/numeric_format.h"

#include "iree/base/internal/math.h"

enum {
  LOOM_NUMERIC_FORMAT_INFO_COUNT = 40,
};

#define LOOM_NUMERIC_FORMAT_FLOAT_FLAGS                                 \
  (LOOM_NUMERIC_FORMAT_FLAG_SIGNED | LOOM_NUMERIC_FORMAT_FLAG_HAS_NAN | \
   LOOM_NUMERIC_FORMAT_FLAG_HAS_INFINITY)

#define LOOM_NUMERIC_FORMAT_FINITE_NAN_FLAGS                            \
  (LOOM_NUMERIC_FORMAT_FLAG_SIGNED | LOOM_NUMERIC_FORMAT_FLAG_HAS_NAN | \
   LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY)

#define LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_FLAGS \
  (LOOM_NUMERIC_FORMAT_FINITE_NAN_FLAGS |                  \
   LOOM_NUMERIC_FORMAT_FLAG_UNSIGNED_ZERO)

#define LOOM_NUMERIC_FORMAT_FINITE_SELECTOR_FLAGS                           \
  (LOOM_NUMERIC_FORMAT_FLAG_SIGNED | LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY | \
   LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR)

#define LOOM_NUMERIC_FORMAT_FINITE_NAN_SELECTOR_FLAGS \
  (LOOM_NUMERIC_FORMAT_FINITE_NAN_FLAGS |             \
   LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR)

#define LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_SELECTOR_FLAGS \
  (LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_FLAGS |             \
   LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR)

#define LOOM_NUMERIC_FORMAT_INFO_ROW(index_, format_, kind_, family_,         \
                                     storage_, exponent_, mantissa_, scalar_, \
                                     flags_)                                  \
  [index_] = {                                                                \
      .format = format_,                                                      \
      .kind = kind_,                                                          \
      .float_family = family_,                                                \
      .storage_bit_count = storage_,                                          \
      .exponent_bit_count = exponent_,                                        \
      .mantissa_bit_count = mantissa_,                                        \
      .direct_scalar_type = scalar_,                                          \
      .flags = flags_,                                                        \
  }

static const loom_numeric_format_info_t
    kLoomNumericFormatInfos[LOOM_NUMERIC_FORMAT_INFO_COUNT] = {
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            0, LOOM_VALUE_FACT_NUMERIC_FORMAT_F64,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_IEEE, 64,
            11, 52, LOOM_SCALAR_TYPE_F64, LOOM_NUMERIC_FORMAT_FLOAT_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            1, LOOM_VALUE_FACT_NUMERIC_FORMAT_F32,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_IEEE, 32,
            8, 23, LOOM_SCALAR_TYPE_F32, LOOM_NUMERIC_FORMAT_FLOAT_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            2, LOOM_VALUE_FACT_NUMERIC_FORMAT_TF32,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_IEEE, 32,
            8, 10, LOOM_SCALAR_TYPE_COUNT_, LOOM_NUMERIC_FORMAT_FLOAT_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            3, LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_IEEE, 16,
            5, 10, LOOM_SCALAR_TYPE_F16, LOOM_NUMERIC_FORMAT_FLOAT_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            4, LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_BFLOAT,
            16, 8, 7, LOOM_SCALAR_TYPE_BF16, LOOM_NUMERIC_FORMAT_FLOAT_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(5, LOOM_VALUE_FACT_NUMERIC_FORMAT_I32,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 32, 0, 0,
                                     LOOM_SCALAR_TYPE_I32,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(6, LOOM_VALUE_FACT_NUMERIC_FORMAT_U32,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 32, 0, 0,
                                     LOOM_SCALAR_TYPE_I32, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(7, LOOM_VALUE_FACT_NUMERIC_FORMAT_I16,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 16, 0, 0,
                                     LOOM_SCALAR_TYPE_I16,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(8, LOOM_VALUE_FACT_NUMERIC_FORMAT_U16,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 16, 0, 0,
                                     LOOM_SCALAR_TYPE_I16, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(9, LOOM_VALUE_FACT_NUMERIC_FORMAT_I8,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 8, 0, 0,
                                     LOOM_SCALAR_TYPE_I8,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(10, LOOM_VALUE_FACT_NUMERIC_FORMAT_U8,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 8, 0, 0,
                                     LOOM_SCALAR_TYPE_I8, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(11, LOOM_VALUE_FACT_NUMERIC_FORMAT_I6,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 6, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(12, LOOM_VALUE_FACT_NUMERIC_FORMAT_U6,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 6, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(13, LOOM_VALUE_FACT_NUMERIC_FORMAT_I5,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 5, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(14, LOOM_VALUE_FACT_NUMERIC_FORMAT_U5,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 5, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(15, LOOM_VALUE_FACT_NUMERIC_FORMAT_I4,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 4, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(16, LOOM_VALUE_FACT_NUMERIC_FORMAT_U4,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 4, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(17, LOOM_VALUE_FACT_NUMERIC_FORMAT_I3,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 3, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(18, LOOM_VALUE_FACT_NUMERIC_FORMAT_U3,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 3, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(19, LOOM_VALUE_FACT_NUMERIC_FORMAT_I2,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 2, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(20, LOOM_VALUE_FACT_NUMERIC_FORMAT_U2,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 2, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(21, LOOM_VALUE_FACT_NUMERIC_FORMAT_I1,
                                     LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 1, 0, 0,
                                     LOOM_SCALAR_TYPE_I1,
                                     LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(22, LOOM_VALUE_FACT_NUMERIC_FORMAT_U1,
                                     LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 1, 0, 0,
                                     LOOM_SCALAR_TYPE_I1, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            23, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_FP8, 8, 4,
            3, LOOM_SCALAR_TYPE_F8E4M3,
            LOOM_NUMERIC_FORMAT_FLOAT_FLAGS |
                LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            24, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_BF8, 8, 5,
            2, LOOM_SCALAR_TYPE_F8E5M2,
            LOOM_NUMERIC_FORMAT_FLOAT_FLAGS |
                LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            25, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_FP8, 8, 4,
            3, LOOM_SCALAR_TYPE_F8E4M3,
            LOOM_NUMERIC_FORMAT_FINITE_NAN_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            26, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_FP8, 8, 4,
            3, LOOM_SCALAR_TYPE_F8E4M3,
            LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            27, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_BF8, 8, 5,
            2, LOOM_SCALAR_TYPE_F8E5M2,
            LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(28, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0,
                                     LOOM_NUMERIC_FORMAT_KIND_FLOAT,
                                     LOOM_NUMERIC_FLOAT_FAMILY_F8_E8M0, 8, 8, 0,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            29, LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_BF8, 8, 5,
            2, LOOM_SCALAR_TYPE_F8E5M2,
            LOOM_NUMERIC_FORMAT_FLOAT_FLAGS |
                LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR),
        LOOM_NUMERIC_FORMAT_INFO_ROW(30, LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2,
                                     LOOM_NUMERIC_FORMAT_KIND_FLOAT,
                                     LOOM_NUMERIC_FLOAT_FAMILY_FP6, 6, 3, 2,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FINITE_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(31, LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3,
                                     LOOM_NUMERIC_FORMAT_KIND_FLOAT,
                                     LOOM_NUMERIC_FLOAT_FAMILY_FP6, 6, 2, 3,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FINITE_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            32, LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6,
            LOOM_NUMERIC_FORMAT_KIND_FLOAT, LOOM_NUMERIC_FLOAT_FAMILY_BF6, 6, 5,
            0, LOOM_SCALAR_TYPE_COUNT_,
            LOOM_NUMERIC_FORMAT_FLAG_SIGNED |
                LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR),
        LOOM_NUMERIC_FORMAT_INFO_ROW(33, LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1,
                                     LOOM_NUMERIC_FORMAT_KIND_FLOAT,
                                     LOOM_NUMERIC_FLOAT_FAMILY_FP4, 4, 2, 1,
                                     LOOM_SCALAR_TYPE_COUNT_,
                                     LOOM_NUMERIC_FORMAT_FINITE_SELECTOR_FLAGS),
        LOOM_NUMERIC_FORMAT_INFO_ROW(34, LOOM_VALUE_FACT_NUMERIC_FORMAT_TERNARY,
                                     LOOM_NUMERIC_FORMAT_KIND_TERNARY,
                                     LOOM_NUMERIC_FLOAT_FAMILY_NONE, 2, 0, 0,
                                     LOOM_SCALAR_TYPE_COUNT_, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            35, LOOM_VALUE_FACT_NUMERIC_FORMAT_SIGN_BIT,
            LOOM_NUMERIC_FORMAT_KIND_SIGN_BIT, LOOM_NUMERIC_FLOAT_FAMILY_NONE,
            1, 0, 0, LOOM_SCALAR_TYPE_COUNT_, LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            36, LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX,
            LOOM_NUMERIC_FORMAT_KIND_CODEBOOK_INDEX,
            LOOM_NUMERIC_FLOAT_FAMILY_NONE, 0, 0, 0, LOOM_SCALAR_TYPE_I8, 0),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            37, LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
            LOOM_NUMERIC_FORMAT_KIND_QUANTIZED_SIGNED_INTEGER,
            LOOM_NUMERIC_FLOAT_FAMILY_NONE, 8, 0, 0, LOOM_SCALAR_TYPE_I8,
            LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            38, LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6,
            LOOM_NUMERIC_FORMAT_KIND_QUANTIZED_SIGNED_INTEGER,
            LOOM_NUMERIC_FLOAT_FAMILY_NONE, 6, 0, 0, LOOM_SCALAR_TYPE_COUNT_,
            LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
        LOOM_NUMERIC_FORMAT_INFO_ROW(
            39, LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4,
            LOOM_NUMERIC_FORMAT_KIND_QUANTIZED_SIGNED_INTEGER,
            LOOM_NUMERIC_FLOAT_FAMILY_NONE, 4, 0, 0, LOOM_SCALAR_TYPE_COUNT_,
            LOOM_NUMERIC_FORMAT_FLAG_SIGNED),
};
static_assert(sizeof(loom_numeric_format_info_t) == 24,
              "direct scalar types must reuse numeric format table padding");

static const loom_value_fact_numeric_format_flags_t
    kLoomNumericFormatsByScalarType[LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_SCALAR_TYPE_I1] = LOOM_VALUE_FACT_NUMERIC_FORMAT_I1,
        [LOOM_SCALAR_TYPE_I8] = LOOM_VALUE_FACT_NUMERIC_FORMAT_I8,
        [LOOM_SCALAR_TYPE_I16] = LOOM_VALUE_FACT_NUMERIC_FORMAT_I16,
        [LOOM_SCALAR_TYPE_I32] = LOOM_VALUE_FACT_NUMERIC_FORMAT_I32,
        [LOOM_SCALAR_TYPE_F8E4M3] = LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
        [LOOM_SCALAR_TYPE_F8E5M2] = LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2,
        [LOOM_SCALAR_TYPE_F16] = LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
        [LOOM_SCALAR_TYPE_BF16] = LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16,
        [LOOM_SCALAR_TYPE_F32] = LOOM_VALUE_FACT_NUMERIC_FORMAT_F32,
        [LOOM_SCALAR_TYPE_F64] = LOOM_VALUE_FACT_NUMERIC_FORMAT_F64,
};
static_assert(IREE_ARRAYSIZE(kLoomNumericFormatsByScalarType) ==
                  LOOM_SCALAR_TYPE_COUNT_,
              "scalar type numeric-format table out of sync");

static bool loom_numeric_format_is_single_bit(
    loom_value_fact_numeric_format_flags_t format) {
  return format != 0 && (format & (format - 1)) == 0;
}

bool loom_numeric_format_info(loom_value_fact_numeric_format_flags_t format,
                              const loom_numeric_format_info_t** out_info) {
  if (out_info != NULL) {
    *out_info = NULL;
  }
  if (!loom_numeric_format_is_single_bit(format)) {
    return false;
  }
  const iree_host_size_t index =
      (iree_host_size_t)iree_math_count_trailing_zeros_u64_width(format, 64);
  if (index >= IREE_ARRAYSIZE(kLoomNumericFormatInfos)) {
    return false;
  }
  const loom_numeric_format_info_t* info = &kLoomNumericFormatInfos[index];
  if (info->format != format) {
    return false;
  }
  if (out_info != NULL) {
    *out_info = info;
  }
  return true;
}

bool loom_numeric_format_is_finite_only(
    loom_value_fact_numeric_format_flags_t format) {
  const loom_numeric_format_info_t* info = NULL;
  return loom_numeric_format_info(format, &info) &&
         iree_any_bit_set(info->flags, LOOM_NUMERIC_FORMAT_FLAG_FINITE_ONLY);
}

bool loom_numeric_format_needs_encoded_payload_selector(
    loom_value_fact_numeric_format_flags_t format) {
  const loom_numeric_format_info_t* info = NULL;
  return loom_numeric_format_info(format, &info) &&
         iree_any_bit_set(info->flags,
                          LOOM_NUMERIC_FORMAT_FLAG_ENCODED_PAYLOAD_SELECTOR);
}

bool loom_numeric_format_direct_scalar_type(
    loom_value_fact_numeric_format_flags_t format,
    loom_scalar_type_t* out_type) {
  const loom_numeric_format_info_t* info = NULL;
  if (!loom_numeric_format_info(format, &info) ||
      info->direct_scalar_type == LOOM_SCALAR_TYPE_COUNT_) {
    return false;
  }
  *out_type = info->direct_scalar_type;
  return true;
}

bool loom_numeric_format_uses_unsigned_integer_semantics(
    loom_value_fact_numeric_format_flags_t format) {
  const loom_numeric_format_info_t* info = NULL;
  return loom_numeric_format_info(format, &info) &&
         (info->kind == LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER ||
          info->kind == LOOM_NUMERIC_FORMAT_KIND_CODEBOOK_INDEX);
}

loom_value_fact_numeric_format_flags_t loom_numeric_format_from_scalar_type(
    loom_scalar_type_t type) {
  return type < IREE_ARRAYSIZE(kLoomNumericFormatsByScalarType)
             ? kLoomNumericFormatsByScalarType[type]
             : LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
}

#undef LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_SELECTOR_FLAGS
#undef LOOM_NUMERIC_FORMAT_FINITE_NAN_SELECTOR_FLAGS
#undef LOOM_NUMERIC_FORMAT_FINITE_SELECTOR_FLAGS
#undef LOOM_NUMERIC_FORMAT_FINITE_NAN_UNSIGNED_ZERO_FLAGS
#undef LOOM_NUMERIC_FORMAT_FINITE_NAN_FLAGS
#undef LOOM_NUMERIC_FORMAT_FLOAT_FLAGS
