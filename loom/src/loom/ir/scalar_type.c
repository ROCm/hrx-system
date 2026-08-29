// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/scalar_type.h"

#include "loom/ir/scalar_type_table.inc"

static const int32_t loom_scalar_type_bitwidths[] = {
    [LOOM_SCALAR_TYPE_INDEX] = 64, [LOOM_SCALAR_TYPE_OFFSET] = 64,
    [LOOM_SCALAR_TYPE_I1] = 1,     [LOOM_SCALAR_TYPE_I8] = 8,
    [LOOM_SCALAR_TYPE_I16] = 16,   [LOOM_SCALAR_TYPE_I32] = 32,
    [LOOM_SCALAR_TYPE_I64] = 64,   [LOOM_SCALAR_TYPE_F8E4M3] = 8,
    [LOOM_SCALAR_TYPE_F8E5M2] = 8, [LOOM_SCALAR_TYPE_F16] = 16,
    [LOOM_SCALAR_TYPE_BF16] = 16,  [LOOM_SCALAR_TYPE_F32] = 32,
    [LOOM_SCALAR_TYPE_F64] = 64,
};

static_assert(IREE_ARRAYSIZE(loom_scalar_type_bitwidths) ==
                  LOOM_SCALAR_TYPE_COUNT_,
              "loom_scalar_type_bitwidths out of sync with enum");

const char* loom_scalar_type_name(loom_scalar_type_t type) {
  if (loom_scalar_type_is_valid(type)) {
    return loom_scalar_type_names[type];
  }
  return NULL;
}

int32_t loom_scalar_type_bitwidth(loom_scalar_type_t type) {
  if (loom_scalar_type_is_valid(type)) {
    return loom_scalar_type_bitwidths[type];
  }
  return 0;
}

bool loom_scalar_type_integer_domain(loom_scalar_type_t type, int64_t* out_lo,
                                     int64_t* out_hi) {
  switch (type) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I64:
      *out_lo = INT64_MIN;
      *out_hi = INT64_MAX;
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_lo = 0;
      *out_hi = INT64_MAX;
      return true;
    case LOOM_SCALAR_TYPE_I1:
      *out_lo = 0;
      *out_hi = 1;
      return true;
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32: {
      const int32_t bitwidth = loom_scalar_type_bitwidth(type);
      const int64_t positive_extent = INT64_C(1) << (bitwidth - 1);
      *out_lo = -positive_extent;
      *out_hi = positive_extent - 1;
      return true;
    }
    default:
      return false;
  }
}

bool loom_scalar_type_fp8_format(loom_scalar_type_t type,
                                 loom_scalar_type_fp8_format_t* out_format) {
  switch (type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_format = (loom_scalar_type_fp8_format_t){
          .exponent_bits = 4,
          .mantissa_bits = 3,
          .exponent_bias = 7,
          .special_policy = LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN,
      };
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_format = (loom_scalar_type_fp8_format_t){
          .exponent_bits = 5,
          .mantissa_bits = 2,
          .exponent_bias = 15,
          .special_policy = LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE,
      };
      return true;
    default:
      *out_format = (loom_scalar_type_fp8_format_t){0};
      return false;
  }
}

bool loom_scalar_type_parse(iree_string_view_t name,
                            loom_scalar_type_t* out_type) {
  const loom_scalar_type_t type = loom_scalar_type_classify_name(name);
  if (type == LOOM_SCALAR_TYPE_NONE) return false;
  *out_type = type;
  return true;
}
