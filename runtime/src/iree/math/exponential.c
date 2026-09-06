// Copyright (c) 2017-2025, Arm Limited.
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/exponential.h"

#include <stdbool.h>

#include "iree/math/float_bits.h"

// The finite kernels below derive from Arm Optimized Routines revision
// 67126040cf80f956676fbf473c2d9bebdb475283. IREE supplies explicit exceptional
// shells and freezes the resulting f32 mappings as its own named operations.

//===----------------------------------------------------------------------===//
// Base-two exponential
//===----------------------------------------------------------------------===//

// The low five bits select 2^(i/32). Each entry has i*2^(52-5) subtracted
// from its binary64 representation so adding the complete reduction integer
// simultaneously restores the table value and installs its power-of-two
// exponent.
static const uint64_t iree_math_exp2_f32_table[32] = {
    UINT64_C(0x3FF0000000000000), UINT64_C(0x3FEFD9B0D3158574),
    UINT64_C(0x3FEFB5586CF9890F), UINT64_C(0x3FEF9301D0125B51),
    UINT64_C(0x3FEF72B83C7D517B), UINT64_C(0x3FEF54873168B9AA),
    UINT64_C(0x3FEF387A6E756238), UINT64_C(0x3FEF1E9DF51FDEE1),
    UINT64_C(0x3FEF06FE0A31B715), UINT64_C(0x3FEEF1A7373AA9CB),
    UINT64_C(0x3FEEDEA64C123422), UINT64_C(0x3FEECE086061892D),
    UINT64_C(0x3FEEBFDAD5362A27), UINT64_C(0x3FEEB42B569D4F82),
    UINT64_C(0x3FEEAB07DD485429), UINT64_C(0x3FEEA47EB03A5585),
    UINT64_C(0x3FEEA09E667F3BCD), UINT64_C(0x3FEE9F75E8EC5F74),
    UINT64_C(0x3FEEA11473EB0187), UINT64_C(0x3FEEA589994CCE13),
    UINT64_C(0x3FEEACE5422AA0DB), UINT64_C(0x3FEEB737B0CDC5E5),
    UINT64_C(0x3FEEC49182A3F090), UINT64_C(0x3FEED503B23E255D),
    UINT64_C(0x3FEEE89F995AD3AD), UINT64_C(0x3FEEFF76F2FB5E47),
    UINT64_C(0x3FEF199BDD85529C), UINT64_C(0x3FEF3720DCEF9069),
    UINT64_C(0x3FEF5818DCFBA487), UINT64_C(0x3FEF7C97337B9B5F),
    UINT64_C(0x3FEFA4AFA2A490DA), UINT64_C(0x3FEFD0765B6E4540),
};

static bool iree_math_exp2_f32_handle_special(float value, float* out_result) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= UINT32_C(0x42FC0000))) {
    if (magnitude_bits >= IREE_MATH_F32_INFINITY) {
      if (magnitude_bits > IREE_MATH_F32_INFINITY) {
        *out_result = iree_math_f32_canonical_nan();
      } else {
        *out_result = (value_bits & IREE_MATH_F32_SIGN_BIT)
                          ? 0.0f
                          : iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
      }
      return true;
    }
    // -126 is the smallest input whose result is a normal f32. Inputs below
    // it flush to zero; inputs at least 128 overflow to infinity.
    if (value_bits & IREE_MATH_F32_SIGN_BIT) {
      if (magnitude_bits > UINT32_C(0x42FC0000)) {
        *out_result = 0.0f;
        return true;
      }
    } else if (magnitude_bits >= UINT32_C(0x43000000)) {
      *out_result = iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
      return true;
    }
  }
  // No f32 value this close to zero can make 2^x round away from one.
  if (IREE_UNLIKELY(magnitude_bits <= UINT32_C(0x33000000))) {
    *out_result = 1.0f;
    return true;
  }
  return false;
}

IREE_API_EXPORT float iree_math_exp2_f32_approx(float value) {
  float result = 0.0f;
  if (iree_math_exp2_f32_handle_special(value, &result)) return result;

  // Round value*32 to an integer using the binary64 shift trick. The encoded
  // rounded value carries both the table index and output exponent.
  const double input = (double)value;
  const double shift = 0x1.8p47;
  double rounded = input + shift;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= shift;
  const double residual = input - rounded;

  uint64_t scale_bits =
      iree_math_exp2_f32_table[encoded_rounded & UINT64_C(31)];
  scale_bits += encoded_rounded << 47;
  const double scale = iree_math_f64_from_bits(scale_bits);

  // Approximate 2^residual over [-1/64, 1/64] and apply the table scale.
  const double residual_squared = residual * residual;
  const double upper_polynomial =
      0x1.c6af84b912394p-5 * residual + 0x1.ebfce50fac4f3p-3;
  const double lower_polynomial = 0x1.62e42ff0c52d6p-1 * residual + 1.0;
  const double polynomial =
      upper_polynomial * residual_squared + lower_polynomial;
  return (float)(polynomial * scale);
}

//===----------------------------------------------------------------------===//
// Base-two logarithm
//===----------------------------------------------------------------------===//

typedef struct iree_math_log2_f32_table_entry_t {
  // Reciprocal of the subinterval center used to form a small residual.
  double inverse_center;
  // Base-two logarithm of the subinterval center.
  double log2_center;
} iree_math_log2_f32_table_entry_t;

static const iree_math_log2_f32_table_entry_t iree_math_log2_f32_table[16] = {
    {0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2},
    {0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2},
    {0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2},
    {0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2},
    {0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2},
    {0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3},
    {0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3},
    {0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4},
    {0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5},
    {0x1p+0, 0x0p+0},
    {0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4},
    {0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3},
    {0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3},
    {0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2},
    {0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2},
    {0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2},
};

static const double iree_math_log2_f32_polynomial[4] = {
    -0x1.712b6f70a7e4dp-2,
    0x1.ecabf496832ep-2,
    -0x1.715479ffae3dep-1,
    0x1.715475f35c8b8p0,
};

static bool iree_math_log2_f32_handle_special(float value, float* out_result) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits < IREE_MATH_F32_MIN_NORMAL)) {
    *out_result = iree_math_f32_from_bits(UINT32_C(0xFF800000));
    return true;
  }
  if (IREE_UNLIKELY((value_bits & IREE_MATH_F32_SIGN_BIT) != 0)) {
    *out_result = iree_math_f32_canonical_nan();
    return true;
  }
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F32_INFINITY)) {
    *out_result = magnitude_bits == IREE_MATH_F32_INFINITY
                      ? iree_math_f32_from_bits(IREE_MATH_F32_INFINITY)
                      : iree_math_f32_canonical_nan();
    return true;
  }
  return false;
}

IREE_API_EXPORT float iree_math_log2_f32_approx(float value) {
  float result = 0.0f;
  if (iree_math_log2_f32_handle_special(value, &result)) return result;

  // Normalize the input and select one of sixteen centers without a division.
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t adjusted_bits = value_bits - UINT32_C(0x3F330000);
  const uint32_t table_index = (adjusted_bits >> 19) & UINT32_C(15);
  const uint32_t exponent_code = adjusted_bits >> 23;
  const int32_t exponent = exponent_code >= UINT32_C(256)
                               ? (int32_t)exponent_code - 512
                               : (int32_t)exponent_code;
  const uint32_t normalized_bits =
      value_bits - (adjusted_bits & UINT32_C(0xFF800000));
  const double normalized = (double)iree_math_f32_from_bits(normalized_bits);
  const double residual =
      normalized * iree_math_log2_f32_table[table_index].inverse_center - 1.0;
  const double base =
      iree_math_log2_f32_table[table_index].log2_center + (double)exponent;

  const double residual_squared = residual * residual;
  double upper = iree_math_log2_f32_polynomial[1] * residual +
                 iree_math_log2_f32_polynomial[2];
  upper = iree_math_log2_f32_polynomial[0] * residual_squared + upper;
  const double lower = iree_math_log2_f32_polynomial[3] * residual + base;
  return (float)(upper * residual_squared + lower);
}
