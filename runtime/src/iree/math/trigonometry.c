// Shared radian trigonometry and structural turns functions.
//
// Copyright (c) 2022-2025 Alexei Sibidanov.
// Copyright 2026 The IREE Authors
//
// The frozen binary32 turns mapping derives from the CORE-MATH project
// (https://core-math.gitlabpages.inria.fr/).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// SPDX-License-Identifier: MIT

#include "iree/math/trigonometry.h"

#include <stdbool.h>

#include "iree/base/internal/math.h"
#include "iree/math/float_bits.h"

// This adaptation shares the original sinpi/cospi table, removes errno and
// floating-environment side effects, canonicalizes nonfinite results, and
// accepts turns at its public boundary. CORE-MATH revision:
// e0a3599597503f2af90311c82e2ca92dc06c789a.

// Coefficients evaluate sine and cosine of the fixed-point residual used by
// both kernels. Their powers incorporate the residual's binary scaling.
static const double iree_math_turns_sine_coefficients[] = {
    0x1.921fb54442d0fp-37,
    -0x1.4abbce6102b94p-112,
    0x1.4669fa3c58463p-189,
};

static const double iree_math_turns_cosine_coefficients[] = {
    -0x1.3bd3cc9be45cfp-74,
    0x1.03c1f08088742p-150,
    -0x1.55d1e5eff55a5p-228,
};

// Entry i approximates sin(i*pi/64). A complete signed period avoids quadrant
// branches in the reconstruction and serves cosine through a 32-entry offset.
static const double iree_math_turns_sine_table[] = {
    0x0p+0,
    0x1.91f65f10dd814p-5,
    0x1.917a6bc29b42cp-4,
    0x1.2c8106e8e613ap-3,
    0x1.8f8b83c69a60bp-3,
    0x1.f19f97b215f1bp-3,
    0x1.294062ed59f06p-2,
    0x1.58f9a75ab1fddp-2,
    0x1.87de2a6aea963p-2,
    0x1.b5d1009e15ccp-2,
    0x1.e2b5d3806f63bp-2,
    0x1.073879922ffeep-1,
    0x1.1c73b39ae68c8p-1,
    0x1.30ff7fce17035p-1,
    0x1.44cf325091dd6p-1,
    0x1.57d69348cecap-1,
    0x1.6a09e667f3bcdp-1,
    0x1.7b5df226aafafp-1,
    0x1.8bc806b151741p-1,
    0x1.9b3e047f38741p-1,
    0x1.a9b66290ea1a3p-1,
    0x1.b728345196e3ep-1,
    0x1.c38b2f180bdb1p-1,
    0x1.ced7af43cc773p-1,
    0x1.d906bcf328d46p-1,
    0x1.e212104f686e5p-1,
    0x1.e9f4156c62ddap-1,
    0x1.f0a7efb9230d7p-1,
    0x1.f6297cff75cbp-1,
    0x1.fa7557f08a517p-1,
    0x1.fd88da3d12526p-1,
    0x1.ff621e3796d7ep-1,
    0x1p+0,
    0x1.ff621e3796d7ep-1,
    0x1.fd88da3d12526p-1,
    0x1.fa7557f08a517p-1,
    0x1.f6297cff75cbp-1,
    0x1.f0a7efb9230d7p-1,
    0x1.e9f4156c62ddap-1,
    0x1.e212104f686e5p-1,
    0x1.d906bcf328d46p-1,
    0x1.ced7af43cc773p-1,
    0x1.c38b2f180bdb1p-1,
    0x1.b728345196e3ep-1,
    0x1.a9b66290ea1a3p-1,
    0x1.9b3e047f38741p-1,
    0x1.8bc806b151741p-1,
    0x1.7b5df226aafafp-1,
    0x1.6a09e667f3bcdp-1,
    0x1.57d69348cecap-1,
    0x1.44cf325091dd6p-1,
    0x1.30ff7fce17035p-1,
    0x1.1c73b39ae68c8p-1,
    0x1.073879922ffeep-1,
    0x1.e2b5d3806f63bp-2,
    0x1.b5d1009e15ccp-2,
    0x1.87de2a6aea963p-2,
    0x1.58f9a75ab1fddp-2,
    0x1.294062ed59f06p-2,
    0x1.f19f97b215f1bp-3,
    0x1.8f8b83c69a60bp-3,
    0x1.2c8106e8e613ap-3,
    0x1.917a6bc29b42cp-4,
    0x1.91f65f10dd814p-5,
    0x0p+0,
    -0x1.91f65f10dd814p-5,
    -0x1.917a6bc29b42cp-4,
    -0x1.2c8106e8e613ap-3,
    -0x1.8f8b83c69a60bp-3,
    -0x1.f19f97b215f1bp-3,
    -0x1.294062ed59f06p-2,
    -0x1.58f9a75ab1fddp-2,
    -0x1.87de2a6aea963p-2,
    -0x1.b5d1009e15ccp-2,
    -0x1.e2b5d3806f63bp-2,
    -0x1.073879922ffeep-1,
    -0x1.1c73b39ae68c8p-1,
    -0x1.30ff7fce17035p-1,
    -0x1.44cf325091dd6p-1,
    -0x1.57d69348cecap-1,
    -0x1.6a09e667f3bcdp-1,
    -0x1.7b5df226aafafp-1,
    -0x1.8bc806b151741p-1,
    -0x1.9b3e047f38741p-1,
    -0x1.a9b66290ea1a3p-1,
    -0x1.b728345196e3ep-1,
    -0x1.c38b2f180bdb1p-1,
    -0x1.ced7af43cc773p-1,
    -0x1.d906bcf328d46p-1,
    -0x1.e212104f686e5p-1,
    -0x1.e9f4156c62ddap-1,
    -0x1.f0a7efb9230d7p-1,
    -0x1.f6297cff75cbp-1,
    -0x1.fa7557f08a517p-1,
    -0x1.fd88da3d12526p-1,
    -0x1.ff621e3796d7ep-1,
    -0x1p+0,
    -0x1.ff621e3796d7ep-1,
    -0x1.fd88da3d12526p-1,
    -0x1.fa7557f08a517p-1,
    -0x1.f6297cff75cbp-1,
    -0x1.f0a7efb9230d7p-1,
    -0x1.e9f4156c62ddap-1,
    -0x1.e212104f686e5p-1,
    -0x1.d906bcf328d46p-1,
    -0x1.ced7af43cc773p-1,
    -0x1.c38b2f180bdb1p-1,
    -0x1.b728345196e3ep-1,
    -0x1.a9b66290ea1a3p-1,
    -0x1.9b3e047f38741p-1,
    -0x1.8bc806b151741p-1,
    -0x1.7b5df226aafafp-1,
    -0x1.6a09e667f3bcdp-1,
    -0x1.57d69348cecap-1,
    -0x1.44cf325091dd6p-1,
    -0x1.30ff7fce17035p-1,
    -0x1.1c73b39ae68c8p-1,
    -0x1.073879922ffeep-1,
    -0x1.e2b5d3806f63bp-2,
    -0x1.b5d1009e15ccp-2,
    -0x1.87de2a6aea963p-2,
    -0x1.58f9a75ab1fddp-2,
    -0x1.294062ed59f06p-2,
    -0x1.f19f97b215f1bp-3,
    -0x1.8f8b83c69a60bp-3,
    -0x1.2c8106e8e613ap-3,
    -0x1.917a6bc29b42cp-4,
    -0x1.91f65f10dd814p-5,
};

// Evaluates sin(pi*value) for finite binary32 value.
static float iree_math_sinpi_f32_finite(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const int32_t exponent = (int32_t)((bits >> 23) & UINT32_C(0xFF));
  int32_t mantissa =
      (int32_t)((bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000));
  const int32_t sign = (int32_t)(bits >> 31);
  mantissa = (mantissa ^ -sign) + sign;
  const int32_t shift = 143 - exponent;

  // Values with at most six fractional bits are either cardinal or are an
  // exact table entry. Values at least 2^23 are integral binary32 values.
  if (IREE_UNLIKELY(shift < 0)) {
    if (IREE_UNLIKELY(shift < -6)) {
      return iree_math_f32_from_bits(bits & IREE_MATH_F32_SIGN_BIT);
    }
    uint32_t table_index = (uint32_t)mantissa << (-shift - 1);
    table_index &= UINT32_C(127);
    if (table_index == 0 || table_index == 64) {
      return iree_math_f32_from_bits(bits & IREE_MATH_F32_SIGN_BIT);
    }
    return (float)iree_math_turns_sine_table[table_index];
  }

  // Very small inputs avoid fixed-point reduction and use the leading terms
  // of sin(pi*x) directly.
  if (IREE_UNLIKELY(shift > 30)) {
    const double input = value;
    const double input_squared = input * input;
    return (float)(input * (0x1.921fb54442d18p+1 +
                            input_squared * -0x1.4abbce625be53p+2));
  }

  // Round to the nearest 1/64 table entry. residual_integer is the signed
  // fixed-point remainder consumed by the pre-scaled coefficient arrays.
  const int32_t exact_shift = 25 - shift;
  if (IREE_UNLIKELY(exact_shift >= 0 &&
                    ((uint32_t)mantissa << exact_shift) == 0)) {
    return iree_math_f32_from_bits(bits & IREE_MATH_F32_SIGN_BIT);
  }
  const int32_t residual_integer =
      (int32_t)((uint32_t)mantissa << (31 - shift));
  const double residual = residual_integer;
  const double residual_squared = residual * residual;
  const double residual_sine =
      iree_math_turns_sine_coefficients[0] +
      residual_squared *
          (iree_math_turns_sine_coefficients[1] +
           residual_squared * iree_math_turns_sine_coefficients[2]);
  const double residual_cosine =
      iree_math_turns_cosine_coefficients[0] +
      residual_squared *
          (iree_math_turns_cosine_coefficients[1] +
           residual_squared * iree_math_turns_cosine_coefficients[2]);
  uint32_t quadrant = (uint32_t)(mantissa >> shift);
  quadrant = (quadrant + 1) >> 1;
  const uint32_t sine_index = quadrant & UINT32_C(127);
  const uint32_t cosine_index = (quadrant + 32) & UINT32_C(127);
  const double table_sine = iree_math_turns_sine_table[sine_index];
  const double table_cosine = iree_math_turns_sine_table[cosine_index];
  return (float)(table_sine + table_sine * residual_squared * residual_cosine +
                 table_cosine * residual * residual_sine);
}

// Evaluates cos(pi*value) for finite binary32 value.
static float iree_math_cospi_f32_finite(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const int32_t exponent = (int32_t)((bits >> 23) & UINT32_C(0xFF));
  const int32_t mantissa =
      (int32_t)((bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000));
  const int32_t shift = 143 - exponent;
  const int32_t table_shift = exponent - 112;

  // cos(pi*x) rounds to one for the smallest inputs. The split avoids an
  // intermediate coefficient underflow on the very smallest subnormals.
  if (IREE_UNLIKELY(table_shift < 0)) {
    const uint32_t magnitude_bits = bits & IREE_MATH_F32_MAGNITUDE_MASK;
    if (magnitude_bits >= UINT32_C(0x0019F030)) {
      return (float)(1.0f + (-0x1.3bd3ccp+2f * value) * value);
    }
    return 1.0f + (-value) * value;
  }

  // Large values have no residual beyond an exact table entry. Values at
  // least 2^23 are integral binary32 values and therefore return one.
  if (IREE_UNLIKELY(table_shift > 31)) {
    if (IREE_UNLIKELY(table_shift > 63)) {
      return 1.0f;
    }
    const uint32_t table_index = (uint32_t)mantissa << (table_shift - 32);
    return (
        float)iree_math_turns_sine_table[(table_index + 32) & UINT32_C(127)];
  }

  const int32_t residual_integer = (int32_t)((uint32_t)mantissa << table_shift);
  if (IREE_UNLIKELY(residual_integer == 0)) {
    const uint32_t table_index = (uint32_t)mantissa >> (32 - table_shift);
    return (
        float)iree_math_turns_sine_table[(table_index + 32) & UINT32_C(127)];
  }

  const double residual = residual_integer;
  const double residual_squared = residual * residual;
  const double residual_sine =
      iree_math_turns_sine_coefficients[0] +
      residual_squared *
          (iree_math_turns_sine_coefficients[1] +
           residual_squared * iree_math_turns_sine_coefficients[2]);
  const double residual_cosine =
      iree_math_turns_cosine_coefficients[0] +
      residual_squared *
          (iree_math_turns_cosine_coefficients[1] +
           residual_squared * iree_math_turns_cosine_coefficients[2]);
  uint32_t quadrant = (uint32_t)mantissa >> shift;
  quadrant = (quadrant + 1) >> 1;
  const uint32_t sine_index = quadrant & UINT32_C(127);
  const uint32_t cosine_index = (quadrant + 32) & UINT32_C(127);
  const double table_sine = iree_math_turns_sine_table[cosine_index];
  const double table_cosine = iree_math_turns_sine_table[sine_index];
  return (float)(table_sine + table_sine * residual_squared * residual_cosine -
                 table_cosine * residual * residual_sine);
}

IREE_API_EXPORT float iree_math_sin_turns_f32_approx(float turns) {
  const uint32_t bits = iree_math_f32_to_bits(turns);
  const uint32_t magnitude_bits = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  // Every binary32 value at least 2^23 is an integer number of turns.
  if (IREE_UNLIKELY(magnitude_bits >= UINT32_C(0x4B000000))) {
    return iree_math_f32_from_bits(bits & IREE_MATH_F32_SIGN_BIT);
  }
  return iree_math_sinpi_f32_finite(turns * 2.0f);
}

IREE_API_EXPORT float iree_math_cos_turns_f32_approx(float turns) {
  const uint32_t magnitude_bits =
      iree_math_f32_to_bits(turns) & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  // Every binary32 value at least 2^23 is an integer number of turns.
  if (IREE_UNLIKELY(magnitude_bits >= UINT32_C(0x4B000000))) {
    return 1.0f;
  }
  return iree_math_cospi_f32_finite(turns * 2.0f);
}

// Polynomial kernels and compensated medium reduction:
// Copyright (C) 1993, 2004 Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// F32 conversion by Ian Lance Taylor; optimized by Bruce D. Evans.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// F32 integer reduction:
// Copyright (c) 2018-2024, Arm Limited.
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*-
 * Copyright (c) 2017, 2023 Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//==============================================================================
// Ordinary radian and binary64 turns functions
//==============================================================================

// The polynomial kernels and compensated medium reduction derive from
// Sun/OpenLibm 5fe399749f9276eaa0b8403e507470da05cbbb3f. Large F32 reduction
// derives from Arm Optimized Routines 2ee4b0d45680a9880412c1d186224b08652c35f6;
// large F64 reduction translates AOCL 29fd054f383e6c5e2dec2fce781d5220059f1836
// remainder_piby2 into portable integer arithmetic. F64 turns uses FreeBSD
// 2d3b0a687b910c84606e4bc36176945ad5c60406 sinpi/cospi reconstruction.

static float iree_math_sine_kernel_f32(double value) {
  const double square = value * value;
  const double fourth = square * square;
  const double tail = -0x1a00f9e2cae774.0p-65 + square * 0x16cd878c3b46a7.0p-71;
  const double cube = square * value;
  return (float)((value + cube * (-0x15555554cbac77.0p-55 +
                                  square * 0x111110896efbb2.0p-59)) +
                 cube * fourth * tail);
}

static float iree_math_cosine_kernel_f32(double value) {
  const double square = value * value;
  const double fourth = square * square;
  const double tail = -0x16c087e80f1e27.0p-62 + square * 0x199342e0ee5069.0p-68;
  return (float)(((1.0 + square * -0x1ffffffd0c5e81.0p-54) +
                  fourth * 0x155553e1053a42.0p-57) +
                 (fourth * square) * tail);
}

static float iree_math_tangent_kernel_f32(double value, bool reciprocal) {
  const double square = value * value;
  double tail = 0x185dadfcecf44e.0p-61 + square * 0x1362b9bf971bcd.0p-59;
  const double middle =
      0x1b54c91d865afe.0p-57 + square * 0x191df3908c33ce.0p-58;
  const double fourth = square * square;
  const double cube = square * value;
  const double leading =
      0x15554d3418c99f.0p-54 + square * 0x1112fd38999f72.0p-55;
  tail = (value + cube * leading) + (cube * fourth) * (middle + fourth * tail);
  return (float)(reciprocal ? -1.0 / tail : tail);
}

static double iree_math_sine_kernel_f64(double value, double tail,
                                        bool has_tail) {
  const double square = value * value;
  const double fourth = square * square;
  const double correction =
      8.33333333332248946124e-03 +
      square *
          (-1.98412698298579493134e-04 + square * 2.75573137070700676789e-06) +
      square * fourth *
          (-2.50507602534068634195e-08 + square * 1.58969099521155010221e-10);
  const double cube = square * value;
  const double leading = -1.66666666666666324348e-01;
  if (!has_tail) {
    return value + cube * (leading + square * correction);
  }
  return value -
         ((square * (0.5 * tail - cube * correction) - tail) - cube * leading);
}

static double iree_math_cosine_kernel_f64(double value, double tail) {
  const double square = value * value;
  const double fourth = square * square;
  const double correction =
      square * (4.16666666666666019037e-02 +
                square * (-1.38888888888741095749e-03 +
                          square * 2.48015872894767294178e-05)) +
      fourth * fourth *
          (-2.75573143513906633035e-07 +
           square * (2.08757232129817482790e-09 +
                     square * -1.13596475577881948265e-11));
  const double half_square = 0.5 * square;
  const double leading = 1.0 - half_square;
  return leading + (((1.0 - leading) - half_square) +
                    (square * correction - value * tail));
}

static double iree_math_tangent_kernel_f64(double value, double tail,
                                           bool reciprocal) {
  static const double coefficients[] = {
      3.33333333333334091986e-01, 1.33333333333201242699e-01,
      5.39682539762260521377e-02, 2.18694882948595424599e-02,
      8.86323982359930005737e-03, 3.59207910759131235356e-03,
      1.45620945432529025516e-03, 5.88041240820264096874e-04,
      2.46463134818469906812e-04, 7.81794442939557092300e-05,
      7.14072491382608190305e-05, -1.85586374855275456654e-05,
      2.59073051863633712884e-05,
  };
  const uint64_t bits = iree_math_f64_to_bits(value);
  const bool negative = (bits & IREE_MATH_F64_SIGN_BIT) != 0;
  const bool near_quarter_pi =
      (bits & IREE_MATH_F64_MAGNITUDE_MASK) >= UINT64_C(0x3FE5942800000000);
  if (near_quarter_pi) {
    if (negative) {
      value = -value;
      tail = -tail;
    }
    const double head = 7.85398163397448278999e-01 - value;
    const double low = 3.06161699786838301793e-17 - tail;
    value = head + low;
    tail = 0.0;
  }
  const double square = value * value;
  const double fourth = square * square;
  double correction =
      coefficients[1] +
      fourth * (coefficients[3] +
                fourth * (coefficients[5] +
                          fourth * (coefficients[7] +
                                    fourth * (coefficients[9] +
                                              fourth * coefficients[11]))));
  const double alternate =
      square *
      (coefficients[2] +
       fourth * (coefficients[4] +
                 fourth * (coefficients[6] +
                           fourth * (coefficients[8] +
                                     fourth * (coefficients[10] +
                                               fourth * coefficients[12])))));
  const double cube = square * value;
  correction = tail + square * (cube * (correction + alternate) + tail);
  correction += coefficients[0] * cube;
  const double result = value + correction;
  if (near_quarter_pi) {
    const double direction = reciprocal ? -1.0 : 1.0;
    const double sign = negative ? -1.0 : 1.0;
    return sign *
           (direction - 2.0 * (value - (result * result / (result + direction) -
                                        correction)));
  }
  if (!reciprocal) {
    return result;
  }
  // Keep the denominator's low part when reconstructing -1/tan near a pole.
  const double head = iree_math_f64_from_bits(iree_math_f64_to_bits(result) &
                                              UINT64_C(0xFFFFFFFF00000000));
  const double low = correction - (head - value);
  const double inverse = -1.0 / result;
  const double inverse_head = iree_math_f64_from_bits(
      iree_math_f64_to_bits(inverse) & UINT64_C(0xFFFFFFFF00000000));
  const double product_error = 1.0 + inverse_head * head;
  return inverse_head + inverse * (product_error + inverse_head * low);
}

// 4/pi with overlapping byte windows. The F32 significand needs only a
// 32-by-96-bit product to retain at least 33 accurate residual bits.
static const uint32_t iree_math_four_over_pi_f32[24] = {
    0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
};

// Reduces a positive finite F32 magnitude greater than pi/4.
static uint32_t iree_math_reduce_radians_f32(uint32_t magnitude,
                                             double* out_remainder) {
  const double value = iree_math_f32_from_bits(magnitude);
  uint32_t quadrant = 0;
  if (magnitude <= UINT32_C(0x40E231D5)) {
    quadrant = magnitude <= UINT32_C(0x4016CBE3)   ? 1
               : magnitude <= UINT32_C(0x407B53D1) ? 2
               : magnitude <= UINT32_C(0x40AFEDDF) ? 3
                                                   : 4;
    *out_remainder = value - quadrant * 0x1.921fb54442d18p0;
    return quadrant;
  }
  if (magnitude < UINT32_C(0x42F00000)) {
    const double multiple =
        (value * 6.36619772367581382433e-01 + 0x1.8p52) - 0x1.8p52;
    *out_remainder = (value - multiple * 1.57079631090164184570e+00) -
                     multiple * 1.58932547735281966916e-08;
    return (uint32_t)multiple;
  }
  const uint32_t* table = &iree_math_four_over_pi_f32[(magnitude >> 26) & 15];
  const uint32_t significand =
      ((magnitude & UINT32_C(0xFFFFFF)) | UINT32_C(0x800000))
      << ((magnitude >> 23) & 7);
  // This limb intentionally retains the low 32 bits of the product. The
  // following shift discards the high half before composing the 64-bit window.
  const uint32_t low_limb = significand * table[0];
  const uint64_t middle = (uint64_t)significand * table[4];
  const uint64_t high = (uint64_t)significand * table[8];
  uint64_t low = ((high >> 32) | ((uint64_t)low_limb << 32)) + middle;
  const uint64_t rounded = (low + (UINT64_C(1) << 61)) >> 62;
  low -= rounded << 62;
  *out_remainder = (double)(int64_t)low * 0x1.921fb54442d18p-62;
  return (uint32_t)rounded;
}

// Little-endian windows of 2/pi. The exponent chooses the relevant fractional
// window; only three limbs are needed even at the largest finite exponent.
static const uint8_t iree_math_two_over_pi_f64[158] = {
    224, 241, 27,  193, 12,  88,  33,  116, 53,  126, 196, 126, 237, 175, 169,
    75,  74,  41,  222, 231, 28,  244, 236, 197, 151, 175, 31,  235, 158, 212,
    181, 168, 127, 121, 154, 253, 24,  61,  221, 38,  44,  159, 60,  251, 217,
    180, 125, 180, 41,  104, 45,  70,  188, 188, 63,  96,  22,  120, 255, 95,
    226, 127, 236, 160, 228, 247, 46,  126, 17,  114, 210, 231, 76,  13,  230,
    88,  71,  230, 4,   249, 125, 209, 154, 192, 113, 166, 19,  18,  237, 186,
    212, 215, 8,   162, 251, 156, 166, 196, 114, 172, 119, 248, 115, 72,  70,
    39,  168, 187, 36,  25,  128, 75,  55,  9,   233, 184, 145, 220, 134, 21,
    239, 122, 175, 142, 69,  249, 7,   65,  14,  241, 100, 86,  138, 109, 3,
    119, 211, 212, 71,  95,  157, 240, 167, 84,  16,  57,  185, 13,  230, 139,
    2,   0,   0,   0,   0,   0,   0,   0,
};

static uint32_t iree_math_reduce_radians_f64_large(uint64_t magnitude,
                                                   double remainder[2]) {
  const int exponent = (int)(magnitude >> 52) - 1023;
  const uint8_t* window = &iree_math_two_over_pi_f64[134 - (exponent >> 3)];
  const uint64_t significand =
      (magnitude & UINT64_C(0x000FFFFFFFFFFFFF)) | UINT64_C(0x0010000000000000);
  uint64_t low, middle, low_carry, middle_carry;
  iree_math_mul_u64_to_u128(significand, iree_unaligned_load_le_u64(window),
                            &low_carry, &low);
  iree_math_mul_u64_to_u128(significand, iree_unaligned_load_le_u64(window + 8),
                            &middle_carry, &middle);
  middle += low_carry;
  middle_carry += middle < low_carry;
  uint64_t high =
      significand * iree_unaligned_load_le_u64(window + 16) + middle_carry;
  const int fraction_bits = 54 - (exponent & 7);
  const uint32_t integer_part = (uint32_t)(high >> fraction_bits);
  const uint32_t round_up = (uint32_t)((high >> (fraction_bits - 1)) & 1);
  if (round_up) {
    high = ~high;
    middle = ~middle;
    low = ~low;
  }
  high &= (UINT64_C(1) << fraction_bits) - 1;
  int result_exponent = -fraction_bits;
  if (!high) {
    high = middle;
    middle = low;
    low = 0;
    result_exponent -= 64;
  }
  // Binary64 arguments cannot cancel both retained leading fraction words.
  assert(high != 0);
  const int leading_bit = 63 - iree_math_count_leading_zeros_u64(high);
  result_exponent += leading_bit;
  const int shift = leading_bit - 52;
  if (shift > 0) {
    middle = (middle >> shift) | (high << (64 - shift));
    high >>= shift;
  } else if (shift < 0) {
    high = (high << -shift) | (middle >> (64 + shift));
    middle = (middle << -shift) | (low >> (64 + shift));
  }
  const uint64_t sign = (uint64_t)round_up << 63;
  const double fraction = iree_math_f64_from_bits(
      sign | ((uint64_t)(result_exponent + 1023) << 52) |
      (high & UINT64_C(0x000FFFFFFFFFFFFF)));
  double fraction_tail = 0.0;
  if (middle) {
    const int leading = iree_math_count_leading_zeros_u64(middle);
    const uint64_t mantissa =
        ((middle << leading) & IREE_MATH_F64_MAGNITUDE_MASK) >> 11;
    fraction_tail = iree_math_f64_from_bits(
        sign | ((uint64_t)(result_exponent - 53 - leading + 1023) << 52) |
        mantissa);
  }
  const double fraction_high = iree_math_f64_from_bits(
      iree_math_f64_to_bits(fraction) & UINT64_C(0xFFFFFFFFF8000000));
  const double fraction_low = fraction - fraction_high;
  const double product = fraction * 0x1.921fb54442d18p0;
  double correction = (((fraction_high * 0x1.921fb5p0 - product) +
                        fraction_low * 0x1.921fb5p0) +
                       fraction_high * 0x1.110b46p-26) +
                      fraction_low * 0x1.110b46p-26;
  correction +=
      fraction * 0x1.1a62633145c06p-54 + fraction_tail * 0x1.921fb54442d18p0;
  remainder[0] = product + correction;
  remainder[1] = (product - remainder[0]) + correction;
  return (integer_part + round_up) & 3;
}

// Reduces a positive finite F64 magnitude greater than pi/4. The residual
// retains a low part for kernel evaluation close to zero and tangent poles.
static uint32_t iree_math_reduce_radians_f64(uint64_t magnitude,
                                             double remainder[2]) {
  const uint32_t high = (uint32_t)(magnitude >> 32);
  const double value = iree_math_f64_from_bits(magnitude);
  const double half_pi_head = 1.57079632673412561417e+00;
  const double half_pi_tail = 6.07710050650619224932e-11;
  uint32_t small_multiple = 0;
  if (high <= UINT32_C(0x400F6A7A)) {
    if ((high & UINT32_C(0xFFFFF)) != UINT32_C(0x921FB)) {
      small_multiple = high <= UINT32_C(0x4002D97C) ? 1 : 2;
    }
  } else if (high <= UINT32_C(0x401C463B)) {
    if (high != UINT32_C(0x4012D97C) && high != UINT32_C(0x401921FB)) {
      small_multiple = high <= UINT32_C(0x4015FDBC) ? 3 : 4;
    }
  }
  if (small_multiple) {
    const double head = value - small_multiple * half_pi_head;
    const double tail = small_multiple * half_pi_tail;
    remainder[0] = head - tail;
    remainder[1] = (head - remainder[0]) - tail;
    return small_multiple;
  }
  if (high >= UINT32_C(0x413921FB)) {
    return iree_math_reduce_radians_f64_large(magnitude, remainder);
  }
  const double multiple =
      (value * 6.36619772367581382433e-01 + 0x1.8p52) - 0x1.8p52;
  double head = value - multiple * half_pi_head;
  double tail = multiple * half_pi_tail;
  remainder[0] = head - tail;
  const int input_exponent = (int)(magnitude >> 52);
  int cancellation =
      input_exponent -
      (int)((iree_math_f64_to_bits(remainder[0]) >> 52) & UINT64_C(0x7FF));
  if (cancellation > 16) {
    const double previous = head;
    tail = multiple * 6.07710050630396597660e-11;
    head = previous - tail;
    tail = multiple * 2.02226624879595063154e-21 - ((previous - head) - tail);
    remainder[0] = head - tail;
    cancellation =
        input_exponent -
        (int)((iree_math_f64_to_bits(remainder[0]) >> 52) & UINT64_C(0x7FF));
    if (cancellation > 49) {
      const double previous = head;
      tail = multiple * 2.02226624871116645580e-21;
      head = previous - tail;
      tail = multiple * 8.47842766036889956997e-32 - ((previous - head) - tail);
      remainder[0] = head - tail;
    }
  }
  remainder[1] = (head - remainder[0]) - tail;
  return (uint32_t)multiple;
}

IREE_API_EXPORT float iree_math_sin_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (magnitude <= UINT32_C(0x3F490FDA)) {
    if (magnitude < UINT32_C(0x39800000)) {
      return value;
    }
    return iree_math_sine_kernel_f32(value);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  double remainder;
  const uint32_t quadrant = iree_math_reduce_radians_f32(magnitude, &remainder);
  float result;
  switch (quadrant & 3) {
    case 0:
      result = iree_math_sine_kernel_f32(remainder);
      break;
    case 1:
      result = iree_math_cosine_kernel_f32(remainder);
      break;
    case 2:
      result = iree_math_sine_kernel_f32(-remainder);
      break;
    default:
      result = -iree_math_cosine_kernel_f32(remainder);
      break;
  }
  return bits & IREE_MATH_F32_SIGN_BIT ? -result : result;
}

IREE_API_EXPORT float iree_math_cos_f32(float value) {
  const uint32_t magnitude =
      iree_math_f32_to_bits(value) & IREE_MATH_F32_MAGNITUDE_MASK;
  if (magnitude <= UINT32_C(0x3F490FDA)) {
    if (magnitude < UINT32_C(0x39800000)) {
      return 1.0f;
    }
    return iree_math_cosine_kernel_f32(value);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  double remainder;
  const uint32_t quadrant = iree_math_reduce_radians_f32(magnitude, &remainder);
  switch (quadrant & 3) {
    case 0:
      return iree_math_cosine_kernel_f32(remainder);
    case 1:
      return iree_math_sine_kernel_f32(-remainder);
    case 2:
      return -iree_math_cosine_kernel_f32(remainder);
    default:
      return iree_math_sine_kernel_f32(remainder);
  }
}

IREE_API_EXPORT float iree_math_tan_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (magnitude <= UINT32_C(0x3F490FDA)) {
    if (magnitude < UINT32_C(0x39800000)) {
      return value;
    }
    return iree_math_tangent_kernel_f32(value, false);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  double remainder;
  const uint32_t quadrant = iree_math_reduce_radians_f32(magnitude, &remainder);
  const float result =
      iree_math_tangent_kernel_f32(remainder, (quadrant & 1) != 0);
  return bits & IREE_MATH_F32_SIGN_BIT ? -result : result;
}

IREE_API_EXPORT double iree_math_sin_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (magnitude < UINT64_C(0x3FE921FC00000000)) {
    if (magnitude < UINT64_C(0x3E50000000000000)) {
      return value;
    }
    return iree_math_sine_kernel_f64(value, 0.0, false);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  double remainder[2];
  const uint32_t quadrant = iree_math_reduce_radians_f64(magnitude, remainder);
  double result;
  switch (quadrant & 3) {
    case 0:
      result = iree_math_sine_kernel_f64(remainder[0], remainder[1], true);
      break;
    case 1:
      result = iree_math_cosine_kernel_f64(remainder[0], remainder[1]);
      break;
    case 2:
      result = -iree_math_sine_kernel_f64(remainder[0], remainder[1], true);
      break;
    default:
      result = -iree_math_cosine_kernel_f64(remainder[0], remainder[1]);
      break;
  }
  return bits & IREE_MATH_F64_SIGN_BIT ? -result : result;
}

IREE_API_EXPORT double iree_math_cos_f64(double value) {
  const uint64_t magnitude =
      iree_math_f64_to_bits(value) & IREE_MATH_F64_MAGNITUDE_MASK;
  if (magnitude < UINT64_C(0x3FE921FC00000000)) {
    if (magnitude < UINT64_C(0x3E46A09E00000000)) {
      return 1.0;
    }
    return iree_math_cosine_kernel_f64(value, 0.0);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  double remainder[2];
  const uint32_t quadrant = iree_math_reduce_radians_f64(magnitude, remainder);
  switch (quadrant & 3) {
    case 0:
      return iree_math_cosine_kernel_f64(remainder[0], remainder[1]);
    case 1:
      return -iree_math_sine_kernel_f64(remainder[0], remainder[1], true);
    case 2:
      return -iree_math_cosine_kernel_f64(remainder[0], remainder[1]);
    default:
      return iree_math_sine_kernel_f64(remainder[0], remainder[1], true);
  }
}

IREE_API_EXPORT double iree_math_tan_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (magnitude < UINT64_C(0x3FE921FC00000000)) {
    if (magnitude < UINT64_C(0x3E40000000000000)) {
      return value;
    }
    return iree_math_tangent_kernel_f64(value, 0.0, false);
  }
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  double remainder[2];
  const uint32_t quadrant = iree_math_reduce_radians_f64(magnitude, remainder);
  const double result = iree_math_tangent_kernel_f64(remainder[0], remainder[1],
                                                     (quadrant & 1) != 0);
  return bits & IREE_MATH_F64_SIGN_BIT ? -result : result;
}

// Multiplies a quarter-period residual by split pi, retaining the low part
// discarded by an ordinary multiplication before entering the radian kernels.
static void iree_math_turns_angle_f64(double residual, double angle[2]) {
  const double pi_high = 3.1415926814079285e+00;
  const double pi_low = -2.7818135228334233e-08;
  const double head = (float)residual;
  const double tail = residual - head;
  const double low = tail * (pi_low + pi_high) + head * pi_low;
  const double high = head * pi_high;
  angle[0] = high + low;
  angle[1] = (high - angle[0]) + low;
}

IREE_API_EXPORT double iree_math_sin_turns_f64(double turns) {
  const uint64_t bits = iree_math_f64_to_bits(turns);
  const uint64_t magnitude_bits = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (magnitude_bits >= UINT64_C(0x4320000000000000) || magnitude_bits == 0) {
    return iree_math_f64_from_bits(bits & IREE_MATH_F64_SIGN_BIT);
  }
  double magnitude = iree_math_f64_from_bits(magnitude_bits) * 2.0;
  if (magnitude < 0x1p-29) {
    // Scaling keeps the split products normal even for subnormal turns.
    const double high =
        iree_math_f64_from_bits(iree_math_f64_to_bits(magnitude) &
                                UINT64_C(0xFFFFFFFF00000000)) *
        0x1p53;
    const double low = magnitude * 0x1p53 - high;
    const double result =
        ((-2.7818135228334233e-08 + 3.1415926814079285e+00) * low +
         -2.7818135228334233e-08 * high + 3.1415926814079285e+00 * high) *
        0x1p-53;
    return bits & IREE_MATH_F64_SIGN_BIT ? -result : result;
  }
  uint64_t integer = 0;
  if (magnitude >= 1.0) {
    integer = (uint64_t)magnitude;
    magnitude -= (double)integer;
    if (magnitude == 0.0) {
      return iree_math_f64_from_bits(bits & IREE_MATH_F64_SIGN_BIT);
    }
  }
  double angle[2];
  double result;
  if (magnitude < 0.25 || magnitude >= 0.75) {
    iree_math_turns_angle_f64(magnitude < 0.25 ? magnitude : 1.0 - magnitude,
                              angle);
    result = iree_math_sine_kernel_f64(angle[0], angle[1], true);
  } else {
    iree_math_turns_angle_f64(
        magnitude < 0.5 ? 0.5 - magnitude : magnitude - 0.5, angle);
    result = iree_math_cosine_kernel_f64(angle[0], angle[1]);
  }
  return ((bits >> 63) ^ (integer & 1)) ? -result : result;
}

IREE_API_EXPORT double iree_math_cos_turns_f64(double turns) {
  const uint64_t magnitude_bits =
      iree_math_f64_to_bits(turns) & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (magnitude_bits >= UINT64_C(0x4330000000000000)) {
    return 1.0;
  }
  if (magnitude_bits >= UINT64_C(0x4320000000000000)) {
    return magnitude_bits & 1 ? -1.0 : 1.0;
  }
  double magnitude = iree_math_f64_from_bits(magnitude_bits) * 2.0;
  if (magnitude < 0x1p-29) {
    return 1.0;
  }
  uint64_t integer = 0;
  if (magnitude >= 1.0) {
    integer = (uint64_t)magnitude;
    magnitude -= (double)integer;
  }
  if (magnitude == 0.5) {
    return 0.0;
  }
  double angle[2];
  double result;
  if (magnitude < 0.25 || magnitude >= 0.75) {
    iree_math_turns_angle_f64(magnitude < 0.25 ? magnitude : 1.0 - magnitude,
                              angle);
    result = iree_math_cosine_kernel_f64(angle[0], angle[1]);
  } else {
    iree_math_turns_angle_f64(
        magnitude < 0.5 ? 0.5 - magnitude : magnitude - 0.5, angle);
    result = iree_math_sine_kernel_f64(angle[0], angle[1], true);
  }
  return ((integer & 1) ^ (magnitude >= 0.5)) ? -result : result;
}
