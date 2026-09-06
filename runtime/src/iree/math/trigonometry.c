// Correctly-rounded binary32 sine and cosine of angles in turns.
//
// Copyright (c) 2022-2025 Alexei Sibidanov.
// Copyright 2026 The IREE Authors
//
// This file derives from the CORE-MATH project
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
    if (IREE_UNLIKELY(table_shift > 63)) return 1.0f;
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
  if (IREE_UNLIKELY(magnitude_bits >= UINT32_C(0x4B000000))) return 1.0f;
  return iree_math_cospi_f32_finite(turns * 2.0f);
}
