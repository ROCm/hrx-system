// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));
using Bytes8 = unsigned char __attribute__((ext_vector_type(8)));
using Words2 = unsigned __attribute__((ext_vector_type(2)));
using Floats2 = float __attribute__((ext_vector_type(2)));
using Halves2 = _Float16 __attribute__((ext_vector_type(2)));
using BFloats2 = __bf16 __attribute__((ext_vector_type(2)));
using Float8x4 = __float8_e4m3fn __attribute__((ext_vector_type(4)));
using BFloat8x4 = __float8_e5m2 __attribute__((ext_vector_type(4)));
using Floats4 = float __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void scale_packed_fp8(const unsigned* input, unsigned* output, float factor) {
  unsigned lane = loom::kernel::workitem::id.x;
  auto packed = __builtin_bit_cast(Float8x4, input[lane]);
  auto wide = __builtin_convertvector(packed, Floats4);
  auto scaled = __builtin_convertvector(wide * factor, Float8x4);
  output[lane] = __builtin_bit_cast(unsigned, scaled);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void packed_bytes(const unsigned* input, unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  auto bytes = __builtin_bit_cast(Bytes4, input[lane]);
  // Reorder and add distinct lane values so a round-trip identity cannot hide
  // a disagreement about the source byte order or carry between byte lanes.
  Bytes4 reordered{bytes[3], bytes[2], bytes[1], bytes[0]};
  output[lane] = __builtin_bit_cast(unsigned, reordered + Bytes4{1, 2, 3, 4});
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void floating_payloads(const unsigned* input, Float8x4* fp8, BFloat8x4* bf8,
                       Halves2* half, BFloats2* bfloat, float* scalar) {
  unsigned lane = loom::kernel::workitem::id.x;
  const unsigned bits = input[lane];
  fp8[lane] = __builtin_bit_cast(Float8x4, bits);
  bf8[lane] = __builtin_bit_cast(BFloat8x4, bits);
  half[lane] = __builtin_bit_cast(Halves2, bits);
  bfloat[lane] = __builtin_bit_cast(BFloats2, bits);
  const auto bytes = __builtin_bit_cast(Bytes4, bits);
  scalar[lane] = __builtin_bit_cast(float, bytes);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void floating_to_words(const Float8x4* fp8, const BFloat8x4* bf8,
                       const Halves2* half, const BFloats2* bfloat,
                       const float* scalar, unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  output[lane] = __builtin_bit_cast(unsigned, fp8[lane]);
  output[32 + lane] = __builtin_bit_cast(unsigned, bf8[lane]);
  output[64 + lane] = __builtin_bit_cast(unsigned, half[lane]);
  output[96 + lane] = __builtin_bit_cast(unsigned, bfloat[lane]);
  const auto bytes = __builtin_bit_cast(Bytes4, scalar[lane]);
  output[128 + lane] = __builtin_bit_cast(unsigned, bytes);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void wide_payloads(const unsigned long long* input, Floats2* floats,
                   double* scalar, unsigned long long* reversed) {
  const auto bytes = __builtin_bit_cast(Bytes8, input[0]);
  const Bytes8 swapped{bytes[7], bytes[6], bytes[5], bytes[4],
                       bytes[3], bytes[2], bytes[1], bytes[0]};
  reversed[0] = __builtin_bit_cast(unsigned long long, swapped);
  floats[0] = __builtin_bit_cast(Floats2, input[1]);
  const auto words = __builtin_bit_cast(Words2, input[2]);
  scalar[0] = __builtin_bit_cast(double, words);
}

LOOM_CHECK_CASE(packed_byte_order) {
  const auto input = loom::check::fill<32, unsigned>(0xFCFEFF80u);
  const auto output = loom::check::fill<32, unsigned>(0u);
  loom::check::launch<packed_bytes>(input, output);
  loom::check::expect_bitwise(output,
                              loom::check::fill<32, unsigned>(0x840200FDu));
  loom::check::expect_bitwise(input,
                              loom::check::fill<32, unsigned>(0xFCFEFF80u));
}

LOOM_CHECK_CASE(packed_fp8_rescale) {
  // E4M3 [1, 1.25, 448, -0] rounds to [1, 1.375, 448, -0], covering a
  // midpoint, ordinary rounding, maximum finite saturation and signed zero.
  const auto input = loom::check::fill<32, unsigned>(0x807E3A38u);
  const auto output = loom::check::fill<32, unsigned>(0u);
  loom::check::launch<scale_packed_fp8>(input, output, 1.0625f);
  loom::check::expect_bitwise(output,
                              loom::check::fill<32, unsigned>(0x807E3B38u));
  loom::check::expect_bitwise(input,
                              loom::check::fill<32, unsigned>(0x807E3A38u));
}

LOOM_CHECK_CASE(raw_floating_payloads) {
  // Each source word is checked after a real store through every float type,
  // then loaded with that type by a separate kernel. Signed zero, subnormal
  // and NaN encodings are transported as bits, without floating arithmetic.
  const auto input = loom::check::fill<32, unsigned>(0x7F817E80u);
  const auto fp8 = loom::check::fill<32, unsigned>(0u);
  const auto bf8 = loom::check::fill<32, unsigned>(0u);
  const auto half = loom::check::fill<32, unsigned>(0u);
  const auto bfloat = loom::check::fill<32, unsigned>(0u);
  const auto scalar = loom::check::fill<32, unsigned>(0u);
  const auto output = loom::check::fill<160, unsigned>(0u);
  loom::check::launch<floating_payloads>(input, fp8, bf8, half, bfloat, scalar);
  loom::check::launch<floating_to_words>(fp8, bf8, half, bfloat, scalar,
                                         output);
  loom::check::expect_bitwise(fp8, input);
  loom::check::expect_bitwise(bf8, input);
  loom::check::expect_bitwise(half, input);
  loom::check::expect_bitwise(bfloat, input);
  loom::check::expect_bitwise(scalar, input);
  loom::check::expect_bitwise(output,
                              loom::check::fill<160, unsigned>(0x7F817E80u));
}

LOOM_CHECK_CASE(signed_zero_payloads) {
  const auto input = loom::check::fill<32, unsigned>(0x80000000u);
  const auto fp8 = loom::check::fill<32, unsigned>(0u);
  const auto bf8 = loom::check::fill<32, unsigned>(0u);
  const auto half = loom::check::fill<32, unsigned>(0u);
  const auto bfloat = loom::check::fill<32, unsigned>(0u);
  const auto scalar = loom::check::fill<32, unsigned>(0u);
  const auto output = loom::check::fill<160, unsigned>(0u);
  loom::check::launch<floating_payloads>(input, fp8, bf8, half, bfloat, scalar);
  loom::check::launch<floating_to_words>(fp8, bf8, half, bfloat, scalar,
                                         output);
  loom::check::expect_bitwise(fp8, input);
  loom::check::expect_bitwise(bf8, input);
  loom::check::expect_bitwise(half, input);
  loom::check::expect_bitwise(bfloat, input);
  loom::check::expect_bitwise(scalar, input);
  loom::check::expect_bitwise(output,
                              loom::check::fill<160, unsigned>(0x80000000u));
}

LOOM_CHECK_CASE(wide_word_order) {
  const auto input =
      loom::check::fill<3, unsigned long long>(0xFEDCBA9876543210ull);
  const auto floats = loom::check::fill<1, unsigned long long>(0ull);
  const auto scalar = loom::check::fill<1, unsigned long long>(0ull);
  const auto reversed = loom::check::fill<1, unsigned long long>(0ull);
  loom::check::launch<wide_payloads>(input, floats, scalar, reversed);
  loom::check::expect_bitwise(
      reversed,
      loom::check::fill<1, unsigned long long>(0x1032547698BADCFEull));
  loom::check::expect_bitwise(floats, loom::check::slice<1>(input, 0));
  loom::check::expect_bitwise(scalar, loom::check::slice<1>(input, 0));
}

LOOM_CHECK_CASE(wide_nan_payload) {
  const auto input =
      loom::check::fill<3, unsigned long long>(0x7FF123456789ABCDull);
  const auto floats = loom::check::fill<1, unsigned long long>(0ull);
  const auto scalar = loom::check::fill<1, unsigned long long>(0ull);
  const auto reversed = loom::check::fill<1, unsigned long long>(0ull);
  loom::check::launch<wide_payloads>(input, floats, scalar, reversed);
  loom::check::expect_bitwise(
      reversed,
      loom::check::fill<1, unsigned long long>(0xCDAB89674523F17Full));
  loom::check::expect_bitwise(floats, loom::check::slice<1>(input, 0));
  loom::check::expect_bitwise(scalar, loom::check::slice<1>(input, 0));
}

constexpr Float8x4 kPackedScale = __builtin_bit_cast(Float8x4, 0x403c3830u);
static_assert(kPackedScale[0] == 0.5f && kPackedScale[3] == 2.0f);

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void constexpr_scale(const Floats4* input, Floats4* output) {
  *output = *input * __builtin_convertvector(kPackedScale, Floats4);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void constexpr_payloads(Float8x4* fp8, BFloat8x4* bf8, Halves2* half,
                        BFloats2* bfloat, float* scalar, double* wide) {
  constexpr auto fp8_value = __builtin_bit_cast(Float8x4, 0x7E7F0080u);
  constexpr auto bf8_value = __builtin_bit_cast(BFloat8x4, 0x7D7E0080u);
  constexpr auto half_value = __builtin_bit_cast(Halves2, 0x7C018000u);
  constexpr auto bfloat_value = __builtin_bit_cast(BFloats2, 0x7F818000u);
  constexpr auto scalar_value = __builtin_bit_cast(float, 0x7F812345u);
  constexpr auto wide_value = __builtin_bit_cast(double, 0x7FF0000012345678ULL);
  *fp8 = fp8_value;
  *bf8 = bf8_value;
  *half = half_value;
  *bfloat = bfloat_value;
  *scalar = scalar_value;
  *wide = wide_value;
}

LOOM_CHECK_CASE(constexpr_packed_scales) {
  const auto input = loom::check::fill<4, float>(2.0f);
  const auto output = loom::check::fill<4, float>(0.0f);
  const auto first = loom::check::slice<1>(output, 0);
  const auto second = loom::check::slice<1>(output, 1);
  const auto third = loom::check::slice<1>(output, 2);
  const auto fourth = loom::check::slice<1>(output, 3);
  loom::check::launch<constexpr_scale>(input, output);
  loom::check::expect_bitwise(first, loom::check::fill<1, float>(1.0f));
  loom::check::expect_bitwise(second, loom::check::fill<1, float>(2.0f));
  loom::check::expect_bitwise(third, loom::check::fill<1, float>(3.0f));
  loom::check::expect_bitwise(fourth, loom::check::fill<1, float>(4.0f));
}

LOOM_CHECK_CASE(constexpr_raw_payloads) {
  const auto fp8 = loom::check::fill<1, unsigned>(0u);
  const auto bf8 = loom::check::fill<1, unsigned>(0u);
  const auto half = loom::check::fill<1, unsigned>(0u);
  const auto bfloat = loom::check::fill<1, unsigned>(0u);
  const auto scalar = loom::check::fill<1, unsigned>(0u);
  const auto wide = loom::check::fill<1, unsigned long long>(0ULL);
  loom::check::launch<constexpr_payloads>(fp8, bf8, half, bfloat, scalar, wide);
  loom::check::expect_bitwise(fp8, loom::check::fill<1, unsigned>(0x7E7F0080u));
  loom::check::expect_bitwise(bf8, loom::check::fill<1, unsigned>(0x7D7E0080u));
  loom::check::expect_bitwise(half,
                              loom::check::fill<1, unsigned>(0x7C018000u));
  loom::check::expect_bitwise(bfloat,
                              loom::check::fill<1, unsigned>(0x7F818000u));
  loom::check::expect_bitwise(scalar,
                              loom::check::fill<1, unsigned>(0x7F812345u));
  loom::check::expect_bitwise(
      wide, loom::check::fill<1, unsigned long long>(0x7FF0000012345678ULL));
}
