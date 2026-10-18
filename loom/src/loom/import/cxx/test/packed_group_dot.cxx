// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/scalar.h>
#include <loomcxx/vector.h>

#include <stdfloat>

using Bytes16 = unsigned char __attribute__((ext_vector_type(16)));
using Codes16 = signed char __attribute__((ext_vector_type(16)));
using BFloat16 = std::bfloat16_t __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

// Require FP32 accumulation-scale accuracy: gamma_32 times the sum of absolute
// scaled contributions, with unit roundoff 2^-24. This requirement does not
// depend on any target's chosen accumulation sequence or observed result bits.
constexpr double kGroupErrorFactor = 32.0 * 0x1p-24 / (1.0 - 32.0 * 0x1p-24);

[[loom::force_inline]] static Codes16 lookup(Codes16 table, Bytes16 indices) {
  return {table[indices[0]],  table[indices[1]],  table[indices[2]],
          table[indices[3]],  table[indices[4]],  table[indices[5]],
          table[indices[6]],  table[indices[7]],  table[indices[8]],
          table[indices[9]],  table[indices[10]], table[indices[11]],
          table[indices[12]], table[indices[13]], table[indices[14]],
          table[indices[15]]};
}

// One 32-weight group: unpack codebook indices, convert exact signed codes to
// BF16, accumulate paired products, then apply the group's scale and bias.
float packed_group_dot(Codes16 table, Bytes16 packed, BFloat16 activations_low,
                       BFloat16 activations_high, float scale, float seed) {
  auto low_codes = lookup(table, packed & 15);
  auto high_codes = lookup(table, packed >> 4);
  auto low_weights = loom::convert<BFloat16>(low_codes);
  auto high_weights = loom::convert<BFloat16>(high_codes);
  Float8 partial = {};
  partial = loom::vector::dot2f(low_weights, activations_low, partial);
  partial = loom::vector::dot2f(high_weights, activations_high, partial);
  return loom::scalar::fmaf(scale, loom::vector::reduce::addf(partial, 0.0f),
                            seed);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void decode_dot(
    [[loom::noalias, loom::assume_aligned(64)]] const Codes16* table,
    [[loom::noalias, loom::assume_aligned(64)]] const Bytes16* packed,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat16* activations_low,
    [[loom::noalias,
      loom::assume_aligned(64)]] const BFloat16* activations_high,
    [[loom::noalias]] float* output, float scale, float seed) {
  *output = packed_group_dot(*table, *packed, *activations_low,
                             *activations_high, scale, seed);
}

LOOM_CHECK_CASE(parallel_codes) {
  // Repeated table [-8, -3, 0, 7, -6, 5, -1, 2]. The low nibbles visit
  // entries 0..7 and the high nibbles visit 8..15 in each eight-byte group.
  const auto table =
      loom::check::fill<2, unsigned long long>(0x02FF05FA0700FDF8ull);
  const auto packed =
      loom::check::fill<2, unsigned long long>(0xF7E6D5C4B3A29180ull);
  // BF16 activations repeat [1, 2, 3, 4] and [-4, 3, -2, 1].
  const auto low =
      loom::check::fill<4, unsigned long long>(0x4080404040003F80ull);
  const auto high =
      loom::check::fill<4, unsigned long long>(0x3F80C0004040C080ull);
  const auto storage = loom::check::fill<3, float>(999.0f);
  const auto output = loom::check::slice<1>(storage, 1);
  loom::check::launch<decode_dot>(table, packed, low, high, output, 0.5f,
                                  3.25f);
  // Independent integer products per eight lanes sum to 96. Two groups,
  // scaled by one half and biased by 3.25, give exactly 99.25. Their absolute
  // products sum to 320, giving a scaled magnitude of 160 + 3.25 = 653/4.
  loom::check::expect_close(output, loom::check::fill<1, float>(99.25f),
                            (653.0 / 4.0) * kGroupErrorFactor, 0.0,
                            "different");
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<1, float>(999.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2),
                              loom::check::fill<1, float>(999.0f));
  loom::check::expect_bitwise(
      table, loom::check::fill<2, unsigned long long>(0x02FF05FA0700FDF8ull));
  loom::check::expect_bitwise(
      packed, loom::check::fill<2, unsigned long long>(0xF7E6D5C4B3A29180ull));
  loom::check::expect_bitwise(
      low, loom::check::fill<4, unsigned long long>(0x4080404040003F80ull));
  loom::check::expect_bitwise(
      high, loom::check::fill<4, unsigned long long>(0x3F80C0004040C080ull));
}

LOOM_CHECK_CASE(crossed_codes) {
  const auto table =
      loom::check::fill<2, unsigned long long>(0x02FF05FA0700FDF8ull);
  // Reverse only the high nibbles: the two lookups select different codes.
  const auto packed =
      loom::check::fill<2, unsigned long long>(0x8796A5B4C3D2E1F0ull);
  const auto low =
      loom::check::fill<4, unsigned long long>(0x4080404040003F80ull);
  const auto high =
      loom::check::fill<4, unsigned long long>(0x3F80C0004040C080ull);
  const auto storage = loom::check::fill<3, float>(999.0f);
  const auto output = loom::check::slice<1>(storage, 1);
  loom::check::launch<decode_dot>(table, packed, low, high, output, 0.5f,
                                  3.25f);
  // Independent integer products per eight lanes sum to -34. The full group's
  // absolute products sum to 276, giving a scaled magnitude of 138 + 3.25 =
  // 565/4.
  loom::check::expect_close(output, loom::check::fill<1, float>(-30.75f),
                            (565.0 / 4.0) * kGroupErrorFactor, 0.0,
                            "different");
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<1, float>(999.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2),
                              loom::check::fill<1, float>(999.0f));
  loom::check::expect_bitwise(
      table, loom::check::fill<2, unsigned long long>(0x02FF05FA0700FDF8ull));
  loom::check::expect_bitwise(
      packed, loom::check::fill<2, unsigned long long>(0x8796A5B4C3D2E1F0ull));
  loom::check::expect_bitwise(
      low, loom::check::fill<4, unsigned long long>(0x4080404040003F80ull));
  loom::check::expect_bitwise(
      high, loom::check::fill<4, unsigned long long>(0x3F80C0004040C080ull));
}
