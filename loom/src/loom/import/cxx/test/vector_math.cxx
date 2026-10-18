// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/vector.h>

#include <stdfloat>

using Float2 = float __attribute__((ext_vector_type(2)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Half4 = std::float16_t __attribute__((ext_vector_type(4)));
using BFloat4 = std::bfloat16_t __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void ordered(float a, float b, float c, float d, float seed,
             [[loom::noalias]] float* output) {
  Float4 values = {a, b, c, d};
  Float4 ones = {1.0f, 1.0f, 1.0f, 1.0f};
  output[0] = loom::vector::reduce::addf(values, seed);
  output[1] = loom::vector::dotf(values, ones, seed);
}

LOOM_CHECK_CASE(logical_lane_order) {
  const auto storage = loom::check::fill<4, float>(99.0f);
  const auto output = loom::check::slice<2>(storage, 1);
  loom::check::launch<ordered>(16777216.0f, 1.0f, -16777216.0f, 2.0f, 0.0f,
                               output);
  // The first addition loses 1 at F32 precision, then cancellation leaves 2.
  // A reassociated tree can instead retain that 1 and produce 3.
  loom::check::expect_bitwise(output, loom::check::fill<2, float>(2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<1, float>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 3),
                              loom::check::fill<1, float>(99.0f));
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fused(float a, float b, float seed, [[loom::noalias]] float* output) {
  Float2 lhs = {a, 0.0f};
  Float2 rhs = {b, 0.0f};
  *output = loom::vector::dotf(lhs, rhs, seed);
}

LOOM_CHECK_CASE(fused_accumulation) {
  const auto storage = loom::check::fill<3, float>(99.0f);
  const auto output = loom::check::slice<1>(storage, 1);
  loom::check::launch<fused>(0x1.000002p0f, 0x1.fffffcp-1f, -1.0f, output);
  // (1 + 2^-23) * (1 - 2^-23) - 1 = -2^-46. Rounding the product
  // before adding the seed would lose the residual and produce zero.
  loom::check::expect_bitwise(output, loom::check::fill<1, float>(-0x1p-46f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 0),
                              loom::check::fill<1, float>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(storage, 2),
                              loom::check::fill<1, float>(99.0f));
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void pair_arithmetic(
    [[loom::noalias, loom::assume_aligned(64)]] const Half4* half_lhs,
    [[loom::noalias, loom::assume_aligned(64)]] const Half4* half_rhs,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat4* bfloat_lhs,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat4* bfloat_rhs,
    float first, float second, [[loom::noalias]] Float2* output) {
  Float2 seed = {first, second};
  output[0] = loom::vector::dot2f(*half_lhs, *half_rhs, seed);
  output[1] = loom::vector::dot2f(*bfloat_lhs, *bfloat_rhs, seed);
  Float4 half_left = loom::convert<Float4>(*half_lhs);
  Float4 half_right = loom::convert<Float4>(*half_rhs);
  Float4 bfloat_left = loom::convert<Float4>(*bfloat_lhs);
  Float4 bfloat_right = loom::convert<Float4>(*bfloat_rhs);
  output[2] = {
      loom::vector::dotf(Float2{half_left[0], half_left[1]},
                         Float2{half_right[0], half_right[1]}, first),
      loom::vector::dotf(Float2{half_left[2], half_left[3]},
                         Float2{half_right[2], half_right[3]}, second)};
  output[3] = {
      loom::vector::dotf(Float2{bfloat_left[0], bfloat_left[1]},
                         Float2{bfloat_right[0], bfloat_right[1]}, first),
      loom::vector::dotf(Float2{bfloat_left[2], bfloat_left[3]},
                         Float2{bfloat_right[2], bfloat_right[3]}, second)};
}

LOOM_CHECK_CASE(pair_formats_and_seeds) {
  // Both formats encode lhs [1, 2, -3, 4] and rhs [5, -6, 7, 8].
  const auto half_lhs =
      loom::check::fill<1, unsigned long long>(0x4400C20040003C00ull);
  const auto half_rhs =
      loom::check::fill<1, unsigned long long>(0x48004700C6004500ull);
  const auto bfloat_lhs =
      loom::check::fill<1, unsigned long long>(0x4080C04040003F80ull);
  const auto bfloat_rhs =
      loom::check::fill<1, unsigned long long>(0x410040E0C0C040A0ull);
  const auto storage = loom::check::fill<16, float>(99.0f);
  const auto output = loom::check::slice<8>(storage, 4);
  loom::check::launch<pair_arithmetic>(half_lhs, half_rhs, bfloat_lhs,
                                       bfloat_rhs, 9.0f, -10.0f, output);
  // Each format computes [9 + 5 - 12, -10 - 21 + 32] = [2, 1].
  // Require FP32 accumulation-scale accuracy from native grouped arithmetic:
  // gamma_3 times the sum of absolute contributions, with unit roundoff 2^-24.
  // This is the case's accuracy requirement, independent of native result bits.
  constexpr double error_factor = 3.0 * 0x1p-24 / (1.0 - 3.0 * 0x1p-24);
  loom::check::expect_close(loom::check::slice<1>(output, 0),
                            loom::check::fill<1, float>(2.0f),
                            26.0 * error_factor, 0.0, "different");
  loom::check::expect_close(loom::check::slice<1>(output, 1),
                            loom::check::fill<1, float>(1.0f),
                            63.0 * error_factor, 0.0, "different");
  loom::check::expect_close(loom::check::slice<1>(output, 2),
                            loom::check::fill<1, float>(2.0f),
                            26.0 * error_factor, 0.0, "different");
  loom::check::expect_close(loom::check::slice<1>(output, 3),
                            loom::check::fill<1, float>(1.0f),
                            63.0 * error_factor, 0.0, "different");
  // Explicitly widened dotf uses ordered F32 FMAs, exact for these inputs.
  loom::check::expect_bitwise(loom::check::slice<1>(output, 4),
                              loom::check::fill<1, float>(2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 5),
                              loom::check::fill<1, float>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 6),
                              loom::check::fill<1, float>(2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 7),
                              loom::check::fill<1, float>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 0),
                              loom::check::fill<4, float>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 12),
                              loom::check::fill<4, float>(99.0f));
}
