// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

#include <stdfloat>

using Half4 = std::float16_t __attribute__((ext_vector_type(4)));
using BFloat4 = std::bfloat16_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Byte4 = signed char __attribute__((ext_vector_type(4)));
using Int4 = int __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void expand_float16(
    [[loom::noalias, loom::assume_aligned(64)]] const Half4* half,
    [[loom::noalias, loom::assume_aligned(64)]] const BFloat4* bfloat,
    [[loom::noalias]] Float4* output) {
  output[0] = loom::convert<Float4>(*half);
  output[1] = loom::convert<Float4>(*bfloat);
}

LOOM_CHECK_CASE(uniform_float16_conversion) {
  // Literal payloads encode [1, -2, 0.5, -0.25] in each input format.
  const auto half =
      loom::check::fill<1, unsigned long long>(0xB4003800C0003C00ull);
  const auto bfloat =
      loom::check::fill<1, unsigned long long>(0xBE803F00C0003F80ull);
  const auto storage = loom::check::fill<16, float>(99.0f);
  const auto output = loom::check::slice<8>(storage, 4);
  loom::check::launch<expand_float16>(half, bfloat, output);
  loom::check::expect_bitwise(loom::check::slice<1>(output, 0),
                              loom::check::fill<1, float>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 1),
                              loom::check::fill<1, float>(-2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 2),
                              loom::check::fill<1, float>(0.5f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 3),
                              loom::check::fill<1, float>(-0.25f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 4),
                              loom::check::fill<1, float>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 5),
                              loom::check::fill<1, float>(-2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 6),
                              loom::check::fill<1, float>(0.5f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 7),
                              loom::check::fill<1, float>(-0.25f));
  loom::check::expect_bitwise(
      half, loom::check::fill<1, unsigned long long>(0xB4003800C0003C00ull));
  loom::check::expect_bitwise(
      bfloat, loom::check::fill<1, unsigned long long>(0xBE803F00C0003F80ull));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 0),
                              loom::check::fill<4, float>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 12),
                              loom::check::fill<4, float>(99.0f));
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_float16_integers(
    [[loom::noalias, loom::assume_aligned(64)]] const Byte4* codebook,
    [[loom::noalias, loom::assume_aligned(64)]] const Half4* fractional,
    [[loom::noalias]] BFloat4* decoded, [[loom::noalias]] Int4* truncated) {
  *decoded = loom::convert<BFloat4>(*codebook);
  *truncated = loom::convert<Int4>(*fractional);
}

LOOM_CHECK_CASE(integer_float16_conversion) {
  // Codebook [-128, -1, 0, 127] and FP16 [1.5, -2.5, 0.5, -0.5].
  const auto codebook = loom::check::fill<1, unsigned>(0x7F00FF80u);
  const auto fractional =
      loom::check::fill<1, unsigned long long>(0xB8003800C1003E00ull);
  const auto decoded = loom::check::fill<1, unsigned long long>(0);
  const auto truncated = loom::check::fill<4, int>(42);
  loom::check::launch<convert_float16_integers>(codebook, fractional, decoded,
                                                truncated);
  loom::check::expect_bitwise(
      decoded, loom::check::fill<1, unsigned long long>(0x42FE0000BF80C300ull));
  loom::check::expect_bitwise(loom::check::slice<1>(truncated, 0),
                              loom::check::fill<1, int>(1));
  loom::check::expect_bitwise(loom::check::slice<1>(truncated, 1),
                              loom::check::fill<1, int>(-2));
  loom::check::expect_bitwise(loom::check::slice<2>(truncated, 2),
                              loom::check::fill<2, int>(0));
}
