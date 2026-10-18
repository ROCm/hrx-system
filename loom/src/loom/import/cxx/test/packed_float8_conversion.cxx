// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

using Float8E4M3x4 =
    loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(4)));
using Float8E5M2x4 =
    loom::type::float8_e5m2_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Byte4 = signed char __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void expand_float8(
    [[loom::noalias, loom::assume_aligned(64)]] const Float8E4M3x4* e4m3,
    [[loom::noalias, loom::assume_aligned(64)]] const Float8E5M2x4* e5m2,
    [[loom::noalias]] Float4* output) {
  output[0] = loom::convert<Float4>(*e4m3);
  output[1] = loom::convert<Float4>(*e5m2);
}

LOOM_CHECK_CASE(uniform_float8_conversion) {
  // Literal payloads encode [1, -2, 0.5, -0.25] in each input format.
  const auto e4m3 = loom::check::fill<1, unsigned>(0xA830C038u);
  const auto e5m2 = loom::check::fill<1, unsigned>(0xB438C03Cu);
  const auto storage = loom::check::fill<16, float>(99.0f);
  const auto output = loom::check::slice<8>(storage, 4);
  loom::check::launch<expand_float8>(e4m3, e5m2, output);
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
  loom::check::expect_bitwise(e4m3,
                              loom::check::fill<1, unsigned>(0xA830C038u));
  loom::check::expect_bitwise(e5m2,
                              loom::check::fill<1, unsigned>(0xB438C03Cu));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 0),
                              loom::check::fill<4, float>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 12),
                              loom::check::fill<4, float>(99.0f));
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_float8_integers(
    [[loom::noalias, loom::assume_aligned(64)]] const Byte4* codebook,
    [[loom::noalias, loom::assume_aligned(64)]] const Float8E5M2x4* fractional,
    [[loom::noalias]] Float8E4M3x4* e4m3, [[loom::noalias]] Float8E5M2x4* e5m2,
    [[loom::noalias]] Byte4* truncated) {
  *e4m3 = loom::convert<Float8E4M3x4>(*codebook);
  *e5m2 = loom::convert<Float8E5M2x4>(*codebook);
  *truncated = loom::convert<Byte4>(*fractional);
}

LOOM_CHECK_CASE(integer_float8_conversion) {
  // Codebook [-128, -1, 0, 127] and E5M2 [1.5, -2.5, 0.5, -0.5].
  const auto codebook = loom::check::fill<1, unsigned>(0x7F00FF80u);
  const auto fractional = loom::check::fill<1, unsigned>(0xB838C13Eu);
  const auto e4m3 = loom::check::fill<1, unsigned>(0);
  const auto e5m2 = loom::check::fill<1, unsigned>(0);
  const auto truncated = loom::check::fill<1, unsigned>(0);
  loom::check::launch<convert_float8_integers>(codebook, fractional, e4m3, e5m2,
                                               truncated);
  loom::check::expect_bitwise(e4m3,
                              loom::check::fill<1, unsigned>(0x7000B8F0u));
  loom::check::expect_bitwise(e5m2,
                              loom::check::fill<1, unsigned>(0x5800BCD8u));
  loom::check::expect_bitwise(truncated,
                              loom::check::fill<1, unsigned>(0x0000FE01u));
}
