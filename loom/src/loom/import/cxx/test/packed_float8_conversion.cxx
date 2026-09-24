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

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void expand_float8(
    [[loom::noalias, loom::assume_aligned(64)]] const Float8E4M3x4* e4m3,
    [[loom::noalias, loom::assume_aligned(64)]] const Float8E5M2x4* e5m2,
    [[loom::noalias]] Float4* output) {
  output[0] = __builtin_convertvector(*e4m3, Float4);
  output[1] = __builtin_convertvector(*e5m2, Float4);
}

LOOM_CHECK_CASE(uniform_float8_conversion) {
  // Literal payloads encode [1, -2, 0.5, -0.25] in each input format.
  const auto e4m3 = loom::check::fill<unsigned, 1>(0xA830C038u);
  const auto e5m2 = loom::check::fill<unsigned, 1>(0xB438C03Cu);
  const auto storage = loom::check::fill<float, 16>(99.0f);
  const auto output = loom::check::slice<8>(storage, 4);
  loom::check::launch<expand_float8>(e4m3, e5m2, output);
  loom::check::expect_bitwise(loom::check::slice<1>(output, 0),
                              loom::check::fill<float, 1>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 1),
                              loom::check::fill<float, 1>(-2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 2),
                              loom::check::fill<float, 1>(0.5f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 3),
                              loom::check::fill<float, 1>(-0.25f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 4),
                              loom::check::fill<float, 1>(1.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 5),
                              loom::check::fill<float, 1>(-2.0f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 6),
                              loom::check::fill<float, 1>(0.5f));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 7),
                              loom::check::fill<float, 1>(-0.25f));
  loom::check::expect_bitwise(e4m3,
                              loom::check::fill<unsigned, 1>(0xA830C038u));
  loom::check::expect_bitwise(e5m2,
                              loom::check::fill<unsigned, 1>(0xB438C03Cu));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 0),
                              loom::check::fill<float, 4>(99.0f));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 12),
                              loom::check::fill<float, 4>(99.0f));
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}
