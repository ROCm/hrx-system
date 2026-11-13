// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --8<-- [start:source]
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

static loom::kernel::configuration configure_saxpy(unsigned element_count) {
  return {{(element_count + 255) / 256, 1, 1}, {256, 1, 1}};
}

[[loom::kernel(configure_saxpy)]]
void saxpy_f32(unsigned element_count, float alpha,
               [[loom::noalias]] const float* x, [[loom::noalias]] float* y) {
  unsigned element_index =
      loom::kernel::workgroup::id.x * loom::kernel::workgroup::size.x +
      loom::kernel::workitem::id.x;
  if (element_index < element_count) {
    y[element_index] =
        loom::scalar::fmaf(alpha, x[element_index], y[element_index]);
  }
}

#define SAXPY_CASE(name, count)                                           \
  LOOM_CHECK_CASE(name) {                                                 \
    constexpr unsigned element_count = count;                             \
    const auto x = loom::check::fill<element_count, float>(2.0f);         \
    const auto y = loom::check::fill<element_count, float>(3.0f);         \
    const auto expected = loom::check::fill<element_count, float>(11.0f); \
    loom::check::launch<saxpy_f32>(loom::kernel::workload(element_count), \
                                   element_count, 4.0f, x, y);            \
    loom::check::expect_bitwise(y, expected);                             \
  }

SAXPY_CASE(saxpy_f32_1, 1)
SAXPY_CASE(saxpy_f32_255, 255)
SAXPY_CASE(saxpy_f32_256, 256)
SAXPY_CASE(saxpy_f32_257, 257)
SAXPY_CASE(saxpy_f32_1009, 1009)
SAXPY_CASE(saxpy_f32_65536, 65536)

LOOM_CHECK_BENCHMARK(saxpy_f32_64k, saxpy_f32_65536);
// --8<-- [end:source]
