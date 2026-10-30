// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

#ifndef CHECK_ALIAS_EXPECTED
#define CHECK_ALIAS_EXPECTED 17u
#endif

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void alias_update(unsigned* left, unsigned* right) {
  left[1] = 13;
  right[0] += 4;
}

LOOM_CHECK_CASE(alias_values) {
  const auto storage = loom::check::fill<5, unsigned>(37u);
  const auto copy = storage;
  const auto left = loom::check::slice<3>(copy, 1);
  const auto right = loom::check::slice<2>(left, 1);
  loom::check::launch<alias_update>(left, right);
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 0),
                              loom::check::fill<2, unsigned>(37u));
  loom::check::expect_bitwise(
      loom::check::slice<1>(storage, 2),
      loom::check::fill<1, unsigned>(CHECK_ALIAS_EXPECTED));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 3),
                              loom::check::fill<2, unsigned>(37u));
}

LOOM_CHECK_BENCHMARK(alias_values_benchmark, alias_values);
