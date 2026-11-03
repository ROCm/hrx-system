// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

struct Pair {
  unsigned sum;
  unsigned mixed;
};

[[loom::noinline]]
static Pair update(const unsigned* input, unsigned index, unsigned bias) {
  const unsigned value = input[index];
  return {value + bias, value ^ bias};
}

[[loom::kernel, loom::workgroup_count(1, 1, 1), loom::workgroup_size(64, 1, 1)]]
void retained_calls(const unsigned* input, unsigned* sums, unsigned* mixed) {
  const unsigned index = loom::kernel::workitem::id.x;
  const Pair first = update(input, index, 7u);
  const Pair second = update(input, index, 13u);
  sums[index] = first.sum + second.sum;
  mixed[index] = first.mixed + second.mixed;
}

LOOM_CHECK_CASE(retained_call_values) {
  const auto input = loom::check::fill<64, unsigned>(21u);
  const auto sums = loom::check::fill<64, unsigned>(0u);
  const auto mixed = loom::check::fill<64, unsigned>(0u);
  loom::check::launch<retained_calls>(input, sums, mixed);
  loom::check::expect_bitwise(input, loom::check::fill<64, unsigned>(21u));
  loom::check::expect_bitwise(sums, loom::check::fill<64, unsigned>(62u));
  loom::check::expect_bitwise(mixed, loom::check::fill<64, unsigned>(42u));
}
