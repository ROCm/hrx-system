// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/target.h>

LOOM_FORCE_INLINE unsigned launch_width() {
  return loom::target::subgroup::size();
}

LOOM_FORCE_INLINE unsigned divide_up(unsigned value, unsigned divisor) {
  return (value + divisor - 1) / divisor;
}

static loom::kernel::configuration configure_copy(unsigned item_count) {
  unsigned width = launch_width();
  return {{divide_up(item_count, width), 1, 1}, {width, 1, 1}};
}

[[loom::kernel(configure_copy)]] void configured_copy(unsigned item_count,
                                                      const float* input,
                                                      float* output) {
  unsigned item =
      loom::kernel::workgroup::id.x * loom::kernel::workgroup::size.x +
      loom::kernel::workitem::id.x;
  if (item < item_count) {
    output[item] = input[item];
  }
}

LOOM_CHECK_CASE(configured_copy_case) {
  const auto input = loom::check::iota<67, float>(-1.0f, 0.25f, 8);
  const auto output = loom::check::fill<67, float>(0.0f);
  loom::check::launch<configured_copy>(loom::kernel::workload(67u), 67u, input,
                                       output);
  loom::check::expect_bitwise(output, input);
}
