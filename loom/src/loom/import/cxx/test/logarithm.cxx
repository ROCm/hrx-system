// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/check.h>
#include <loomcxx/view.h>

// Box-Muller callers need scalar AFN logarithms inside an ordinary CPU HAL
// kernel. The inputs cross the invocation boundary, preventing constant folds.
__global__ [[loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void logarithms(float* output, float input) {
  output[0] = loom::scalar::approximate::logf(input);
  output[1] = loom::scalar::approximate::log2f(input);
  output[2] = loom::scalar::approximate::log10f(input);
  output[3] = loom::scalar::approximate::log1pf(input);
}

__global__ [[loom::fastmath, loom::workgroup_size(1, 1, 1),
             loom::workgroup_count(1, 1, 1)]]
void logarithms_fast(float* output, float input) {
  output[0] = loom::scalar::logf(input);
  output[1] = loom::scalar::log2f(input);
  output[2] = loom::scalar::log10f(input);
  output[3] = loom::scalar::log1pf(input);
}

static void logarithms_oracle(loom::type::buffer<float> output, float input) {
  auto values =
      loom::buffer::view<4>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(loom::scalar::logf(input), values, 0);
  loom::view::store(loom::scalar::log2f(input), values, 1);
  loom::view::store(loom::scalar::log10f(input), values, 2);
  loom::view::store(loom::scalar::log1pf(input), values, 3);
}

static float input_value(loom::check::ordinal trial) {
  return (float(trial) + 1.0f) / 16.0f;
}

LOOM_CHECK_SCENARIO(logarithm_kernel) {
  loom::check::trial<32>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::generate<input_value>(trial);
        const auto output = loom::check::fill<4, float>(0.0f);
        loom::check::compare<logarithms, logarithms_oracle>(output, input, [&] {
          loom::check::expect_approximate(output, output, 0.0, 0x1p-21);
        });
      });
  loom::check::trial<32>([](loom::check::ordinal trial,
                            loom::check::entropy entropy) {
    const auto input = loom::check::generate<input_value>(trial);
    const auto output = loom::check::fill<4, float>(0.0f);
    loom::check::compare<logarithms_fast, logarithms_oracle>(
        output, input,
        [&] { loom::check::expect_approximate(output, output, 0.0, 0x1p-21); });
  });
}
