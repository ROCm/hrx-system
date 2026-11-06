// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/check.h>
#include <loomcxx/view.h>

template <typename T>
__global__ [[loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void rounding(T* output, T input) {
  output[0] = loom::scalar::ceilf(input);
  output[1] = loom::scalar::floorf(input);
  output[2] = loom::scalar::truncf(input);
  output[3] = loom::scalar::roundevenf(input);
  output[4] = loom::scalar::roundf(input);
  // The two candidate-quantization forms in rocRAND's Poisson sampler.
  output[5] = loom::scalar::floorf(input + T(0.5));
  output[6] = loom::scalar::roundf(T(256) * input + T(65536));
}

template <typename T>
static void rounding_oracle(loom::type::buffer<T> output, T input) {
  auto values =
      loom::buffer::view<7>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(loom::scalar::ceilf(input), values, 0);
  loom::view::store(loom::scalar::floorf(input), values, 1);
  loom::view::store(loom::scalar::truncf(input), values, 2);
  loom::view::store(loom::scalar::roundevenf(input), values, 3);
  loom::view::store(loom::scalar::roundf(input), values, 4);
  loom::view::store(loom::scalar::floorf(input + T(0.5)), values, 5);
  loom::view::store(loom::scalar::roundf(T(256) * input + T(65536)), values, 6);
}

template <typename T>
static T input_value(loom::check::ordinal trial) {
  return (T(trial) - T(16)) / T(4);
}

LOOM_CHECK_SCENARIO(rounding_kernels) {
  loom::check::trial<32>([](loom::check::ordinal trial,
                            loom::check::entropy entropy) {
    const auto input = loom::check::generate<input_value<float>>(trial);
    const auto output = loom::check::fill<7, float>(0.0f);
    loom::check::compare<rounding<float>, rounding_oracle<float>>(
        output, input, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<32>([](loom::check::ordinal trial,
                            loom::check::entropy entropy) {
    const auto input = loom::check::generate<input_value<double>>(trial);
    const auto output = loom::check::fill<7, double>(0.0);
    loom::check::compare<rounding<double>, rounding_oracle<double>>(
        output, input, [&] { loom::check::expect_bitwise(output, output); });
  });
}
