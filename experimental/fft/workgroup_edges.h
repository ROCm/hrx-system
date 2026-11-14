// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_WORKGROUP_EDGES_H_
#define IREE_EXPERIMENTAL_FFT_WORKGROUP_EDGES_H_

#include <stdfloat>

#include "complex_stages.h"
namespace fft_experiment::workgroup {
template <class T>
using StoragePair = T __attribute__((ext_vector_type(2)));
template <class Storage>
struct RealInput {
  // Borrowed frame, aligned to a pair of storage scalars.
  const Storage* values;
};
template <class Storage>
struct WindowedInput {
  // Borrowed frame, aligned to a pair of storage scalars.
  const Storage* values;
  // Borrowed FP32 coefficients, aligned to eight bytes.
  const float* window;
};
template <class Storage>
struct SpectrumOutput {
  // Borrowed interleaved output, aligned to a pair of storage scalars.
  Storage* values;
};
template <class Storage>
struct PowerOutput {
  // Borrowed power output with natural scalar alignment.
  Storage* values;
};
template <class Storage>
[[loom::force_inline]] static Complex<float> load(RealInput<Storage> input,
                                                  unsigned position) {
  const auto pair =
      reinterpret_cast<const StoragePair<Storage>*>(input.values)[position];
  return {static_cast<float>(pair[0]), static_cast<float>(pair[1])};
}
template <class Storage>
[[loom::force_inline]] static Complex<float> load(WindowedInput<Storage> input,
                                                  unsigned position) {
  const auto value = load(RealInput<Storage>{input.values}, position);
  const auto weight = load(RealInput<float>{input.window}, position);
  return {value.real * weight.real, value.imag * weight.imag};
}
template <class Storage>
[[loom::force_inline]] static void store(SpectrumOutput<Storage> output,
                                         unsigned frequency,
                                         Complex<float> value) {
  reinterpret_cast<StoragePair<Storage>*>(output.values)[frequency] =
      StoragePair<Storage>{static_cast<Storage>(value.real),
                           static_cast<Storage>(value.imag)};
}
template <class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void store(
    PowerOutput<Storage> output, unsigned frequency, Complex<float> value) {
  output.values[frequency] =
      static_cast<Storage>(value.real * value.real + value.imag * value.imag);
}
}  // namespace fft_experiment::workgroup

#endif  // IREE_EXPERIMENTAL_FFT_WORKGROUP_EDGES_H_
