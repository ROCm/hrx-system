// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_FRAME_IO_H_
#define IREE_EXPERIMENTAL_FFT_FRAME_IO_H_
#include <stdfloat>

#include "complex_stages.h"
namespace fft_experiment {
enum class SpectrumOutput { Complex, Power };

template <class T, unsigned Width, unsigned Lane, class Input>
[[loom::force_inline]] static T gather_frame_lanes(Input input,
                                                   loom::type::index frame,
                                                   unsigned position, T value) {
  value[Lane] = static_cast<float>(
      loom::view::load(input, frame + loom::type::index{Lane}, position));
  if constexpr (Lane + 1 < Width) {
    return gather_frame_lanes<T, Width, Lane + 1>(input, frame, position,
                                                  value);
  } else {
    return value;
  }
}

template <class T, unsigned Width, class Input>
[[loom::force_inline]] static T read_frames(Input input,
                                            loom::type::index frame,
                                            unsigned position) {
  if constexpr (Width == 1) {
    return static_cast<float>(loom::view::load(input, frame, position));
  } else {
    return gather_frame_lanes<T, Width, 0>(input, frame, position, T{});
  }
}

template <class OutputScalar, unsigned Width, unsigned Lane = 0, class Output,
          class T>
[[loom::force_inline]] static void write_frames(Output output,
                                                loom::type::index frame,
                                                unsigned position, T value) {
  if constexpr (Width == 1) {
    loom::view::store(static_cast<OutputScalar>(value), output, frame,
                      position);
  } else {
    loom::view::store(static_cast<OutputScalar>(value[Lane]), output,
                      frame + loom::type::index{Lane}, position);
    if constexpr (Lane + 1 < Width) {
      write_frames<OutputScalar, Width, Lane + 1>(output, frame, position,
                                                  value);
    }
  }
}

template <SpectrumOutput Mode, class OutputScalar, unsigned Width, class Output,
          class T>
[[loom::force_inline, loom::fastmath("contract")]] static void consume_spectrum(
    Output output, loom::type::index frame, unsigned frequency,
    Complex<T> value) {
  if constexpr (Mode == SpectrumOutput::Power) {
    const T power = value.real * value.real + value.imag * value.imag;
    write_frames<OutputScalar, Width>(output, frame, frequency, power);
  } else {
    write_frames<OutputScalar, Width>(output, frame, 2 * frequency, value.real);
    write_frames<OutputScalar, Width>(output, frame, 2 * frequency + 1,
                                      value.imag);
  }
}

}  // namespace fft_experiment
#endif
