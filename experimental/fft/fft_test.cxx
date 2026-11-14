// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fft.h"

void forward_test(loom::type::buffer<FFT_INPUT> input,
                  loom::type::buffer<FFT_OUTPUT> output,
                  loom::type::buffer<float> table,
                  loom::type::buffer<float> workspace, unsigned frames) {
#if FFT_FREQUENCY_ONLY
  fft_experiment::forward<FFT_LENGTH, float, 1,
                          fft_experiment::SpectrumOutput::Complex>(
      input, output, table, workspace, frames);
#else
  fft_experiment::cascade_transform<false, FFT_LENGTH, FFT_WIDTH>(
      input, output, table, workspace, frames);
#endif
}

void inverse_test(loom::type::buffer<FFT_INPUT> input,
                  loom::type::buffer<FFT_OUTPUT> output,
                  loom::type::buffer<float> table,
                  loom::type::buffer<float> workspace, unsigned frames) {
#if FFT_FREQUENCY_ONLY
  fft_experiment::inverse<FFT_LENGTH, float, 1>(input, output, table, workspace,
                                                frames);
#else
  fft_experiment::cascade_transform<true, FFT_LENGTH, FFT_WIDTH>(
      input, output, table, workspace, frames);
#endif
}
