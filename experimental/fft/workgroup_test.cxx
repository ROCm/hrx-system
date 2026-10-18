// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "workgroup.h"

#include <loomcxx/target.h>

#include "workgroup_edges.h"

using namespace fft_experiment::workgroup;

static loom::kernel::configuration configure(unsigned frames) {
  const unsigned width = loom::target::subgroup::size();
  constexpr unsigned minimum_width = minimum_workgroup_width<FFT_LENGTH>();
  return {{frames, 1, 1},
          {width < minimum_width ? minimum_width : width, 1, 1}};
}

[[loom::kernel(configure), loom::fastmath("contract")]] void forward_test(
    [[loom::noalias, loom::assume_aligned(8)]] const FFT_INPUT* input,
    [[loom::noalias, loom::assume_aligned(8)]] FFT_OUTPUT* output,
    unsigned frames) {
  LOOM_WORKGROUP float
      workspace[fft_experiment::workgroup::workspace_elements<FFT_LENGTH>()];
  const unsigned long long frame =
      static_cast<unsigned>(loom::kernel::workgroup::id.x);
  if (frame < frames) {
    forward<FFT_LENGTH>(
        RealInput<FFT_INPUT>{input + frame * FFT_LENGTH},
        SpectrumOutput<FFT_OUTPUT>{output + frame * (FFT_LENGTH + 2)},
        workspace);
  }
}

[[loom::kernel(configure), loom::fastmath("contract")]] void power_test(
    [[loom::noalias, loom::assume_aligned(8)]] const FFT_INPUT* input,
    [[loom::noalias, loom::assume_aligned(8)]] FFT_OUTPUT* output,
    [[loom::noalias, loom::assume_aligned(8)]] const float* window,
    unsigned frames) {
  LOOM_WORKGROUP float
      workspace[fft_experiment::workgroup::workspace_elements<FFT_LENGTH>()];
  const unsigned long long frame =
      static_cast<unsigned>(loom::kernel::workgroup::id.x);
  if (frame < frames) {
    forward<FFT_LENGTH>(
        WindowedInput<FFT_INPUT>{input + frame * FFT_LENGTH, window},
        PowerOutput<FFT_OUTPUT>{output + frame * (FFT_LENGTH / 2 + 1)},
        workspace);
  }
}
