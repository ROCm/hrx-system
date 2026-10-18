// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/target.h>

#include "workgroup.h"
#include "workgroup_edges.h"

using namespace fft_experiment::workgroup;

// The complete consumed result stays in one dispatch. Other producers and
// consumers compose through the same load/store boundary.
#define SPECTRAL_KERNEL(Length)                                              \
  static loom::kernel::configuration configure##Length(unsigned frames) {    \
    const unsigned width = loom::target::subgroup::size();                   \
    constexpr unsigned minimum_width = minimum_workgroup_width<Length>();    \
    return {{frames, 1, 1},                                                  \
            {width < minimum_width ? minimum_width : width, 1, 1}};          \
  }                                                                          \
  [[loom::kernel(configure##Length), loom::fastmath("contract")]]            \
  void gpu_spectral##Length(                                                 \
      [[loom::noalias, loom::assume_aligned(8)]] const float* input,         \
      [[loom::noalias, loom::assume_aligned(8)]] float* output,              \
      [[loom::noalias, loom::assume_aligned(8)]] const float* window,        \
      unsigned frames) {                                                     \
    LOOM_WORKGROUP float                                                     \
        workspace[fft_experiment::workgroup::workspace_elements<Length>()];  \
    const unsigned long long frame =                                         \
        static_cast<unsigned>(loom::kernel::workgroup::id.x);                \
    if (frame < frames) {                                                    \
      forward<Length>(WindowedInput<float>{input + frame * Length, window},  \
                      PowerOutput<float>{output + frame * (Length / 2 + 1)}, \
                      workspace);                                            \
    }                                                                        \
  }

SPECTRAL_KERNEL(400)
SPECTRAL_KERNEL(1024)
#undef SPECTRAL_KERNEL
