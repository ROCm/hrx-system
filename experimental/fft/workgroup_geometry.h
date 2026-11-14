// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_WORKGROUP_GEOMETRY_H_
#define IREE_EXPERIMENTAL_FFT_WORKGROUP_GEOMETRY_H_

#include "geometry.h"

namespace fft_experiment::workgroup {
static constexpr unsigned stage_radix(unsigned remaining) {
  return remaining % 8 == 0   ? 8
         : remaining % 5 == 0 ? 5
         : remaining % 4 == 0 ? 4
                              : 2;
}

// One workitem owns one radix job; later stages may have more jobs than the
// first. Target launch validation admits the resulting width and LDS extent.
template <unsigned Length>
static constexpr unsigned minimum_workgroup_width() {
  static_assert(supported_real_length(Length),
                "workgroup real FFT length must be positive, even, and have "
                "only factors two and five");
  unsigned width = 1;
  for (unsigned previous = 1; previous < Length / 2;) {
    const unsigned radix = stage_radix(Length / 2 / previous);
    const unsigned jobs = Length / 2 / radix;
    if (jobs > width) {
      width = jobs;
    }
    previous *= radix;
  }
  return width;
}

// Interleaved FP32 scratch, with one complex padding slot per eight values.
template <unsigned Length>
static constexpr unsigned workspace_elements() {
  static_assert(supported_real_length(Length),
                "workgroup real FFT length must be positive, even, and have "
                "only factors two and five");
  static_assert(
      Length <= (~0u / 9) * 8,
      "workgroup FFT scratch extent exceeds the index representation");
  return 2 * (Length / 2 + Length / 16);
}
}  // namespace fft_experiment::workgroup

#endif  // IREE_EXPERIMENTAL_FFT_WORKGROUP_GEOMETRY_H_
