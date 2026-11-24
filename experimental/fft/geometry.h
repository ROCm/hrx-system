// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_GEOMETRY_H_
#define IREE_EXPERIMENTAL_FFT_GEOMETRY_H_

namespace fft_experiment {

static constexpr unsigned forward_radix(unsigned remaining) {
  return remaining % 5 == 0 ? 5 : remaining % 4 == 0 ? 4 : 2;
}
static constexpr unsigned inverse_radix(unsigned remaining) {
  if (remaining <= 5) {
    return remaining;
  }
  return inverse_radix(remaining / forward_radix(remaining));
}

static constexpr bool supported_real_length(unsigned length) {
  if (length == 0 || length % 2 != 0) {
    return false;
  }
  while (length % 2 == 0) {
    length /= 2;
  }
  while (length % 5 == 0) {
    length /= 5;
  }
  return length == 1;
}

static constexpr unsigned real_table_offset(unsigned length,
                                            unsigned stage_length) {
  unsigned position = 1;
  unsigned offset = 0;
  while (position < stage_length) {
    const unsigned radix = forward_radix(length / position);
    offset += (radix - 1) * (position - 1);
    position *= radix;
  }
  return offset;
}
// Exact stage-major real-twiddle extent, including stage-local lane padding.
static constexpr unsigned real_table_elements(unsigned length) {
  return real_table_offset(length, length);
}

template <unsigned Length>
static constexpr unsigned table_elements() {
  static_assert(supported_real_length(Length),
                "real FFT length must be positive, even, and have only factors "
                "two and five");
  static_assert(Length <= ~0u / 3,
                "FFT table extent exceeds the index representation");
  return real_table_elements(Length) + 2 * Length;
}

template <unsigned Length, unsigned Width>
static constexpr unsigned workspace_elements() {
  static_assert(Width > 0 && (Width & (Width - 1)) == 0,
                "FFT frame packet width must be a power of two");
  static_assert(Length <= ~0u / 2 / Width,
                "FFT workspace extent exceeds the index representation");
  return 2 * Length * Width;
}

}  // namespace fft_experiment
#endif  // IREE_EXPERIMENTAL_FFT_GEOMETRY_H_
