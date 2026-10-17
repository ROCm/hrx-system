// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_DEINTERLEAVE_H_
#define IREE_EXPERIMENTAL_FFT_DEINTERLEAVE_H_
#include <loomcxx/kernel.h>

#include "complex_stages.h"
namespace fft_experiment {
struct ScalarFrameOrigin {
  // Runtime frame in the caller's view.
  loom::type::index frame;
  // Sample position within that frame.
  unsigned position;
};
LOOM_TEMPLATE_DECL("fft.deinterleave4")
Complex<Packet> deinterleave(Packet first, Packet second);
}  // namespace fft_experiment
#endif
