// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fft.h"

// A caller selects its exact transform length and storage types when importing
// this source; the number of complete, contiguous frames remains a runtime
// value.
void forward400(loom::type::buffer<float> input,
                loom::type::buffer<float> output,
                loom::type::buffer<float> table,
                loom::type::buffer<float> workspace, unsigned frames) {
  fft_experiment::cascade_transform<false, 400, 8>(input, output, table,
                                                   workspace, frames);
}

void inverse1024(loom::type::buffer<float> input,
                 loom::type::buffer<float> output,
                 loom::type::buffer<float> table,
                 loom::type::buffer<float> workspace, unsigned frames) {
  fft_experiment::cascade_transform<true, 1024, 8>(input, output, table,
                                                   workspace, frames);
}
