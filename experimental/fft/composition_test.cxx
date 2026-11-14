// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fft.h"

// Defined by a separately imported translation unit using the same headers.
void forward_test(loom::type::buffer<float> input,
                  loom::type::buffer<float> output,
                  loom::type::buffer<float> table,
                  loom::type::buffer<float> workspace, unsigned frames);

void composed_test(loom::type::buffer<float> input,
                   loom::type::buffer<float> output,
                   loom::type::buffer<float> table,
                   loom::type::buffer<float> workspace,
                   loom::type::buffer<float> intermediate, unsigned frames) {
  forward_test(input, intermediate, table, workspace, frames);
  fft_experiment::cascade_transform<true, 10, 2>(intermediate, output, table,
                                                 workspace, frames);
}
