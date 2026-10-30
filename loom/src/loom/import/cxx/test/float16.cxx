// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

using Half4 = _Float16 __attribute__((ext_vector_type(4)));

[[loom::force_inline]] static Half4 scale(Half4 value, _Float16 factor) {
  return value * factor;
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void scale_half_vectors(const Half4* input, Half4* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  output[lane + 1] = scale(input[lane + 1], 5.0f16);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_half(const float* input, _Float16* output, float* expanded) {
  unsigned lane = loom::kernel::workitem::id.x;
  _Float16 value = input[lane];
  output[lane] = value;
  expanded[lane] = value;
}
