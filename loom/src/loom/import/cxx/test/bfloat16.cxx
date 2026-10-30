// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include <stdfloat>

using std::bfloat16_t;
using std::float16_t;
using BFloat4 = bfloat16_t __attribute__((ext_vector_type(4)));

struct Packet {
  // Four independently scaled samples.
  BFloat4 samples;
  // Numeric BF16 factor shared by all samples.
  bfloat16_t gain;
};

[[loom::force_inline]] static BFloat4 scale(Packet value) {
  return value.samples * loom::scalar::absf(value.gain);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void scale_bfloat_vectors(const BFloat4* input, BFloat4* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  output[lane + 1] = scale(Packet{input[lane + 1], -2.0bf16});
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_bfloat(const float* input, bfloat16_t* output, float* expanded) {
  unsigned lane = loom::kernel::workitem::id.x;
  bfloat16_t value = input[lane];
  output[lane] = value;
  expanded[lane] = value;
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_formats(const float16_t* input_half, bfloat16_t* output_bfloat,
                     const bfloat16_t* input_bfloat, float16_t* output_half) {
  unsigned lane = loom::kernel::workitem::id.x;
  output_bfloat[lane] = (bfloat16_t)input_half[lane];
  output_half[lane] = (float16_t)input_bfloat[lane];
}
