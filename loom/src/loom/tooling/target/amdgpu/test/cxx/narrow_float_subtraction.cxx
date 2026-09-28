// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

#include <stdfloat>

using Half8 = std::float16_t __attribute__((ext_vector_type(8)));
using Brain8 = std::bfloat16_t __attribute__((ext_vector_type(8)));
using Float8 = float __attribute__((ext_vector_type(8)));

// Widen the narrow result after subtraction so the checks observe its rounding.
[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void subtract_f16_8(const Half8* lhs, const Half8* rhs, Float8* output,
                    unsigned count) {
  for (unsigned packet = loom::workitem_id.x; packet < count / 8;
       packet += 64) {
    Half8 difference = lhs[packet] - rhs[packet];
    output[packet] = __builtin_convertvector(difference, Float8);
  }
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void subtract_bf16_8(const Brain8* lhs, const Brain8* rhs, Float8* output,
                     unsigned count) {
  for (unsigned packet = loom::workitem_id.x; packet < count / 8;
       packet += 64) {
    Brain8 difference = lhs[packet] - rhs[packet];
    output[packet] = __builtin_convertvector(difference, Float8);
  }
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void subtract_f16_1(const std::float16_t* lhs, const std::float16_t* rhs,
                    float* output, unsigned count) {
  for (unsigned position = loom::workitem_id.x; position < count;
       position += 64) {
    std::float16_t difference = lhs[position] - rhs[position];
    output[position] = difference;
  }
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void subtract_bf16_1(const std::bfloat16_t* lhs, const std::bfloat16_t* rhs,
                     float* output, unsigned count) {
  for (unsigned position = loom::workitem_id.x; position < count;
       position += 64) {
    std::bfloat16_t difference = lhs[position] - rhs[position];
    output[position] = difference;
  }
}
