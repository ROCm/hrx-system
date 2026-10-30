// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

#include <stdfloat>

using loom::type::float8_e4m3fn_t;
using loom::type::float8_e5m2_t;
using Float8E4M3FNx4 = float8_e4m3fn_t __attribute__((ext_vector_type(4)));
using Float8E5M2x4 = float8_e5m2_t __attribute__((ext_vector_type(4)));

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void encode_float8(const float* input, float8_e4m3fn_t* e4m3,
                   float8_e5m2_t* e5m2) {
  unsigned lane = loom::kernel::workitem::id.x;
  e4m3[lane] = input[lane];
  e5m2[lane] = input[lane];
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void decode_float8(const float8_e4m3fn_t* e4m3, const float8_e5m2_t* e5m2,
                   std::bfloat16_t* bfloat, std::float16_t* half,
                   float8_e5m2_t* converted_e5m2,
                   float8_e4m3fn_t* converted_e4m3) {
  unsigned lane = loom::kernel::workitem::id.x;
  bfloat[lane] = e4m3[lane];
  half[lane] = e5m2[lane];
  converted_e5m2[lane] = static_cast<float8_e5m2_t>(e4m3[lane]);
  converted_e4m3[lane] = static_cast<float8_e4m3fn_t>(e5m2[lane]);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void copy_float8_vectors(const Float8E4M3FNx4* input, Float8E4M3FNx4* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  output[lane + 1] = input[lane + 1];
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void constant_float8_vectors(Float8E4M3FNx4* e4m3, Float8E5M2x4* e5m2) {
  unsigned lane = loom::kernel::workitem::id.x;
  constexpr float8_e4m3fn_t even = 1.0625;
  constexpr float8_e4m3fn_t odd = 1.1875;
  constexpr float8_e4m3fn_t saturated = 1000.0;
  constexpr float8_e5m2_t infinity = 61440.0;
  e4m3[lane] = Float8E4M3FNx4{even, odd, saturated, float8_e4m3fn_t(-0.0)};
  e5m2[lane] = Float8E5M2x4{float8_e5m2_t(1.125), float8_e5m2_t(1.375),
                            infinity, float8_e5m2_t(-0.0)};
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void constant_float8_midpoints(Float8E4M3FNx4* e4m3, Float8E5M2x4* e5m2) {
  unsigned lane = loom::kernel::workitem::id.x;
  constexpr float8_e4m3fn_t above_e4 = 0x1.1000000000001p0;
  constexpr float8_e4m3fn_t below_e4 = 0x1.2ffffffffffffp0;
  constexpr float8_e5m2_t above_e5 = 0x1.2000000000001p0;
  constexpr float8_e5m2_t below_e5 = 0x1.5ffffffffffffp0;
  e4m3[lane] = Float8E4M3FNx4{above_e4, float8_e4m3fn_t(0x1.1000000000001p0),
                              below_e4, float8_e4m3fn_t(0x1.2ffffffffffffp0)};
  e5m2[lane] = Float8E5M2x4{above_e5, float8_e5m2_t(0x1.2000000000001p0),
                            below_e5, float8_e5m2_t(0x1.5ffffffffffffp0)};
}
