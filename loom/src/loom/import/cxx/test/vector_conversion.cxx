// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>

#include <stdfloat>

using Float4 = float __attribute__((ext_vector_type(4)));
using Int4 = int __attribute__((ext_vector_type(4)));
using UInt4 = unsigned __attribute__((ext_vector_type(4)));
using Byte4 = unsigned char __attribute__((ext_vector_type(4)));
using SignedByte4 = signed char __attribute__((ext_vector_type(4)));
using Half2 = std::float16_t __attribute__((ext_vector_type(2)));
using BFloat2 = std::bfloat16_t __attribute__((ext_vector_type(2)));
using Float8x4 =
    loom::type::float8_e4m3fn_t __attribute__((ext_vector_type(4)));
using Float8E5x4 =
    loom::type::float8_e5m2_t __attribute__((ext_vector_type(4)));

template <typename To, typename From>
static constexpr To convert(From value) {
  return __builtin_convertvector(value, To);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void scale_float8(const Float8x4* input, Float8x4* output,
                  Float8E5x4* other_format) {
  unsigned lane = loom::kernel::workitem::id.x;
  Float4 wide = convert<Float4>(input[lane]);
  output[lane] = convert<Float8x4>(wide * 1.0625f);
  other_format[lane] = convert<Float8E5x4>(input[lane]);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_integer_vectors(const Byte4* input, Int4* unsigned_widened,
                             Int4* signed_widened, Float4* unsigned_float,
                             Float4* signed_float) {
  unsigned lane = loom::kernel::workitem::id.x;
  auto bytes = input[lane];
  auto signed_bytes = (SignedByte4)bytes;
  unsigned_widened[lane] = convert<Int4>(bytes);
  signed_widened[lane] = convert<Int4>(signed_bytes);
  unsigned_float[lane] = convert<Float4>(bytes);
  signed_float[lane] = convert<Float4>(signed_bytes);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void truncate_float_vectors(const Float4* input, Int4* signed_integer,
                            UInt4* unsigned_integer, Byte4* bytes) {
  unsigned lane = loom::kernel::workitem::id.x;
  signed_integer[lane] = convert<Int4>(input[lane]);
  unsigned_integer[lane] = convert<UInt4>(input[lane]);
  bytes[lane] = convert<Byte4>(convert<Int4>(input[lane]));
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void convert_half_vectors(const Half2* input, BFloat2* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  output[lane] = convert<BFloat2>(input[lane]);
}
