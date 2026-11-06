// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <loomcxx/check.h>
#include <loomcxx/view.h>

__global__
    [[loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]] void
    hip_vector_values(const unsigned* input, unsigned* output) {
  uint4 words = *reinterpret_cast<const uint4*>(input);
  words.x = words.w + words.y;
  words.y ^= input[4];
  words.z += words.y;
  words.w -= words.x;
  *reinterpret_cast<uint4*>(output) = words;

  uint2 pair = make_uint2(input[4], input[2]);
  pair.x ^= pair.y;
  pair.y += input[1];
  *reinterpret_cast<uint2*>(output + 4) = pair;

  uint3 triple = make_uint3(input[1], input[2], input[3]);
  triple.x += 6u;
  triple.y -= triple.x;
  triple.z ^= triple.y;
  output[6] = triple.x;
  output[7] = triple.y;
  output[8] = triple.z;

  uchar1 byte = make_uchar1((unsigned char)input[3]);
  byte.x += 2u;
  output[9] = byte.x;

  short3 signed_values = make_short3(-32768, -1, 32767);
  signed_values.y += 2;
  signed_values.z -= 7;
  output[10] = (unsigned)(int)signed_values.x;
  output[11] = (unsigned)(int)signed_values.y;
  output[12] = (unsigned)(int)signed_values.z;

  ulonglong2 wide =
      make_ulonglong2(((unsigned long long)input[2] << 32u) | input[3],
                      ((unsigned long long)input[4] << 32u) | input[1]);
  wide.x ^= wide.y;
  wide.y += 0xffffffffull;
  output[13] = (unsigned)wide.x;
  output[14] = (unsigned)(wide.x >> 32u);
  output[15] = (unsigned)wide.y;
  output[16] = (unsigned)(wide.y >> 32u);

  float4 floating = make_float4((float)(int)input[1], 2.0f, -3.5f, 4.0f);
  floating.x = -floating.x;
  floating.w += floating.z;
  output[17] = __builtin_bit_cast(unsigned, floating.x);
  output[18] = __builtin_bit_cast(unsigned, floating.w);

  double2 doubles = make_double2((double)input[1], -0.0);
  doubles.x += 0.5;
  unsigned long long double_bits =
      __builtin_bit_cast(unsigned long long, doubles.y);
  output[19] = (unsigned)(doubles.x * 2.0);
  output[20] = (unsigned)double_bits;
  output[21] = (unsigned)(double_bits >> 32u);

  longlong1 signed_wide = make_longlong1(-9223372036854775807ll);
  signed_wide.x += 7;
  output[22] = signed_wide.x < 0 ? 1u : 0u;

  ulonglong4 counters = make_ulonglong4(input[1], 0xffffffffull, 0x100000000ull,
                                        0xffffffffffffffffull);
  counters.w += counters.x;
  output[23] = (unsigned)counters.w;
}

static void hip_vector_input(loom::type::buffer<unsigned> input) {
  auto values =
      loom::buffer::view<8>(input, {}, loom::encoding::layout::dense<1>());
  loom::view::store(0u, values, 0);
  loom::view::store(1u, values, 1);
  loom::view::store(0x7fffffffu, values, 2);
  loom::view::store(0xffffffffu, values, 3);
  loom::view::store(0x80000000u, values, 4);
  loom::view::store(0x3f800000u, values, 5);
  loom::view::store(0x80000000u, values, 6);
  loom::view::store(0x40000000u, values, 7);
}

static void hip_vector_oracle(loom::type::buffer<unsigned> input,
                              loom::type::buffer<unsigned> output) {
  auto source =
      loom::buffer::view<8>(input, {}, loom::encoding::layout::dense<1>());
  auto result =
      loom::buffer::view<24>(output, {}, loom::encoding::layout::dense<1>());
  unsigned input1 = loom::view::load(source, 1);
  unsigned input2 = loom::view::load(source, 2);
  unsigned input3 = loom::view::load(source, 3);
  unsigned input4 = loom::view::load(source, 4);

  unsigned word0 = input3 + input1;
  unsigned word1 = input1 ^ input4;
  unsigned word2 = input2 + word1;
  unsigned word3 = input3 - word0;
  loom::view::store(word0, result, 0);
  loom::view::store(word1, result, 1);
  loom::view::store(word2, result, 2);
  loom::view::store(word3, result, 3);

  loom::view::store(input4 ^ input2, result, 4);
  loom::view::store(input2 + input1, result, 5);
  unsigned triple0 = input1 + 6u;
  unsigned triple1 = input2 - triple0;
  loom::view::store(triple0, result, 6);
  loom::view::store(triple1, result, 7);
  loom::view::store(input3 ^ triple1, result, 8);
  loom::view::store((unsigned)(unsigned char)((unsigned char)input3 + 2u),
                    result, 9);
  loom::view::store((unsigned)(int)(short)-32768, result, 10);
  loom::view::store(1u, result, 11);
  loom::view::store(32760u, result, 12);

  unsigned long long wide0 = ((unsigned long long)input2 << 32u) | input3;
  unsigned long long wide1 = ((unsigned long long)input4 << 32u) | input1;
  wide0 ^= wide1;
  wide1 += 0xffffffffull;
  loom::view::store((unsigned)wide0, result, 13);
  loom::view::store((unsigned)(wide0 >> 32u), result, 14);
  loom::view::store((unsigned)wide1, result, 15);
  loom::view::store((unsigned)(wide1 >> 32u), result, 16);
  loom::view::store(0xbf800000u, result, 17);
  loom::view::store(0x3f000000u, result, 18);
  loom::view::store(3u, result, 19);
  loom::view::store(0u, result, 20);
  loom::view::store(0x80000000u, result, 21);
  loom::view::store(1u, result, 22);
  loom::view::store(0u, result, 23);
}

LOOM_CHECK_SCENARIO(hip_vector_type_values) {
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<8, unsigned>(0u);
        loom::check::generate<hip_vector_input>(input);
        const auto output = loom::check::fill<24, unsigned>(0xdeadbeefu);
        loom::check::compare<hip_vector_values, hip_vector_oracle>(
            input, output, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(output, output);
            });
      });
}
