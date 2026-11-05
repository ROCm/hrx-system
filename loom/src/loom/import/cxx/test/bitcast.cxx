// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));
using Bytes8 = unsigned char __attribute__((ext_vector_type(8)));
using Halves2 = _Float16 __attribute__((ext_vector_type(2)));
using Bfloats2 = __bf16 __attribute__((ext_vector_type(2)));
using Float8E4 = __float8_e4m3fn __attribute__((ext_vector_type(4)));
using Float8E5 = __float8_e5m2 __attribute__((ext_vector_type(4)));
using Words2 = unsigned __attribute__((ext_vector_type(2)));

[[loom::inline]] void bitcast_blocks(const unsigned* input, unsigned* output) {
  for (unsigned index = 0; index < 256; ++index) {
    unsigned word = input[index];
    Bytes4 bytes = __builtin_bit_cast(Bytes4, word);
    Bytes4 reversed{bytes[3], bytes[2], bytes[1], bytes[0]};
    output[index * 8] =
        __builtin_bit_cast(unsigned, reversed + Bytes4{1, 2, 3, 4});
    Halves2 half = __builtin_bit_cast(Halves2, word);
    output[index * 8 + 1] =
        __builtin_bit_cast(unsigned, Halves2{half[1], half[0]});
    Bfloats2 bfloat = __builtin_bit_cast(Bfloats2, word);
    output[index * 8 + 2] =
        __builtin_bit_cast(unsigned, Bfloats2{bfloat[1], bfloat[0]});
    Float8E4 e4m3 = __builtin_bit_cast(Float8E4, word);
    output[index * 8 + 3] = __builtin_bit_cast(
        unsigned, Float8E4{e4m3[1], e4m3[3], e4m3[0], e4m3[2]});
    Float8E5 e5m2 = __builtin_bit_cast(Float8E5, word);
    output[index * 8 + 4] = __builtin_bit_cast(
        unsigned, Float8E5{e5m2[3], e5m2[2], e5m2[1], e5m2[0]});
    unsigned long long wide = word;
    wide |= static_cast<unsigned long long>(input[(index + 1) & 255u]) << 32;
    Bytes8 wide_bytes = __builtin_bit_cast(Bytes8, wide);
    Bytes8 wide_reversed{wide_bytes[7], wide_bytes[6], wide_bytes[5],
                         wide_bytes[4], wide_bytes[3], wide_bytes[2],
                         wide_bytes[1], wide_bytes[0]};
    double payload = __builtin_bit_cast(double, wide_reversed);
    Words2 words = __builtin_bit_cast(Words2, payload);
    output[index * 8 + 5] = words[0];
    output[index * 8 + 6] = words[1];
    // Mix constexpr signaling-NaN and signed-zero payloads with runtime lanes.
    constexpr auto constants = __builtin_bit_cast(Float8E5, 0x00817d80u);
    Float8E5 mixed{constants[1], e5m2[1], constants[0], e5m2[3]};
    output[index * 8 + 7] = __builtin_bit_cast(unsigned, mixed);
  }
}
