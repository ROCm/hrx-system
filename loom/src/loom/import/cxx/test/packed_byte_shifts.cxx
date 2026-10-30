// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

using Bytes8 = unsigned char __attribute__((ext_vector_type(8)));
using SignedBytes8 = signed char __attribute__((ext_vector_type(8)));

// Distinct neighboring bytes exercise both physical words at every shift count.
[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(8, 1, 1)]]
void packed_byte_shifts(const unsigned char* input_bytes,
                        unsigned char* output_bytes) {
  const Bytes8* input = reinterpret_cast<const Bytes8*>(input_bytes);
  Bytes8* output = reinterpret_cast<Bytes8*>(output_bytes);
  unsigned index =
      loom::kernel::workgroup::id.x * 32 + loom::kernel::workitem::id.x;
  Bytes8 values = input[index];
  [[loom::unroll(8)]]
  for (unsigned amount = 0; amount < 8; ++amount) {
    output[index * 19 + amount * 2] = values << amount;
    output[index * 19 + amount * 2 + 1] = values >> amount;
  }
  // A shared bound does not imply that every byte has the same shift amount.
  Bytes8 amounts = values & 7;
  output[index * 19 + 16] = values << amounts;
  output[index * 19 + 17] = values >> amounts;
  // The left shift can set the sign bit; its facts must preserve this
  // comparison.
  Bytes8 sign_bits = (values & 1) << 7;
  output[index * 19 + 18] = __builtin_convertvector(
      __builtin_convertvector(sign_bits, SignedBytes8) < 0, Bytes8);
}

void packed_byte_shift_input(loom::type::buffer<unsigned char> input) {
  auto values =
      loom::buffer::view<256, 8>(input, {}, loom::encoding::layout::dense<2>());
  for (unsigned packet = 0; packet < 256u; ++packet) {
    for (unsigned lane = 0; lane < 8u; ++lane) {
      loom::view::store((unsigned char)(packet + lane * 73u), values, packet,
                        lane);
    }
  }
}

void packed_byte_shift_oracle(loom::type::buffer<unsigned char> input,
                              loom::type::buffer<unsigned char> output) {
  auto input_values =
      loom::buffer::view<256, 8>(input, {}, loom::encoding::layout::dense<2>());
  auto output_values = loom::buffer::view<256, 19, 8>(
      output, {}, loom::encoding::layout::dense<3>());
  for (unsigned packet = 0; packet < 256u; ++packet) {
    for (unsigned lane = 0; lane < 8u; ++lane) {
      unsigned char value = loom::view::load(input_values, packet, lane);
      for (unsigned amount = 0; amount < 8u; ++amount) {
        loom::view::store((unsigned char)(value << amount), output_values,
                          packet, amount * 2u, lane);
        loom::view::store((unsigned char)(value >> amount), output_values,
                          packet, amount * 2u + 1u, lane);
      }
      unsigned amount = value & 7u;
      loom::view::store((unsigned char)(value << amount), output_values, packet,
                        16u, lane);
      loom::view::store((unsigned char)(value >> amount), output_values, packet,
                        17u, lane);
      loom::view::store((unsigned char)((value & 1u) ? 0xffu : 0u),
                        output_values, packet, 18u, lane);
    }
  }
}

LOOM_CHECK_SCENARIO(packed_byte_shift_values) {
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto input = loom::check::fill<2048, unsigned char>(0);
        loom::check::generate<packed_byte_shift_input>(input);
        const auto storage =
            loom::check::fill<38944, unsigned char>((unsigned char)-123);
        const auto output = loom::check::slice<38912>(storage, 16);
        loom::check::compare<packed_byte_shifts, packed_byte_shift_oracle>(
            input, output, [&] {
              loom::check::expect_bitwise(input, input);
              loom::check::expect_bitwise(storage, storage);
            });
      });
}
