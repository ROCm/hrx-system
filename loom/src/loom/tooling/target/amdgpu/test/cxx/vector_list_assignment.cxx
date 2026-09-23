// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

using Words4 = unsigned __attribute__((ext_vector_type(4)));
using Floats4 = float __attribute__((vector_size(16)));
using Bytes4 = unsigned char __attribute__((ext_vector_type(4)));

struct State {
  // Vector payload updated through a member lvalue.
  Words4 value;
};

static Words4 identity(Words4 value) { return value; }

static unsigned record(unsigned* trace, unsigned digit, unsigned value) {
  *trace = *trace * 10u + digit;
  return value;
}

static Words4* destination(unsigned* trace, Words4* output) {
  record(trace, 5, 0);
  return output;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void assign_vector(const unsigned* input, Words4* output, unsigned* trace) {
  const unsigned first = input[0];
  *output = {first, 8u, 9u, 10u};
  Words4 local{11u, 12u, 13u, 14u};
  local = {first};
  output[1] = local;
  State state{};
  state.value = {first + 1u, first + 2u};
  output[2] = state.value;
  output[3] = {};
  Floats4 floats{};
  floats = {float(first), -0.0f, 1.5f};
  output[4] = __builtin_bit_cast(Words4, floats);
  // List elements execute in order, before evaluating the assignment target.
  *destination(trace, output + 5) = {
      record(trace, 1, first), record(trace, 2, first + 1u),
      record(trace, 3, first + 2u), record(trace, 4, first + 3u)};
  output[6] = identity({first, 8u});
  Bytes4 bytes{};
  bytes = {static_cast<unsigned char>(first), 255u, 128u};
  *reinterpret_cast<Bytes4*>(output + 7) = bytes;
}

LOOM_CHECK_CASE(braced_vector_assignment) {
  const auto input = loom::check::fill<unsigned, 1>(7u);
  const auto storage = loom::check::fill<unsigned, 40>(0xA5A5A5A5u);
  const auto output = loom::check::slice<32>(storage, 4);
  const auto trace = loom::check::fill<unsigned, 1>(0u);
  loom::check::launch<assign_vector>(input, output, trace);
  loom::check::expect_bitwise(loom::check::slice<1>(output, 0),
                              loom::check::fill<unsigned, 1>(7u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 1),
                              loom::check::fill<unsigned, 1>(8u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 2),
                              loom::check::fill<unsigned, 1>(9u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 3),
                              loom::check::fill<unsigned, 1>(10u));
  // Partial assignments replace every lane, including zero-filled tails.
  loom::check::expect_bitwise(loom::check::slice<1>(output, 4),
                              loom::check::fill<unsigned, 1>(7u));
  loom::check::expect_bitwise(loom::check::slice<3>(output, 5),
                              loom::check::fill<unsigned, 3>(0u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 8),
                              loom::check::fill<unsigned, 1>(8u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 9),
                              loom::check::fill<unsigned, 1>(9u));
  loom::check::expect_bitwise(loom::check::slice<6>(output, 10),
                              loom::check::fill<unsigned, 6>(0u));
  // Floating lanes retain their numeric conversion and signed zero.
  loom::check::expect_bitwise(loom::check::slice<1>(output, 16),
                              loom::check::fill<unsigned, 1>(0x40E00000u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 17),
                              loom::check::fill<unsigned, 1>(0x80000000u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 18),
                              loom::check::fill<unsigned, 1>(0x3FC00000u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 19),
                              loom::check::fill<unsigned, 1>(0u));
  loom::check::expect_bitwise(loom::check::slice<4>(output, 20),
                              loom::check::slice<4>(output, 0));
  loom::check::expect_bitwise(trace, loom::check::fill<unsigned, 1>(12345u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 24),
                              loom::check::fill<unsigned, 1>(7u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 25),
                              loom::check::fill<unsigned, 1>(8u));
  loom::check::expect_bitwise(loom::check::slice<2>(output, 26),
                              loom::check::fill<unsigned, 2>(0u));
  loom::check::expect_bitwise(loom::check::slice<1>(output, 28),
                              loom::check::fill<unsigned, 1>(0x0080FF07u));
  loom::check::expect_bitwise(loom::check::slice<3>(output, 29),
                              loom::check::fill<unsigned, 3>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 0),
                              loom::check::fill<unsigned, 4>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<4>(storage, 36),
                              loom::check::fill<unsigned, 4>(0xA5A5A5A5u));
  loom::check::expect_bitwise(input, loom::check::fill<unsigned, 1>(7u));
  loom::check::expect_event("device", "type", "asan_report", "count", 0);
}
