// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

using Short = unsigned __attribute__((ext_vector_type(2)));
using Long = unsigned __attribute__((ext_vector_type(4)));
[[loom::noinline]] static unsigned tick(unsigned value, unsigned& count) {
  ++count;
  return value;
}
// Extended-vector components are non-addressable designations: a conditional
// can read their selected value, but cannot produce an ordinary C++ reference.
unsigned widths(bool choose, unsigned observation) {
  Short first{1u, 2u};
  Long second{3u, 4u, 5u, 6u};
  unsigned count = 0;
  const unsigned selected =
      choose ? first[tick(1u, count)] : second[tick(3u, count)];
  return observation == 0u ? selected : count;
}
unsigned nested(bool outer, bool inner, unsigned observation) {
  Short first{1u, 2u};
  Long second{3u, 4u, 5u, 6u}, third{7u, 8u, 1u, 2u};
  unsigned count = 0;
  const unsigned selected =
      outer ? (inner ? first[tick(1u, count)] : second[tick(3u, count)])
            : third[tick(2u, count)];
  return observation == 0u ? selected : count;
}
unsigned same_vector(bool choose) {
  Long lanes{1u, 2u, 3u, 4u};
  unsigned first = 1u;
  unsigned long long second = 3u;
  return choose ? lanes[first] : lanes[second];
}
unsigned scalar_lane(bool choose) {
  unsigned scalar = 5u;
  Long lanes{1u, 2u, 3u, 4u};
  return choose ? scalar : lanes[3u];
}
unsigned memory_assignment(bool choose) {
  unsigned scalar = 5u;
  unsigned memory[2]{1u, 2u};
  unsigned index = 1u;
  (choose ? scalar : memory[index--]) += 4u;
  return scalar * 1000u + memory[0] * 100u + memory[1] * 10u + index;
}
unsigned vector_base(bool choose) {
  Long first{1u, 2u, 3u, 4u}, second{5u, 6u, 7u, 8u};
  unsigned index = 1u;
  return (choose ? first : second)[index];
}
