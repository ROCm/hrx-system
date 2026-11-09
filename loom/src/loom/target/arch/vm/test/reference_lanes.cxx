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
[[loom::noinline]] static void replace(unsigned& value) { value = 9u; }
unsigned widths(bool choose, unsigned field) {
  Short first{1u, 2u};
  Long second{3u, 4u, 5u, 6u};
  unsigned count = 0;
  unsigned& selected =
      choose ? first[tick(1u, count)] : second[tick(3u, count)];
  replace(selected);
  if (field == 0u) {
    return first[0] * 10u + first[1];
  }
  if (field == 1u) {
    return second[0] * 1000u + second[1] * 100u + second[2] * 10u + second[3];
  }
  return count;
}
unsigned nested(bool outer, bool inner, unsigned field) {
  Short first{1u, 2u};
  Long second{3u, 4u, 5u, 6u}, third{7u, 8u, 1u, 2u};
  unsigned count = 0;
  unsigned& selected =
      outer ? (inner ? first[tick(1u, count)] : second[tick(3u, count)])
            : third[tick(2u, count)];
  replace(selected);
  if (field == 0u) {
    return first[0] * 10u + first[1];
  }
  if (field == 1u) {
    return second[0] * 1000u + second[1] * 100u + second[2] * 10u + second[3];
  }
  if (field == 2u) {
    return third[0] * 1000u + third[1] * 100u + third[2] * 10u + third[3];
  }
  return count;
}
unsigned same_vector(bool choose) {
  Long lanes{1u, 2u, 3u, 4u};
  unsigned first = 1u;
  unsigned long long second = 3u;
  unsigned& selected = choose ? lanes[first] : lanes[second];
  first = 0u;
  second = 2u;
  replace(selected);
  return lanes[0] * 1000u + lanes[1] * 100u + lanes[2] * 10u + lanes[3];
}
unsigned scalar_lane(bool choose) {
  unsigned scalar = 5u;
  Long lanes{1u, 2u, 3u, 4u};
  unsigned& selected = choose ? scalar : lanes[3u];
  replace(selected);
  return scalar * 10000u + lanes[0] * 1000u + lanes[1] * 100u + lanes[2] * 10u +
         lanes[3];
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
  unsigned& selected = (choose ? first : second)[index];
  index = 0u;
  replace(selected);
  return first[0] * 1000u + first[1] * 100u + second[0] * 10u + second[1];
}
