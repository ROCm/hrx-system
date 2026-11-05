// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

LOOM_INLINE static unsigned sum(unsigned count, unsigned factor) {
  unsigned total = 0;
  [[loom::unroll(factor)]]
  for (unsigned index = 0; index < count; ++index) {
    total += index;
  }
  return total;
}

unsigned schedule_call(unsigned count) {
  unsigned factor = 4;
  --factor;
  return sum(count, factor);
}

unsigned schedule_snapshot(unsigned count) {
  unsigned factor = 3;
  unsigned total = 0;
  [[loom::unroll(factor)]]
  for (unsigned index = 0; index < count; ++index) {
    total += index;
    ++factor;
  }
  return total + factor;
}

unsigned schedule_wide(unsigned count) {
  unsigned long long factor = (1ULL << 32) + 3;
  unsigned total = 0;
  [[loom::unroll(factor - (1ULL << 32))]]
  for (unsigned index = 0; index < count; ++index) {
    total += index;
  }
  return total;
}

unsigned schedule_narrow(unsigned count) {
  unsigned char factor = 131;
  factor -= 128;
  unsigned total = 0;
  [[loom::unroll(factor)]]
  for (unsigned index = 0; index < count; ++index) {
    total += index;
  }
  return total;
}

unsigned schedule_signed(int count) {
  signed char factor = -1;
  factor += 4;
  unsigned total = 0;
  [[loom::unroll(factor)]]
  for (int index = 0; index < count; ++index) {
    total += index;
  }
  return total;
}

unsigned schedule_unevaluated(unsigned count) {
  unsigned total = 0;
  [[loom::unroll(sizeof(++count) / alignof(unsigned))]]
  for (unsigned index = 0; index < count; ++index) {
    total += index;
  }
  return total + count + sizeof(long) + 'A';
}

unsigned schedule_initializer(unsigned count) {
  int factor = -1;
  unsigned total = 0;
  [[loom::unroll(factor)]]
  for (unsigned index = ++factor; index < count; ++index) {
    total += index;
  }
  return total + factor;
}

unsigned schedule_serial(unsigned count) { return sum(count, 0); }

unsigned schedule_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value < 6u) {
    return trial_value;
  }
  if (trial_value == 6u) {
    return 7u;
  }
  if (trial_value == 7u) {
    return 16u;
  }
  if (trial_value == 8u) {
    return 17u;
  }
  return 33u;
}

void schedule_observations(unsigned count,
                           loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<8>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(schedule_call(count), values, 0);
  loom::view::store(schedule_snapshot(count), values, 1);
  loom::view::store(schedule_wide(count), values, 2);
  loom::view::store(schedule_narrow(count), values, 3);
  loom::view::store(schedule_signed(count), values, 4);
  loom::view::store(schedule_unevaluated(count), values, 5);
  loom::view::store(schedule_initializer(count), values, 6);
  loom::view::store(schedule_serial(count), values, 7);
}

void schedule_oracle(unsigned count, loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<8>(output, {}, loom::encoding::layout::dense<1>());
  unsigned sum = count * (count - 1u) / 2u;
  loom::view::store(sum, values, 0);
  loom::view::store(sum + count + 3u, values, 1);
  loom::view::store(sum, values, 2);
  loom::view::store(sum, values, 3);
  loom::view::store(sum, values, 4);
  loom::view::store(sum + count + 73u, values, 5);
  loom::view::store(sum, values, 6);
  loom::view::store(sum, values, 7);
}

LOOM_CHECK_SCENARIO(schedule_values_functions) {
  loom::check::trial<10>([](loom::check::ordinal trial,
                            loom::check::entropy entropy) {
    const auto count = loom::check::generate<schedule_input>(trial);
    const auto output = loom::check::fill<8, unsigned>(-123u);
    loom::check::compare<schedule_observations, schedule_oracle>(
        count, output, [&] { loom::check::expect_bitwise(output, output); });
  });
}
