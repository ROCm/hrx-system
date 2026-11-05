// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

struct Observation {
  // Value produced through the borrowed element pointer.
  unsigned value;
  // Element preceding the helper's writable range.
  unsigned before;
  // Element following the helper's writable range.
  unsigned after;
};

[[loom::inline]] static void update(unsigned* value, unsigned enabled) {
  if (enabled) {
    *value += 4;
  }
}

static Observation guarded(unsigned input, unsigned enabled) {
  unsigned values[3] = {37, input, 41};
  auto* element = values + 1;
  update(element, enabled);
  return {values[1], values[0], values[2]};
}

// Each clause can observe the preceding initialization, and list evaluation
// must update ordinary bindings in order. The last element is zero-initialized.
static unsigned ordered() {
  unsigned next = 3;
  unsigned values[4] = {next++, next++, values[0] + values[1]};
  return values[0] + 10 * values[1] + 100 * values[2] + 1000 * values[3] +
         10000 * next;
}

static unsigned value_initialized() {
  const unsigned empty[3]{};
  constexpr unsigned deduced[] = {3, 5};
  const unsigned parenthesized[3](7, 11);
  float converted[3] = {2, 3};
  unsigned short nested[2] = {{13}, {}};
  return empty[0] + empty[1] + empty[2] + deduced[0] + deduced[1] +
         parenthesized[0] + parenthesized[1] + parenthesized[2] +
         static_cast<unsigned>(converted[0] + converted[1] + converted[2]) +
         nested[0] + nested[1];
}

[[loom::inline]] static void fill(unsigned* values, unsigned count) {
  for (unsigned i = 0; i < count; ++i) {
    values[i] = i + 3;
  }
}

static unsigned output_parameter(unsigned index) {
  unsigned values[4];
  fill(values, 4);
  return values[index];
}

static unsigned indexed(int first, unsigned long long second) {
  const unsigned values[4] = {3, 5, 7, 11};
  return 10 * values[first] + values[second];
}

// The pointer names the whole array, with its complete stride. Alias writes
// must meet the same root as direct subscripts and array decay.
static unsigned whole_array() {
  unsigned values[3] = {3, 5, 7};
  auto* array = &values;
  (*array)[1] += 4;
  return values[0] + values[1] + values[2];
}

static unsigned repeated(unsigned count) {
  unsigned total = 0;
  for (unsigned i = 0; i < count; ++i) {
    unsigned values[2] = {i};
    if (i == 1) {
      continue;
    }
    update(&values[0], 1);
    total += values[0] + values[1];
  }
  return total;
}

static unsigned volatile_elements() {
  volatile unsigned values[3] = {7};
  values[1] = values[0] + 3;
  return values[0] + values[1] + values[2];
}

LOOM_CHECK_CASE(array_aliases) {
  const auto unchanged = guarded(7, 0);
  const auto changed = guarded(7, 1);
  const auto wrapped = guarded(0xfffffffeu, 1);
  const auto whole = whole_array();
  loom::check::expect_equal(unchanged.value, 7u);
  loom::check::expect_equal(changed.value, 11u);
  loom::check::expect_equal(wrapped.value, 2u);
  loom::check::expect_equal(unchanged.before, 37u);
  loom::check::expect_equal(changed.before, 37u);
  loom::check::expect_equal(wrapped.before, 37u);
  loom::check::expect_equal(unchanged.after, 41u);
  loom::check::expect_equal(changed.after, 41u);
  loom::check::expect_equal(wrapped.after, 41u);
  loom::check::expect_equal(whole, 19u);
}

LOOM_CHECK_CASE(array_initialization) {
  const auto sequence = ordered();
  const auto zeros = value_initialized();
  const auto first = output_parameter(0);
  const auto last = output_parameter(3);
  const auto indices = indexed(2, 3);
  const auto fresh = repeated(4);
  const auto empty = repeated(0);
  const auto observed = volatile_elements();
  loom::check::expect_equal(sequence, 50743u);
  loom::check::expect_equal(zeros, 44u);
  loom::check::expect_equal(first, 3u);
  loom::check::expect_equal(last, 6u);
  loom::check::expect_equal(indices, 81u);
  loom::check::expect_equal(fresh, 17u);
  loom::check::expect_equal(empty, 0u);
  loom::check::expect_equal(observed, 17u);
}
