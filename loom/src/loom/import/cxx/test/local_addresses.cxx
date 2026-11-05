// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

[[loom::inline]] static void increment(unsigned* value) { ++*value; }
[[loom::inline]] static unsigned* identity(unsigned* value) { return value; }
[[loom::inline]] static void initialize(unsigned* value, unsigned input) {
  *value = input + 7;
}

// Direct writes before the first address expression must update the same object
// that the helper sees. Taking an address only on one path still owns one
// object.
static unsigned conditional(unsigned input, unsigned choice) {
  unsigned value = input;
  if (choice) {
    value += 3;
    increment(&value);
  }
  return value;
}

// Borrowed helper results and ordinary pointer copies preserve alias identity.
static unsigned aliases(unsigned input) {
  unsigned value = input;
  auto* first = &value;
  auto* second = identity(first);
  *first += 3;
  increment(second);
  return value + *first + *second;
}

// An output-only helper can initialize a declared object without a dummy store.
static unsigned output_parameter(unsigned input) {
  unsigned value;
  initialize(&value, input);
  return value;
}

static unsigned parameter(unsigned value) {
  increment(&value);
  return value;
}

// An aliased bound cannot be hoisted, even without a syntactic write to limit.
static unsigned mutable_bound(unsigned limit) {
  auto* bound = &limit;
  unsigned total = 0;
  for (unsigned i = 0; i < limit; ++i) {
    total += i;
    if (i == 1) {
      *bound = 2;
    }
  }
  return total;
}

// Both the body helper and the source increment change the induction object.
static unsigned mutable_induction(unsigned limit) {
  unsigned total = 0;
  for (unsigned i = 0; i < limit; ++i) {
    total += i;
    increment(&i);
  }
  return total;
}

static unsigned repeated_declaration(unsigned limit) {
  unsigned total = 0;
  for (unsigned i = 0; i < limit; ++i) {
    unsigned value = i;
    increment(&value);
    total += value;
  }
  return total;
}

// The condition recreates its value before every check, including the final
// false check. A continue still performs that next initialization.
static unsigned decision(unsigned count) {
  unsigned checks = 0;
  unsigned total = 0;
  while (unsigned value = (++checks, count--)) {
    increment(&value);
    if (value == 3) {
      continue;
    }
    total += value;
  }
  return total + 100 * checks;
}

// Atomic observation prevents promotion but shares the same local object with
// ordinary reads and the compare-exchange expected-value output parameter.
static unsigned atomic_local(unsigned input) {
  unsigned value = input;
  unsigned expected = input + 1;
  bool failed = __atomic_compare_exchange_n(&value, &expected, 41u, false,
                                            __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  bool succeeded = __atomic_compare_exchange_n(
      &value, &expected, 41u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return value + expected + (failed ? 1000u : 0u) + (succeeded ? 100u : 0u);
}

using Words = unsigned __attribute__((vector_size(16)));
[[loom::inline]] static void update_vector(Words* value) {
  *value += Words{1, 2, 3, 4};
}
static unsigned vector_local(unsigned input) {
  Words value = {input, input, input, input};
  update_vector(&value);
  return value[0] + value[1] + value[2] + value[3];
}

LOOM_CHECK_CASE(local_aliases) {
  const auto false_arm = conditional(37, 0);
  const auto true_arm = conditional(37, 1);
  const auto aliased = aliases(37);
  const auto initialized = output_parameter(37);
  const auto parameter_copy = parameter(37);
  const auto wrapped = parameter(0xffffffffu);
  const auto atomic = atomic_local(37);
  const auto vector = vector_local(37);
  loom::check::expect_equal(false_arm, 37u);
  loom::check::expect_equal(true_arm, 41u);
  loom::check::expect_equal(aliased, 123u);
  loom::check::expect_equal(initialized, 44u);
  loom::check::expect_equal(parameter_copy, 38u);
  loom::check::expect_equal(wrapped, 0u);
  loom::check::expect_equal(atomic, 178u);
  loom::check::expect_equal(vector, 158u);
}

LOOM_CHECK_CASE(local_lifetimes) {
  const auto changed_bound = mutable_bound(8);
  const auto changed_induction = mutable_induction(8);
  const auto fresh_locals = repeated_declaration(5);
  const auto empty_locals = repeated_declaration(0);
  const auto fresh_decisions = decision(3);
  const auto empty_decisions = decision(0);
  loom::check::expect_equal(changed_bound, 1u);
  loom::check::expect_equal(changed_induction, 12u);
  loom::check::expect_equal(fresh_locals, 15u);
  loom::check::expect_equal(empty_locals, 0u);
  loom::check::expect_equal(fresh_decisions, 406u);
  loom::check::expect_equal(empty_decisions, 100u);
}
