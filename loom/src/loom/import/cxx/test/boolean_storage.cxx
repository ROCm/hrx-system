// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>

[[loom::inline]] static void invert(bool* value) { *value = !*value; }
[[loom::inline]] static bool* identity(bool* value) { return value; }
[[loom::inline]] static void initialize(bool* value, unsigned input) {
  *value = input;
}

static bool parameter(bool value) {
  invert(&value);
  return value;
}

static unsigned aliases(unsigned input) {
  bool value;
  initialize(&value, input);
  auto* first = &value;
  auto* second = identity(first);
  *first = !value;
  return unsigned(value) + 2 * unsigned(*first) + 4 * unsigned(*second);
}

static bool conditional(bool input, bool enabled) {
  bool value = input;
  if (enabled) {
    invert(&value);
  }
  return value;
}

static bool array_value(unsigned input, unsigned index) {
  bool values[4] = {bool(input), true};
  auto* alias = values + 1;
  *alias = !values[0];
  return values[index];
}

static bool observed(unsigned input) {
  volatile bool value = input;
  (void)value;
  value = !value;
  return value;
}

enum class Flag : bool { no, yes };

static unsigned enum_array(bool input, unsigned index) {
  Flag values[3] = {static_cast<Flag>(input), Flag::yes};
  return static_cast<unsigned>(values[index]);
}

struct Flags {
  // Guard preceding the Boolean fields.
  unsigned char prefix;
  // Independent Boolean object.
  bool enabled;
  // Nominal enum with Boolean storage.
  Flag selected;
  // Consecutive Boolean objects with byte stride.
  bool lanes[3];
  // Guard following the Boolean fields.
  unsigned char suffix;
};
static_assert(sizeof(Flags) == 7);
static_assert(alignof(Flags) == 1);

bool boolean_transfer(volatile bool* destination, const volatile bool* source) {
  bool value = *source;
  *destination = value;
  return value;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void boolean_objects(const volatile bool* input, volatile bool* output,
                     unsigned number) {
  bool value = boolean_transfer(output + 1, input);
  invert(&value);
  output[2] = value;
  output[3] = array_value(number, 2);
  bool initialized;
  initialize(&initialized, number);
  output[4] = initialized;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void boolean_records(const Flags* input, Flags* output) {
  output[1].enabled = !input[1].enabled;
  output[1].selected = input[1].selected;
  output[1].lanes[0] = input[0].lanes[2];
  output[1].lanes[1] = !input[1].lanes[0];
  output[1].lanes[2] = true;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void boolean_shared(const bool* input, bool* output) {
  LOOM_WORKGROUP bool slots[2];
  slots[0] = input[0];
  slots[1] = !slots[0];
  output[1] = slots[0];
  output[2] = slots[1];
}

LOOM_CHECK_CASE(boolean_locals) {
  const auto inverted_false = parameter(false);
  const auto inverted_true = parameter(true);
  const auto zero_alias = aliases(0);
  const auto even_alias = aliases(256);
  const auto disabled = conditional(true, false);
  const auto enabled = conditional(true, true);
  const auto observed_zero = observed(0);
  const auto observed_even = observed(2);
  loom::check::expect_equal(inverted_false, true);
  loom::check::expect_equal(inverted_true, false);
  loom::check::expect_equal(zero_alias, 7u);
  loom::check::expect_equal(even_alias, 0u);
  loom::check::expect_equal(disabled, true);
  loom::check::expect_equal(enabled, false);
  loom::check::expect_equal(observed_zero, true);
  loom::check::expect_equal(observed_even, false);
}

LOOM_CHECK_CASE(boolean_arrays) {
  const auto false_first = array_value(0, 0);
  const auto false_alias = array_value(0, 1);
  const auto true_first = array_value(256, 0);
  const auto true_alias = array_value(256, 1);
  const auto zero_third = array_value(256, 2);
  const auto zero_fourth = array_value(256, 3);
  const auto enum_false = enum_array(false, 0);
  const auto enum_true = enum_array(true, 0);
  const auto enum_named = enum_array(false, 1);
  const auto enum_zero = enum_array(true, 2);
  loom::check::expect_equal(false_first, false);
  loom::check::expect_equal(false_alias, true);
  loom::check::expect_equal(true_first, true);
  loom::check::expect_equal(true_alias, false);
  loom::check::expect_equal(zero_third, false);
  loom::check::expect_equal(zero_fourth, false);
  loom::check::expect_equal(enum_false, 0u);
  loom::check::expect_equal(enum_true, 1u);
  loom::check::expect_equal(enum_named, 1u);
  loom::check::expect_equal(enum_zero, 0u);
}
