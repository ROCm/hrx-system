// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

static int direct(unsigned selector, int value) {
  switch (selector) {
    case 0xffffffffu:
      value += 9;
      break;
    case 2u:
      value += 3;
      break;
    default:
      value -= 1;
      break;
  }
  return value;
}

static int fallthrough(unsigned selector, int value) {
  switch (selector) {
    case 0u:
    case 1u:
      value += 1;
    case 2u:
      value *= 2;
      break;
    default:
      value -= 3;
    case 4u:
      value += 5;
  }
  return value;
}

static int no_default(int selector, int value) {
  switch (selector) {
    case -3:
      value *= 2;
      break;
    case 4:
      value += 5;
      break;
  }
  return value;
}

static int continued(int selector, int value) {
  for (int index = 0; index < 4; ++index) {
    switch (selector) {
      case -1:
        value += 2;
        break;
      case 1:
        continue;
      default:
        value += index;
        break;
    }
    value += 10;
  }
  return value;
}

static int partial_break(unsigned selector, int value) {
  switch (selector) {
    case 0u:
      value += 1;
      if ((value & 1) != 0) {
        break;
      }
      value += 2;
    case 1u:
      value += 4;
      break;
    default:
      value += 8;
  }
  return value;
}

static int conditional_continue(int selector) {
  int value = 0;
  for (int index = 0; index < 4; ++index) {
    switch (selector) {
      case 0:
        if ((index & 1) != 0) {
          continue;
        }
        value += 1;
        break;
      default:
        value += 2;
        break;
    }
    value += 10;
  }
  return value;
}

static unsigned duff_tail(unsigned remainder) {
  unsigned visited = 0;
  switch (remainder & 15u) {
    case 0:
      visited |= 1u << 0u;
    case 15:
      visited |= 1u << 15u;
    case 14:
      visited |= 1u << 14u;
    case 13:
      visited |= 1u << 13u;
    case 12:
      visited |= 1u << 12u;
    case 11:
      visited |= 1u << 11u;
    case 10:
      visited |= 1u << 10u;
    case 9:
      visited |= 1u << 9u;
    case 8:
      visited |= 1u << 8u;
    case 7:
      visited |= 1u << 7u;
    case 6:
      visited |= 1u << 6u;
    case 5:
      visited |= 1u << 5u;
    case 4:
      visited |= 1u << 4u;
    case 3:
      visited |= 1u << 3u;
    case 2:
      visited |= 1u << 2u;
    case 1:
      visited |= 1u << 1u;
  }
  return visited;
}

LOOM_CHECK_CASE(switch_values) {
  const auto direct_max = direct(0xffffffffu, 5);
  const auto direct_two = direct(2u, 5);
  const auto direct_default = direct(7u, 5);
  const auto fallthrough_zero = fallthrough(0u, 10);
  const auto fallthrough_one = fallthrough(1u, 10);
  const auto fallthrough_two = fallthrough(2u, 10);
  const auto fallthrough_default = fallthrough(3u, 10);
  const auto fallthrough_four = fallthrough(4u, 10);
  const auto no_default_negative = no_default(-3, 7);
  const auto no_default_four = no_default(4, 7);
  const auto no_default_miss = no_default(5, 7);
  const auto continue_all = continued(1, 7);
  const auto continue_none = continued(0, 0);
  const auto continue_negative = continued(-1, 0);
  const auto partial_break_taken = partial_break(0u, 0);
  const auto partial_break_fallthrough = partial_break(0u, 1);
  const auto partial_break_case = partial_break(1u, 0);
  const auto partial_break_default = partial_break(7u, 0);
  const auto conditional_continue_some = conditional_continue(0);
  const auto conditional_continue_none = conditional_continue(1);
  const auto duff_zero = duff_tail(0u);
  const auto duff_fifteen = duff_tail(15u);
  const auto duff_eight = duff_tail(8u);
  const auto duff_one = duff_tail(1u);

  loom::check::expect_equal(direct_max, 14);
  loom::check::expect_equal(direct_two, 8);
  loom::check::expect_equal(direct_default, 4);
  loom::check::expect_equal(fallthrough_zero, 22);
  loom::check::expect_equal(fallthrough_one, 22);
  loom::check::expect_equal(fallthrough_two, 20);
  loom::check::expect_equal(fallthrough_default, 12);
  loom::check::expect_equal(fallthrough_four, 15);
  loom::check::expect_equal(no_default_negative, 14);
  loom::check::expect_equal(no_default_four, 12);
  loom::check::expect_equal(no_default_miss, 7);
  loom::check::expect_equal(continue_all, 7);
  loom::check::expect_equal(continue_none, 46);
  loom::check::expect_equal(continue_negative, 48);
  loom::check::expect_equal(partial_break_taken, 1);
  loom::check::expect_equal(partial_break_fallthrough, 8);
  loom::check::expect_equal(partial_break_case, 4);
  loom::check::expect_equal(partial_break_default, 8);
  loom::check::expect_equal(conditional_continue_some, 22);
  loom::check::expect_equal(conditional_continue_none, 48);
  loom::check::expect_equal(duff_zero, 65535u);
  loom::check::expect_equal(duff_fifteen, 65534u);
  loom::check::expect_equal(duff_eight, 510u);
  loom::check::expect_equal(duff_one, 2u);
}
