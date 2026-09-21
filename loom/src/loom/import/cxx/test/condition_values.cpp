// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

template <unsigned Choice>
static unsigned selected_value(unsigned input) {
  if constexpr (const unsigned value = sizeof(value) ? Choice : 0u) {
    return input + value;
  } else {
    return input + value + 20;
  }
}

static unsigned runtime_value(unsigned input) {
  if (unsigned value = input++) {
    value += 2;
    input += value;
  } else {
    input += value + 30;
  }
  return input;
}

static unsigned while_values(unsigned count) {
  unsigned checks = 0;
  unsigned total = 0;
  while (unsigned value = (++checks, count--)) {
    if (value == 2) {
      value += 10;
      total += value;
      continue;
    }
    total += value;
  }
  // The final false check also decrements count and increments checks.
  return total + 100 * checks + 10000 * (count + 1);
}

static unsigned for_values(unsigned count) {
  unsigned checks = 0;
  unsigned total = 0;
  for (unsigned remaining = count; unsigned value = (++checks, remaining);
       remaining = value - 1) {
    if (value > 2) {
      --value;
      continue;
    }
    total += value;
  }
  return total + 100 * checks;
}

static unsigned discarded_values(unsigned count) {
  unsigned total = 0;
  for (; unsigned value = count; (void)++total, (void)--count) {
    total += value;
  }
  return total;
}

LOOM_CHECK_CASE(decision_values) {
  const auto selected = selected_value<7>(3);
  const auto discarded = selected_value<0>(3);
  const auto nonzero = runtime_value(3);
  const auto zero = runtime_value(0);
  loom::check::expect_equal(selected, 10u);
  loom::check::expect_equal(discarded, 23u);
  loom::check::expect_equal(nonzero, 9u);
  loom::check::expect_equal(zero, 31u);
}

LOOM_CHECK_CASE(loop_decisions) {
  const auto while_empty = while_values(0);
  const auto while_populated = while_values(3);
  const auto for_empty = for_values(0);
  const auto for_short = for_values(2);
  const auto for_continue = for_values(5);
  const auto discarded = discarded_values(3);
  loom::check::expect_equal(while_empty, 100u);
  loom::check::expect_equal(while_populated, 416u);
  loom::check::expect_equal(for_empty, 100u);
  loom::check::expect_equal(for_short, 303u);
  loom::check::expect_equal(for_continue, 401u);
  loom::check::expect_equal(discarded, 9u);
}
