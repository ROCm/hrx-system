// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

LOOM_CHECK_CASE(switch_cases_and_default) {
  const auto first = choose_case(0u);
  const auto second = choose_case(2u);
  const auto fallback = choose_case(1u);
  const auto unsigned_key = choose_case(0xffffffffu);
  const auto only_default = default_case(4u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 22u);
  loom::check::expect_equal(fallback, 33u);
  loom::check::expect_equal(unsigned_key, 33u);
  loom::check::expect_equal(only_default, 5u);
}

LOOM_CHECK_CASE(signed_and_wide_case_keys) {
  const auto minimum = signed_case(-9223372036854775807ll - 1ll);
  const auto negative = signed_case(-3ll);
  const auto wide = signed_case(4294967296ll);
  const auto maximum = signed_case(9223372036854775807ll);
  const auto fallback = signed_case(0ll);
  loom::check::expect_equal(minimum, 1u);
  loom::check::expect_equal(negative, 2u);
  loom::check::expect_equal(wide, 3u);
  loom::check::expect_equal(maximum, 4u);
  loom::check::expect_equal(fallback, 5u);
}

LOOM_CHECK_CASE(loop_carried_parallel_assignments) {
  const auto empty = rotate(0u, 0u);
  const auto swapped = rotate(1u, 0u);
  const auto restored = rotate(2u, 0u);
  const auto nested = rotate(2u, 1u);
  const auto fallback = rotate(2u, 9u);
  loom::check::expect_equal(empty, 305u);
  loom::check::expect_equal(swapped, 503u);
  loom::check::expect_equal(restored, 305u);
  loom::check::expect_equal(nested, 813u);
  loom::check::expect_equal(fallback, 1305u);
}

LOOM_CHECK_CASE(selector_effect_runs_once) {
  const auto first = observe_selector_effects(0u);
  const auto second = observe_selector_effects(1u);
  const auto third = observe_selector_effects(2u);
  const auto count = observe_selector_effects(3u);
  loom::check::expect_equal(first, 17u);
  loom::check::expect_equal(second, 19u);
  loom::check::expect_equal(third, 23u);
  loom::check::expect_equal(count, 3u);
}

LOOM_CHECK_CASE(returns_before_nonreturning_paths) {
  const auto defined = defined_return(true);
  const auto ready = wait_or_return(true);
  loom::check::expect_equal(defined, 7u);
  loom::check::expect_equal(ready, 9u);
}

LOOM_CHECK_CASE(views_and_pointer_offsets) {
  const auto first_view = observe_views(0u);
  const auto second_view = observe_views(1u);
  const auto first_pointer = observe_views(2u);
  const auto second_pointer = observe_views(3u);
  loom::check::expect_equal(first_view, 30u);
  loom::check::expect_equal(second_view, 400u);
  loom::check::expect_equal(first_pointer, 30u);
  loom::check::expect_equal(second_pointer, 200u);
}
