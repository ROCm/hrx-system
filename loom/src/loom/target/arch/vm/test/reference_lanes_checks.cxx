// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>

#include <reference_lanes.cxx>
LOOM_CHECK_CASE(widths_first) {
  const auto selected = widths(true, 0u);
  const auto index_evaluations = widths(true, 1u);
  loom::check::expect_equal(selected, 2u);
  loom::check::expect_equal(index_evaluations, 1u);
}
LOOM_CHECK_CASE(widths_second) {
  const auto selected = widths(false, 0u);
  const auto index_evaluations = widths(false, 1u);
  loom::check::expect_equal(selected, 6u);
  loom::check::expect_equal(index_evaluations, 1u);
}
LOOM_CHECK_CASE(nested_first) {
  const auto selected = nested(true, true, 0u);
  const auto index_evaluations = nested(true, true, 1u);
  loom::check::expect_equal(selected, 2u);
  loom::check::expect_equal(index_evaluations, 1u);
}
LOOM_CHECK_CASE(nested_second) {
  const auto selected = nested(true, false, 0u);
  const auto index_evaluations = nested(true, false, 1u);
  loom::check::expect_equal(selected, 6u);
  loom::check::expect_equal(index_evaluations, 1u);
}
LOOM_CHECK_CASE(nested_third) {
  const auto selected = nested(false, true, 0u);
  const auto index_evaluations = nested(false, true, 1u);
  loom::check::expect_equal(selected, 1u);
  loom::check::expect_equal(index_evaluations, 1u);
}
LOOM_CHECK_CASE(same_vector_observations) {
  const auto first_selected = same_vector(true);
  const auto second_selected = same_vector(false);
  loom::check::expect_equal(first_selected, 2u);
  loom::check::expect_equal(second_selected, 4u);
}
LOOM_CHECK_CASE(scalar_lane_observations) {
  const auto first_selected = scalar_lane(true);
  const auto second_selected = scalar_lane(false);
  loom::check::expect_equal(first_selected, 5u);
  loom::check::expect_equal(second_selected, 4u);
}
LOOM_CHECK_CASE(memory_assignment_observations) {
  const auto first_selected = memory_assignment(true);
  const auto second_selected = memory_assignment(false);
  loom::check::expect_equal(first_selected, 9121u);
  loom::check::expect_equal(second_selected, 5160u);
}
LOOM_CHECK_CASE(vector_base_observations) {
  const auto first_selected = vector_base(true);
  const auto second_selected = vector_base(false);
  loom::check::expect_equal(first_selected, 2u);
  loom::check::expect_equal(second_selected, 6u);
}
