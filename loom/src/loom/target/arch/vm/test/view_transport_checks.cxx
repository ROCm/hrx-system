// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The runner supplies either authored or exported subjects on the include path.
// These assertions stay unchanged and outside the program being roundtripped.
#include <loomcxx/check.h>

#include <view_transport.cxx>

LOOM_CHECK_CASE(dense_branch_preserves_view) {
  const auto first = dense_branch(true);
  const auto second = dense_branch(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(dense_counted_zero_trip_preserves_view) {
  const auto first = dense_counted(true, 0u);
  const auto second = dense_counted(false, 0u);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 12u);
}

LOOM_CHECK_CASE(dense_counted_repeated_preserves_view) {
  const auto first = dense_counted(true, 3u);
  const auto second = dense_counted(false, 3u);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(dense_swap_preserves_view) {
  const auto first = dense_swap(true);
  const auto second = dense_swap(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(dense_do_while_preserves_view) {
  const auto first = dense_do_while(true);
  const auto second = dense_do_while(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(strided_branch_preserves_view) {
  const auto first = strided_branch(true);
  const auto second = strided_branch(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(strided_counted_zero_trip_preserves_view) {
  const auto first = strided_counted(true, 0u);
  const auto second = strided_counted(false, 0u);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 12u);
}

LOOM_CHECK_CASE(strided_counted_repeated_preserves_view) {
  const auto first = strided_counted(true, 3u);
  const auto second = strided_counted(false, 3u);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(strided_swap_preserves_view) {
  const auto first = strided_swap(true);
  const auto second = strided_swap(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(strided_do_while_preserves_view) {
  const auto first = strided_do_while(true);
  const auto second = strided_do_while(false);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 34u);
}

LOOM_CHECK_CASE(independent_dimension_retains_arithmetic) {
  const auto first_length = independent_dimension(true, true);
  const auto second_length = independent_dimension(false, true);
  loom::check::expect_equal(first_length, 4u);
  loom::check::expect_equal(second_length, 5u);
}

LOOM_CHECK_CASE(independent_dimension_constructs_view) {
  const auto first_byte = independent_dimension(true, false);
  const auto second_byte = independent_dimension(false, false);
  loom::check::expect_equal(first_byte, 15u);
  loom::check::expect_equal(second_byte, 16u);
}

LOOM_CHECK_CASE(unused_view_carrier_preserves_producer_writes) {
  const auto first_write = unused_carrier(0u);
  const auto second_write = unused_carrier(1u);
  const auto third_write = unused_carrier(2u);
  loom::check::expect_equal(first_write, 1u);
  loom::check::expect_equal(second_write, 2u);
  loom::check::expect_equal(third_write, 3u);
}
