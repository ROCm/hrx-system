// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assertions remain outside the subject and execute unchanged after public
// C++ export and reimport.
#include <loomcxx/check.h>

#include <reference_selection.cxx>

LOOM_CHECK_CASE(saved_reference_selects_first) {
  const auto first = selected_object(true, 0u);
  const auto second = selected_object(true, 1u);
  const auto watched = selected_object(true, 2u);
  const auto snapshot = selected_object(true, 3u);
  const auto selections = selected_object(true, 4u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(watched, 33u);
  loom::check::expect_equal(snapshot, 11u);
  loom::check::expect_equal(selections, 1u);
}

LOOM_CHECK_CASE(field_reference_selects_first) {
  const auto first = selected_field(true, 0u);
  const auto second = selected_field(true, 1u);
  const auto watched = selected_field(true, 2u);
  const auto snapshot = selected_field(true, 3u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(watched, 33u);
  loom::check::expect_equal(snapshot, 11u);
}

LOOM_CHECK_CASE(base_reference_selects_first) {
  const auto first = selected_base(true, 0u);
  const auto second = selected_base(true, 1u);
  const auto first_marker = selected_base(true, 2u);
  const auto second_marker = selected_base(true, 3u);
  const auto snapshot = selected_base(true, 4u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(first_marker, 101u);
  loom::check::expect_equal(second_marker, 102u);
  loom::check::expect_equal(snapshot, 11u);
}

LOOM_CHECK_CASE(returned_member_changed) {
  const auto prefix = returned_member(true, 0u);
  const auto first = returned_member(true, 1u);
  const auto second = returned_member(true, 2u);
  const auto suffix = returned_member(true, 3u);
  loom::check::expect_equal(prefix, 100u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(suffix, 200u);
}

LOOM_CHECK_CASE(free_call_evaluates_arguments_before_reading_first) {
  const auto first = selected_argument_order(true, 0u, 0u);
  const auto second = selected_argument_order(true, 0u, 1u);
  const auto through_reference = selected_argument_order(true, 0u, 2u);
  const auto previous = selected_argument_order(true, 0u, 3u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(member_call_evaluates_arguments_before_reading_first) {
  const auto first = selected_argument_order(true, 1u, 0u);
  const auto second = selected_argument_order(true, 1u, 1u);
  const auto through_reference = selected_argument_order(true, 1u, 2u);
  const auto previous = selected_argument_order(true, 1u, 3u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(operator_call_evaluates_arguments_before_reading_first) {
  const auto first = selected_argument_order(true, 2u, 0u);
  const auto second = selected_argument_order(true, 2u, 1u);
  const auto through_reference = selected_argument_order(true, 2u, 2u);
  const auto previous = selected_argument_order(true, 2u, 3u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(saved_reference_selects_second) {
  const auto first = selected_object(false, 0u);
  const auto second = selected_object(false, 1u);
  const auto watched = selected_object(false, 2u);
  const auto snapshot = selected_object(false, 3u);
  const auto selections = selected_object(false, 4u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(watched, 33u);
  loom::check::expect_equal(snapshot, 12u);
  loom::check::expect_equal(selections, 1u);
}

LOOM_CHECK_CASE(field_reference_selects_second) {
  const auto first = selected_field(false, 0u);
  const auto second = selected_field(false, 1u);
  const auto watched = selected_field(false, 2u);
  const auto snapshot = selected_field(false, 3u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(watched, 33u);
  loom::check::expect_equal(snapshot, 12u);
}

LOOM_CHECK_CASE(base_reference_selects_second) {
  const auto first = selected_base(false, 0u);
  const auto second = selected_base(false, 1u);
  const auto first_marker = selected_base(false, 2u);
  const auto second_marker = selected_base(false, 3u);
  const auto snapshot = selected_base(false, 4u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(first_marker, 101u);
  loom::check::expect_equal(second_marker, 102u);
  loom::check::expect_equal(snapshot, 12u);
}

LOOM_CHECK_CASE(returned_member_unchanged) {
  const auto prefix = returned_member(false, 0u);
  const auto first = returned_member(false, 1u);
  const auto second = returned_member(false, 2u);
  const auto suffix = returned_member(false, 3u);
  loom::check::expect_equal(prefix, 100u);
  loom::check::expect_equal(first, 12u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(suffix, 200u);
}

LOOM_CHECK_CASE(free_call_evaluates_arguments_before_reading_second) {
  const auto first = selected_argument_order(false, 0u, 0u);
  const auto second = selected_argument_order(false, 0u, 1u);
  const auto through_reference = selected_argument_order(false, 0u, 2u);
  const auto previous = selected_argument_order(false, 0u, 3u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(member_call_evaluates_arguments_before_reading_second) {
  const auto first = selected_argument_order(false, 1u, 0u);
  const auto second = selected_argument_order(false, 1u, 1u);
  const auto through_reference = selected_argument_order(false, 1u, 2u);
  const auto previous = selected_argument_order(false, 1u, 3u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(operator_call_evaluates_arguments_before_reading_second) {
  const auto first = selected_argument_order(false, 2u, 0u);
  const auto second = selected_argument_order(false, 2u, 1u);
  const auto through_reference = selected_argument_order(false, 2u, 2u);
  const auto previous = selected_argument_order(false, 2u, 3u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(previous, 22u);
}

LOOM_CHECK_CASE(nested_selection_1_1) {
  const auto first = selected_nested(true, true, 0u);
  const auto second = selected_nested(true, true, 1u);
  const auto third = selected_nested(true, true, 2u);
  const auto through_reference = selected_nested(true, true, 3u);
  const auto snapshot = selected_nested(true, true, 4u);
  const auto decisions = selected_nested(true, true, 5u);
  loom::check::expect_equal(first, 33u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(third, 22u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(snapshot, 11u);
  loom::check::expect_equal(decisions, 1u);
}

LOOM_CHECK_CASE(nested_selection_1_0) {
  const auto first = selected_nested(true, false, 0u);
  const auto second = selected_nested(true, false, 1u);
  const auto third = selected_nested(true, false, 2u);
  const auto through_reference = selected_nested(true, false, 3u);
  const auto snapshot = selected_nested(true, false, 4u);
  const auto decisions = selected_nested(true, false, 5u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 33u);
  loom::check::expect_equal(third, 22u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(snapshot, 12u);
  loom::check::expect_equal(decisions, 1u);
}

LOOM_CHECK_CASE(nested_selection_0_1) {
  const auto first = selected_nested(false, true, 0u);
  const auto second = selected_nested(false, true, 1u);
  const auto third = selected_nested(false, true, 2u);
  const auto through_reference = selected_nested(false, true, 3u);
  const auto snapshot = selected_nested(false, true, 4u);
  const auto decisions = selected_nested(false, true, 5u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(third, 33u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(snapshot, 22u);
  loom::check::expect_equal(decisions, 0u);
}

LOOM_CHECK_CASE(nested_selection_0_0) {
  const auto first = selected_nested(false, false, 0u);
  const auto second = selected_nested(false, false, 1u);
  const auto third = selected_nested(false, false, 2u);
  const auto through_reference = selected_nested(false, false, 3u);
  const auto snapshot = selected_nested(false, false, 4u);
  const auto decisions = selected_nested(false, false, 5u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(third, 33u);
  loom::check::expect_equal(through_reference, 33u);
  loom::check::expect_equal(snapshot, 22u);
  loom::check::expect_equal(decisions, 0u);
}

LOOM_CHECK_CASE(forwarded_receiver_retains_effects) {
  const auto unchanged = forwarded_receiver(false);
  const auto updated = forwarded_receiver(true);
  loom::check::expect_equal(unchanged, 22u);
  loom::check::expect_equal(updated, 66u);
}
