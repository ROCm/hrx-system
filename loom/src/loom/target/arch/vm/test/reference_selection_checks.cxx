// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assertions remain outside the subject and execute unchanged after public
// C++ export and reimport.
#include <loomcxx/check.h>

#include <reference_selection.cxx>

LOOM_CHECK_CASE(repeated_field_alternatives) {
  const auto first = repeated_field(0u);
  const auto second = repeated_field(1u);
  const auto repeated_first = repeated_field(2u);
  const auto repeated_second = repeated_field(3u);
  loom::check::expect_equal(first, 12111233u);
  loom::check::expect_equal(second, 12113311u);
  loom::check::expect_equal(repeated_first, 12111233u);
  loom::check::expect_equal(repeated_second, 12113311u);
}

LOOM_CHECK_CASE(partial_field_alternatives) {
  const auto first = partial_field(true);
  const auto second = partial_field(false);
  loom::check::expect_equal(first, 98u);
  loom::check::expect_equal(second, 71u);
}

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

LOOM_CHECK_CASE(independent_selections_first_and_third) {
  const auto first = independent_selections(true, true, 0u);
  const auto second = independent_selections(true, true, 1u);
  const auto third = independent_selections(true, true, 2u);
  const auto fourth = independent_selections(true, true, 3u);
  const auto returned = independent_selections(true, true, 4u);
  const auto snapshot = independent_selections(true, true, 5u);
  loom::check::expect_equal(first, 22u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(third, 11u);
  loom::check::expect_equal(fourth, 33u);
  loom::check::expect_equal(returned, 11u);
  loom::check::expect_equal(snapshot, 22u);
}

LOOM_CHECK_CASE(independent_selections_first_and_fourth) {
  const auto first = independent_selections(true, false, 0u);
  const auto second = independent_selections(true, false, 1u);
  const auto third = independent_selections(true, false, 2u);
  const auto fourth = independent_selections(true, false, 3u);
  const auto returned = independent_selections(true, false, 4u);
  const auto snapshot = independent_selections(true, false, 5u);
  loom::check::expect_equal(first, 22u);
  loom::check::expect_equal(second, 12u);
  loom::check::expect_equal(third, 22u);
  loom::check::expect_equal(fourth, 11u);
  loom::check::expect_equal(returned, 11u);
  loom::check::expect_equal(snapshot, 33u);
}

LOOM_CHECK_CASE(independent_selections_second_and_third) {
  const auto first = independent_selections(false, true, 0u);
  const auto second = independent_selections(false, true, 1u);
  const auto third = independent_selections(false, true, 2u);
  const auto fourth = independent_selections(false, true, 3u);
  const auto returned = independent_selections(false, true, 4u);
  const auto snapshot = independent_selections(false, true, 5u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 22u);
  loom::check::expect_equal(third, 11u);
  loom::check::expect_equal(fourth, 33u);
  loom::check::expect_equal(returned, 11u);
  loom::check::expect_equal(snapshot, 22u);
}

LOOM_CHECK_CASE(independent_selections_second_and_fourth) {
  const auto first = independent_selections(false, false, 0u);
  const auto second = independent_selections(false, false, 1u);
  const auto third = independent_selections(false, false, 2u);
  const auto fourth = independent_selections(false, false, 3u);
  const auto returned = independent_selections(false, false, 4u);
  const auto snapshot = independent_selections(false, false, 5u);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(second, 22u);
  loom::check::expect_equal(third, 22u);
  loom::check::expect_equal(fourth, 11u);
  loom::check::expect_equal(returned, 11u);
  loom::check::expect_equal(snapshot, 33u);
}

LOOM_CHECK_CASE(selected_objects_keep_distinct_lifetimes) {
  const auto first_total = selected_destruction(true, 0u);
  const auto second_total = selected_destruction(false, 0u);
  const auto first_snapshot = selected_destruction(true, 1u);
  const auto second_snapshot = selected_destruction(false, 1u);
  const auto first_before_cleanup = selected_destruction(true, 2u);
  const auto second_before_cleanup = selected_destruction(false, 2u);
  loom::check::expect_equal(first_total, 44u);
  loom::check::expect_equal(second_total, 44u);
  loom::check::expect_equal(first_snapshot, 11u);
  loom::check::expect_equal(second_snapshot, 11u);
  loom::check::expect_equal(first_before_cleanup, 0u);
  loom::check::expect_equal(second_before_cleanup, 0u);
}

LOOM_CHECK_CASE(semantic_array_selection) {
  const auto first = selected_array(0u);
  const auto second = selected_array(1u);
  const auto third = selected_array(2u);
  loom::check::expect_equal(first, 11221244u);
  loom::check::expect_equal(second, 12224411u);
  loom::check::expect_equal(third, 22441211u);
}

LOOM_CHECK_CASE(nested_semantic_array_selection) {
  const auto first = selected_array_matrix(0u, 0u);
  const auto second = selected_array_matrix(0u, 1u);
  const auto third = selected_array_matrix(1u, 0u);
  const auto fourth = selected_array_matrix(1u, 1u);
  loom::check::expect_equal(first, 1133221244u);
  loom::check::expect_equal(second, 1233224411u);
  loom::check::expect_equal(third, 2233441211u);
  loom::check::expect_equal(fourth, 3344221211u);
}

LOOM_CHECK_CASE(forwarded_semantic_array_reference) {
  const auto first_0 = returned_array_reference(0u, 0u);
  const auto first_1 = returned_array_reference(0u, 1u);
  const auto first_2 = returned_array_reference(0u, 2u);
  const auto first_watched = returned_array_reference(0u, 3u);
  const auto first_snapshot = returned_array_reference(0u, 4u);
  const auto first_selected = returned_array_reference(0u, 5u);
  const auto second_0 = returned_array_reference(1u, 0u);
  const auto second_1 = returned_array_reference(1u, 1u);
  const auto second_2 = returned_array_reference(1u, 2u);
  const auto second_watched = returned_array_reference(1u, 3u);
  const auto second_snapshot = returned_array_reference(1u, 4u);
  const auto second_selected = returned_array_reference(1u, 5u);
  const auto third_0 = returned_array_reference(2u, 0u);
  const auto third_1 = returned_array_reference(2u, 1u);
  const auto third_2 = returned_array_reference(2u, 2u);
  const auto third_watched = returned_array_reference(2u, 3u);
  const auto third_snapshot = returned_array_reference(2u, 4u);
  const auto third_selected = returned_array_reference(2u, 5u);

  loom::check::expect_equal(first_0, 22u);
  loom::check::expect_equal(first_1, 33u);
  loom::check::expect_equal(first_2, 12u);
  loom::check::expect_equal(first_watched, 22u);
  loom::check::expect_equal(first_snapshot, 11u);
  loom::check::expect_equal(first_selected, 22u);
  loom::check::expect_equal(second_0, 44u);
  loom::check::expect_equal(second_1, 22u);
  loom::check::expect_equal(second_2, 12u);
  loom::check::expect_equal(second_watched, 22u);
  loom::check::expect_equal(second_snapshot, 12u);
  loom::check::expect_equal(second_selected, 22u);
  loom::check::expect_equal(third_0, 44u);
  loom::check::expect_equal(third_1, 33u);
  loom::check::expect_equal(third_2, 22u);
  loom::check::expect_equal(third_watched, 22u);
  loom::check::expect_equal(third_snapshot, 22u);
  loom::check::expect_equal(third_selected, 22u);
}

LOOM_CHECK_CASE(multiple_return_semantic_array_reference) {
  const auto first = returned_array_edge(true);
  const auto last = returned_array_edge(false);
  loom::check::expect_equal(first, 11u);
  loom::check::expect_equal(last, 22u);
}

LOOM_CHECK_CASE(wide_const_semantic_array_reference) {
  const auto first = returned_wide_const_array_reference(0u);
  const auto middle = returned_wide_const_array_reference(4u);
  const auto last = returned_wide_const_array_reference(8u);
  loom::check::expect_equal(first, 1u);
  loom::check::expect_equal(middle, 5u);
  loom::check::expect_equal(last, 9u);
}

LOOM_CHECK_CASE(exact_semantic_reference_preserves_forwarded_effects) {
  const auto result = returned_exact_reference_effect();
  loom::check::expect_equal(result, 111u);
}

LOOM_CHECK_CASE(mixed_owner_semantic_reference) {
  const auto direct = returned_mixed_reference(true);
  const auto member = returned_mixed_reference(false);
  loom::check::expect_equal(direct, 11u);
  loom::check::expect_equal(member, 22u);
}

LOOM_CHECK_CASE(nested_selected_semantic_reference) {
  const auto left_first = returned_nested_reference(true, true);
  const auto left_second = returned_nested_reference(true, false);
  const auto right_first = returned_nested_reference(false, true);
  const auto right_second = returned_nested_reference(false, false);
  loom::check::expect_equal(left_first, 33221244u);
  loom::check::expect_equal(left_second, 33224411u);
  loom::check::expect_equal(right_first, 33441211u);
  loom::check::expect_equal(right_second, 44221211u);
}

LOOM_CHECK_CASE(mixed_storage_and_value_reference) {
  const auto stored = returned_mixed_storage_reference(true);
  const auto automatic = returned_mixed_storage_reference(false);
  loom::check::expect_equal(stored, 224411u);
  loom::check::expect_equal(automatic, 441211u);
}
