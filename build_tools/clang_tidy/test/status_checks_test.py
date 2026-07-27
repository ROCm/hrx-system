#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

import clang_tidy_test

_ARGS = clang_tidy_test.parse_arguments()


class StatusChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_borrowed_status_parameter_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-borrowed-parameter",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "borrowed_parameter_status",
                "const_status_consumed",
                "[iree-status-borrowed-parameter]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "returned_parameter_status",
                "consumed_parameter_status",
                "const_borrowed_parameter_status",
                "const_clone_stored_parameter_status",
                "const_callback_parameter_status",
                "stored_parameter_status",
                "joined_parameter_status",
                "annotated_parameter_status",
                "cloned_parameter_status",
                "sink_parameter_status",
                "reclaim_callback_parameter_status",
            ],
        )

    def test_borrowed_status_parameter_accepts_cpp_status_observer(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-borrowed-parameter",
            source=clang_tidy_test.source_path(__file__, "status_checks.cc"),
            compiler_args=["-std=c++17"],
        )
        self.assertContainsNone(
            output,
            [
                "ToString",
                "[iree-status-borrowed-parameter]",
            ],
        )

    def test_discarded_status_result_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-discarded",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "iree_clang_tidy_status_dropped_source",
                "iree_clang_tidy_status_named_call_source",
                "iree_clang_tidy_status_void_cast_source",
                "[iree-status-discarded]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_status_assigned_source",
                "iree_clang_tidy_status_ignored_source",
                "iree_clang_tidy_status_returned_source",
            ],
        )

    def test_status_lifetime_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-lifetime",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "leaked_status",
                "overwritten_status",
                "double_consumed_status",
                "used_after_consume_status",
                "used_after_transfer_status",
                "lost_on_return_status",
                "code_predicate_status",
                "reported_ignore_status",
                "formatted_ignore_status",
                "immediate_ok_overwrite_status",
                "[iree-status-lifetime]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "returned_status",
                "return_if_error_status",
                "macro_return_status",
                "macro_fallthrough_status",
                "transfer_source_status",
                "transfer_target_status",
                "conditional_source_status",
                "conditional_target_status",
                "joined_primary_status",
                "joined_secondary_status",
                "loop_break_status",
                "ok_and_status",
                "conditional_ok_status",
                "joined_ok_status",
                "consumed_predicate_status",
                "atomic_escape_status",
                "atomic_builtin_status",
                "sink_argument_status",
                "borrowed_status",
                "joined_status",
                "annotated_status",
                "stored_status",
                "consumed_code_status",
                "reported_free_status",
                "observer_only_status",
                "const_borrowed_observed_status",
                "const_callback_observed_status",
            ],
        )

    def test_status_lifetime_accepts_cpp_test_consumers(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-lifetime",
            source=clang_tidy_test.source_path(__file__, "status_checks.cc"),
            compiler_args=["-std=c++17"],
        )
        self.assertContainsNone(
            output,
            [
                "consume_for_test_status",
                "consume_for_test_move_status",
                "string_to_free_status",
                "gtest_condition_status",
                "cpp_status_wrapper_status",
                "cpp_status_wrapper_move_status",
                "cpp_status_or_status",
                "[iree-status-lifetime]",
            ],
        )

    def test_status_lifetime_fixes_immediate_ok_overwrite(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-lifetime",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "fix_status",
                "[iree-status-lifetime]",
            ],
        )
        self.assertIn(
            "iree_status_t fix_status = iree_clang_tidy_status_fix_source();",
            fixed_source,
        )
        self.assertIn("  return fix_status;", fixed_source)
        self.assertNotIn(
            "  fix_status = iree_clang_tidy_status_fix_source();",
            fixed_source,
        )
        self.assertIn(
            "iree_status_t conditional_ok_status = iree_ok_status();",
            fixed_source,
        )

    def test_status_transfer_order_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-transfer-order",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "transfer_order_join_same_status",
                "transfer_order_nested_status",
                "[iree-status-transfer-order]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "explicit_sequence_status",
                "clone_before_fanout_status",
                "cloned_status",
                "callback_status",
                "observer_only_status",
            ],
        )


if __name__ == "__main__":
    unittest.main()
