# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

import clang_tidy_test

_ARGS = clang_tidy_test.parse_arguments()


class TraceChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_trace_zone_balance_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-trace-zone-balance",
            source=clang_tidy_test.source_path(__file__, "trace_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "raw_return_zone",
                "hip_return_zone",
                "plain_return_if_error_zone",
                "hrx_plain_return_zone",
                "unmatched_end_zone",
                "mismatch_outer_zone",
                "mismatch_inner_zone",
                "return_helper_wrong_zone",
                "return_helper_active_zone",
                "missing_end_zone",
                "branch_missing_end_inner_zone",
                "branch_conditional_end_zone",
                "[iree-trace-zone-balance]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_trace_zone_balanced",
                "iree_clang_tidy_trace_zone_nested_balanced",
                "iree_clang_tidy_trace_zone_single_exit_cleanup",
                "iree_clang_tidy_trace_zone_branch_return_balanced",
                "iree_clang_tidy_trace_zone_branch_macro_return_balanced",
                "iree_clang_tidy_trace_zone_branch_local_zone",
                "iree_clang_tidy_trace_zone_branch_replaces_handle",
                "iree_clang_tidy_trace_zone_adopted_branch",
                "iree_clang_tidy_trace_zone_transfer",
                "iree_clang_tidy_trace_zone_return_and_end_if_error",
                "iree_clang_tidy_trace_zone_dynamic_end",
                "iree_clang_tidy_trace_zone_dynamic_return_helper",
                "iree_clang_tidy_trace_zone_switch_return_balanced",
                "iree_clang_tidy_trace_zone_infinite_loop_return_balanced",
                "iree_clang_tidy_trace_zone_return_and_end",
                "iree_clang_tidy_trace_zone_hrx_return_and_end_if_error",
            ],
        )


if __name__ == "__main__":
    unittest.main()
