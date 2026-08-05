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


class RecursionCheckTest(clang_tidy_test.ClangTidyAssertions):
    def test_reports_one_diagnostic_per_recursive_scc(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-unbounded-recursion",
            source=clang_tidy_test.source_path(__file__, "recursion_check.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertEqual(output.count("[iree-unbounded-recursion]"), 6, output)
        self.assertContainsAll(
            output,
            [
                "direct_recursion",
                "mutual_recursion_a",
                "mutual_recursion_b",
                "through callback dispatcher invoke_callback",
                "second_parameter_callback_a",
                "second_parameter_callback_b",
                "through callback dispatcher invoke_record",
                "indirect call from dispatch_table_callback",
            ],
        )
        self.assertNotIn("explicit_worklist", output)


if __name__ == "__main__":
    unittest.main()
