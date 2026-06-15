#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest
from pathlib import Path

import clang_tidy_test

_ARGS = clang_tidy_test.parse_arguments()


class StyleChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_cpp_designated_initializers_are_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=clang_tidy_test.source_path(__file__, "style_checks.cc"),
            compiler_args=["-std=c++20"],
        )
        self.assertContainsAll(
            output,
            [
                "C++ code must use comment field labels instead of designated "
                "initializers for MSVC portability",
                ".ordinal = 1",
                '.name = "device"',
                '.name = "skipped"',
                ".traits = 1",
                '.name = "sparse"',
                "IREE_CLANG_TIDY_STYLE_CPP_ACCEPT_CONFIG",
                "[iree-cpp-designated-initializer]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_cpp_comment_labels",
            ],
        )

    def test_cpp_designated_initializers_are_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=clang_tidy_test.source_path(__file__, "style_checks.cc"),
            compiler_args=["-std=c++20"],
        )
        self.assertContainsAll(
            output,
            [
                "C++ code must use comment field labels instead of designated "
                "initializers for MSVC portability",
                "[iree-cpp-designated-initializer]",
            ],
        )
        self.assertIn("/*.ordinal=*/1,", fixed_source)
        self.assertIn('/*.name=*/"device",', fixed_source)
        self.assertIn('/*.ordinal=*/{}, /*.name=*/"skipped"', fixed_source)
        self.assertIn("/*.traits=*/1,", fixed_source)
        self.assertIn('/*.effective_traits=*/{}, /*.name=*/"sparse"', fixed_source)
        self.assertIn(".ordinal = 3", fixed_source)
        self.assertIn('.name = "macro-arg"', fixed_source)
        self.assertNotIn(".ordinal = 1", fixed_source)
        self.assertNotIn('.name = "device"', fixed_source)
        self.assertNotIn('.name = "skipped"', fixed_source)
        self.assertNotIn(".traits = 1", fixed_source)
        self.assertNotIn('.name = "sparse"', fixed_source)

    def test_c_designated_initializers_are_allowed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsNone(
            output,
            [
                "C++ code must use comment field labels instead of designated "
                "initializers for MSVC portability",
                "[iree-cpp-designated-initializer]",
            ],
        )

    def test_direct_goto_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-direct-goto",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "direct goto is not allowed",
                "goto cleanup",
                "IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO",
                "[iree-direct-goto]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_computed_goto",
            ],
        )

    def test_guarded_release_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-guarded-release",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "release functions are null-safe",
                "iree_clang_tidy_style_resource_release",
                "if (resource) iree_clang_tidy_style_resource_release(resource);",
                "if (resource != NULL) {",
                "[iree-guarded-release]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_null_guard_ignored",
                "iree_clang_tidy_style_different_predicate_ignored",
                "iree_clang_tidy_style_extra_behavior_ignored",
                "iree_clang_tidy_style_deinitialize_ignored",
                "iree_clang_tidy_style_destroy_ignored",
                "iree_clang_tidy_style_multi_argument_release_ignored",
                "iree_clang_tidy_style_non_pointer_release_ignored",
                "iree_clang_tidy_style_indirect_release_ignored",
            ],
        )

    def test_guarded_release_is_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-guarded-release",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "release functions are null-safe",
                "[iree-guarded-release]",
            ],
        )
        self.assertIn(
            """void iree_clang_tidy_style_guarded_release(
    iree_clang_tidy_style_resource_t* resource) {
  iree_clang_tidy_style_resource_release(resource);
}""",
            fixed_source,
        )
        self.assertIn(
            """void iree_clang_tidy_style_guarded_release_with_clear(
    iree_clang_tidy_style_resource_t* resource) {
  iree_clang_tidy_style_resource_release(resource);
  resource = NULL;
}""",
            fixed_source,
        )
        self.assertNotIn(
            "if (resource) iree_clang_tidy_style_resource_release(resource);",
            fixed_source,
        )
        self.assertNotIn("if (resource != NULL) {", fixed_source)

    def test_test_status_macros_are_diagnosed_outside_tests(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-test-status-macro-scope",
            source=Path(__file__).resolve().parents[1]
            / "fixtures"
            / "status_macro_scope.c",
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "IREE_ASSERT_OK is a test-only status assertion macro",
                "IREE_EXPECT_OK is a test-only status assertion macro",
                "IREE_EXPECT_STATUS_IS is a test-only status assertion macro",
                "[iree-test-status-macro-scope]",
            ],
        )

    def test_test_status_macros_are_allowed_in_tests(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-test-status-macro-scope",
            source=clang_tidy_test.source_path(__file__, "status_macro_scope_test.cc"),
            compiler_args=["-std=c++17"],
        )
        self.assertContainsNone(
            output,
            [
                "test-only status assertion macro",
                "[iree-test-status-macro-scope]",
            ],
        )

    def test_raw_status_predicates_are_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-test-status-predicate",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "use IREE status test macros instead of EXPECT_TRUE with "
                "iree_status_is_ok",
                "use IREE status test macros instead of ASSERT_TRUE with "
                "iree_status_is_ok",
                "use IREE status test macros instead of EXPECT_FALSE with "
                "iree_status_is_ok",
                "use IREE status test macros instead of ASSERT_FALSE with "
                "iree_status_is_ok",
                "[iree-test-status-predicate]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_boolean_test_predicates_are_allowed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
