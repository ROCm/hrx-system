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


class ExtentChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_empty_initializers_are_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-extent-empty-initializer",
            source=clang_tidy_test.source_path(__file__, "extent_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "use iree_byte_span_empty() instead of aggregate zero "
                "initialization for empty iree_byte_span_t values",
                "use iree_const_byte_span_empty() instead of aggregate zero "
                "initialization for empty iree_const_byte_span_t values",
                "use iree_string_view_empty() instead of aggregate zero "
                "initialization for empty iree_string_view_t values",
                "use iree_mutable_string_view_empty() instead of aggregate "
                "zero initialization for empty iree_mutable_string_view_t values",
                "use iree_hal_semaphore_list_empty() instead of aggregate "
                "zero initialization for empty iree_hal_semaphore_list_t values",
                "[iree-extent-empty-initializer]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_extent_allowed_initializers",
            ],
        )

    def test_empty_initializers_are_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-extent-empty-initializer",
            source=clang_tidy_test.source_path(__file__, "extent_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(output, ["[iree-extent-empty-initializer]"])
        self.assertIn(
            "iree_byte_span_t byte_span = iree_byte_span_empty();",
            fixed_source,
        )
        self.assertIn(
            "iree_const_byte_span_t const_byte_span = iree_const_byte_span_empty();",
            fixed_source,
        )
        self.assertIn(
            "iree_string_view_t string_view = iree_string_view_empty();",
            fixed_source,
        )
        self.assertIn(
            "iree_mutable_string_view_t mutable_string_view = "
            "iree_mutable_string_view_empty();",
            fixed_source,
        )
        self.assertIn(
            "iree_hal_semaphore_list_t semaphore_list = "
            "iree_hal_semaphore_list_empty();",
            fixed_source,
        )
        self.assertNotIn("iree_byte_span_t byte_span = {0};", fixed_source)
        self.assertNotIn(
            "iree_const_byte_span_t const_byte_span = {NULL, 0};",
            fixed_source,
        )

    def test_cpp_empty_initializers_are_diagnosed_and_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-extent-empty-initializer",
            source=clang_tidy_test.source_path(__file__, "extent_checks.cc"),
            compiler_args=["-std=c++17"],
        )
        self.assertContainsAll(
            output,
            [
                "use iree_string_view_empty() instead of aggregate zero "
                "initialization for empty iree_string_view_t values",
                "use iree_const_byte_span_empty() instead of aggregate zero "
                "initialization for empty iree_const_byte_span_t values",
                "[iree-extent-empty-initializer]",
            ],
        )
        self.assertIn(
            "iree_string_view_t string_view = iree_string_view_empty();",
            fixed_source,
        )
        self.assertIn(
            "iree_const_byte_span_t const_byte_span = iree_const_byte_span_empty();",
            fixed_source,
        )
        self.assertNotIn("iree_string_view_t string_view = {};", fixed_source)
        self.assertNotIn(
            "iree_const_byte_span_t const_byte_span = {};",
            fixed_source,
        )

    def test_pointer_empty_predicates_are_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-extent-empty-predicate",
            source=clang_tidy_test.source_path(__file__, "extent_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "test empty iree_string_view_t values with "
                "iree_string_view_is_empty(); pointer-null is malformedness, "
                "not emptiness",
                "test empty iree_const_byte_span_t values with "
                "iree_const_byte_span_is_empty(); pointer-null is "
                "malformedness, not emptiness",
                "[iree-extent-empty-predicate]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_extent_allowed_predicates",
            ],
        )

    def test_pointer_empty_predicates_are_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-extent-empty-predicate",
            source=clang_tidy_test.source_path(__file__, "extent_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(output, ["[iree-extent-empty-predicate]"])
        self.assertIn(
            "if (iree_string_view_is_empty(view)) return 1;",
            fixed_source,
        )
        self.assertIn("if (!view.data || !view.size) return;", fixed_source)
        self.assertIn(
            "if (iree_const_byte_span_is_empty(span)) return;",
            fixed_source,
        )
        self.assertIn(
            "if (span.data == NULL || span.data_length == 0) return;",
            fixed_source,
        )
        self.assertNotIn("view.data == NULL || view.size == 0", fixed_source)
        self.assertNotIn("!view.data && !view.size", fixed_source)
        self.assertNotIn("!span.data && !span.data_length", fixed_source)


if __name__ == "__main__":
    unittest.main()
