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


class DesignatedInitializerCheckTest(clang_tidy_test.ClangTidyAssertions):
    def test_comment_labels_and_setup_blocks_are_fixed_and_compile(self):
        source = clang_tidy_test.source_path(
            __file__, "designated_initializer_check.cc"
        )
        header = clang_tidy_test.source_path(__file__, "designated_initializer_check.h")
        output, fixed_source, fixed_companions = (
            clang_tidy_test.run_clang_tidy_fix_and_compile(
                clang_tidy=_ARGS.clang_tidy,
                clangxx=_ARGS.clangxx,
                plugin=_ARGS.plugin,
                checks="-*,iree-cpp-designated-initializer",
                source=source,
                companion_files=[header],
                compiler_args=["-std=c++20", "-pedantic-errors", "-Werror"],
                clang_tidy_args=["--header-filter=.*"],
            )
        )

        self.assertContainsAll(
            output,
            [
                "replace comment field label with a C++20 designated initializer",
                "comment label names 'name', but this positional initializer "
                "selects 'ordinal'",
                "comment label names 'real', but this positional initializer "
                "selects 'integer'",
                "comment field label cannot be represented as a C++20 designator",
                "comment field label cannot be converted until every initializer "
                "element is representable as a designator",
                "fold aggregate setup into C++20 designated initialization",
                "aggregate setup cannot be folded: member assignments are not in "
                "declaration order",
                "aggregate setup cannot be folded: an assignment value may change "
                "initialization semantics",
                "aggregate setup cannot be folded: member assignments are separated "
                "by observation or control flow",
                "aggregate setup cannot be folded: union member activation differs "
                "between initialization and assignment",
                "aggregate setup cannot be folded: the aggregate has a default "
                "member initializer",
                "aggregate setup cannot be folded: the aggregate has nontrivial "
                "initialization or assignment",
                "aggregate setup cannot be folded: the empty initializer is "
                "produced by a macro",
                "[iree-cpp-designated-initializer]",
                "Suppressed 1 warnings (1 NOLINT)",
            ],
        )
        self.assertIn(".ordinal = 1,", fixed_source)
        self.assertIn(".inner = {", fixed_source)
        self.assertIn(".x = 3,", fixed_source)
        self.assertIn("Choice labeled_union = {.integer = 9};", fixed_source)
        self.assertIn("WithAnonymous anonymous_label = {.integer = 12", fixed_source)
        self.assertIn(".ordinal = CONFIG_VALUE(22),", fixed_source)
        self.assertIn(".flags = CONFIG_VALUE(23),", fixed_source)
        self.assertIn("Config macro_config = MAKE_CONFIG(21);", fixed_source)
        self.assertIn("Numbers configured = {.first = 20, .second = 21};", fixed_source)
        self.assertIn("Config sparse = {.flags = 22};", fixed_source)
        self.assertIn(
            "Numbers evaluated_in_order = {.first = Next(), .second = Next()};",
            fixed_source,
        )

        self.assertIn("/*.ordinal=*/7", fixed_source)
        self.assertIn("/*.flags=*/8", fixed_source)
        self.assertIn("Config stale_label = {/*.name=*/10};", fixed_source)
        self.assertIn("Choice stale_union_label = {/*.real=*/11};", fixed_source)
        self.assertIn(
            "WithAnonymous stale_anonymous_label = {/*.real=*/14",
            fixed_source,
        )
        self.assertIn("DerivedConfig base_label = {/*.base=*/{16}", fixed_source)
        self.assertIn("OuterConfig brace_elided = {/*.inner=*/18", fixed_source)
        self.assertIn("Config { /*.ordinal=*/ value }", fixed_source)
        self.assertIn("reordered.second = 23;", fixed_source)
        self.assertIn("self_referencing.first = 25;", fixed_source)
        self.assertIn("aliased.first = 26;", fixed_source)
        self.assertIn("observed.first = 28;", fixed_source)
        self.assertIn("conditional.first = 30;", fixed_source)
        self.assertIn("union_setup.integer = 31;", fixed_source)
        self.assertIn("defaulted.second = 32;", fixed_source)
        self.assertIn("nontrivial.member = 33;", fixed_source)
        self.assertIn("narrowing.value = -1;", fixed_source)
        self.assertIn("macro_initialized.first = 34;", fixed_source)
        self.assertIn("suppressed.first = 35;", fixed_source)
        self.assertIn("/* Preserve the explicit zero setup. */", fixed_source)
        self.assertIn("commented_initializer.first = 36;", fixed_source)
        self.assertIn(
            "// Preserve the explanation attached to the assignment.",
            fixed_source,
        )
        self.assertIn("comment_before_assignment.first = 37;", fixed_source)
        self.assertIn(
            "// Preserve the explanation attached to the second assignment.",
            fixed_source,
        )
        self.assertIn("comment_between_assignments.second = 39;", fixed_source)

        self.assertIn("return {/*.value=*/1};", fixed_companions[header.name])
        self.assertNotIn("designated_initializer_check.h:", output)

    def test_setup_block_folding_can_be_disabled(self):
        source = clang_tidy_test.source_path(
            __file__, "designated_initializer_check.cc"
        )
        output, fixed_source, _ = clang_tidy_test.run_clang_tidy_fix_and_compile(
            clang_tidy=_ARGS.clang_tidy,
            clangxx=_ARGS.clangxx,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=source,
            companion_files=[
                clang_tidy_test.source_path(__file__, "designated_initializer_check.h")
            ],
            compiler_args=["-std=c++20", "-pedantic-errors", "-Werror"],
            clang_tidy_args=[
                '--config={"CheckOptions": {'
                '"iree-cpp-designated-initializer.EnableSetupBlockFolding": '
                '"false"}}'
            ],
        )

        self.assertIn(
            "replace comment field label with a C++20 designated initializer",
            output,
        )
        self.assertNotIn("aggregate setup", output)
        self.assertIn("Numbers configured = {};", fixed_source)
        self.assertIn("configured.first = 20;", fixed_source)

    def test_comment_label_conversion_can_be_disabled(self):
        source = clang_tidy_test.source_path(
            __file__, "designated_initializer_check.cc"
        )
        output, fixed_source, _ = clang_tidy_test.run_clang_tidy_fix_and_compile(
            clang_tidy=_ARGS.clang_tidy,
            clangxx=_ARGS.clangxx,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=source,
            companion_files=[
                clang_tidy_test.source_path(__file__, "designated_initializer_check.h")
            ],
            compiler_args=["-std=c++20", "-pedantic-errors", "-Werror"],
            clang_tidy_args=[
                '--config={"CheckOptions": {'
                '"iree-cpp-designated-initializer.EnableCommentLabelConversion": '
                '"false"}}'
            ],
        )

        self.assertNotIn("comment field label", output)
        self.assertNotIn("comment label names", output)
        self.assertIn("fold aggregate setup", output)
        self.assertIn("Config labeled_config = {", fixed_source)
        self.assertIn("/*.ordinal=*/1,", fixed_source)
        self.assertIn("Numbers configured = {.first = 20, .second = 21};", fixed_source)

    def test_check_is_inactive_before_cxx20(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=clang_tidy_test.source_path(
                __file__, "designated_initializer_check.cc"
            ),
            compiler_args=["-std=c++17"],
        )
        self.assertNotIn("[iree-cpp-designated-initializer]", output)

    def test_c_designated_initializers_are_unchanged(self):
        source = clang_tidy_test.source_path(__file__, "designated_initializer_check.c")
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-cpp-designated-initializer",
            source=source,
            compiler_args=["-std=c11"],
        )
        self.assertNotIn("[iree-cpp-designated-initializer]", output)
        self.assertEqual(fixed_source, source.read_text())


if __name__ == "__main__":
    unittest.main()
