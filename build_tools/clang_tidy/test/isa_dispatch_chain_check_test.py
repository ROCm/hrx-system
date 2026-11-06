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
_CHECK = "iree-isa-dispatch-chain"


class IsaDispatchChainCheckTest(clang_tidy_test.ClangTidyAssertions):
    def test_c_maximal_compound_multiline_and_macro_chains(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks=f"-*,{_CHECK}",
            source=clang_tidy_test.source_path(__file__, "isa_dispatch_chain_check.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertEqual(output.count(f"[{_CHECK}]"), 4)
        self.assertEqual(output.count("the first predicate over the limit is here"), 4)
        self.assertContainsAll(
            output,
            [
                "concrete isa dispatch contains 4 predicates; the configured "
                "maximum is 3",
                "concrete isa dispatch contains 5 predicates; the configured "
                "maximum is 3",
                "chain_fourth_isa",
                "compound_fourth_isa",
                "macro_fourth_isa",
                "five_fourth_isa",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "short_third_isa",
                "body_fourth_isa",
                "suppressed_fourth_isa",
            ],
        )

    def test_cpp_template_specializations_are_counted(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks=f"-*,{_CHECK}",
            source=clang_tidy_test.source_path(__file__, "isa_dispatch_chain_check.cc"),
            compiler_args=["-std=c++20"],
        )
        self.assertEqual(output.count(f"[{_CHECK}]"), 2)
        self.assertEqual(
            output.count(
                "concrete isa dispatch contains 4 predicates; the configured "
                "maximum is 3"
            ),
            2,
        )
        self.assertNotIn("cpp_non_template", output)

    def test_maximum_predicate_option_changes_the_boundary(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks=f"-*,{_CHECK}",
            source=clang_tidy_test.source_path(__file__, "isa_dispatch_chain_check.c"),
            compiler_args=["-std=gnu11"],
            clang_tidy_args=[
                '--config={"CheckOptions": {'
                '"iree-isa-dispatch-chain.MaximumIsaPredicates": "4"}}'
            ],
        )
        self.assertEqual(output.count(f"[{_CHECK}]"), 1)
        self.assertContainsAll(
            output,
            [
                "concrete isa dispatch contains 5 predicates; the configured "
                "maximum is 4",
                "five_fifth_isa",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "chain_fourth_isa",
                "compound_fourth_isa",
                "macro_fourth_isa",
            ],
        )


if __name__ == "__main__":
    unittest.main()
