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


class LifecycleChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_lifecycle_naming_classifies_caller_owned_storage(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-lifecycle-naming",
            source=clang_tidy_test.source_path(__file__, "lifecycle_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "caller-owned output out_resource from "
                "iree_clang_tidy_in_place_resource_allocate uses allocate "
                "naming but is cleaned up by "
                "iree_clang_tidy_in_place_resource_deinitialize",
                "pointer-to-pointer output out_resource from "
                "iree_clang_tidy_view_initialize uses initialize naming "
                "without an explicit storage parameter",
                "[iree-lifecycle-naming]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_in_place_resource_initialize",
                "iree_clang_tidy_heap_resource_allocate",
                "iree_clang_tidy_arena_bitset_allocate",
                "iree_clang_tidy_other_resource_deinitialize",
                "iree_clang_tidy_view_with_storage_initialize",
                "iree_clang_tidy_view_with_named_storage_initialize",
            ],
        )


if __name__ == "__main__":
    unittest.main()
