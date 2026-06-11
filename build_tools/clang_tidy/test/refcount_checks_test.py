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


class RefCountChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_refcount_lifecycle_requires_void_and_null_safe_release(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-refcount-lifecycle",
            source=clang_tidy_test.source_path(__file__, "refcount_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "refcounted retain/release function "
                "iree_clang_tidy_refcount_status_retain must return void",
                "refcounted retain/release function "
                "iree_clang_tidy_refcount_status_release must return void",
                "refcounted release function "
                "iree_clang_tidy_refcount_unguarded_release "
                "must be null-safe",
                "refcounted release function "
                "iree_clang_tidy_refcount_asserting_release "
                "must be null-safe",
                "iree_atomic_ref_count_dec return value must be checked",
                "iree_atomic_ref_count_t field ref_count must be the first "
                "field in a refcounted object",
                "iree_atomic_ref_count_t field pending_submissions must model "
                "object lifetime",
                "iree_atomic_ref_count_t field queued_callbacks must model "
                "object lifetime",
                "resource is dereferenced after "
                "iree_clang_tidy_refcount_void_release releases it",
                "resources is dereferenced after "
                "iree_clang_tidy_refcount_void_release releases it",
                "resource is released by "
                "iree_clang_tidy_refcount_void_release after "
                "iree_clang_tidy_refcount_void_release already released it",
                "resource is used after "
                "iree_clang_tidy_refcount_void_release releases it",
                "cast_resource is used after "
                "iree_clang_tidy_refcount_void_release releases it",
                "assigned_resource is used after "
                "iree_clang_tidy_refcount_void_release releases it",
                "aliased_resource is used after "
                "iree_clang_tidy_refcount_void_release releases it",
                "returned_resource is used after "
                "iree_clang_tidy_refcount_void_release releases it",
                "guarded_resource is null-guarded before "
                "iree_clang_tidy_refcount_void_release",
                "null_compared_resource is null-guarded before "
                "iree_clang_tidy_refcount_void_release",
                "reversed_null_compared_resource is null-guarded before "
                "iree_clang_tidy_refcount_void_release",
                "[iree-refcount-lifecycle]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_refcount_void_retain",
                "refcounted retain/release function "
                "iree_clang_tidy_refcount_void_release",
                "iree_clang_tidy_refcount_early_null_release",
                "iree_clang_tidy_refcount_inline_null_release",
                "iree_clang_tidy_refcount_likely_null_release",
                "iree_clang_tidy_refcount_retain_then_double_release",
                "iree_clang_tidy_refcount_release_then_clear",
                "iree_clang_tidy_refcount_release_then_replace",
                "conditionally_released_resource",
                "iree_clang_tidy_refcount_release_in_branch_then_use",
                "iree_clang_tidy_refcount_local_counter_release",
                "iree_clang_tidy_refcount_lookup_retain",
                "iree_clang_tidy_virtual_memory_release",
                "status_released_resource",
                "guarded_status_released_resource",
                "side_counter",
            ],
        )


if __name__ == "__main__":
    unittest.main()
