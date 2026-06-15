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


class SmokeTest(clang_tidy_test.ClangTidyAssertions):
    def test_plugin_loads_and_emits_diagnostics(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-smoke",
            source=clang_tidy_test.source_path(__file__, "smoke.c"),
        )
        self.assertIn("IREE clang-tidy plugin smoke diagnostic [iree-smoke]", output)


if __name__ == "__main__":
    unittest.main()
