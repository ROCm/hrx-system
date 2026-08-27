#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    args, unittest_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unittest_args]
    return args


_ARGS = parse_arguments()


class RecursionCrossTranslationUnitAnalysisTest(unittest.TestCase):
    def test_reports_aggregated_cycle(self):
        report = _ARGS.report.read_text(encoding="utf-8")
        self.assertEqual(report.count("[iree-unbounded-recursion]"), 1, report)
        self.assertIn(
            "cross-translation-unit call cycle containing 2 functions", report
        )
        self.assertIn("recursion_cross_tu_a", report)
        self.assertIn("recursion_cross_tu_b", report)


if __name__ == "__main__":
    unittest.main()
