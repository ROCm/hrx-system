#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregator", required=True, type=Path)
    parser.add_argument("--clang-tidy", required=True, type=Path)
    parser.add_argument("--plugin", type=Path)
    args, unittest_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unittest_args]
    return args


_ARGS = parse_arguments()


class RecursionCrossTranslationUnitDriverTest(unittest.TestCase):
    def test_plugin_summaries_feed_shared_aggregator(self):
        source_directory = Path(__file__).resolve().parent
        sources = [
            source_directory / "recursion_cross_tu_a.c",
            source_directory / "recursion_cross_tu_b.c",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            summaries: list[Path] = []
            for source in sources:
                summary = temporary_path / f"{source.stem}.json"
                environment = os.environ.copy()
                environment["IREE_CLANG_TIDY_RECURSION_SUMMARY"] = str(summary)
                environment["IREE_CLANG_TIDY_RECURSION_DIAGNOSTICS"] = "0"
                command = [str(_ARGS.clang_tidy)]
                if _ARGS.plugin:
                    command.append(f"--load={_ARGS.plugin}")
                completed = subprocess.run(
                    [
                        *command,
                        "--checks=-*,iree-unbounded-recursion",
                        str(source),
                        "--",
                        "-std=c11",
                    ],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertTrue(summary.is_file(), completed.stdout)
                summaries.append(summary)

            report = temporary_path / "report.txt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(_ARGS.aggregator),
                    "--output",
                    str(report),
                    "--allow-diagnostics",
                    *map(str, summaries),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(
            report_text.count("[iree-unbounded-recursion]"), 1, report_text
        )
        self.assertIn(
            "cross-translation-unit call cycle containing 2 functions", report_text
        )
        self.assertIn("recursion_cross_tu_a", report_text)
        self.assertIn("recursion_cross_tu_b", report_text)


if __name__ == "__main__":
    unittest.main()
