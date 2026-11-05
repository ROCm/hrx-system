#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import run_clang_tidy_action


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang-tidy", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--vfsoverlay", required=True)
    parser.add_argument("--source", required=True)
    args, unittest_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unittest_args]
    return args


_ARGS = parse_arguments()


class ActionRunnerTest(unittest.TestCase):
    def run_analysis(
        self,
        compiler_args: list[str],
        *,
        compiler: str = "clang",
        source: str | None = None,
        checks: str | None = "-*,iree-smoke",
        warnings_as_errors: str | None = None,
    ) -> tuple[int, str]:
        source = source or _ARGS.source
        options = []
        if checks is not None:
            options.append(f"--checks={checks}")
        if warnings_as_errors is not None:
            options.append(f"--warnings-as-errors={warnings_as_errors}")
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "diagnostics.txt"
            result = run_clang_tidy_action.main(
                [
                    "--clang-tidy",
                    _ARGS.clang_tidy,
                    "--compiler",
                    compiler,
                    "--plugin",
                    _ARGS.plugin,
                    "--config-file",
                    _ARGS.config_file,
                    "--vfsoverlay",
                    _ARGS.vfsoverlay,
                    "--source",
                    source,
                    "--output",
                    str(report),
                    *options,
                    "--",
                    *compiler_args,
                    source,
                ]
            )
            return result, report.read_text(encoding="utf-8")

    def test_enabled_policy_rejects_callback_outputs_before_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "completion.c"
            source.write_text(
                "typedef void (*completion_fn_t)(void* user_data, "
                "int* out_copy, void* scope);\n",
                encoding="utf-8",
            )
            result, output = self.run_analysis(
                ["-x", "c", "-std=c11"],
                source=str(source),
                checks="iree-output-parameter-order",
                warnings_as_errors="*",
            )
            self.assertNotEqual(result, 0, output)
            self.assertIn("[iree-output-parameter-order,-warnings-as-errors]", output)
            self.assertIn("output parameter 'out_copy'", output)

    def test_native_compiler_inputs_reach_analysis(self):
        # The input is the build action's relative path. Analysis must preserve
        # that spelling when the MSVC driver runs on a POSIX build worker.
        for compiler_args in (
            ["-x", "c", "-std=c11"],
            ["-x", "c++", "-std=c++20"],
            ["--driver-mode=cl", "--target=x86_64-pc-windows-msvc", "/TC", "/std:c11"],
            [
                "--driver-mode=cl",
                "--target=x86_64-pc-windows-msvc",
                "/TP",
                "/std:c++20",
            ],
        ):
            with self.subTest(compiler_args=compiler_args):
                result, output = self.run_analysis(compiler_args)
                self.assertEqual(result, 0, output)
                self.assertIn(
                    "IREE clang-tidy plugin smoke diagnostic [iree-smoke]", output
                )

    def test_compiler_relative_standard_headers_reach_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler = root / "bin" / "clang++"
            compiler.parent.mkdir()
            headers = root / "include" / "c++" / "v1"
            headers.mkdir(parents=True)
            header_name = "iree_action_runner_standard_header.h"
            (headers / header_name).write_text(
                "using iree_action_runner_standard_type = int;\n",
                encoding="utf-8",
            )
            result, output = self.run_analysis(
                ["-x", "c++", "-std=c++20", "-stdlib=libc++", "-include", header_name],
                compiler=str(compiler),
            )
            self.assertEqual(result, 0, output)
            self.assertIn(
                "IREE clang-tidy plugin smoke diagnostic [iree-smoke]", output
            )

    def test_compiler_failure_reaches_the_caller_and_report(self):
        result, output = self.run_analysis(
            ["-include", "__iree_missing_action_runner_test_header__.h"]
        )
        self.assertNotEqual(result, 0, output)
        self.assertIn("__iree_missing_action_runner_test_header__.h", output)
        self.assertIn("file not found", output)


if __name__ == "__main__":
    unittest.main()
