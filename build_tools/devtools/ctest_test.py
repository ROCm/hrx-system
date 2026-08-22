# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import unittest
from pathlib import Path

from build_tools.devtools import ctest as ctest_dev


def ctest_model(*tests: dict) -> str:
    return json.dumps(
        {
            "kind": "ctestInfo",
            "version": {"major": 1, "minor": 0},
            "tests": list(tests),
        }
    )


def test_record(name: str) -> dict:
    return {"name": name, "properties": []}


def build_target_catalog(*entries: tuple[str, object]) -> str:
    return json.dumps(
        {
            "kind": "ireeCtestBuildTargets",
            "version": 1,
            "tests": dict(entries),
        }
    )


class CTestTest(unittest.TestCase):
    def test_parses_and_deduplicates_build_targets(self):
        selection = ctest_dev.parse_ctest_selection(
            ctest_model(
                test_record("compiled"),
                test_record("native"),
                test_record("source-only"),
                test_record("shared-tool"),
            ),
            build_target_catalog(
                ("compiled", ["compiled_root"]),
                ("native", ["native_root", "tool_root"]),
                ("source-only", []),
                ("shared-tool", ["tool_root"]),
            ),
        )

        self.assertEqual(
            selection.test_names,
            ("compiled", "native", "source-only", "shared-tool"),
        )
        self.assertEqual(
            selection.build_targets,
            ("compiled_root", "native_root", "tool_root"),
        )

    def test_rejects_missing_or_malformed_build_metadata(self):
        invalid_models = [
            (
                ctest_model(test_record("unmanaged")),
                build_target_catalog(),
                "selected CTest test unmanaged is missing from the "
                "CMake build-target catalog",
            ),
            (
                ctest_model(test_record("malformed")),
                build_target_catalog(("malformed", [1])),
                "selected CTest test malformed has an invalid "
                "build-target catalog entry: 1",
            ),
            (
                ctest_model(test_record("malformed")),
                build_target_catalog(("malformed", {})),
                "selected CTest test malformed has a non-list "
                "build-target catalog entry",
            ),
        ]
        for model, catalog, expected_message in invalid_models:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(
                    ctest_dev.CTestMetadataError,
                    expected_message,
                ):
                    ctest_dev.parse_ctest_selection(model, catalog)

    def test_build_commands_are_stable_and_command_length_safe(self):
        commands = ctest_dev.cmake_build_commands(
            "cmake",
            Path("build"),
            tuple(f"target_{index:02d}" for index in range(20)),
            build_config="RelWithDebInfo",
            max_command_length=100,
        )

        self.assertGreater(len(commands), 1)
        self.assertEqual(
            [
                target
                for command in commands
                for target in command[command.index("--target") + 1 :]
            ],
            [f"target_{index:02d}" for index in range(20)],
        )
        for command in commands:
            self.assertLessEqual(
                sum(len(argument) + 1 for argument in command),
                100,
            )
            self.assertEqual(
                command[:6],
                [
                    "cmake",
                    "--build",
                    "build",
                    "--config",
                    "RelWithDebInfo",
                    "--target",
                ],
            )

    def test_ctest_build_config_uses_the_last_explicit_value(self):
        self.assertEqual(
            ctest_dev.ctest_build_config(
                [
                    "-CDebug",
                    "--build-config",
                    "RelWithDebInfo",
                    "--build-config=Release",
                ]
            ),
            "Release",
        )
        self.assertIsNone(ctest_dev.ctest_build_config(["-R", "test"]))

    def test_ctest_arguments_drive_both_selection_and_execution(self):
        arguments = [
            "-R",
            "required",
            "-E",
            "excluded",
            "-L",
            "runtime-resource=cpu",
            "-LE",
            "benchmark",
            "-FA",
            "fixture-any",
            "-FS",
            "fixture-setup",
            "-FC",
            "fixture-cleanup",
            "--rerun-failed",
            "--tests-from-file",
            "selected.txt",
            "--exclude-from-file",
            "excluded.txt",
            "--no-tests=error",
        ]

        selection_command = ctest_dev.ctest_selection_command(
            "ctest",
            Path("build"),
            arguments,
        )
        run_command = ctest_dev.ctest_run_command(
            "ctest",
            Path("build"),
            arguments,
        )

        self.assertEqual(selection_command[3:-1], arguments)
        self.assertEqual(selection_command[-1], "--show-only=json-v1")
        self.assertEqual(run_command[4:], arguments)

    def test_inspection_modes_do_not_request_a_build(self):
        for arguments in (
            ["-N"],
            ["--show-only=json-v1"],
            ["--print-labels"],
            ["--list-presets"],
        ):
            with self.subTest(arguments=arguments):
                self.assertTrue(ctest_dev.is_inspection_only(arguments))
        self.assertFalse(ctest_dev.is_inspection_only(["-R", "test"]))


if __name__ == "__main__":
    unittest.main()
