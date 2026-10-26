# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native reports survive per-function scratch and compare through the CLI."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from loom.reporting.compile_report import load_compile_report
from loom.tools import compile_report


class NativeReportTest(unittest.TestCase):
    def check_report(self, source, target, roots, expected_exports, helper):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            documents = {}
            for mode in ("none", "summary", "details"):
                artifact = directory / "native.bin"
                report = directory / (mode + ".json")
                command = [
                    compiler,
                    source,
                    "--target=" + target,
                    "--output=" + str(artifact),
                ] + ["--root=" + root for root in roots]
                if mode != "none":
                    command += [
                        "--compile-report=" + mode,
                        "--compile-report-output=" + str(report),
                    ]
                subprocess.run(command, check=True)
                if mode == "none":
                    reference = artifact.read_bytes()
                    continue
                # Reporting must not change code, layout, or artifact metadata.
                self.assertEqual(artifact.read_bytes(), reference)
                document = load_compile_report(report)
                documents[mode] = document
                self.assertEqual(document.report["target_family"], "x86")
                names = {entry["function"] for entry in document.entries}
                self.assertTrue(expected_exports <= names, names)
                self.assertEqual(len(names), len(document.entries))
                self.assertTrue(any(name.startswith(helper) for name in names), names)
                for entry in document.entries:
                    self.assertTrue(entry["target_bundle"])
                    self.assertGreater(entry["instruction_count"], 0)
                    self.assertGreaterEqual(
                        entry["code_byte_count"], entry["instruction_count"]
                    )
                    self.assertGreater(entry["planning"]["frame_build_count"], 0)
                    self.assertGreater(entry["allocation_assignment_count"], 0)
            self.assertEqual(
                [entry["function"] for entry in documents["summary"].entries],
                [entry["function"] for entry in documents["details"].entries],
            )
            # Self-comparison exercises complete multi-function matching through
            # the public consumer, including helpers and generated library glue.
            with contextlib.redirect_stdout(io.StringIO()):
                report = str(directory / "details.json")
                self.assertEqual(compile_report.main(["diff", report, report]), 0)

    def test_callable_and_private_helper(self):
        self.check_report(
            callable_source,
            "x86:znver5",
            ["call_pair"],
            {"call_pair"},
            "draw_arguments",
        )

    def test_task_entries_and_callable_helper(self):
        self.check_report(
            task_source,
            "x86:avx2",
            [],
            {
                "grid",
                "fill_byte",
                "constant_carriers",
                "iree_hal_executable_library_query",
            },
            "mix",
        )

    def test_selected_processor_model_controls_object_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "schedule.o"
            subprocess.run(
                [
                    compiler,
                    schedule_model_source,
                    "--pipeline=none",
                    "--format=x86-elf",
                    "--output=" + str(artifact),
                ],
                check=True,
            )
            contents = artifact.read_bytes()
            interleaved_roots = bytes.fromhex("c5fc51c0 c5fc51c9 c5fc51c0 c5fc51c9")
            source_order_roots = bytes.fromhex("c5fc51c0 c5fc51c0 c5fc51c9 c5fc51c9")
            self.assertIn(interleaved_roots, contents)
            self.assertNotIn(source_order_roots, contents)


if __name__ == "__main__":
    compiler, callable_source, task_source, schedule_model_source = sys.argv[1:]
    unittest.main(argv=[sys.argv[0]])
