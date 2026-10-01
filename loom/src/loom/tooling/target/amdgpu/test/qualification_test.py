# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises native qualification admission and failures through loom-check."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class QualificationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.corpus = self.root / "corpus.loom-test"
        source = Path(_ARGS.accepted).read_text()
        self.corpus.write_text(source, newline="\n")
        self.fixture = self.root / "fixture.loom-test"
        self.fixture.write_text(
            "// TEMPLATE: corpus.loom-test\n" + source, newline="\n"
        )
        self.rejected = self.root / "rejected.loom-test"
        self.rejected.write_bytes(Path(_ARGS.rejected).read_bytes())

    def check(self, source, *arguments):
        return subprocess.run(
            [
                _ARGS.checker,
                "--target=amdgpu:gfx942",
                "--json=all",
                *arguments,
                str(source),
            ],
            capture_output=True,
            text=True,
        )

    def test_native_compilation_uses_authored_cases(self):
        result = self.check(self.fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["summary"], {"total": 2, "passed": 2, "failed": 0, "skipped": 0}
        )
        self.assertTrue(all(case["mode"] == "compile" for case in report["cases"]))

    def test_expected_target_diagnostic_passes(self):
        result = self.check(self.rejected)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["summary"], {"total": 1, "passed": 1, "failed": 0, "skipped": 0}
        )

    def test_wrong_diagnostic_identity_fails(self):
        self.rejected.write_text(
            self.rejected.read_text().replace("AMDGPU/026", "AMDGPU/025"),
            newline="\n",
        )
        result = self.check(self.rejected)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["skipped"], 0)

    def test_unannotated_unsupported_source_fails(self):
        self.rejected.write_text(
            "\n".join(
                line
                for line in self.rejected.read_text().splitlines()
                if not line.startswith("// ERROR")
            )
            + "\n",
            newline="\n",
        )
        result = self.check(self.rejected)
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["summary"]["passed"], 0)
        self.assertEqual(report["summary"]["failed"], 1)

    def test_compilation_uses_concrete_cases_without_template_synchronization(self):
        self.fixture.write_text(
            self.fixture.read_text().split("// ====", 1)[0], newline="\n"
        )
        result = self.check(self.fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["summary"], {"total": 1, "passed": 1, "failed": 0, "skipped": 0}
        )

    def test_compilation_needs_no_template_source(self):
        self.corpus.unlink()
        original = self.fixture.read_bytes()
        result = self.check(self.fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["summary"], {"total": 2, "passed": 2, "failed": 0, "skipped": 0}
        )
        self.assertEqual(self.fixture.read_bytes(), original)

    def test_update_does_not_rewrite_targeted_input(self):
        original = self.fixture.read_bytes()
        result = self.check(self.fixture, "--update")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "cannot maintain template sources or update RUN goldens", result.stderr
        )
        self.assertEqual(self.fixture.read_bytes(), original)

    def test_unknown_profile_is_not_a_skip(self):
        result = self.check(self.corpus, "--target=amdgpu:not-a-profile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not-a-profile", result.stderr)

    def test_malformed_bytecode_is_not_a_pass(self):
        source = self.root / "invalid.loombc"
        source.write_bytes(b"not a Loom bytecode module")
        result = self.check(source)
        self.assertNotEqual(result.returncode, 0)

    def test_empty_source_is_not_a_pass(self):
        source = self.root / "empty.loom-test"
        source.write_text("")
        result = self.check(source)
        self.assertNotEqual(result.returncode, 0)

    def test_input_directive_overrides_unknown_suffix(self):
        source = self.root / "input.unknown-test"
        source.write_text(
            "// INPUT: loom\n" + self.corpus.read_text().split("// ====", 1)[0]
        )
        result = self.check(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_entry_report_and_publication(self):
        artifact_path = self.root / "module.hsaco"
        report_path = self.root / "report.json"
        roots = ["address_guarded_rows", "address_materialized_wide_offset"]
        result = subprocess.run(
            [
                _ARGS.compiler,
                _ARGS.realizations,
                "--target=amdgpu:gfx942",
                *[f"--root=@{root}" for root in roots],
                f"--output={artifact_path}",
                "--compile-report=summary",
                f"--compile-report-output={report_path}",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(artifact_path.stat().st_size, 0)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["target_key"], "gfx942")
        self.assertEqual(report["entries"]["count"], len(roots))
        self.assertCountEqual(
            [row["source_function"] for row in report["entries"]["rows"]], roots
        )
        self.assertTrue(
            all(row["code_byte_count"] > 0 for row in report["entries"]["rows"])
        )

    def test_rejected_compilation_does_not_publish_output(self):
        artifact_path = self.root / "rejected.hsaco"
        for contents in (None, b"previous artifact"):
            with self.subTest(existing_output=contents is not None):
                if contents is not None:
                    artifact_path.write_bytes(contents)
                result = subprocess.run(
                    [
                        _ARGS.compiler,
                        str(self.rejected),
                        "--target=amdgpu:gfx942",
                        "--root=@unsupported_wave_size",
                        f"--output={artifact_path}",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("AMDGPU/026", result.stderr)
                self.assertEqual(result.stdout, "")
                if contents is None:
                    self.assertFalse(artifact_path.exists())
                else:
                    self.assertEqual(artifact_path.read_bytes(), contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checker")
    parser.add_argument("accepted")
    parser.add_argument("compiler")
    parser.add_argument("realizations")
    parser.add_argument("rejected")
    _ARGS = parser.parse_args()
    unittest.main(argv=[sys.argv[0]])
