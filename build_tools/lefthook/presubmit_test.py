# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

PRESUBMIT_PATH = Path(__file__).with_name("presubmit.py")
PRESUBMIT_SPEC = importlib.util.spec_from_file_location("presubmit", PRESUBMIT_PATH)
if PRESUBMIT_SPEC is None or PRESUBMIT_SPEC.loader is None:
    raise RuntimeError(f"unable to load {PRESUBMIT_PATH}")
presubmit = importlib.util.module_from_spec(PRESUBMIT_SPEC)
sys.modules[PRESUBMIT_SPEC.name] = presubmit
PRESUBMIT_SPEC.loader.exec_module(presubmit)


class PresubmitTest(unittest.TestCase):
    def test_semgrep_candidates_require_configured_prefix_and_extension(self):
        with (
            mock.patch.object(presubmit, "SEMGREP_PATH_PREFIXES", ("project/src/",)),
            mock.patch.object(presubmit, "SEMGREP_EXTENSIONS", {".c"}),
        ):
            self.assertTrue(presubmit.is_semgrep_candidate_file("project/src/file.c"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("project/src/file.h"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("other/src/file.c"))

    def test_semgrep_scan_command_uses_local_error_rules(self):
        with mock.patch.dict(os.environ, {"IREE_SEMGREP_JOBS": "7"}):
            command = presubmit.semgrep_scan_command(["runtime/src/iree/base/status.c"])

        self.assertEqual(command[0:2], ["semgrep", "scan"])
        self.assertIn("--metrics=off", command)
        self.assertIn("--disable-version-check", command)
        self.assertIn("--strict", command)
        self.assertIn("--error", command)
        self.assertIn("ERROR", command)
        self.assertIn(presubmit.SEMGREP_CONFIG, command)
        self.assertIn("7", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_semgrep_default_jobs_are_capped_on_large_machines(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=192),
        ):
            self.assertEqual(presubmit.semgrep_jobs(), 14)

    def test_default_profile_has_no_static_analysis_provider(self):
        ok = presubmit.run_static_analysis(
            ["runtime/src/iree/base/status.c"], profile="default", verbose=False
        )

        self.assertTrue(ok)

    def test_missing_static_tool_is_only_fatal_in_ci(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(presubmit.shutil, "which", return_value=None),
        ):
            self.assertTrue(
                presubmit.require_static_tool(
                    "missing-tool", "Missing tool", "paranoid"
                )
            )
            self.assertFalse(
                presubmit.require_static_tool("missing-tool", "Missing tool", "ci")
            )

        self.assertIn("[skip]", output.getvalue())
        self.assertIn("[fail]", output.getvalue())

    def test_requirements_files_trigger_root_devtools_tests(self):
        self.assertTrue(presubmit.is_root_devtools_trigger("requirements-analysis.in"))
        self.assertTrue(
            presubmit.is_root_devtools_trigger("requirements-analysis.lock.txt")
        )
        self.assertFalse(presubmit.is_root_devtools_trigger("runtime/requirements.txt"))

    def test_existing_project_scripts_include_loom(self):
        self.assertIn(
            "loom",
            {project.name for project in presubmit.existing_project_scripts()},
        )


if __name__ == "__main__":
    unittest.main()
