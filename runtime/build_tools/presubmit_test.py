# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def load_presubmit_module():
    presubmit_path = Path(__file__).with_name("presubmit.py")
    spec = importlib.util.spec_from_file_location("runtime_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimePresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_tests_leave_resource_selection_to_execution_policy(self):
        command = self.presubmit.BAZEL_TESTS.test_command()

        self.assertEqual(command[:3], ["bazel", "test", "--config=presubmit"])
        self.assertIn("--", command)
        self.assertIn("//runtime/...", command)

        self.assertIn("--test_tag_filters=-manual", command)
        self.assertIn(
            "-//runtime/src/iree/hal/driver/task/executable/elf:elf_module_test",
            command,
        )

    def test_bazel_selection_receives_inputs_outside_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            file_list = Path(directory) / "files.txt"
            file_list.write_text("build_tools/shared/tool.py\n", encoding="utf-8")
            args = types.SimpleNamespace(
                tests=True, lane="bazel", files_from=str(file_list)
            )
            with (
                mock.patch.object(self.presubmit, "parse_arguments", return_value=args),
                mock.patch.object(
                    self.presubmit.bazel_test_selection.TestSuite,
                    "run",
                    return_value=True,
                ) as run,
            ):
                self.assertEqual(self.presubmit.main(), 0)
            run.assert_called_once_with(
                self.presubmit.REPO_ROOT, paths=["build_tools/shared/tool.py"]
            )

    def test_bazel_tests_use_selected_execution_policy(self):
        with mock.patch.dict(
            os.environ,
            {
                self.presubmit.project_presubmit.BAZEL_CONFIGS_ENV: json.dumps(
                    ["remote-execution", "local-tests"]
                )
            },
            clear=True,
        ):
            command = self.presubmit.BAZEL_TESTS.test_command()

        self.assertIn("--config=remote-execution", command)
        self.assertIn("--config=local-tests", command)


if __name__ == "__main__":
    unittest.main()
