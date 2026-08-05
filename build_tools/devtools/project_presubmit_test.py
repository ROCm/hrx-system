# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import project_presubmit


class ProjectPresubmitTest(unittest.TestCase):
    def test_common_arguments_expose_project_hygiene_phase(self):
        parser = argparse.ArgumentParser()
        project_presubmit.add_common_arguments(parser, project_name="example")

        args = parser.parse_args(
            ["--hygiene", "--lane", "cmake", "--files-from", "paths.txt"]
        )

        self.assertTrue(args.hygiene)
        self.assertFalse(args.tests)
        self.assertEqual(args.lane, "cmake")
        self.assertEqual(args.files_from, "paths.txt")

    def test_cmake_build_dir_uses_environment_override(self):
        with mock.patch.dict(
            os.environ,
            {project_presubmit.CMAKE_BUILD_DIR_ENV: "/tmp/iree-cmake-build"},
        ):
            self.assertEqual(
                project_presubmit.cmake_build_dir(Path("/repo")),
                Path("/tmp/iree-cmake-build"),
            )

    def test_validate_rejects_incomplete_ninja_build_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            build_dir = Path(temporary_directory)
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertFalse(
                    project_presubmit.validate_cmake_build_tree("runtime", build_dir)
                )

            self.assertIn("mixed generator state", output.getvalue())


if __name__ == "__main__":
    unittest.main()
