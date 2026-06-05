# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest
from pathlib import Path

from build_tools.devtools import cmake as cmake_dev
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment, ToolMode


class CMakeTest(unittest.TestCase):
    def test_build_dir_defaults_to_repo_local_cmake_tree(self):
        self.assertEqual(
            cmake_dev.build_dir(
                environ={},
                state_file=Path("/nonexistent/cmake_build_dir"),
            ),
            REPO_ROOT / "build/cmake",
        )

    def test_build_dir_resolves_relative_paths_from_repo_root(self):
        self.assertEqual(
            cmake_dev.build_dir(Path("build/cmake-debug")),
            REPO_ROOT / "build/cmake-debug",
        )

    def test_build_dir_preserves_absolute_paths(self):
        self.assertEqual(cmake_dev.build_dir(Path("/tmp/cmake")), Path("/tmp/cmake"))

    def test_build_dir_prefers_environment_over_recorded_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_file = Path(temporary_dir) / "cmake_build_dir"
            state_file.write_text(
                str(Path(temporary_dir) / "recorded"),
                encoding="utf-8",
            )

            self.assertEqual(
                cmake_dev.build_dir(
                    environ={"IREE_CMAKE_BUILD_DIR": "build/from-env"},
                    state_file=state_file,
                ),
                REPO_ROOT / "build/from-env",
            )

    def test_build_dir_uses_recorded_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorded_build_dir = Path(temporary_dir) / "configured"
            state_file = Path(temporary_dir) / "cmake_build_dir"
            state_file.write_text(str(recorded_build_dir), encoding="utf-8")

            self.assertEqual(
                cmake_dev.build_dir(environ={}, state_file=state_file),
                recorded_build_dir,
            )

    def test_build_args_translate_target_shorthand(self):
        self.assertEqual(
            cmake_dev.build_args(["hrx", "iree-run-module", "--parallel", "8"]),
            ["--target", "hrx", "--target", "iree-run-module", "--parallel", "8"],
        )

    def test_configure_build_and_test_plans_use_selected_build_dir(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        configure_plan = cmake_dev.configure_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["-DIREE_HAL_DRIVER_AMDGPU=OFF"],
        )
        configure_description = configure_plan.describe()
        self.assertIn("build/cmake-debug", configure_description)
        self.assertIn("codemodel-v2", configure_description)
        self.assertIn("record CMake build directory", configure_description)
        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=OFF", configure_description)

        build_plan = cmake_dev.build_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["hrx"],
        )
        self.assertIn("--target hrx", build_plan.describe())

        test_plan = cmake_dev.test_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["-R", "hrx"],
        )
        self.assertIn("ctest", test_plan.describe())
        self.assertIn("-R hrx", test_plan.describe())

    def test_run_plan_resolves_target_with_cmake_file_api(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        plan = cmake_dev.run_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["iree-run-module", "--", "--help"],
        )
        description = plan.describe()

        self.assertIn("# cmake run iree-run-module", description)
        self.assertIn("CMake File API", description)
        self.assertIn("exec '<built executable>' --help", description)

    def test_run_parse_supports_print_path(self):
        command = cmake_dev.parse_run_args(["-p", "iree-run-module"])

        self.assertTrue(command.print_path)
        self.assertEqual(command.target, "iree-run-module")


if __name__ == "__main__":
    unittest.main()
