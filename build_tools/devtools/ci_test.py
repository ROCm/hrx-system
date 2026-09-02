# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import ci, ci_config


class CiTest(unittest.TestCase):
    def uses_cmake_build_dir(self, step: ci.CiStep, command_name: str) -> bool:
        expected_args = (
            "--cmake-build-dir",
            str(ci.CMAKE_CI_BUILD_ROOT / command_name),
        )
        return any(
            step.argv[index : index + 2] == expected_args
            for index in range(len(step.argv) - 1)
        )

    def ctest_exclude_regexes(self, step: ci.CiStep) -> list[str]:
        return [
            step.argv[index + 1]
            for index, arg in enumerate(step.argv[:-1])
            if arg == "-E"
        ]

    def workflow_job_block(self, path: str, job_name: str) -> str:
        """Returns the body of ``job_name`` in the workflow at ``path``.

        A job body is blank lines and lines indented at least four spaces, so
        the first line that is neither ends it: the next job's key, a comment
        introducing that job, or anything after ``jobs:``. Letting a comment
        run on into the preceding body would make assertions over that body
        report the wrong thing in both directions -- ``assertIn`` could pass on
        text belonging to the next job, and ``assertNotIn`` could fail on it.
        """
        text = Path(path).read_text()
        match = re.search(
            rf"^  {re.escape(job_name)}:\n"
            r"(?P<body>.*?)(?=^(?![ \t]*\n)(?! {4})|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        return match.group("body")

    def test_cpu_dry_run_exposes_copyable_commands(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertIn("dev.py bazel build -- //runtime/...", text)
        self.assertIn("dev.py bazel test --test_tag_filters=", text)
        self.assertIn(" -- //runtime/...", text)
        self.assertIn("-//runtime/src/iree/hal/drivers/amdgpu/...", text)
        self.assertIn("-//runtime/src/iree/hal/drivers/vulkan/...", text)
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES),
            text,
        )

    def test_bazel_profiles_are_named_for_each_build_and_test_phase(self):
        profile_dir = Path("/tmp/iree-bazel-profiles")
        resolved_profile_dir = profile_dir.resolve()
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu",
                "--target",
                "//runtime/...",
                "--bazel-profile-dir",
                str(profile_dir),
            ]
        )

        steps = ci.steps_from_args(args)

        self.assertEqual(
            [
                arg
                for step in steps
                for arg in step.argv
                if arg.startswith("--profile=")
            ],
            [
                f"--profile={resolved_profile_dir / 'build-iree.profile.gz'}",
                f"--profile={resolved_profile_dir / 'test-iree.profile.gz'}",
            ],
        )

    def test_bazel_profile_names_must_be_unique(self):
        duplicate_steps = [
            ci.bazel_build_step("Compile IREE", ("//runtime/...",)),
            ci.bazel_test_step("Compile IREE", ("//runtime/...",)),
        ]

        with self.assertRaisesRegex(ValueError, "Bazel profile path collision"):
            ci.add_bazel_profiles(duplicate_steps, Path("/tmp/profiles"))

    def test_non_bazel_commands_reject_bazel_profile_directory(self):
        for command in ("iree-cmake-cpu", "iree-importers-tilelang"):
            with self.subTest(command=command):
                args = ci.parse_arguments(
                    [command, "--bazel-profile-dir", "/tmp/profiles"]
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "--bazel-profile-dir is only supported for Bazel CI commands",
                ):
                    ci.steps_from_args(args)

    def test_main_creates_bazel_profile_directory_only_when_executing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir) / "profiles"
            argv = [
                "iree-bazel-cpu",
                "--target",
                "//runtime/...",
                "--bazel-profile-dir",
                str(profile_dir),
            ]
            with mock.patch.object(ci, "run_steps", return_value=0):
                self.assertEqual(ci.main([*argv, "--dry-run"]), 0)
                self.assertFalse(profile_dir.exists())

                self.assertEqual(ci.main(argv), 0)
                self.assertTrue(profile_dir.is_dir())

    def test_bazel_default_targets_include_loom(self):
        args = ci.parse_arguments(["iree-bazel-cpu"])

        steps = ci.steps_from_args(args)
        build_step = next(step for step in steps if step.name == "Build IREE")
        test_step = next(step for step in steps if step.name == "Test IREE")

        self.assertIn("//runtime/...", build_step.argv)
        self.assertIn("//loom/...", build_step.argv)
        self.assertIn("//runtime/...", test_step.argv)
        self.assertIn("//loom/...", test_step.argv)

    def test_bazel_repository_build_covers_supported_linux_graph(self):
        args = ci.parse_arguments(["iree-bazel-repository-build"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        self.assertEqual(
            [step.name for step in steps], ["Configure Bazel", "Build repository"]
        )
        configure_step = steps[0]
        for define in (
            "IREE_HAL_DRIVER_AMDGPU",
            "IREE_HAL_DRIVER_TASK",
            "IREE_HAL_DRIVER_VULKAN",
            "IREE_HAL_DRIVER_WEBGPU",
        ):
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        self.assertIn(
            "--//loom/config/target:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_TARGETS),
            configure_step.argv,
        )
        self.assertIn(
            "--//loom/config/import:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_IMPORTERS),
            configure_step.argv,
        )
        self.assertIn("-DIREE_ROCM_PATH=/tmp/rocm-root", configure_step.argv)
        build_step = steps[-1]
        self.assertEqual(build_step.argv[-1], "//...")
        self.assertFalse(any(arg.startswith("-//") for arg in build_step.argv))
        self.assertFalse(any(step.name.startswith("Test") for step in steps))

    def test_bazel_repository_build_rejects_partial_target_scope(self):
        args = ci.parse_arguments(
            ["iree-bazel-repository-build", "--target", "//runtime/..."]
        )

        with self.assertRaisesRegex(ValueError, "repository-wide"):
            ci.steps_from_args(args)

    def test_bazel_repository_integration_exercises_supported_repository_tests(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-repository-integration",
                "--amdgpu-target",
                "gfx11-generic",
            ]
        )

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        self.assertEqual(
            [step.name for step in steps],
            [
                "Configure Bazel",
                "Build repository",
                "Build AMDGPU device toolchain smoke",
                "Test repository",
                "Test lock-free Bazel launch",
                "Run dynamic library environment smoke",
                "Run executable alias smoke",
            ],
        )
        configure_step = steps[0]
        self.assertIn("-DIREE_ROCM_PATH=/tmp/rocm-root", configure_step.argv)
        device_build_step = steps[2]
        for target in ci_config.BAZEL_REPOSITORY_INTEGRATION_DEVICE_TARGETS:
            self.assertIn(target, device_build_step.argv)
        self.assertIn(
            "--//runtime/src/iree/hal/drivers/amdgpu:targets=gfx11-generic",
            device_build_step.argv,
        )
        integration_test_step = steps[3]
        self.assertEqual(integration_test_step.argv[-1], "//...")
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES),
            integration_test_step.argv,
        )
        self.assertEqual(
            steps[5].argv[-1],
            ci_config.BAZEL_REPOSITORY_INTEGRATION_DYNAMIC_LIBRARY_TARGET,
        )
        self.assertEqual(
            steps[6].argv[-3:],
            (
                ci_config.BAZEL_REPOSITORY_INTEGRATION_ALIAS_TARGET,
                "--",
                "--help",
            ),
        )
        self.assertEqual(
            steps[4].argv[-1],
            "build_tools/devtools/bazel_launcher_integration_test.py",
        )

    def test_bazel_repository_integration_requires_rocm_device_tools(self):
        args = ci.parse_arguments(["iree-bazel-repository-integration"])

        with mock.patch.dict(ci.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires HRX_ROCM_ROOT"):
                ci.steps_from_args(args)

    def test_bazel_repository_integration_rejects_partial_target_scope(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-repository-integration",
                "--target",
                "//runtime/...",
            ]
        )

        with self.assertRaisesRegex(ValueError, "repository-wide"):
            ci.steps_from_args(args)

    def test_cmake_repository_build_covers_supported_graph_with_prebuilts(self):
        args = ci.parse_arguments(["iree-cmake-repository-build"])

        steps = ci.steps_from_args(args)

        self.assertEqual(
            [step.name for step in steps],
            ["Configure CMake", "Build repository", "Test repository smoke"],
        )
        configure_step = steps[0]
        for _driver, define in ci.CMAKE_HAL_DRIVER_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _target, define in ci.CMAKE_LOOM_TARGET_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _importer, define in ci.CMAKE_LOOM_IMPORTER_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for option in ci.AMDGPU_DEVICE_BINARY_PREBUILT_OPTIONS:
            self.assertIn(option, configure_step.argv)
        for option in ci.AMDGPU_DEVICE_BINARY_SOURCE_OPTIONS:
            self.assertNotIn(option, configure_step.argv)
        self.assertFalse(
            any(
                option.startswith("-DIREE_HAL_AMDGPU_TARGETS=")
                or option.startswith("-DLOOM_TARGET_AMDGPU_TARGETS=")
                for option in configure_step.argv
            )
        )
        build_step = steps[1]
        self.assertEqual(build_step.argv[-2:], ("build", "--parallel"))
        smoke_step = steps[2]
        for regex in ci_config.CMAKE_REPOSITORY_SMOKE_CTEST_REGEXES:
            self.assertTrue(any(regex in arg for arg in smoke_step.argv))
        self.assertTrue(
            any(
                ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX in arg
                for arg in smoke_step.argv
            )
        )
        self.assertTrue(
            any(
                ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX in arg
                for arg in smoke_step.argv
            )
        )

    def test_amdgpu_dry_run_does_not_embed_machine_paths(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with (
            mock.patch.dict(ci.os.environ, {}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertIn("-DIREE_ROCM_DEPENDENCY_MODE=pinned", text)
        self.assertIn(
            "--//runtime/src/iree/hal/drivers/amdgpu:targets=gfx942",
            text,
        )
        self.assertIn("--//loom/config/target/amdgpu:targets=iree_hal", text)
        self.assertNotIn("IREE_ROCM_PATH", text)
        self.assertNotIn("/opt/rocm", text)

    def test_amdgpu_target_selects_runtime_and_loom_build_settings(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--amdgpu-target",
                "gfx1150",
            ]
        )

        steps = ci.steps_from_args(args)
        bazel_action_steps = [
            step
            for step in steps
            if step.argv[2:4] in (("bazel", "build"), ("bazel", "test"))
        ]

        self.assertTrue(bazel_action_steps)
        for step in bazel_action_steps:
            self.assertIn(
                "--//runtime/src/iree/hal/drivers/amdgpu:targets=gfx1150",
                step.argv,
            )
            self.assertIn(
                "--//loom/config/target/amdgpu:targets=iree_hal",
                step.argv,
            )

    def test_amdgpu_target_rejected_by_non_amdgpu_command(self):
        args = ci.parse_arguments(["iree-bazel-cpu", "--amdgpu-target", "gfx1150"])

        with self.assertRaisesRegex(ValueError, "only supported for AMDGPU"):
            ci.steps_from_args(args)

    def test_amdgpu_bazel_tests_pin_libhsa_from_rocm_root(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//runtime/...",
            ]
        )

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        amdgpu_test = next(step for step in steps if step.name == "Test IREE / AMDGPU")
        relative_path = (
            Path("bin/hsa-runtime64.dll")
            if ci.sys.platform == "win32"
            else Path("lib/libhsa-runtime64.so.1")
        )
        self.assertIn(
            "--test_env=IREE_HAL_AMDGPU_LIBHSA_PATH="
            + str(Path("/tmp/rocm-root") / relative_path),
            amdgpu_test.argv,
        )

    def test_amdgpu_bazel_device_toolchain_uses_fetched_rocm_root(self):
        args = ci.parse_arguments(["iree-bazel-amdgpu"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        configure_step = next(step for step in steps if step.name == "Configure Bazel")
        self.assertIn("-DIREE_ROCM_PATH=/tmp/rocm-root", configure_step.argv)

    def test_amdgpu_bazel_tests_use_explicit_libhsa_override(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//runtime/...",
            ]
        )

        with mock.patch.dict(
            ci.os.environ,
            {
                "HRX_ROCM_ROOT": "/tmp/rocm-root",
                "IREE_HAL_AMDGPU_LIBHSA_PATH": "/tmp/custom/libhsa-runtime64.so.1",
            },
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        amdgpu_test = next(step for step in steps if step.name == "Test IREE / AMDGPU")
        self.assertIn(
            "--test_env=IREE_HAL_AMDGPU_LIBHSA_PATH=/tmp/custom/libhsa-runtime64.so.1",
            amdgpu_test.argv,
        )

    def test_amdgpu_loom_target_scope_builds_and_tests_loom(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        build_step = next(step for step in steps if step.name == "Build IREE / AMDGPU")
        test_step = next(step for step in steps if step.name == "Test IREE / AMDGPU")

        self.assertIn("//loom/...", build_step.argv)
        self.assertNotIn("//runtime/...", build_step.argv)
        self.assertIn("//loom/...", test_step.argv)
        self.assertNotIn("//runtime/...", test_step.argv)

    def test_bazel_loom_amdgpu_command_runs_compile_coverage_without_driver(self):
        args = ci.parse_arguments(["iree-bazel-loom-amdgpu"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(len(steps), 2)
        self.assertEqual(command_lines[0], "python3 dev.py bazel configure")
        test_step = steps[1]
        self.assertEqual(test_step.name, "Test Loom AMDGPU compile coverage")
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES),
            test_step.argv,
        )
        for target in ci_config.LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS:
            self.assertIn(target, test_step.argv)
        self.assertNotIn(
            "//loom/src/loom/target/emit/native/amdgpu:hsaco_hsa_test",
            test_step.argv,
        )
        self.assertFalse(
            any("-DIREE_HAL_DRIVER_AMDGPU=ON" in line for line in command_lines)
        )

    def test_tilelang_importer_command_sets_up_and_tests_bazel_and_cmake(self):
        args = ci.parse_arguments(["iree-importers-tilelang"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(
            [step.name for step in steps],
            [
                "Setup TileLang importer environment",
                "Report TileLang importer environment",
                "Test TileLang importer with Bazel",
                "Configure TileLang importer CMake",
                "Build TileLang importer CMake verifier",
                "Test TileLang importer with CMake",
            ],
        )
        self.assertIn("python3 dev.py importers setup tilelang", command_lines)
        self.assertIn("python3 dev.py importers env tilelang", command_lines)
        self.assertIn("--importer-env tilelang", command_lines[2])
        self.assertIn(
            "//loom/py/loom/importers/tilelang:tilelang_import_test",
            command_lines[2],
        )
        self.assertTrue(self.uses_cmake_build_dir(steps[3], "iree-importers-tilelang"))
        self.assertIn("--importer-env tilelang", command_lines[3])
        self.assertIn("cmake build loom-opt --parallel", command_lines[4])
        self.assertIn(
            "loom/py/loom/importers/tilelang/tilelang_import_test",
            command_lines[5],
        )

    def test_importer_command_rejects_target_override(self):
        args = ci.parse_arguments(["iree-importers-tilelang", "--target", "//loom/..."])

        with self.assertRaisesRegex(
            ValueError,
            "--target is not supported for importer CI commands",
        ):
            ci.steps_from_args(args)

    def test_sanitizer_command_runs_tests_and_msan_build(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu-sanitizers",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                "bazel test --config=asan --test_tag_filters="
                + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES)
                in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test --config=ubsan --test_tag_filters="
                + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES)
                in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "bazel test --config=tsan --test_tag_filters="
                + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES)
                in line
                for line in command_lines
            )
        )
        tsan_test_step = next(
            step for step in steps if step.name == "Test IREE with TSAN"
        )
        tsan_test_env = [
            arg for arg in tsan_test_step.argv if arg.startswith("--test_env=")
        ]
        self.assertEqual(len(tsan_test_env), 1)
        self.assertTrue(
            tsan_test_env[0].startswith("--test_env=TSAN_OPTIONS=suppressions=")
        )
        tsan_suppression_path = Path(
            tsan_test_env[0].removeprefix("--test_env=TSAN_OPTIONS=suppressions=")
        )
        self.assertTrue(tsan_suppression_path.is_absolute())
        self.assertTrue(tsan_suppression_path.is_file())
        self.assertTrue(
            any(
                step.argv[:7]
                == (
                    "python3",
                    "dev.py",
                    "bazel",
                    "build",
                    "--config=msan",
                    "--",
                    "//runtime/...",
                )
                and "--config=msan" in step.argv
                for step in steps
            )
        )
        sanitizer_test_steps = [
            step for step in steps if step.name.startswith("Test IREE")
        ]
        for xfail_target in ci_config.CPU_SANITIZERS_XFAIL_TARGETS:
            self.assertTrue(
                any(xfail_target in step.argv for step in sanitizer_test_steps)
            )

    def test_bazel_cpu_single_sanitizer_command_runs_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu-asan",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(command_lines[0], "python3 dev.py bazel configure")
        self.assertTrue(
            any(
                "bazel test --config=asan --test_tag_filters="
                + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES)
                in line
                for line in command_lines
            )
        )
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=tsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_cpu_msan_command_builds_without_tests(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu-msan",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                step.argv[:7]
                == (
                    "python3",
                    "dev.py",
                    "bazel",
                    "build",
                    "--config=msan",
                    "--",
                    "//runtime/...",
                )
                and "--config=msan" in step.argv
                for step in steps
            )
        )
        self.assertFalse(any("bazel test" in line for line in command_lines))

    def test_amdgpu_command_builds_scope_before_one_semantic_test_graph(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//runtime/...",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        self.assertEqual(
            [step.name for step in steps],
            ["Configure Bazel", "Build IREE / AMDGPU", "Test IREE / AMDGPU"],
        )
        build_step = steps[1]
        test_step = steps[2]
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, build_step.argv)
            self.assertIn(target, test_step.argv)
        for target in ci_config.AMDGPU_BAZEL_TARGET_EXCLUDES:
            self.assertIn(target, build_step.argv)
            self.assertIn(target, test_step.argv)
        self.assertFalse(
            any(
                arg.startswith("-//runtime/src/iree/hal/drivers/amdgpu")
                for arg in build_step.argv + test_step.argv
            )
        )
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.AMDGPU_BAZEL_TEST_TAG_FILTERS),
            test_step.argv,
        )

    def test_bazel_amdgpu_single_sanitizer_command_runs_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-tsan",
                "--target",
                "//runtime/...",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(
            command_lines[0],
            "python3 dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON "
            "-DIREE_ROCM_DEPENDENCY_MODE=pinned",
        )
        self.assertEqual(
            [step.name for step in steps],
            [
                "Configure Bazel",
                "Build IREE / AMDGPU / TSAN",
                "Test IREE / AMDGPU / TSAN",
            ],
        )
        tsan_build = steps[1]
        tsan_test = steps[2]
        self.assertIn("--config=tsan", tsan_build.argv)
        self.assertIn("--config=tsan", tsan_test.argv)
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, tsan_build.argv)
            self.assertIn(target, tsan_test.argv)
        self.assertTrue(
            any(
                arg.startswith("--test_env=TSAN_OPTIONS=suppressions=")
                for arg in tsan_test.argv
            )
        )
        self.assertIn(
            "--test_tag_filters="
            + ",".join(
                ci_config.AMDGPU_BAZEL_TEST_TAG_FILTERS
                + (f"-{ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL}",)
            ),
            tsan_test.argv,
        )
        self.assertFalse(
            any(
                arg.startswith("-//runtime/src/iree/hal/drivers/amdgpu")
                for step in steps
                for arg in step.argv
            )
        )
        self.assertFalse(any("--config=asan" in line for line in command_lines))
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_amdgpu_asan_and_ubsan_keep_all_resource_tests(self):
        for sanitizer in ("asan", "ubsan"):
            with self.subTest(sanitizer=sanitizer):
                args = ci.parse_arguments(
                    [
                        f"iree-bazel-amdgpu-{sanitizer}",
                        "--target",
                        "//loom/...",
                    ]
                )

                steps = ci.steps_from_args(args)
                test_step = next(
                    step
                    for step in steps
                    if step.name == f"Test IREE / AMDGPU / {sanitizer.upper()}"
                )
                self.assertIn(
                    "--test_tag_filters="
                    + ",".join(ci_config.AMDGPU_BAZEL_TEST_TAG_FILTERS),
                    test_step.argv,
                )
                self.assertFalse(
                    any(
                        ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL in arg
                        for arg in test_step.argv
                    )
                )

    def test_vulkan_command_builds_scope_before_one_semantic_test_graph(self):
        args = ci.parse_arguments(["iree-bazel-vulkan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(
            command_lines[0],
            "python3 dev.py bazel configure -DIREE_HAL_DRIVER_VULKAN=ON",
        )
        self.assertEqual(
            [step.name for step in steps],
            ["Configure Bazel", "Build IREE / Vulkan", "Test IREE / Vulkan"],
        )
        build_step = steps[1]
        test_step = steps[2]
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, build_step.argv)
            self.assertIn(target, test_step.argv)
        for target in ci_config.VULKAN_BAZEL_TARGET_EXCLUDES:
            self.assertIn(target, build_step.argv)
            self.assertIn(target, test_step.argv)
        self.assertFalse(
            any(
                arg.startswith("-//runtime/src/iree/hal/drivers/vulkan")
                for arg in build_step.argv + test_step.argv
            )
        )
        for xfail_target in ci_config.VULKAN_XFAIL_TARGETS:
            self.assertIn(xfail_target, test_step.argv)
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.VULKAN_BAZEL_TEST_TAG_FILTERS),
            test_step.argv,
        )

    def test_vulkan_commands_preserve_explicit_mesa_device_selection(self):
        with mock.patch.dict(os.environ, {"DRI_PRIME": "1!"}):
            bazel_steps = ci.steps_from_args(ci.parse_arguments(["iree-bazel-vulkan"]))
            cmake_steps = ci.steps_from_args(ci.parse_arguments(["iree-cmake-vulkan"]))

        bazel_test_step = next(
            step for step in bazel_steps if step.name == "Test IREE / Vulkan"
        )
        self.assertIn("--test_env=DRI_PRIME=1!", bazel_test_step.argv)

        cmake_test_steps = [
            step for step in cmake_steps if step.name.startswith("Test IREE CMake")
        ]
        self.assertGreater(len(cmake_test_steps), 0)
        for step in cmake_test_steps:
            self.assertIn(("DRI_PRIME", "1!"), step.env)

    def test_bazel_gpu_command_surface_omits_nonexecuting_lanes(self):
        for command in (
            "iree-bazel-amdgpu-msan",
            "iree-bazel-amdgpu-sanitizers",
            "iree-bazel-vulkan-asan",
            "iree-bazel-vulkan-msan",
            "iree-bazel-vulkan-tsan",
            "iree-bazel-vulkan-ubsan",
            "iree-bazel-vulkan-sanitizers",
        ):
            with self.subTest(command=command):
                self.assertNotIn(command, ci.BAZEL_COMMANDS)

        for command in (
            "iree-bazel-amdgpu-asan",
            "iree-bazel-amdgpu-tsan",
            "iree-bazel-amdgpu-ubsan",
            "iree-bazel-vulkan",
        ):
            with self.subTest(command=command):
                self.assertIn(command, ci.BAZEL_COMMANDS)

    def test_bazel_cpu_sanitizer_workflow_is_split_by_configuration(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_bazel.yml", "linux_bazel_cpu"
        )
        self.assertIn("name: Linux / CPU", block)
        for sanitizer in ("ASAN", "MSAN", "TSAN", "UBSAN"):
            self.assertIn(f"name: Linux / CPU / {sanitizer}", block)
            self.assertIn(f"command: iree-bazel-cpu-{sanitizer.lower()}", block)
        self.assertNotIn("command: iree-bazel-cpu-sanitizers", block)

    def test_cmake_workflow_uses_sanitizer_smoke(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_cmake.yml", "linux_cmake_cpu"
        )
        self.assertIn("name: Linux / CPU", block)
        self.assertIn("name: Linux / CPU / Sanitizer Smoke", block)
        self.assertIn("command: iree-cmake-sanitizer-smoke", block)
        for sanitizer in ("asan", "msan", "tsan", "ubsan"):
            self.assertNotIn(f"command: iree-cmake-cpu-{sanitizer}", block)
        self.assertNotIn("command: iree-cmake-cpu-sanitizers", block)

        for job_name in ("linux_cmake_amdgpu", "linux_cmake_vulkan"):
            with self.subTest(job=job_name):
                block = self.workflow_job_block(
                    ".github/workflows/ci_iree_cmake.yml", job_name
                )
                self.assertNotIn("/ Sanitizers", block)
                self.assertNotIn("-sanitizers", block)

    def test_bazel_amdgpu_sanitizer_workflow_is_split_by_configuration(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_bazel.yml", "linux_bazel_amdgpu"
        )
        for sanitizer in ("ASAN", "TSAN", "UBSAN"):
            self.assertIn(f"name: Linux / AMDGPU / gfx942 / {sanitizer}", block)
            self.assertIn(f"command: iree-bazel-amdgpu-{sanitizer.lower()}", block)
        self.assertNotIn("/ Sanitizers", block)
        self.assertNotIn("iree-bazel-amdgpu-msan", block)
        self.assertNotIn("iree-bazel-amdgpu-sanitizers", block)

    def test_amdgpu_container_jobs_initialize_git_home(self):
        for path, job_name in (
            (".github/workflows/ci_iree_bazel.yml", "linux_bazel_amdgpu"),
            (".github/workflows/ci_iree_cmake.yml", "linux_cmake_amdgpu"),
            (
                ".github/workflows/test_core_linux_gpu_source.yml",
                "test_core_linux_gpu_source",
            ),
        ):
            with self.subTest(path=path):
                block = self.workflow_job_block(path, job_name)
                mkdir_command = 'mkdir -p "$HOME"'
                git_config_command = 'git config --global --add safe.directory "$PWD"'
                self.assertIn(mkdir_command, block)
                self.assertIn(git_config_command, block)
                self.assertLess(
                    block.index(mkdir_command), block.index(git_config_command)
                )

    def test_bazel_workflow_uploads_profiles_for_each_attempted_job(self):
        cases = (
            (
                "linux_bazel_cpu",
                "bazel-profiles-${{ matrix.command }}-attempt-"
                "${{ github.run_attempt }}",
            ),
            (
                "linux_bazel_vulkan",
                "bazel-profiles-iree-bazel-vulkan-attempt-${{ github.run_attempt }}",
            ),
            (
                "linux_bazel_amdgpu",
                "bazel-profiles-${{ matrix.command }}-"
                "${{ matrix.target_selector }}-attempt-${{ github.run_attempt }}",
            ),
        )
        for job_name, artifact_name in cases:
            with self.subTest(job=job_name):
                block = self.workflow_job_block(
                    ".github/workflows/ci_iree_bazel.yml", job_name
                )
                self.assertIn("id: iree_bazel_ci", block)
                self.assertIn(
                    '--bazel-profile-dir "${RUNNER_TEMP}/bazel-profiles"', block
                )
                self.assertIn(
                    "if: ${{ always() && steps.iree_bazel_ci.outcome != 'skipped' }}",
                    block,
                )
                self.assertIn(
                    "uses: actions/upload-artifact@"
                    "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1",
                    block,
                )
                self.assertIn(f"name: {artifact_name}", block)
                self.assertIn(
                    "path: ${{ runner.temp }}/bazel-profiles/*.profile.gz", block
                )
                self.assertIn("if-no-files-found: error", block)
                self.assertIn("compression-level: 0", block)
                self.assertLess(
                    block.index("name: Run IREE Bazel CI"),
                    block.index("name: Upload Bazel profiles"),
                )

    def test_bazel_vulkan_workflow_has_no_nonexecuting_sanitizer_lane(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_bazel.yml", "linux_bazel_vulkan"
        )
        self.assertIn("name: Linux / Vulkan", block)
        self.assertIn(
            "python3 build_tools/devtools/ci.py iree-bazel-vulkan",
            block,
        )
        self.assertNotIn("matrix.", block)
        self.assertNotIn("strategy:", block)
        self.assertNotIn("/ Sanitizers", block)
        self.assertNotRegex(block, r"iree-bazel-vulkan-(asan|msan|tsan|ubsan)")

    def test_fetch_toolchain_uses_owned_ci_entry_point(self):
        script = Path(".github/scripts/fetch_rocm_toolchain.sh").read_text()
        self.assertIn(
            '"${HRX_PYTHON}" build_tools/ci/ci_core_linux.py fetch-rocm', script
        )
        self.assertNotIn("build_tools/ci_core_", script)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux CI toolchain setup")
    def test_fetch_toolchain_versions_root_and_rebases_rocm_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "output"
            rocm_root = output_dir / "rocm-root"
            github_env = root / "github-env"
            github_path = root / "github-path"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python3"
elif [[ "$1" == "-m" && "$2" == "pip" ]]; then
  exit 0
elif [[ "$1" == "build_tools/ci/ci_core_linux.py" && "$2" == "fetch-rocm" ]]; then
  mkdir -p "${HRX_ROCM_ROOT}/lib/llvm/bin"
  printf '%s\n' \
    '{"artifact_identity":"nightly-123456-linux-release-core"}' \
    >"${HRX_ROCM_ROOT}/.hrx-rocm-artifacts.json"
elif [[ "$1" == "-c" ]]; then
  exec __REAL_PYTHON__ "$@"
else
  exit 1
fi
""".replace("__REAL_PYTHON__", shlex.quote(sys.executable))
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "AR": "ar",
                    "CC": os.fspath(rocm_root / "lib/llvm/bin/clang"),
                    "CXX": os.fspath(rocm_root / "lib/llvm/bin/clang++"),
                    "GITHUB_ENV": os.fspath(github_env),
                    "GITHUB_PATH": os.fspath(github_path),
                    "HRX_OUTPUT_DIR": os.fspath(output_dir),
                    "HRX_PYTHON": os.fspath(output_dir / "python/bin/python3"),
                    "HRX_ROCM_ROOT": os.fspath(rocm_root),
                    "IREE_CLANG_TIDY_LLVM_ROOT": os.fspath(rocm_root / "lib/llvm"),
                    "IREE_ROCM_PATH": os.fspath(rocm_root),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                }
            )
            subprocess.run(
                ["bash", ".github/scripts/fetch_rocm_toolchain.sh"],
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            versioned_rocm_root = Path(f"{rocm_root}-nightly-123456-linux-release-core")
            exported_env = dict(
                line.split("=", 1) for line in github_env.read_text().splitlines()
            )
            self.assertFalse(rocm_root.exists())
            self.assertTrue(versioned_rocm_root.is_dir())
            self.assertEqual(exported_env["HRX_ROCM_ROOT"], str(versioned_rocm_root))
            self.assertEqual(
                exported_env["CC"], str(versioned_rocm_root / "lib/llvm/bin/clang")
            )
            self.assertEqual(
                exported_env["CXX"],
                str(versioned_rocm_root / "lib/llvm/bin/clang++"),
            )
            self.assertNotIn("AR", exported_env)
            self.assertEqual(
                exported_env["IREE_CLANG_TIDY_LLVM_ROOT"],
                str(versioned_rocm_root / "lib/llvm"),
            )
            self.assertEqual(exported_env["IREE_ROCM_PATH"], str(versioned_rocm_root))
            self.assertEqual(
                exported_env["IREE_HAL_AMDGPU_LIBHSA_PATH"],
                str(versioned_rocm_root / "lib/libhsa-runtime64.so.1"),
            )

    def test_core_gpu_workflow_routes_exact_gpu_labels(self):
        gfx942_block = self.workflow_job_block(
            ".github/workflows/ci_core_linux.yml", "gpu_linux_gfx942"
        )
        self.assertIn("name: Linux / CMake / GPU gfx942", gfx942_block)
        self.assertIn(
            "uses: ./.github/workflows/test_core_linux_gpu_source.yml",
            gfx942_block,
        )
        self.assertIn(
            "runner_labels: '[\"linux-gfx942-1gpu-ccs-csp-ossci-rocm\"]'",
            gfx942_block,
        )
        self.assertIn("amdgpu_targets: gfx942", gfx942_block)
        self.assertNotIn("artifact_run_id", gfx942_block)
        self.assertNotIn("gfx1201", gfx942_block)

        gfx1201_block = self.workflow_job_block(
            ".github/workflows/ci_core_linux.yml", "gpu_linux_gfx1201"
        )
        self.assertIn("name: Linux / CMake / GPU gfx1201", gfx1201_block)
        self.assertIn(
            "uses: ./.github/workflows/test_core_linux_gpu.yml", gfx1201_block
        )
        self.assertIn(
            'runner_labels: \'["self-hosted", "Linux", "X64", "gpu_navi4x"]\'',
            gfx1201_block,
        )
        self.assertIn("artifact_run_id: ${{ github.run_id }}", gfx1201_block)

        summary_block = self.workflow_job_block(
            ".github/workflows/ci_core_linux.yml", "summary"
        )
        for job in ("cmake_linux", "gpu_linux_gfx942", "gpu_linux_gfx1201"):
            self.assertIn(f"- {job}", summary_block)
            self.assertIn("needs." + job + ".result", summary_block)

        for path, job_name in (
            (".github/workflows/test_core_linux_gpu.yml", "test_core_linux_gpu"),
            (
                ".github/workflows/test_core_linux_gpu_source.yml",
                "test_core_linux_gpu_source",
            ),
        ):
            with self.subTest(path=path):
                reusable_workflow = Path(path).read_text()
                # Both workflows declare runner_labels once per trigger, so
                # asserting one match would let either declaration satisfy the
                # test for the other.
                declarations = re.findall(
                    r"^ +runner_labels:$", reusable_workflow, re.MULTILINE
                )
                required_declarations = re.findall(
                    r"runner_labels:\n\s+type: string\n\s+description: "
                    r'"JSON array of labels that must all match the GPU runner\."\n'
                    r"\s+required: true",
                    reusable_workflow,
                )
                self.assertEqual(len(declarations), 2)
                self.assertEqual(len(required_declarations), len(declarations))
                reusable_block = self.workflow_job_block(path, job_name)
                self.assertRegex(
                    reusable_block,
                    r"(?m)^\s+runs-on: \$\{\{ fromJSON\(inputs\.runner_labels\) \}\}$",
                )

    def test_core_gpu_source_workflow_builds_instead_of_downloading(self):
        block = self.workflow_job_block(
            ".github/workflows/test_core_linux_gpu_source.yml",
            "test_core_linux_gpu_source",
        )

        self.assertNotIn("actions/download-artifact", block)
        self.assertNotIn("ci_core_linux.py extract-packages", block)
        self.assertIn("bash .github/scripts/fetch_rocm_toolchain.sh", block)
        self.assertIn('"${HRX_PYTHON}" build_tools/ci/ci_core_linux.py build', block)
        self.assertIn('"${HRX_PYTHON}" build_tools/ci/ci_core_linux.py test', block)

        # prepare_public_deps_root() synthesizes the runtime dependency overlay
        # from the fetched ROCm root because no dependency package is unpacked.
        self.assertIn('HRX_PREPARE_PUBLIC_DEPS: "true"', block)
        self.assertIn('HRX_PACKAGE_SMOKE: "false"', block)
        self.assertIn('HRX_TEST_GPU: "true"', block)
        self.assertIn(
            "HRX_CMAKE_OPTIONS: "
            '"-DIREE_HAL_AMDGPU_TARGETS=${{ inputs.amdgpu_targets }}"',
            block,
        )

    def test_core_windows_workflow_uses_generic_windows_runner(self):
        build_block = self.workflow_job_block(
            ".github/workflows/build_core_windows.yml", "build_core_windows"
        )
        self.assertIn("runs-on: azure-windows-scale-rocm", build_block)
        self.assertIn(
            "python build_tools/ci/ci_core_windows.py fetch-rocm", build_block
        )
        self.assertIn("python build_tools/ci/ci_core_windows.py build", build_block)
        self.assertIn("python build_tools/ci/ci_core_windows.py test", build_block)
        self.assertIn("python build_tools/ci/ci_core_windows.py package", build_block)
        self.assertIn(
            "python build_tools/ci/ci_core_windows.py extract-packages", build_block
        )
        self.assertIn(
            "HRX_ROCM_ROOT: ${{ github.workspace }}/build/windows/${{ inputs.windows_toolchain }}/rocm-root",
            build_block,
        )
        self.assertIn(
            "HRX_DOWNLOAD_CACHE_DIR: ${{ github.workspace }}/build/windows/${{ inputs.windows_toolchain }}/downloads",
            build_block,
        )
        self.assertIn(
            "PIP_CACHE_DIR: ${{ github.workspace }}/build/windows/${{ inputs.windows_toolchain }}/caches/pip",
            build_block,
        )
        self.assertIn('python -m venv "$relocated/python-venv"', build_block)
        self.assertIn(
            '$env:HRX_TEST_PYTHON = "$relocated/python-venv/Scripts/python.exe"',
            build_block,
        )
        self.assertIn("--no-prepare-public-deps", build_block)
        self.assertIn('Move-Item -LiteralPath "$env:HRX_BUILD_DIR"', build_block)
        self.assertIn(
            "hrx-public-windows-${{ inputs.windows_toolchain }}-x86_64", build_block
        )
        self.assertIn(
            "hrx-public-deps-windows-${{ inputs.windows_toolchain }}-x86_64",
            build_block,
        )
        self.assertIn(
            "hrx-tests-windows-${{ inputs.windows_toolchain }}-x86_64", build_block
        )
        self.assertNotIn("windows-gfx1151-gpu-rocm", build_block)
        self.assertNotIn("runtime-resource=amd-npu", build_block)

        ci_block = self.workflow_job_block(
            ".github/workflows/ci_core_windows.yml", "cmake_windows"
        )
        self.assertIn("name: Windows / CMake / ${{ matrix.toolchain }}", ci_block)
        self.assertIn("variant: windows-msvc", ci_block)
        self.assertIn("toolchain: msvc", ci_block)
        self.assertRegex(
            ci_block,
            r"variant: windows-msvc\n\s+toolchain: msvc\n"
            r"\s+assertions: false\n\s+run_tests: true\n"
            r"\s+ctest_exclude_regex: \"\"\n\s+package: true",
        )
        self.assertIn("variant: windows-clang-cl", ci_block)
        self.assertIn("toolchain: clang-cl", ci_block)
        self.assertRegex(
            ci_block,
            r"variant: windows-clang-cl\n\s+toolchain: clang-cl\n"
            r"\s+assertions: true\n\s+run_tests: true\n"
            r"\s+ctest_exclude_regex: \"\"\n\s+package: false",
        )
        self.assertIn("windows_toolchain: ${{ matrix.toolchain }}", ci_block)
        self.assertIn("uses: ./.github/workflows/build_core_windows.yml", ci_block)
        self.assertIn('ctest_label_exclude_regex: "runtime-resource=|manual"', ci_block)

    def test_iree_cmake_windows_workflow_separates_build_and_test_roles(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_cmake.yml", "windows_cmake"
        )

        self.assertIn("runs-on: azure-windows-scale-rocm", block)
        self.assertRegex(
            block,
            r"name: Windows / Repository / MSVC Build\n"
            r"\s+command: iree-cmake-repository-build\n"
            r"\s+host_toolchain: msvc\n"
            r"\s+fetch_rocm: false",
        )
        self.assertRegex(
            block,
            r"name: Windows / CPU\n"
            r"\s+command: iree-cmake-cpu\n"
            r"\s+host_toolchain: clang-cl\n"
            r"\s+fetch_rocm: true",
        )
        self.assertIn("if: matrix.fetch_rocm", block)
        self.assertIn("python build_tools/ci/ci_core_windows.py fetch-rocm", block)
        self.assertNotRegex(
            block,
            r"ci/ci_core_windows\.py (build|test|package|extract-packages)",
        )
        self.assertNotIn("uses: ./.github/workflows/build_core_windows.yml", block)
        self.assertIn("VsDevCmd.bat", block)
        self.assertIn('if "${{ matrix.host_toolchain }}"=="msvc"', block)
        self.assertIn("build_tools/devtools/ci.py ${{ matrix.command }}", block)

    def test_iree_bazel_windows_workflow_is_enabled_and_separates_toolchains(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_iree_bazel.yml", "windows_bazel"
        )

        self.assertIn("name: ${{ matrix.name }}", block)
        self.assertRegex(
            block,
            r"name: Windows / Repository / clang-cl \+ ROCm\n"
            r"\s+host_toolchain: clang-cl",
        )
        self.assertRegex(
            block,
            r"name: Windows / Repository / MSVC \+ ROCm\n"
            r"\s+host_toolchain: msvc",
        )
        self.assertNotIn("if: ${{ false }}", block)
        self.assertIn("runs-on: azure-windows-scale-rocm", block)
        self.assertIn(
            "BAZEL_LLVM: ${{ github.workspace }}\\build\\iree-bazel-windows\\rocm-root\\lib\\llvm",
            block,
        )
        self.assertIn("python build_tools/ci/ci_core_windows.py fetch-rocm", block)
        self.assertIn("python dev.py bazel setup --venv", block)
        self.assertIn('Join-Path $env:SystemDrive "b"', block)
        self.assertNotIn('Join-Path $env:RUNNER_TEMP "bazel"', block)
        self.assertIn("startup --output_user_root=$bazelOutputRoot", block)
        self.assertNotIn("output_user_root=C:", block)
        self.assertIn('if ("${{ matrix.host_toolchain }}" -eq "msvc")', block)
        self.assertIn('"build --config=windows-msvc"', block)
        self.assertIn("VsDevCmd.bat", block)
        self.assertIn('if "${{ matrix.host_toolchain }}"=="msvc"', block)
        self.assertIn('set "CC=cl.exe"', block)
        self.assertIn('set "CXX=cl.exe"', block)
        self.assertIn('set "AR=lib.exe"', block)
        self.assertIn('set "CC=%BAZEL_LLVM%\\bin\\clang-cl.exe"', block)
        self.assertIn('set "CXX=%BAZEL_LLVM%\\bin\\clang-cl.exe"', block)
        self.assertIn('set "AR=%BAZEL_LLVM%\\bin\\llvm-lib.exe"', block)
        self.assertIn(
            "build_tools/devtools/ci.py iree-bazel-repository-integration",
            block,
        )
        self.assertIn("--amdgpu-target gfx11-generic", block)
        self.assertIn(
            "bazel-profiles-iree-bazel-repository-integration-${{ matrix.host_toolchain }}",
            block,
        )

    def test_iree_workflows_do_not_trigger_on_libhrx_only_paths(self):
        for path in (
            ".github/workflows/ci_iree_bazel.yml",
            ".github/workflows/ci_iree_cmake.yml",
        ):
            with self.subTest(path=path):
                text = Path(path).read_text()
                self.assertIn('- "runtime/**"', text)
                self.assertIn('- "loom/**"', text)
                self.assertNotIn('- "libhrx/**"', text)

    def test_importer_workflow_covers_loom_changes_and_uses_locked_cache_key(self):
        text = Path(".github/workflows/ci_importers.yml").read_text()

        self.assertIn("name: CI Importers", text)
        self.assertIn('- "requirements-importers-*.lock.txt"', text)
        self.assertIn('- "requirements-importers-*.in"', text)
        self.assertIn('- "build_tools/devtools/**"', text)
        self.assertIn('- "loom/**"', text)
        self.assertNotIn('- "loom/config/**"', text)
        self.assertNotIn('- "loom/py/loom/importers/**"', text)
        self.assertNotIn('- "runtime/**"', text)
        self.assertNotIn('- "libhrx/**"', text)

        block = self.workflow_job_block(
            ".github/workflows/ci_importers.yml", "linux_importer"
        )
        self.assertIn("profile: tilelang", block)
        self.assertIn("command: iree-importers-tilelang", block)
        self.assertIn("lock_file: requirements-importers-tilelang.lock.txt", block)
        self.assertIn("RUNNER_OS", block)
        self.assertIn("RUNNER_ARCH", block)
        self.assertIn("IMPORTER_LOCK_FILE", block)
        self.assertIn("requirements-dev.lock.txt", block)
        self.assertIn("requirements-analysis.lock.txt", block)
        self.assertIn("actions/cache@", block)
        self.assertIn("PIP_CACHE_DIR", block)
        self.assertIn("python3 dev.py bazel setup --venv", block)
        self.assertIn("python3 dev.py cmake setup --venv", block)
        self.assertIn(
            'python3 build_tools/devtools/ci.py "${IMPORTER_COMMAND}" --keep-going',
            block,
        )

    def test_loom_docs_workflow_reviews_every_change_and_deploys_only_main(self):
        text = Path(".github/workflows/docs.yml").read_text()
        build_block = self.workflow_job_block(
            ".github/workflows/docs.yml", "build_docs"
        )
        deploy_block = self.workflow_job_block(
            ".github/workflows/docs.yml", "deploy_docs"
        )

        for path in (
            '"requirements-analysis.lock.txt"',
            '"requirements-dev.lock.txt"',
            '"build_tools/devtools/**"',
            '"loom/binding/c/**"',
            '"loom/docs/**"',
            '"loom/py/loom/**"',
            '"loom/src/loom/editor/textmate/**"',
        ):
            self.assertIn(f"- {path}", text)
        self.assertIn("runs-on: ubuntu-24.04", build_block)
        self.assertIn("python3 dev.py setup --docs", build_block)
        self.assertIn("loom_docs.highlight_test", build_block)
        self.assertIn(
            "--site-dir build/loom-pages/loom",
            build_block,
        )
        self.assertIn(
            "test -f build/loom-pages/loom/reference/dialects/index.html",
            build_block,
        )
        self.assertIn(
            "test -f build/loom-pages/loom/reference/c-api/generated/index.html",
            build_block,
        )
        self.assertIn(
            "uses: actions/upload-pages-artifact@"
            "fc324d3547104276b827a68afc52ff2a11cc49c9 # v5.0.0",
            build_block,
        )
        self.assertIn("path: build/loom-pages", build_block)
        self.assertIn(
            "group: ${{ github.workflow }}-build-${{ github.ref }}", build_block
        )
        self.assertIn("cancel-in-progress: true", build_block)
        self.assertNotIn("pages: write", build_block)
        self.assertNotIn("id-token: write", build_block)
        self.assertNotIn("actions/deploy-pages", build_block)

        self.assertIn("if: github.ref == 'refs/heads/main'", deploy_block)
        self.assertIn("needs: build_docs", deploy_block)
        self.assertIn("pages: write", deploy_block)
        self.assertIn("id-token: write", deploy_block)
        self.assertIn("name: github-pages", deploy_block)
        self.assertIn("url: ${{ steps.deployment.outputs.page_url }}", deploy_block)
        self.assertIn("group: github-pages", deploy_block)
        self.assertIn("cancel-in-progress: false", deploy_block)
        self.assertIn("id: deployment", deploy_block)
        self.assertIn(
            "uses: actions/deploy-pages@"
            "cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5.0.0",
            deploy_block,
        )
        self.assertEqual(text.count("pages: write"), 1)
        self.assertEqual(text.count("id-token: write"), 1)
        self.assertEqual(text.count("actions/deploy-pages@"), 1)
        self.assertNotIn("apt-get", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("sudo", text)

    def test_xfails_project_to_ctest_regexes(self):
        self.assertIn(
            "^iree/tokenizer/",
            ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/task/executable/elf/elf_module_test$",
            ci_config.CPU_CTEST_EXCLUDE_REGEX,
        )
        self.assertEqual(ci_config.AMDGPU_XFAIL_TARGETS, ())
        self.assertEqual(ci_config.AMDGPU_CTEST_EXCLUDE_REGEX, "")
        self.assertEqual(ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS, ())
        self.assertEqual(ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX, "")
        self.assertEqual(ci_config.AMDGPU_TSAN_XFAIL_TARGETS, ())
        self.assertEqual(ci_config.AMDGPU_TSAN_CTEST_EXCLUDE_REGEX, "")
        self.assertEqual(ci_config.AMDGPU_TSAN_SANITIZERS_XFAIL_TARGETS, ())
        self.assertEqual(ci_config.AMDGPU_TSAN_SANITIZERS_CTEST_EXCLUDE_REGEX, "")
        self.assertEqual(
            ci_config.bazel_pattern_to_ctest_regex("//loom/src/loom/codegen/low:test"),
            "^loom/codegen/low/test$",
        )
        self.assertEqual(
            ci_config.bazel_pattern_to_ctest_regex(
                "//loom/binding/c/example:emit_spirv_vulkan_test"
            ),
            "^loom/binding/c/example/emit_spirv_vulkan_test$",
        )

    def test_cmake_cpu_sanitizer_command_uses_cmake_build_dir_and_xfails(self):
        args = ci.parse_arguments(["iree-cmake-cpu-ubsan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                self.uses_cmake_build_dir(step, "iree-cmake-cpu-ubsan")
                for step in steps
            )
        )
        self.assertTrue(any("-DIREE_ENABLE_UBSAN=ON" in line for line in command_lines))
        test_steps = [step for step in steps if step.name.startswith("Test IREE")]
        self.assertTrue(
            any(
                any(
                    ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.NON_CPU_HAL_DRIVER_CTEST_REGEX in arg for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )

    def test_cmake_cpu_sanitizers_command_runs_each_configuration(self):
        args = ci.parse_arguments(["iree-cmake-cpu-sanitizers"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("iree-cmake-cpu-asan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-cpu-ubsan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-cpu-tsan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-cpu-msan" in line for line in command_lines))
        tsan_options = [
            value for step in steps for key, value in step.env if key == "TSAN_OPTIONS"
        ]
        self.assertNotEqual(tsan_options, [])
        for value in tsan_options:
            path = Path(value.removeprefix("suppressions="))
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_file())
        self.assertTrue(
            any("-DIREE_BUILD_BENCHMARKS=OFF" in line for line in command_lines)
        )
        self.assertTrue(any("-DIREE_BUILD_TESTS=OFF" in line for line in command_lines))
        self.assertFalse(
            any("Test IREE CMake with MSAN" in step.name for step in steps)
        )

    def test_cmake_sanitizer_smoke_builds_selected_test_closures(self):
        args = ci.parse_arguments(["iree-cmake-sanitizer-smoke"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        for sanitizer in ("asan", "ubsan", "tsan", "msan"):
            self.assertTrue(
                any(
                    self.uses_cmake_build_dir(
                        step, f"iree-cmake-sanitizer-smoke-{sanitizer}"
                    )
                    for step in steps
                )
            )
            self.assertTrue(
                any(
                    f"-DIREE_ENABLE_{sanitizer.upper()}=ON" in line
                    for line in command_lines
                )
            )
        for sanitizer in ("asan", "ubsan", "tsan"):
            self.assertTrue(
                any(
                    f"iree-cmake-sanitizer-smoke-{sanitizer}" in line
                    and "-DIREE_BUILD_TESTS=ON" in line
                    and "-DIREE_BUILD_BENCHMARKS=ON" in line
                    for line in command_lines
                )
            )
        self.assertTrue(
            any(
                "iree-cmake-sanitizer-smoke-msan" in line
                and "-DIREE_BUILD_TESTS=OFF" in line
                and "-DIREE_BUILD_BENCHMARKS=OFF" in line
                for line in command_lines
            )
        )

        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        msan_build_step = next(step for step in build_steps if "MSAN" in step.name)
        for target in ci_config.CMAKE_SANITIZER_SMOKE_LIBRARY_BUILD_TARGETS:
            self.assertIn(target, msan_build_step.argv)
        self.assertEqual(len(build_steps), 1)
        self.assertNotIn("all", msan_build_step.argv)

        test_steps = [step for step in steps if step.name.startswith("Test IREE")]
        self.assertEqual(len(test_steps), 3)
        self.assertFalse(any("with MSAN" in step.name for step in test_steps))
        for regex in ci_config.CMAKE_SANITIZER_SMOKE_CTEST_REGEXES:
            self.assertTrue(
                any(regex in arg for step in test_steps for arg in step.argv)
            )
        self.assertFalse(
            any(
                ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX in arg
                for step in test_steps
                for arg in step.argv
            )
        )

    def test_cmake_amdgpu_command_scopes_build_and_tests_to_amdgpu(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu", "--amdgpu-target", "gfx1150"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=ON" in line for line in command_lines)
        )
        self.assertTrue(
            any("-DIREE_ROCM_DEPENDENCY_MODE=pinned" in line for line in command_lines)
        )
        self.assertTrue(
            any("-DIREE_HAL_AMDGPU_TARGETS=gfx1150" in line for line in command_lines)
        )
        self.assertTrue(
            any(
                "-DLOOM_TARGET_AMDGPU_TARGETS=iree_hal" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm" in line
                for line in command_lines
            )
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        for target in ci_config.AMDGPU_CMAKE_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        resource_target = ci.cmake_runtime_resource_build_target(
            ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX
        )
        self.assertTrue(any(resource_target in step.argv for step in build_steps))
        self.assertFalse(
            any(
                "loom_tools_iree-test-loom_amdgpu_execution_test" in arg
                for step in build_steps
                for arg in step.argv
            )
        )
        self.assertTrue(
            any("-R '^iree/hal/drivers/amdgpu/'" in line for line in command_lines)
        )
        self.assertTrue(
            any(
                ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX in step.argv
                for step in steps
            )
        )
        resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake AMDGPU resource tests"
        )
        package_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake AMDGPU package tests"
        )
        self.assertEqual(self.ctest_exclude_regexes(package_test), [])
        self.assertEqual(
            self.ctest_exclude_regexes(resource_test),
            [ci.combine_ctest_regex("^iree/hal/drivers/amdgpu/")],
        )
        self.assertIn(ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX, resource_test.argv)

    def test_cmake_amdgpu_device_binary_source_build_uses_fetched_rocm_root(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        configure_step = next(step for step in steps if step.name == "Configure CMake")
        self.assertIn(
            "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH=/tmp/rocm-root",
            configure_step.argv,
        )

    def test_cmake_amdgpu_tests_pin_libhsa_from_rocm_root(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)
            expected_env = ci.amdgpu_libhsa_test_env()

        test_steps = [
            step for step in steps if step.name.startswith("Test IREE CMake AMDGPU")
        ]
        self.assertEqual(len(test_steps), 2)
        for step in test_steps:
            self.assertEqual(step.env, expected_env)

    def test_cmake_amdgpu_tsan_excludes_only_host_incompatible_tests(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu-tsan"])

        steps = ci.steps_from_args(args)
        resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake AMDGPU resource tests with TSAN"
        )
        self.assertIn(
            ci.combine_ctest_regex(
                ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL,
            ),
            resource_test.argv,
        )

    def test_cmake_amdgpu_msan_builds_driver_targets_without_test_deps(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu-msan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("-DIREE_ENABLE_MSAN=ON" in line for line in command_lines))
        self.assertTrue(any("-DIREE_BUILD_TESTS=OFF" in line for line in command_lines))
        self.assertTrue(
            any("-DIREE_BUILD_BENCHMARKS=OFF" in line for line in command_lines)
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        for target in ci_config.AMDGPU_CMAKE_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        resource_target = ci.cmake_runtime_resource_build_target(
            ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX
        )
        self.assertFalse(any(resource_target in step.argv for step in build_steps))
        self.assertFalse(any("Test IREE CMake AMDGPU" in step.name for step in steps))

    def test_cmake_loom_amdgpu_command_runs_compile_coverage_without_driver(self):
        args = ci.parse_arguments(["iree-cmake-loom-amdgpu"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(len(steps), 3)
        self.assertTrue(
            any(
                self.uses_cmake_build_dir(step, "iree-cmake-loom-amdgpu")
                for step in steps
            )
        )
        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=OFF" in line for line in command_lines)
        )
        self.assertFalse(
            any("-DIREE_HAL_AMDGPU_TARGETS=" in line for line in command_lines)
        )
        build_step = next(
            step
            for step in steps
            if step.name == "Build Loom CMake AMDGPU compile coverage"
        )
        for target in ci_config.LOOM_AMDGPU_CMAKE_COMPILE_TEST_BUILD_TARGETS:
            self.assertIn(target, build_step.argv)
        test_step = next(
            step
            for step in steps
            if step.name == "Test Loom CMake AMDGPU compile coverage"
        )
        for regex in ci_config.LOOM_AMDGPU_CMAKE_COMPILE_CTEST_REGEXES:
            self.assertTrue(any(regex in arg for arg in test_step.argv))
        self.assertIn(ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX, test_step.argv)

    def test_cmake_vulkan_command_scopes_build_and_tests_to_vulkan(self):
        args = ci.parse_arguments(["iree-cmake-vulkan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("-DIREE_HAL_DRIVER_VULKAN=ON" in line for line in command_lines)
        )
        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=OFF" in line for line in command_lines)
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        resource_target = ci.cmake_runtime_resource_build_target(
            ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX
        )
        for target in ci_config.VULKAN_CMAKE_DRIVER_TARGETS + (resource_target,):
            self.assertTrue(any(target in step.argv for step in build_steps))
        for target in (
            "loom/src/loom/tools/iree-test-loom/all",
            "loom/binding/c/example/all",
            "loom/binding/c/test/target/spirv/all",
        ):
            self.assertFalse(any(target in step.argv for step in build_steps))
        package_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake Vulkan package tests"
        )
        self.assertTrue(
            any(ci_config.VULKAN_CTEST_REGEX in arg for arg in package_test.argv)
        )
        resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake Vulkan resource tests"
        )
        self.assertIn(ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX, resource_test.argv)
        self.assertIn(ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX, resource_test.argv)
        self.assertIn(ci_config.VULKAN_CTEST_REGEX, resource_test.argv)
        self.assertFalse(
            any("emit_spirv_vulkan_test" in arg for arg in resource_test.argv)
        )
        self.assertFalse(any("-fuse-ld=lld" in line for line in command_lines))

    def test_cmake_vulkan_sanitizers_use_generic_clang_configs(self):
        args = ci.parse_arguments(["iree-cmake-vulkan-sanitizers"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("iree-cmake-vulkan-asan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-vulkan-tsan" in line for line in command_lines))
        self.assertTrue(
            any("iree-cmake-vulkan-ubsan" in line for line in command_lines)
        )
        self.assertTrue(any("iree-cmake-vulkan-msan" in line for line in command_lines))

    def test_cmake_command_rejects_bazel_targets(self):
        args = ci.parse_arguments(["iree-cmake-cpu", "--target", "//runtime/..."])

        with self.assertRaisesRegex(ValueError, "--target"):
            ci.steps_from_args(args)


if __name__ == "__main__":
    unittest.main()
