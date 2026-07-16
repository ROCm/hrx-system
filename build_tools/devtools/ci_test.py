# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import re
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
        text = Path(path).read_text()
        match = re.search(
            rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_]+:\n|\Z)",
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
            "IREE_HAL_DRIVER_HIP",
            "IREE_HAL_DRIVER_LOCAL_SYNC",
            "IREE_HAL_DRIVER_LOCAL_TASK",
            "IREE_HAL_DRIVER_NULL",
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

        runtime_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU runtime resources"
        )
        self.assertIn(
            "--test_env=IREE_HAL_AMDGPU_LIBHSA_PATH="
            + str(Path("/tmp/rocm-root") / "lib" / "libhsa-runtime64.so.1"),
            runtime_resource_test.argv,
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

        runtime_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU runtime resources"
        )
        self.assertIn(
            "--test_env=IREE_HAL_AMDGPU_LIBHSA_PATH=/tmp/custom/libhsa-runtime64.so.1",
            runtime_resource_test.argv,
        )

    def test_amdgpu_loom_target_scope_runs_loom_resources(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        loom_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU Loom resources"
        )

        self.assertIn("//loom/...", loom_resource_test.argv)
        self.assertNotIn("//runtime/...", loom_resource_test.argv)
        self.assertFalse(
            any(step.name == "Test IREE AMDGPU runtime resources" for step in steps)
        )

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

    def test_amdgpu_command_scopes_tests_to_amdgpu(self):
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
        command_lines = [step.command_line() for step in steps]

        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        for target in ci_config.AMDGPU_BAZEL_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        self.assertTrue(
            any(
                "--test_tag_filters=" + ci_config.AMDGPU_RESOURCE_TAG in line
                for line in command_lines
            )
        )
        runtime_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU runtime resources"
        )
        self.assertIn("//runtime/...", runtime_resource_test.argv)
        self.assertNotIn("//loom/...", runtime_resource_test.argv)
        self.assertFalse(
            any(
                arg.startswith("-//runtime/src/iree/hal/drivers/amdgpu")
                for arg in runtime_resource_test.argv
            )
        )
        loom_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU Loom resources"
        )
        self.assertIn("//loom/...", loom_resource_test.argv)
        self.assertNotIn("//runtime/...", loom_resource_test.argv)

    def test_amdgpu_resource_slices_share_resource_tag_without_target_overlap(self):
        args = ci.parse_arguments(["iree-bazel-amdgpu", "--target", "//..."])

        steps = ci.steps_from_args(args)
        runtime_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU runtime resources"
        )
        loom_resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU Loom resources"
        )

        self.assertIn("//runtime/...", runtime_resource_test.argv)
        self.assertNotIn("//loom/...", runtime_resource_test.argv)
        self.assertIn("//loom/...", loom_resource_test.argv)
        self.assertNotIn("//runtime/...", loom_resource_test.argv)
        self.assertIn(
            "--test_tag_filters=" + ci_config.AMDGPU_RESOURCE_TAG,
            runtime_resource_test.argv,
        )
        self.assertIn(
            "--test_tag_filters=" + ci_config.AMDGPU_RESOURCE_TAG,
            loom_resource_test.argv,
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
        self.assertTrue(
            any(
                "bazel test --config=tsan --test_tag_filters="
                + ci_config.AMDGPU_RESOURCE_TAG
                + ",-"
                + ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL
                in line
                for line in command_lines
            )
        )
        tsan_resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE AMDGPU runtime resources and TSAN"
        )
        self.assertTrue(
            any(
                arg.startswith("--test_env=TSAN_OPTIONS=suppressions=")
                for arg in tsan_resource_test.argv
            )
        )
        loom_tsan_resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE AMDGPU Loom resources and TSAN"
        )
        self.assertIn(
            "--test_tag_filters="
            + ci_config.AMDGPU_RESOURCE_TAG
            + ",-"
            + ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL,
            loom_tsan_resource_test.argv,
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
                loom_resource_test = next(
                    step
                    for step in steps
                    if step.name
                    == f"Test IREE AMDGPU Loom resources and {sanitizer.upper()}"
                )
                self.assertIn(
                    "--test_tag_filters=" + ci_config.AMDGPU_RESOURCE_TAG,
                    loom_resource_test.argv,
                )
                self.assertFalse(
                    any(
                        ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL in arg
                        for arg in loom_resource_test.argv
                    )
                )

    def test_vulkan_command_builds_and_runs_vulkan_package_tests(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-vulkan",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(
            command_lines[0],
            "python3 dev.py bazel configure -DIREE_HAL_DRIVER_VULKAN=ON",
        )
        self.assertTrue(
            any(
                "bazel build " + ci_config.VULKAN_BAZEL_DRIVER_TARGETS[0] in line
                for line in command_lines
            )
        )
        package_test = next(
            step for step in steps if step.name == "Test IREE Vulkan package tests"
        )
        self.assertIn(ci_config.VULKAN_BAZEL_DRIVER_TARGETS[0], package_test.argv)
        for xfail_target in ci_config.VULKAN_XFAIL_TARGETS:
            self.assertIn(xfail_target, package_test.argv)
        self.assertFalse(any("--test_tag_filters" in arg for arg in package_test.argv))
        self.assertFalse(
            any(step.name == "Test IREE Vulkan loom resources" for step in steps)
        )

    def test_vulkan_command_runs_resource_slices_by_requirement_label(self):
        args = ci.parse_arguments(["iree-bazel-vulkan"])

        steps = ci.steps_from_args(args)
        resource_test = next(
            step for step in steps if step.name == "Test IREE Vulkan loom resources"
        )
        command_line = resource_test.command_line()

        self.assertIn("//loom/...", resource_test.argv)
        self.assertIn(
            "--test_tag_filters=" + ci_config.RUNTIME_VULKAN_RESOURCE_TAG,
            resource_test.argv,
        )
        self.assertNotIn(":spirv_vulkan_execution_test", command_line)
        self.assertNotIn(":emit_spirv_vulkan_test", command_line)
        self.assertNotIn(":emit_spirv_iree_hal_test", command_line)

    def test_bazel_vulkan_single_sanitizer_command_builds_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-vulkan-asan",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("bazel build --config=asan" in line for line in command_lines)
        )
        self.assertFalse(
            any("bazel test --config=asan" in line for line in command_lines)
        )
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=tsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_vulkan_sanitizers_build_with_generic_clang_configs(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-vulkan-sanitizers",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("bazel build --config=asan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel build --config=tsan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel build --config=ubsan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel build --config=msan" in line for line in command_lines)
        )
        self.assertFalse(any("bazel test --config=" in line for line in command_lines))

    def test_vulkan_workflows_require_hardware_preflight(self):
        for path, job_name in (
            (".github/workflows/ci_iree_bazel.yml", "linux_bazel_vulkan"),
            (".github/workflows/ci_iree_cmake.yml", "linux_cmake_vulkan"),
        ):
            with self.subTest(path=path):
                block = self.workflow_job_block(path, job_name)
                self.assertIn("runs-on: [self-hosted, Linux, X64, gpu_navi4x]", block)
                self.assertIn(
                    "bash .github/scripts/check_vulkan_hardware_environment.sh",
                    block,
                )
                self.assertNotIn("container:", block)
                self.assertNotIn("install_vulkan_host_environment", block)
                self.assertNotIn("VK_ICD_FILENAMES", block)
                self.assertNotIn("VK_DRIVER_FILES", block)
                self.assertNotIn("LIBGL_ALWAYS_SOFTWARE", block)

        preflight = Path(
            ".github/scripts/check_vulkan_hardware_environment.sh"
        ).read_text()
        self.assertNotIn("lvp_icd", preflight)
        self.assertNotIn("mesa-vulkan-drivers", preflight)
        self.assertNotIn("vulkan-loader", preflight)

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

    def test_bazel_gpu_sanitizer_workflows_stay_batched(self):
        for job_name in ("linux_bazel_amdgpu", "linux_bazel_vulkan"):
            with self.subTest(job=job_name):
                block = self.workflow_job_block(
                    ".github/workflows/ci_iree_bazel.yml", job_name
                )
                self.assertIn("/ Sanitizers", block)
                self.assertRegex(
                    block,
                    r"command: iree-bazel-(amdgpu|vulkan)-sanitizers",
                )

    def test_core_gpu_workflow_routes_gpu_labels_to_linux_runners(self):
        block = self.workflow_job_block(
            ".github/workflows/ci_core_linux.yml", "gpu_linux"
        )
        self.assertIn("runner_label: linux-gfx942-1gpu-core42-ossci-rocm", block)
        self.assertIn("runner_label: gpu_navi4x", block)

        reusable_workflow = Path(
            ".github/workflows/test_core_linux_gpu.yml"
        ).read_text()
        self.assertRegex(
            reusable_workflow,
            r"runner_label:\n\s+type: string\n\s+required: true",
        )
        reusable_block = self.workflow_job_block(
            ".github/workflows/test_core_linux_gpu.yml", "test_core_linux_gpu"
        )
        self.assertRegex(
            reusable_block,
            r"runs-on:\n\s+- self-hosted\n\s+- Linux\n\s+- X64\n"
            r"\s+- \"\$\{\{ inputs\.runner_label \}\}\"",
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

    def test_importer_workflow_is_path_scoped_and_uses_locked_cache_key(self):
        text = Path(".github/workflows/ci_importers.yml").read_text()

        self.assertIn("name: CI Importers", text)
        self.assertIn('- "requirements-importers-*.lock.txt"', text)
        self.assertIn('- "requirements-importers-*.in"', text)
        self.assertIn('- "build_tools/devtools/**"', text)
        self.assertIn('- "loom/config/**"', text)
        self.assertIn('- "loom/py/loom/importers/**"', text)
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

    def test_xfails_project_to_ctest_regexes(self):
        self.assertIn(
            "^iree/tokenizer/",
            ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/local/elf/elf_module_test$",
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

    def test_cmake_sanitizer_smoke_command_builds_minimal_targets(self):
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
        for target in ci_config.CMAKE_SANITIZER_SMOKE_TEST_BUILD_TARGETS:
            self.assertTrue(
                any(
                    target in step.argv
                    for step in build_steps
                    if "MSAN" not in step.name
                )
            )
        msan_build_step = next(step for step in build_steps if "MSAN" in step.name)
        for target in ci_config.CMAKE_SANITIZER_SMOKE_LIBRARY_BUILD_TARGETS:
            self.assertIn(target, msan_build_step.argv)
        for target in ci_config.CMAKE_SANITIZER_SMOKE_TEST_BUILD_TARGETS:
            self.assertNotIn(target, msan_build_step.argv)

        self.assertEqual(len(build_steps), 4)
        for step in build_steps:
            expected_targets = (
                ci_config.CMAKE_SANITIZER_SMOKE_LIBRARY_BUILD_TARGETS
                if "MSAN" in step.name
                else ci_config.CMAKE_SANITIZER_SMOKE_TEST_BUILD_TARGETS
            )
            self.assertEqual(
                set(step.argv).intersection(expected_targets),
                set(expected_targets),
            )
            self.assertNotIn("all", step.argv)

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
