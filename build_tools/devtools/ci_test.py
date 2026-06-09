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

from build_tools.devtools import ci, ci_config


class CiTest(unittest.TestCase):
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
        self.assertIn("-DIREE_ROCM_DEPENDENCY_MODE=pinned", text)
        self.assertNotIn("IREE_ROCM_PATH", text)
        self.assertNotIn("/opt/rocm", text)

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
        resource_test = next(
            step for step in steps if step.name == "Test IREE AMDGPU resources"
        )
        self.assertIn("//runtime/...", resource_test.argv)
        for target in ci_config.AMDGPU_XFAIL_TARGETS:
            self.assertIn(target, resource_test.argv)
        self.assertNotIn(
            "-//runtime/src/iree/hal/drivers/amdgpu:system_test",
            resource_test.argv,
        )

    def test_amdgpu_sanitizer_command_uses_amdgpu_sanitizer_xfails(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-sanitizers",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        sanitizer_test_steps = [
            step for step in steps if step.name.startswith("Test IREE")
        ]

        for xfail_target in ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS:
            self.assertTrue(
                any(xfail_target in step.argv for step in sanitizer_test_steps)
            )
        tsan_test_steps = [
            step for step in sanitizer_test_steps if "--config=tsan" in step.argv
        ]
        non_tsan_test_steps = [
            step for step in sanitizer_test_steps if "--config=tsan" not in step.argv
        ]
        for xfail_target in ci_config.AMDGPU_TSAN_XFAIL_TARGETS:
            self.assertTrue(any(xfail_target in step.argv for step in tsan_test_steps))
            self.assertFalse(
                any(xfail_target in step.argv for step in non_tsan_test_steps)
            )

    def test_bazel_amdgpu_single_sanitizer_command_runs_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-tsan",
                "--target",
                "//runtime/...",
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
                in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "-//runtime/src/iree/hal/drivers/amdgpu/util:pm4_dispatch_live_test"
                in line
                for line in command_lines
            )
        )
        self.assertFalse(any("--config=asan" in line for line in command_lines))
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_amdgpu_asan_omits_tsan_specific_xfails(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-asan",
                "--target",
                "//runtime/...",
            ]
        )

        command_lines = [step.command_line() for step in ci.steps_from_args(args)]

        self.assertFalse(
            any(
                "-//runtime/src/iree/hal/drivers/amdgpu/util:pm4_dispatch_live_test"
                in line
                for line in command_lines
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
        host_test = next(
            step for step in steps if step.name == "Test IREE Vulkan tests"
        )
        self.assertIn(ci_config.VULKAN_BAZEL_DRIVER_TARGETS[0], host_test.argv)
        self.assertFalse(any("--test_tag_filters" in arg for arg in host_test.argv))

    def test_bazel_vulkan_single_sanitizer_command_runs_one_configuration(self):
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
            any("bazel test --config=asan" in line for line in command_lines)
        )
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=tsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_vulkan_sanitizers_use_generic_clang_configs(self):
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
            any("bazel test --config=asan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel test --config=tsan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel test --config=ubsan" in line for line in command_lines)
        )
        self.assertTrue(
            any("bazel build --config=msan" in line for line in command_lines)
        )

    def test_vulkan_workflows_require_hardware_preflight(self):
        for path, job_name in (
            (".github/workflows/ci_iree_bazel.yml", "linux_bazel_vulkan"),
            (".github/workflows/ci_iree_cmake.yml", "linux_cmake_vulkan"),
        ):
            with self.subTest(path=path):
                block = self.workflow_job_block(path, job_name)
                self.assertIn("runs-on: gpu_navi4x", block)
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

    def test_xfails_project_to_ctest_regexes(self):
        self.assertIn(
            "^iree/tokenizer/",
            ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/allocator_test$",
            ci_config.AMDGPU_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/system_test$",
            ci_config.AMDGPU_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/util/vmem_test$",
            ci_config.AMDGPU_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/util/block_pool_test$",
            ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/util/pm4_program_test$",
            ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertNotIn(
            "^iree/hal/drivers/amdgpu/util/pm4_dispatch_live_test$",
            ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX,
        )
        self.assertIn(
            "^iree/hal/drivers/amdgpu/util/pm4_dispatch_live_test$",
            ci_config.AMDGPU_TSAN_CTEST_EXCLUDE_REGEX,
        )

    def test_cmake_cpu_sanitizer_command_uses_cmake_build_dir_and_xfails(self):
        args = ci.parse_arguments(["iree-cmake-cpu-ubsan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                "--cmake-build-dir build/ci/iree-cmake-cpu-ubsan" in line
                for line in command_lines
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
        self.assertTrue(any("runtime-resource=" in step.argv for step in test_steps))

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

    def test_cmake_amdgpu_command_scopes_build_and_tests_to_amdgpu(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=ON" in line for line in command_lines)
        )
        self.assertTrue(
            any("-DIREE_ROCM_DEPENDENCY_MODE=pinned" in line for line in command_lines)
        )
        self.assertTrue(
            any(
                "-DIREE_HAL_AMDGPU_TARGETS=" + ci_config.AMDGPU_TARGET_SELECTOR in line
                for line in command_lines
            )
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        for target in ci_config.AMDGPU_CMAKE_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        self.assertTrue(
            any("-R '^iree/hal/drivers/amdgpu/'" in line for line in command_lines)
        )
        self.assertTrue(
            any(
                "^iree/hal/drivers/amdgpu/system_test$" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "^iree/hal/drivers/amdgpu/util/vmem_test$" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any("-L runtime-resource=amd-gpu" in line for line in command_lines)
        )

    def test_cmake_amdgpu_tsan_uses_tsan_specific_xfails(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu-tsan"])

        command_lines = [step.command_line() for step in ci.steps_from_args(args)]

        self.assertTrue(
            any(
                "^iree/hal/drivers/amdgpu/util/pm4_dispatch_live_test$" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "^iree/hal/drivers/amdgpu/host_queue_command_buffer_profiling_test$"
                in line
                for line in command_lines
            )
        )

    def test_cmake_amdgpu_asan_omits_tsan_specific_xfails(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu-asan"])

        command_lines = [step.command_line() for step in ci.steps_from_args(args)]

        self.assertFalse(
            any(
                "^iree/hal/drivers/amdgpu/util/pm4_dispatch_live_test$" in line
                for line in command_lines
            )
        )
        self.assertFalse(
            any(
                "^iree/hal/drivers/amdgpu/host_queue_command_buffer_profiling_test$"
                in line
                for line in command_lines
            )
        )

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
        for target in ci_config.VULKAN_CMAKE_DRIVER_TARGETS:
            self.assertTrue(any(target in step.argv for step in build_steps))
        self.assertTrue(
            any(ci_config.VULKAN_CTEST_REGEX in line for line in command_lines)
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
