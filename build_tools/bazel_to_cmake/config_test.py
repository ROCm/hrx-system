# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for bazel_to_cmake project config routing."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements
import bazel_to_cmake_targets


class ConfigTest(unittest.TestCase):
    def test_selects_longest_matching_project_for_build_path(self):
        runtime = bazel_to_cmake_config.ProjectConfig(
            name="runtime",
            package_prefixes=["runtime"],
        )
        runtime_iree = bazel_to_cmake_config.ProjectConfig(
            name="runtime_iree",
            package_prefixes=["runtime/src/iree"],
        )

        self.assertIs(
            bazel_to_cmake_config.find_project_for_path(
                [runtime, runtime_iree],
                "runtime/src/iree/base",
            ),
            runtime_iree,
        )
        self.assertIsNone(
            bazel_to_cmake_config.find_project_for_path(
                [runtime, runtime_iree],
                "libhrx/src/libhrx",
            )
        )

    def test_routes_unmatched_targets_by_label_owner(self):
        def convert_runtime_target(converter, target):
            return ["runtime:" + converter._convert_to_cmake_path(target)]

        def convert_libhrx_target(converter, target):
            return ["libhrx:" + converter._convert_to_cmake_path(target)]

        def convert_root_target(converter, target):
            return ["root:" + converter._convert_to_cmake_path(target)]

        runtime = bazel_to_cmake_config.ProjectConfig(
            name="runtime",
            package_prefixes=["runtime"],
            convert_unmatched_target=convert_runtime_target,
        )
        libhrx = bazel_to_cmake_config.ProjectConfig(
            name="libhrx",
            package_prefixes=["libhrx"],
            target_mappings={
                "//libhrx:defines": ["libhrx_defs"],
            },
            convert_unmatched_target=convert_libhrx_target,
        )

        converter = bazel_to_cmake_config.ProjectTargetConverter(
            repo_map={"@iree": ""},
            projects=[runtime, libhrx],
            convert_unmatched_target=convert_root_target,
        )

        self.assertEqual(
            converter.convert_target("//runtime/other:thing"),
            ["runtime:runtime::other::thing"],
        )
        self.assertEqual(
            converter.convert_target("@iree//runtime/other:thing"),
            ["runtime:runtime::other::thing"],
        )
        self.assertEqual(
            converter.convert_target("//libhrx/src/libhrx:hrx"),
            ["libhrx:libhrx::src::libhrx::hrx"],
        )
        self.assertEqual(
            converter.convert_target("@iree//libhrx/src/libhrx:hrx"),
            ["libhrx:libhrx::src::libhrx::hrx"],
        )
        self.assertEqual(
            converter.convert_target("//libhrx:defines"),
            ["libhrx_defs"],
        )
        self.assertEqual(
            converter.convert_target("@iree//third_party:catch2"),
            ["iree::third_party::catch2"],
        )
        self.assertEqual(
            converter.convert_target("//other:thing"),
            ["root:other::thing"],
        )

    def test_loom_c_root_targets_strip_filesystem_staging_prefix(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )

        converter = bazel_to_cmake_config.ProjectTargetConverter(
            repo_map={"@iree": ""},
            projects=[loom],
        )

        self.assertEqual(
            converter.convert_target("//loom/src/loom/ir"),
            ["loom::ir"],
        )
        self.assertEqual(
            converter.convert_target("@iree//loom/src/loom/tools/loom-check"),
            ["loom::tools::loom-check"],
        )
        self.assertEqual(
            converter.convert_target("//loom/src/loom/tools/loom-check:loom-check"),
            ["loom::tools::loom-check::loom-check"],
        )
        self.assertEqual(
            converter.convert_target("//loom/src:defines"),
            [],
        )

    def test_rejects_compiler_monorepo_external_targets(self):
        converter = bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""})

        for target in (
            "@llvm-project//llvm:Core",
            "@llvm-project//mlir:IR",
            "@stablehlo//:stablehlo_ops",
            "@torch-mlir//:TorchMLIRTorchDialect",
        ):
            with self.subTest(target=target):
                with self.assertRaises(KeyError):
                    converter.convert_target(target)

    def test_rejects_compiler_monorepo_local_targets(self):
        def convert_root_target(converter, target):
            return ["root:" + converter._convert_to_cmake_path(target)]

        converter = bazel_to_cmake_config.ProjectTargetConverter(
            repo_map={"@iree": ""},
            projects=[],
            convert_unmatched_target=convert_root_target,
        )

        for target in (
            "@iree//compiler/src/iree/compiler/API:CAPI",
            "@iree//llvm-external-projects/iree-dialects:CAPI",
        ):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    converter.convert_target(target)

    def test_rejects_compiler_monorepo_select_conditions(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._convert_select_condition("//runtime/config/hal:driver_hip"),
            "IREE_HAL_DRIVER_HIP",
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:amdgpu_artifacts"
            ),
            "LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU",
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:spirv_vulkan_artifacts"
            ),
            "LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV AND IREE_HAL_DRIVER_VULKAN",
        )
        with self.assertRaises(NotImplementedError):
            functions.select(
                {
                    "//compiler/plugins:input_stablehlo_enabled": [],
                    "//conditions:default": [],
                }
            )

    def test_target_compatible_with_composes_selects_and_requirements(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="",
        )

        target_compatible_with = [
            SimpleNamespace(cmake_condition="IREE_HAL_DRIVER_WEBGPU"),
            SimpleNamespace(cmake_condition="IREE_HAL_DRIVER_WEBGPU"),
        ] + functions.select(
            {
                "@platforms//cpu:wasm32": [],
                "//conditions:default": ["@platforms//:incompatible"],
            }
        )

        self.assertEqual(
            functions._target_compatible_condition(target_compatible_with),
            'IREE_HAL_DRIVER_WEBGPU AND IREE_ARCH STREQUAL "wasm_32"',
        )

    def test_cc_binary_linkshared_emits_shared_library(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="runtime/src/iree/hal/local/elf/testdata",
        )

        functions.cc_binary(
            name="elementwise_mul_library.so",
            srcs=["elementwise_mul_library.c"],
            deps=["//runtime/src/iree/hal/local:executable_library"],
            testonly=True,
            linkshared=True,
        )

        self.assertIn("iree_cc_library(", converter.body)
        self.assertNotIn("iree_cc_binary(", converter.body)
        self.assertIn("  SHARED\n", converter.body)
        self.assertIn("  TESTONLY\n", converter.body)
        self.assertIn("iree::hal::local::executable_library", converter.body)

    def test_c_embed_data_srcs_can_reference_generated_targets(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="runtime/src/iree/hal/local/elf/testdata",
            repo_root=str(repo_root),
        )

        functions.cc_binary(
            name="elementwise_mul_library.so",
            srcs=["elementwise_mul_library.c"],
            deps=["//runtime/src/iree/hal/local:executable_library"],
            testonly=True,
            linkshared=True,
        )
        converter.body = ""

        functions.iree_c_embed_data(
            name="elementwise_mul",
            srcs=[":elementwise_mul_library.so"],
            c_file_output="elementwise_mul.c",
            h_file_output="elementwise_mul.h",
            testonly=True,
            flatten=True,
        )

        self.assertIn(
            "$<TARGET_FILE:iree::hal::local::elf::testdata::elementwise_mul_library.so>",
            converter.body,
        )
        self.assertNotIn('"elementwise_mul_library.so"', converter.body)

    def test_c_embed_data_srcs_preserve_generated_file_labels(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="runtime/src/iree/vm/test",
            repo_root=str(repo_root),
        )

        functions.iree_c_embed_data(
            name="all_bytecode_modules_c",
            srcs=[":arithmetic_ops.vmfb"],
            c_file_output="all_bytecode_modules.c",
            h_file_output="all_bytecode_modules.h",
            flatten=True,
        )

        self.assertIn('"arithmetic_ops.vmfb"', converter.body)
        self.assertNotIn("$<TARGET_FILE:", converter.body)

    def test_c_embed_data_srcs_preserve_source_file_labels(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="runtime/src/iree/hal/local/elf/testdata",
            repo_root=str(repo_root),
        )

        functions.iree_c_embed_data(
            name="elementwise_mul_source",
            srcs=[":elementwise_mul.mlir"],
            c_file_output="elementwise_mul_source.c",
            h_file_output="elementwise_mul_source.h",
            testonly=True,
            flatten=True,
        )

        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/local/elf/testdata/'
            'elementwise_mul.mlir"',
            converter.body,
        )
        self.assertNotIn("$<TARGET_FILE:", converter.body)

    def test_requirement_policy_loads_cross_project_requirement_defs(self):
        repo_root = Path(__file__).resolve().parents[2]
        policy = bazel_to_cmake_requirements.load_project_policy(
            str(repo_root),
            "loom",
        )

        collected = policy.collect("loom/src/loom/tooling/target/amdgpu/execution")
        conditions = [
            condition.cmake_condition for condition in collected.cmake_conditions()
        ]

        self.assertIn("LOOM_EXECUTE_IREE_HAL", conditions)
        self.assertIn("IREE_HAL_DRIVER_AMDGPU", conditions)
        self.assertNotIn("LOOM_EXECUTE_AMDGPU", conditions)

    def test_native_test_emits_target_compatible_guard(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="",
        )

        functions.native_test(
            name="portable_test",
            src="//tools:runner",
            target_compatible_with=functions.select(
                {
                    "@platforms//cpu:wasm32": [],
                    "//conditions:default": ["@platforms//:incompatible"],
                }
            ),
        )

        self.assertIn('if(IREE_ARCH STREQUAL "wasm_32")', converter.body)
        self.assertIn("iree_native_test(", converter.body)
        self.assertIn("endif()", converter.body)

    def test_native_test_converts_location_args_to_file_locators(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.native_test(
            name="location_test",
            src="//tools:runner",
            args=[
                "$(location input.txt)",
                "--flag=$(location nested/input.bin)",
            ],
        )

        self.assertIn('"{{${PROJECT_SOURCE_DIR}/pkg/input.txt}}"', converter.body)
        self.assertIn(
            '"--flag={{${PROJECT_SOURCE_DIR}/pkg/nested/input.bin}}"',
            converter.body,
        )

    def test_native_test_converts_location_env(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.native_test(
            name="location_env_test",
            src="//tools:runner",
            env={
                "FIXTURE": "$(location input.txt)",
                "SPIRV_VAL": "$(rootpath //third_party:spirv_val)",
            },
        )

        self.assertIn("ENV", converter.body)
        self.assertIn(
            '"FIXTURE=${PROJECT_SOURCE_DIR}/pkg/input.txt"',
            converter.body,
        )
        self.assertIn(
            '"SPIRV_VAL=$<TARGET_FILE:iree::third_party::spirv_val>"',
            converter.body,
        )

    def test_native_test_omits_unresolved_external_location_env(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.native_test(
            name="external_env_test",
            src="//tools:runner",
            env={
                "LLVM_OBJDUMP": "$(rootpath @wasi_sdk//:llvm-objdump)",
            },
        )

        self.assertNotIn("ENV", converter.body)
        self.assertNotIn("TARGET_FILE:pkg_@wasi_sdk", converter.body)

    def test_cc_test_emits_sanitizer_suppressions(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="",
        )

        functions.cc_test(
            name="vulkan_test",
            srcs=["vulkan_test.cc"],
            sanitizer_suppressions={
                "lsan": "//build_tools/sanitizer:lsan_suppressions_vulkan.txt",
            },
        )

        self.assertIn("SANITIZER_SUPPRESSIONS", converter.body)
        self.assertIn("    lsan", converter.body)
        self.assertIn("    vulkan", converter.body)

    def test_execution_test_suite_emits_target_compatible_guard(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.iree_execution_test_suite(
            name="execution_test",
            manifests=["test.json"],
            tools={"runner": "//tools:runner"},
            target_compatible_with=functions.select(
                {
                    "@platforms//cpu:wasm32": [],
                    "//conditions:default": ["@platforms//:incompatible"],
                }
            ),
        )

        self.assertIn('if(IREE_ARCH STREQUAL "wasm_32")', converter.body)
        self.assertIn("iree_execution_test_suite(", converter.body)
        self.assertIn("endif()", converter.body)

    def test_execution_test_suite_emits_sanitizer_suppressions(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.iree_execution_test_suite(
            name="execution_test",
            manifests=["test.json"],
            tools={"runner": "//tools:runner"},
            sanitizer_suppressions={
                "lsan": "//build_tools/sanitizer:lsan_suppressions_vulkan.txt",
            },
        )

        self.assertIn("SANITIZER_SUPPRESSIONS", converter.body)
        self.assertIn("    lsan", converter.body)
        self.assertIn("    vulkan", converter.body)


if __name__ == "__main__":
    unittest.main()
