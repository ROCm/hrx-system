# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for bazel_to_cmake project config routing."""

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements
import bazel_to_cmake_targets


class _PythonBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _should_emit_python_target(self):
        return True


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
            repo_map={"@hrx": ""},
            projects=[runtime, libhrx],
            convert_unmatched_target=convert_root_target,
        )

        self.assertEqual(
            converter.convert_target("//runtime/other:thing"),
            ["runtime:runtime::other::thing"],
        )
        self.assertEqual(
            converter.convert_target("@hrx//runtime/other:thing"),
            ["runtime:runtime::other::thing"],
        )
        self.assertEqual(
            converter.convert_target("//libhrx/src/libhrx:hrx"),
            ["libhrx:libhrx::src::libhrx::hrx"],
        )
        self.assertEqual(
            converter.convert_target("@hrx//libhrx/src/libhrx:hrx"),
            ["libhrx:libhrx::src::libhrx::hrx"],
        )
        self.assertEqual(
            converter.convert_target("//libhrx:defines"),
            ["libhrx_defs"],
        )
        self.assertEqual(
            converter.convert_target("@hrx//third_party:catch2"),
            ["iree::third_party::catch2"],
        )
        self.assertEqual(
            converter.convert_target("//other:thing"),
            ["root:other::thing"],
        )

    def test_package_group_has_no_cmake_target(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
package_group(
    name = "implementation_consumers",
    packages = ["//runtime/src/iree/hal/drivers/task/..."],
    includes = ["//runtime:other_consumers"],
)
""",
            repo_cfg,
            str(repo_root / "runtime"),
            repo_root=str(repo_root),
        )

        self.assertEqual(cmake.count("iree_add_all_subdirs()"), 1)
        self.assertNotIn("implementation_consumers", cmake)

    def test_loom_c_root_targets_strip_filesystem_staging_prefix(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )

        converter = bazel_to_cmake_config.ProjectTargetConverter(
            repo_map={"@hrx": ""},
            projects=[loom],
        )

        self.assertEqual(
            converter.convert_target("//loom/src/loom/ir"),
            ["loom::ir"],
        )
        self.assertEqual(
            converter.convert_target("@hrx//loom/src/loom/tools/loom-check"),
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

    def test_loom_check_test_suite_preserves_suite_rule(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//loom/build_tools/bazel:loom_check.bzl", "loom_check_test_suite")

loom_check_test_suite(
    name = "loom_check_file_test",
    srcs = [
        "test/source_low/b.loom-test",
        "test/source_low/a.loom-test",
    ],
    data = [
        "//loom/src/loom/test/corpus/source_low:vector_dot.loom-test",
        "//third_party:spirv_dis",
    ],
    tags = ["gpu"],
    test_name_prefix_to_strip = "test/source_low/",
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/target/arch/amdgpu"),
            repo_root=str(repo_root),
        )

        self.assertIn("if(LOOM_TARGET_ARCH_AMDGPU)", cmake)
        self.assertIn("loom_check_test_suite(", cmake)
        self.assertNotIn("iree_native_test(", cmake)
        self.assertIn('    "test/source_low/b.loom-test"', cmake)
        self.assertIn('    "test/source_low/a.loom-test"', cmake)
        self.assertLess(
            cmake.index('    "test/source_low/b.loom-test"'),
            cmake.index('    "test/source_low/a.loom-test"'),
        )
        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/loom/src/loom/test/corpus/source_low/'
            'vector_dot.loom-test"',
            cmake,
        )
        self.assertIn('"iree::third_party::spirv_dis"', cmake)
        self.assertIn('    "gpu"', cmake)
        self.assertIn('    "test/source_low/"', cmake)
        self.assertNotIn('    "loom-check"', cmake)

    def test_loom_check_test_suite_preserves_glob_srcs(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//loom/build_tools/bazel:loom_check.bzl", "loom_check_test_suite")

loom_check_test_suite(
    name = "loom_check_file_test",
    srcs = glob(["test/source_low/*.loom-test"]),
    test_name_prefix_to_strip = "test/source_low/",
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/target/arch/amdgpu"),
            repo_root=str(repo_root),
        )

        self.assertIn(
            "file(GLOB _GLOB_TEST_SOURCE_LOW_X_LOOM_TEST LIST_DIRECTORIES false",
            cmake,
        )
        self.assertIn(
            "RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS "
            "test/source_low/*.loom-test)",
            cmake,
        )
        self.assertIn("loom_check_test_suite(", cmake)
        self.assertIn('    "${_GLOB_TEST_SOURCE_LOW_X_LOOM_TEST}"', cmake)
        self.assertNotIn('"test/source_low/a.loom-test"', cmake)
        self.assertNotIn("iree_native_test(", cmake)

    def test_ignored_rule_accepts_opaque_loaded_value(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load(
    "//loom/build_tools/bazel:defs.bzl",
    "loom_test",
)
load(
    "//synthetic:policy.bzl",
    TEST_EXECUTION_POLICY = "DEVICE_EXECUTION_POLICY",
)

loom_test(
    name = "one_test",
    srcs = ["one.loom", "two.loom"],
    execution_profile = TEST_EXECUTION_POLICY,
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/tooling/target/amdgpu/test"),
            repo_root=str(repo_root),
        )

        self.assertNotIn("one_test", cmake)

    def test_unhandled_loaded_rule_fails_loudly(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})

        with self.assertRaisesRegex(
            NotImplementedError,
            "loaded symbol 'unhandled_rule'.*has no Bazel-to-CMake representation",
        ):
            bazel_to_cmake_converter.convert_build_file(
                """
load("//synthetic:rules.bzl", "unhandled_rule")

unhandled_rule(name = "must_not_disappear")
""",
                repo_cfg,
                str(repo_root / "synthetic"),
                repo_root=str(repo_root),
            )

    def test_glob_exclusions_have_distinct_cmake_storage(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//loom/build_tools/bazel:defs.bzl", "loom_module")

loom_module(
    name = "filtered",
    srcs = glob(["*.loom"], exclude = ["negative.loom"]),
)

loom_module(
    name = "complete",
    srcs = glob(["*.loom"]),
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/test/corpus/authoring"),
            repo_root=str(repo_root),
        )

        glob_vars = set(
            re.findall(r"file\(GLOB (_GLOB_X_LOOM(?:_[A-F0-9]{8})?)", cmake)
        )
        self.assertEqual(len(glob_vars), 2)
        self.assertIn("_GLOB_X_LOOM", glob_vars)
        filtered_var = next(var for var in glob_vars if var != "_GLOB_X_LOOM")
        self.assertIn(f'    "${{{filtered_var}}}"', cmake)
        self.assertIn('    "${_GLOB_X_LOOM}"', cmake)

    def test_loom_module_registers_generated_location(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//build_tools/bazel:executable.bzl", "iree_executable_test")
load("//loom/build_tools/bazel:defs.bzl", "loom_module")

loom_module(
    name = "linked_checks",
    srcs = ["testdata/checks.loom"],
    libraries = ["kernels.loom"],
    roots = ["@case"],
    configs = ["model.width=16"],
    mode = "link",
    output = "linked.loombc",
    output_format = "bc",
    include_input_exports = True,
    strip_check = True,
    require_resolved_config = True,
)

iree_executable_test(
    name = "linked_checks_test",
    src = "//loom/src/loom/tools/loom-format",
    args = ["$(location :linked_checks)"],
    data = [":linked_checks"],
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/target/arch/amdgpu"),
            repo_root=str(repo_root),
        )

        self.assertIn("loom_module(", cmake)
        self.assertIn('    "testdata/checks.loom"', cmake)
        self.assertIn('    "kernels.loom"', cmake)
        self.assertIn('    "@case"', cmake)
        self.assertIn('    "model.width=16"', cmake)
        self.assertIn('    "link"', cmake)
        self.assertIn('    "linked.loombc"', cmake)
        self.assertIn('    "bc"', cmake)
        self.assertIn("  INCLUDE_INPUT_EXPORTS", cmake)
        self.assertIn("  STRIP_CHECK", cmake)
        self.assertIn("  REQUIRE_RESOLVED_CONFIG", cmake)
        self.assertIn('"{{${CMAKE_CURRENT_BINARY_DIR}/linked.loombc}}"', cmake)
        self.assertIn(
            '  DATA\n    "${CMAKE_CURRENT_BINARY_DIR}/linked.loombc"',
            cmake,
        )

    def test_loom_module_preserves_cross_package_module_targets(self):
        repo_root = Path(__file__).resolve().parents[2]
        loom = bazel_to_cmake_config.include_project(
            str(repo_root / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        repo_cfg = SimpleNamespace(PROJECTS=[loom], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//loom/build_tools/bazel:defs.bzl", "loom_module")

loom_module(
    name = "provider_suite",
    srcs = ["provider_checks.loom"],
    libraries = [
        "//loom/src/loom/test/corpus/encoding:numeric_conversion_cases",
        "//loom/src/loom/test/corpus/encoding:mxfp4_decode_bf16.loom",
    ],
)
""",
            repo_cfg,
            str(repo_root / "loom/src/loom/tooling/target/amdgpu/test"),
            repo_root=str(repo_root),
        )

        self.assertIn(
            '    "loom::test::corpus::encoding::numeric_conversion_cases"',
            cmake,
        )
        self.assertIn(
            '    "${PROJECT_SOURCE_DIR}/loom/src/loom/test/corpus/encoding/'
            'mxfp4_decode_bf16.loom"',
            cmake,
        )

    def test_rejects_compiler_monorepo_external_targets(self):
        converter = bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""})

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
            repo_map={"@hrx": ""},
            projects=[],
            convert_unmatched_target=convert_root_target,
        )

        for target in (
            "@hrx//compiler/src/iree/compiler/API:CAPI",
            "@hrx//llvm-external-projects/iree-dialects:CAPI",
        ):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    converter.convert_target(target)

    def test_rejects_compiler_monorepo_select_conditions(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._convert_select_condition(
                "//build_tools/bazel:cc_compiler_clang"
            ),
            'CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT MSVC',
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//build_tools/bazel:cc_compiler_clang_cl"
            ),
            'CMAKE_C_COMPILER_ID MATCHES "Clang" AND MSVC',
        )
        self.assertEqual(
            functions._convert_select_condition("//build_tools/bazel:cc_compiler_gcc"),
            'CMAKE_C_COMPILER_ID STREQUAL "GNU"',
        )
        self.assertEqual(
            functions._convert_select_condition("//build_tools/bazel:cc_compiler_msvc"),
            'CMAKE_C_COMPILER_ID STREQUAL "MSVC"',
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:amdgpu_artifacts"
            ),
            "LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU",
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:llvmir_amdgpu_target_env"
            ),
            "LOOM_TARGET_ARCH_LLVMIR AND LOOM_EMIT_LLVMIR AND LOOM_TARGET_ARCH_AMDGPU",
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:llvmir_artifacts"
            ),
            "LOOM_TARGET_ARCH_LLVMIR AND LOOM_EMIT_LLVMIR",
        )
        self.assertEqual(
            functions._convert_select_condition(
                "//loom/config/target:llvmir_x86_target_env"
            ),
            "LOOM_TARGET_ARCH_LLVMIR AND LOOM_EMIT_LLVMIR AND LOOM_TARGET_ARCH_X86",
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
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="runtime/src/iree/hal/drivers/task/executable/elf/testdata",
        )

        functions.cc_binary(
            name="elementwise_mul_library.so",
            srcs=["elementwise_mul_library.c"],
            deps=["//runtime/src/iree/hal/drivers/task/executable/library:abi"],
            testonly=True,
            linkshared=True,
        )

        self.assertIn("iree_cc_library(", converter.body)
        self.assertNotIn("iree_cc_binary(", converter.body)
        self.assertIn("  SHARED\n", converter.body)
        self.assertIn("  TESTONLY\n", converter.body)
        self.assertIn(
            "iree::hal::drivers::task::executable::library::abi", converter.body
        )

    def test_cc_library_linkopts_expand_location_make_variables(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="libhrx/src/binding/hip",
        )

        functions.cc_library(
            name="amdhip64",
            srcs=["api.c"],
            linkopts=[
                "-Wl,--undefined-version",
                "-Wl,--version-script=$(location :amdhip64.map)",
            ],
            shared=True,
        )

        self.assertIn("  LINKOPTS\n", converter.body)
        self.assertIn("-Wl,--undefined-version", converter.body)
        # A same-package $(location ...) resolves to a current-source-dir path
        # (correct even in a CMake sub-project), not a literal make-variable nor
        # a ${PROJECT_SOURCE_DIR} path relative to the repo root.
        self.assertNotIn("$(location", converter.body)
        self.assertIn(
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/amdhip64.map",
            converter.body,
        )

    def test_c_embed_data_srcs_can_reference_generated_targets(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="runtime/src/iree/hal/drivers/task/executable/elf/testdata",
            repo_root=str(repo_root),
        )

        functions.cc_binary(
            name="elementwise_mul_library.so",
            srcs=["elementwise_mul_library.c"],
            deps=["//runtime/src/iree/hal/drivers/task/executable/library:abi"],
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
            "$<TARGET_FILE:iree::hal::drivers::task::executable::elf::testdata::elementwise_mul_library.so>",
            converter.body,
        )
        self.assertNotIn('"elementwise_mul_library.so"', converter.body)

    def test_c_embed_data_srcs_preserve_generated_file_labels(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir=("runtime/src/iree/hal/drivers/task/executable/elf/testdata"),
            repo_root=str(repo_root),
        )

        functions.iree_c_embed_data(
            name="generated_kernel_c",
            srcs=[":generated_kernel.bin"],
            c_file_output="generated_kernel.c",
            h_file_output="generated_kernel.h",
            flatten=True,
        )

        self.assertIn('"generated_kernel.bin"', converter.body)
        self.assertNotIn("$<TARGET_FILE:", converter.body)

    def test_c_embed_data_srcs_preserve_source_file_labels(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="runtime/src/iree/hal/drivers/task/executable/elf/testdata",
            repo_root=str(repo_root),
        )

        functions.iree_c_embed_data(
            name="elementwise_mul_source",
            srcs=[":elementwise_mul_library.c"],
            c_file_output="elementwise_mul_source.c",
            h_file_output="elementwise_mul_source.h",
            testonly=True,
            flatten=True,
        )

        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/drivers/task/executable/elf/testdata/'
            'elementwise_mul_library.c"',
            converter.body,
        )
        self.assertNotIn("$<TARGET_FILE:", converter.body)

    def test_filegroup_registers_stamp_output_producer(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="runtime/src/example",
            repo_root="/repo",
        )

        functions.filegroup(name="device_headers", srcs=["device.h"])

        self.assertIn("add_custom_target(device_headers", converter.body)
        self.assertIn(
            "iree_register_generated_compile_input(device_headers\n"
            "  OUTPUTS\n"
            '    "${CMAKE_CURRENT_BINARY_DIR}/device_headers.stamp"',
            converter.body,
        )

    def test_py_test_allows_unlocated_source_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/bazel_to_cmake",
            repo_root=str(repo_root),
        )

        functions.iree_py_test(
            name="source_data_test",
            srcs=["config_test.py"],
            args=["bazel_to_cmake_config_test"],
            data=["//build_tools/bazel_to_cmake:config_test.py"],
            main="config_test.py",
            deps=[],
        )

        self.assertIn("iree_py_test(", converter.body)

    def test_py_test_preserves_all_sources_and_main(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/bazel_to_cmake",
            repo_root=str(repo_root),
        )

        functions.iree_py_test(
            name="multi_source_test",
            srcs=["config_test.py", "bazel_to_cmake_targets_test.py"],
            main="config_test.py",
            deps=[],
        )

        self.assertIn('MAIN\n    "config_test.py"', converter.body)
        self.assertIn(
            'SRCS\n    "config_test.py"\n    "bazel_to_cmake_targets_test.py"',
            converter.body,
        )

    def test_py_test_maps_size_to_default_timeout(self):
        repo_root = Path(__file__).resolve().parents[2]
        for size, timeout in {
            "small": 60,
            "medium": 300,
            "large": 900,
            "enormous": 3600,
        }.items():
            with self.subTest(size=size):
                converter = SimpleNamespace(body="")
                functions = _PythonBuildFileFunctions(
                    converter=converter,
                    targets=bazel_to_cmake_targets.TargetConverter(
                        repo_map={"@hrx": ""}
                    ),
                    build_dir="build_tools/bazel_to_cmake",
                    repo_root=str(repo_root),
                )

                functions.iree_py_test(
                    name="sized_test",
                    srcs=["config_test.py"],
                    deps=[],
                    size=size,
                )

                self.assertIn(f"TIMEOUT\n    {timeout}", converter.body)

    def test_py_test_explicit_timeout_overrides_size(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/bazel_to_cmake",
            repo_root=str(repo_root),
        )

        functions.iree_py_test(
            name="sized_test",
            srcs=["config_test.py"],
            deps=[],
            size="enormous",
            timeout="short",
        )

        self.assertIn("TIMEOUT\n    60", converter.body)
        self.assertNotIn("TIMEOUT\n    3600", converter.body)

    def test_py_library_resolves_cross_package_sources(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="loom/py/loom/example",
            repo_root=str(repo_root),
        )

        functions.iree_py_library(
            name="shared_source",
            srcs=["//build_tools/bazel_to_cmake:config_test.py"],
            deps=[],
        )

        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/build_tools/bazel_to_cmake/config_test.py"',
            converter.body,
        )

    def test_py_test_rejects_unlocated_generated_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/bazel_to_cmake",
            repo_root=str(repo_root),
        )

        with self.assertRaisesRegex(NotImplementedError, "iree_py_test data"):
            functions.iree_py_test(
                name="generated_data_test",
                srcs=["config_test.py"],
                args=["bazel_to_cmake_config_test"],
                data=["//build_tools/bazel_to_cmake:generated_data.txt"],
                main="config_test.py",
                deps=[],
            )

    def test_generated_files_requires_explicit_cmake_projection(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@iree": ""}),
            build_dir="runtime/src/iree/vm/bytecode/tooling",
        )

        functions.iree_generated_files(
            name="tables_gen",
            tags=["skip-bazel_to_cmake"],
        )
        with self.assertRaisesRegex(
            NotImplementedError, "requires an explicit CMake projection"
        ):
            functions.iree_generated_files(name="tables_gen")

    def test_requirement_policy_loads_cross_project_requirement_defs(self):
        repo_root = Path(__file__).resolve().parents[2]
        source = """
load(
    "//loom/requirements:defs.bzl",
    "EXECUTE_IREE_HAL",
    "TARGET_ARCH_AMDGPU",
)
load("//runtime/requirements:defs.bzl", "HAL_AMDGPU")

PACKAGE_POLICIES = [
    package_policy(
        packages = ["synthetic/cross_project/..."],
        build_requirements = [
            TARGET_ARCH_AMDGPU,
            EXECUTE_IREE_HAL,
            HAL_AMDGPU,
        ],
    ),
]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "package_policy.bzl"
            policy_path.write_text(source, encoding="utf-8")
            env = bazel_to_cmake_requirements._requirement_defs_env()
            env["package_policy"] = bazel_to_cmake_requirements.package_policy
            bazel_to_cmake_requirements._exec_bzl(policy_path, env, repo_root)
        policy = bazel_to_cmake_requirements.ProjectRequirementPolicy(
            package_policies=list(env["PACKAGE_POLICIES"]),
        )

        collected = policy.collect("synthetic/cross_project/child")
        conditions = [
            condition.cmake_condition for condition in collected.cmake_conditions()
        ]

        self.assertIn("LOOM_EXECUTE_IREE_HAL", conditions)
        self.assertIn("IREE_HAL_DRIVER_AMDGPU", conditions)
        self.assertNotIn("LOOM_EXECUTE_AMDGPU", conditions)

        requirements = {
            requirement.id: requirement for requirement in collected.build_requirements
        }
        self.assertEqual(
            requirements["loom.target.arch.amdgpu"].label,
            "//loom/requirements:target_arch_amdgpu",
        )
        self.assertEqual(
            requirements["loom.target.arch.amdgpu"].enabled_by,
            "//loom/config/target/arch:amdgpu",
        )
        self.assertEqual(
            requirements["runtime.hal.amdgpu"].label,
            "//runtime/requirements:hal_amdgpu",
        )

    def test_requirement_policy_excludes_matching_subtree(self):
        broad_requirement = bazel_to_cmake_requirements.build_requirement(
            id="synthetic.broad",
            label="//synthetic:requires_broad",
            enabled_by="//synthetic:enable_broad",
            cmake_condition="SYNTHETIC_BROAD",
        )
        host_requirement = bazel_to_cmake_requirements.build_requirement(
            id="synthetic.host",
            label="//synthetic:requires_host",
            enabled_by="//synthetic:enable_host",
            cmake_condition="SYNTHETIC_HOST",
        )
        policy = bazel_to_cmake_requirements.ProjectRequirementPolicy(
            package_policies=[
                bazel_to_cmake_requirements.package_policy(
                    packages=["synthetic/..."],
                    excluded_packages=["synthetic/host/..."],
                    build_requirements=[broad_requirement],
                ),
                bazel_to_cmake_requirements.package_policy(
                    packages=["synthetic/host/..."],
                    build_requirements=[host_requirement],
                ),
            ],
        )

        device_policy = policy.collect("synthetic/device")
        self.assertEqual(
            [requirement.id for requirement in device_policy.build_requirements],
            ["synthetic.broad"],
        )
        host_policy = policy.collect("synthetic/host/child")
        self.assertEqual(
            [requirement.id for requirement in host_policy.build_requirements],
            ["synthetic.host"],
        )

    def test_native_test_emits_target_compatible_guard(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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

    def test_native_test_preserves_file_and_target_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir=str(repo_root / "build_tools/testing/test"),
            repo_root=str(repo_root),
        )

        functions.native_test(
            name="data_test",
            src="//tools:runner",
            data=[
                "input.txt",
                "//third_party:spirv_val",
            ],
        )

        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/build_tools/testing/test/input.txt"',
            converter.body,
        )
        self.assertIn('"iree::third_party::spirv_val"', converter.body)

    def test_cc_binary_benchmark_converts_location_args_to_source_paths(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.cc_binary_benchmark(
            name="location_benchmark",
            srcs=["location_benchmark.cc"],
            args=[
                "$(location input.txt)",
                "--flag=$(rootpath nested/input.bin)",
            ],
        )

        self.assertIn('"${PROJECT_SOURCE_DIR}/pkg/input.txt"', converter.body)
        self.assertIn(
            '"--flag=${PROJECT_SOURCE_DIR}/pkg/nested/input.bin"',
            converter.body,
        )

    def test_runtime_hal_cts_test_suite_converts_location_args_to_source_paths(
        self,
    ):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions._iree_runtime_hal_cts_test_suite(
            backends=":backends",
            name="hal_cts",
            args=[
                "$(location input.txt)",
                "--flag=$(rootpath nested/input.bin)",
            ],
        )

        self.assertIn('"${PROJECT_SOURCE_DIR}/pkg/input.txt"', converter.body)
        self.assertIn(
            '"--flag=${PROJECT_SOURCE_DIR}/pkg/nested/input.bin"',
            converter.body,
        )

    def test_execution_test_suite_converts_location_args_to_source_paths(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.iree_execution_test_suite(
            name="execution_test",
            manifests=["test.json"],
            tools={"runner": "//tools:runner"},
            args=[
                "$(location input.txt)",
                "--flag=$(rootpath nested/input.bin)",
            ],
        )

        self.assertIn('"${PROJECT_SOURCE_DIR}/pkg/input.txt"', converter.body)
        self.assertIn(
            '"--flag=${PROJECT_SOURCE_DIR}/pkg/nested/input.bin"',
            converter.body,
        )

    def test_execution_test_suite_preserves_file_and_target_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir=str(repo_root / "build_tools/testing/test"),
            repo_root=str(repo_root),
        )

        functions.iree_execution_test_suite(
            name="execution_test",
            manifests=["smoke.test.json"],
            tools={"runner": "//tools:runner"},
            data=[
                "input.txt",
                "//third_party:spirv_dis",
            ],
        )

        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/build_tools/testing/test/input.txt"',
            converter.body,
        )
        self.assertIn('"iree::third_party::spirv_dis"', converter.body)

    def test_execution_test_suite_preserves_glob_data(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//build_tools/testing:build_defs.bzl", "iree_execution_test_suite")

iree_execution_test_suite(
    name = "execution_test",
    manifests = ["smoke.test.json"],
    tools = {"runner": "//tools:runner"},
    data = glob(["*.loom"]),
)
""",
            repo_cfg,
            str(repo_root / "build_tools/testing/test"),
            repo_root=str(repo_root),
        )

        self.assertIn("file(GLOB _GLOB_X_LOOM LIST_DIRECTORIES false", cmake)
        self.assertIn('    "${_GLOB_X_LOOM}"', cmake)
        self.assertNotIn("::${_GLOB_X_LOOM}", cmake)

    def test_native_test_converts_location_env(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.native_test(
            name="external_env_test",
            src="//tools:runner",
            data=["@wasi_sdk//:llvm-objdump"],
            env={
                "LLVM_OBJDUMP": "$(rootpath @wasi_sdk//:llvm-objdump)",
            },
        )

        self.assertNotIn("ENV", converter.body)
        self.assertNotIn("DATA", converter.body)
        self.assertNotIn("@wasi_sdk", converter.body)
        self.assertNotIn("TARGET_FILE:pkg_@wasi_sdk", converter.body)

    def test_cc_test_emits_sanitizer_suppressions(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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

    def test_cc_test_converts_location_args(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.cc_binary(
            name="fixture_tool",
            srcs=["fixture_tool.cc"],
        )
        functions.cc_test(
            name="location_test",
            srcs=["location_test.cc"],
            args=[
                "$(location input.txt)",
                "--tool=$(location //pkg:fixture_tool)",
                "--runner=$(location //tools:runner)",
            ],
        )

        self.assertIn('"${PROJECT_SOURCE_DIR}/pkg/input.txt"', converter.body)
        self.assertIn(
            '"--tool=${PROJECT_SOURCE_DIR}/pkg/fixture_tool"',
            converter.body,
        )
        self.assertIn(
            '"--runner=$<TARGET_FILE:iree::tools::runner>"',
            converter.body,
        )
        self.assertNotIn("$(location", converter.body)

    def test_cc_test_converts_env(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.cc_test(
            name="env_test",
            srcs=["env_test.cc"],
            env={
                "FEATURE": "enabled",
                "FIXTURE": "$(location input.txt)",
            },
        )

        self.assertIn("ENV", converter.body)
        self.assertIn('"FEATURE=enabled"', converter.body)
        self.assertIn(
            '"FIXTURE=${PROJECT_SOURCE_DIR}/pkg/input.txt"',
            converter.body,
        )
        self.assertNotIn("$(location", converter.body)

    def test_execution_test_suite_emits_target_compatible_guard(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
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

    def test_execution_test_suite_emits_resource_group(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )

        functions.iree_execution_test_suite(
            name="execution_test",
            manifests=["test.json"],
            tools={"runner": "//tools:runner"},
            resource_group="gpu",
        )

        self.assertIn("RESOURCE_GROUP", converter.body)
        self.assertIn("    gpu", converter.body)


if __name__ == "__main__":
    unittest.main()
