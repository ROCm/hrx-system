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
from loom_binary import LoomBinaryBuildFileFunctions


class _PythonBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _should_emit_python_target(self):
        return True


class _LoomBinaryBuildFileFunctions(
    LoomBinaryBuildFileFunctions, bazel_to_cmake_converter.BuildFileFunctions
):
    pass


class ConfigTest(unittest.TestCase):
    def test_loom_kernel_projects_profile_roots_configs_and_library_edges(self):
        converter = SimpleNamespace(body="")
        functions = _LoomBinaryBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/programs",
            repo_root="/repo",
        )
        compatibility = functions.select(
            {
                "//loom/config/target:xdna_artifacts": [],
                "//conditions:default": ["@platforms//:incompatible"],
            }
        )
        functions.loom_target_profile(
            name="npu4",
            family="amd.xdna.aie2p",
            selector="amd.xdna.strix.17f0_10",
            target_compatible_with=compatibility,
        )
        functions.loom_kernel_binary(
            name="mul",
            srcs=["z.loom", "a.loom"],
            deps=[":z_library", ":a_library"],
            roots=["@second", "@first"],
            configs={"z.limit": "16", "a.value": "3"},
            target=":npu4",
            out="mul.xdna",
            testonly=True,
            target_compatible_with=compatibility,
        )
        self.assertIn('FAMILY\n    "amd.xdna.aie2p"', converter.body)
        self.assertIn('SELECTOR\n    "amd.xdna.strix.17f0_10"', converter.body)
        self.assertIn(
            "REQUIRES\n    LOOM_BUILD AND LOOM_TARGET_ARCH_XDNA AND LOOM_EMIT_XDNA",
            converter.body,
        )
        self.assertIn(
            "if(LOOM_BUILD AND LOOM_TARGET_ARCH_XDNA AND LOOM_EMIT_XDNA)",
            converter.body,
        )
        self.assertIn('COMPONENT\n    "//programs:mul"', converter.body)
        self.assertIn('SRCS\n    "z.loom"\n    "a.loom"', converter.body)
        self.assertIn("LIBRARIES\n    ::z_library\n    ::a_library", converter.body)
        self.assertIn('ROOTS\n    "@second"\n    "@first"', converter.body)
        self.assertIn('CONFIGS\n    "a.value=3"\n    "z.limit=16"', converter.body)
        self.assertIn("  TESTONLY\n", converter.body)

    def test_loom_kernel_outputs_reach_embedding_by_rule_or_output_label(self):
        converter = SimpleNamespace(body="")
        functions = _LoomBinaryBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/programs",
            repo_root="/repo",
        )
        functions.loom_kernel_binary(
            name="mul", srcs=["mul.loom"], target=":npu4", out="mul.xdna"
        )
        functions.loom_kernel_binary(
            name="deps_only", deps=[":library"], target=":npu4"
        )
        self.assertNotIn("  ROOTS\n", converter.body)
        self.assertNotIn("  CONFIGS\n", converter.body)
        self.assertIn('OUTPUT\n    "deps_only"', converter.body)
        for source in (":mul", ":mul.xdna"):
            with self.subTest(source=source):
                converter.body = ""
                functions.iree_c_embed_data(
                    name="embedded",
                    srcs=[source],
                    c_file_output="embedded.c",
                    h_file_output="embedded.h",
                    testonly=True,
                )
                self.assertIn('"${CMAKE_CURRENT_BINARY_DIR}/mul.xdna"', converter.body)
                self.assertNotIn("$<TARGET_FILE:", converter.body)

    def test_loom_rules_are_available_to_other_project_converters(self):
        repo_cfg = SimpleNamespace(
            PROJECTS=[],
            REPO_MAP={"@hrx": ""},
            CustomBuildFileFunctions=_LoomBinaryBuildFileFunctions,
        )
        output = bazel_to_cmake_converter.convert_build_file(
            """
load("//loom/build_tools/bazel:defs.bzl", "loom_kernel_binary", "loom_library", "loom_target_profile")
loom_library(name="support", srcs=["support.cxx"])
loom_target_profile(name="profile", family="amd.xdna.aie2p", selector="exact")
loom_kernel_binary(name="program", srcs=["program.loom"], deps=[":support"], target=":profile")
""",
            repo_cfg,
            "/repo/consumer",
            repo_root="/repo",
        )
        self.assertIn("loom_module(", output)
        self.assertIn("  NAME\n    support", output)
        self.assertIn("loom_target_profile(", output)
        self.assertIn("loom_kernel_binary(", output)
        self.assertIn("  LIBRARIES\n    ::support", output)
        self.assertIn('OUTPUT\n    "program"', output)

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

    def test_loaded_shell_test_honors_explicit_conversion_skip(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})
        build_dir = str(repo_root / "synthetic")

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "bazel_only_test",
    srcs = ["bazel_only_test.sh"],
    tags = ["skip-bazel_to_cmake"],
)
""",
            repo_cfg,
            build_dir,
            repo_root=str(repo_root),
        )
        self.assertNotIn("bazel_only_test", cmake)

        with self.assertRaisesRegex(NotImplementedError, "sh_test: visible_test"):
            bazel_to_cmake_converter.convert_build_file(
                """
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "visible_test",
    srcs = ["visible_test.sh"],
)
""",
                repo_cfg,
                build_dir,
                repo_root=str(repo_root),
            )

    def test_loaded_executable_aliases_honor_explicit_conversion_skip(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})
        build_dir = str(repo_root / "synthetic")

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//build_tools/bazel:executable.bzl", "iree_executable_alias")

iree_executable_alias(
    name = "bazel_only_alias",
    src = ":tool",
    tags = ["skip-bazel_to_cmake"],
)
""",
            repo_cfg,
            build_dir,
            repo_root=str(repo_root),
        )
        self.assertNotIn("bazel_only_alias", cmake)

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//build_tools/bazel:executable.bzl", "iree_wasi_executable_alias")

iree_wasi_executable_alias(
    name = "bazel_only_wasi_alias",
    src = ":tool",
    tags = ["skip-bazel_to_cmake"],
)
""",
            repo_cfg,
            build_dir,
            repo_root=str(repo_root),
        )
        self.assertNotIn("bazel_only_wasi_alias", cmake)

    def test_glob_exclusions_have_distinct_cmake_storage(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
cc_library(
    name = "filtered",
    srcs = glob(["*.cc"], exclude = ["excluded.cc"]),
)

cc_library(
    name = "complete",
    srcs = glob(["*.cc"]),
)
""",
            repo_cfg,
            str(repo_root / "synthetic"),
            repo_root=str(repo_root),
        )

        glob_vars = set(re.findall(r"file\(GLOB (_GLOB_X_CC(?:_[A-F0-9]{8})?)", cmake))
        self.assertEqual(len(glob_vars), 2)
        self.assertIn("_GLOB_X_CC", glob_vars)
        filtered_var = next(var for var in glob_vars if var != "_GLOB_X_CC")
        self.assertIn(f'    "${{{filtered_var}}}"', cmake)
        self.assertIn('    "${_GLOB_X_CC}"', cmake)

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
        with self.assertRaises(NotImplementedError):
            functions.select(
                {
                    "//compiler/plugins:input_stablehlo_enabled": [],
                    "//conditions:default": [],
                }
            )

    def test_converts_libamdf_build_condition(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._convert_select_condition("//libamdf/config:enabled_setting"),
            "AMDF_BUILD",
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

    def test_target_compatible_with_x86_64(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._target_compatible_condition(["@platforms//cpu:x86_64"]),
            'IREE_ARCH STREQUAL "x86_64"',
        )

    def test_converts_wasi_platform_condition(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._convert_select_condition("@platforms//os:wasi"),
            'CMAKE_SYSTEM_NAME STREQUAL "WASI"',
        )

    def test_target_compatible_with_parenthesizes_disjunctions(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        self.assertEqual(
            functions._target_compatible_condition(
                [
                    SimpleNamespace(cmake_condition="AMDF_BUILD"),
                    SimpleNamespace(
                        cmake_condition="AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA"
                    ),
                ]
            ),
            "AMDF_BUILD AND (AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA)",
        )

    def test_platform_select_deps_supports_named_target_blocks(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="libamdf",
        )

        runtime_data = functions.select(
            {
                "@platforms//os:windows": ["//runtime/src/iree/base:base"],
                "//conditions:default": [],
            }
        )
        target_block, select_block = functions._convert_platform_select_deps(
            "amdf_runtime_data", runtime_data, block_name="RUNTIME_DATA"
        )

        self.assertIn("  RUNTIME_DATA\n", target_block)
        self.assertNotIn("  DEPS\n", target_block)
        self.assertIn("${_amdf_runtime_data_platform_runtime_data}", target_block)
        self.assertIn('if(CMAKE_SYSTEM_NAME STREQUAL "Windows")', select_block)

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

    def test_c_embed_data_preserves_strip_prefix(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="runtime/src/example",
            repo_root="/repo",
        )

        functions.iree_c_embed_data(
            name="headers",
            srcs=["include/nested/header.h"],
            c_file_output="headers.c",
            h_file_output="headers.h",
            strip_prefix="runtime/src/example/include/",
        )

        self.assertIn(
            "  STRIP_PREFIX\n"
            '    "${PROJECT_SOURCE_DIR}/runtime/src/example/include/"\n',
            converter.body,
        )

    def test_c_embed_data_srcs_preserve_source_file_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            package = "runtime/src/iree/hal/drivers/task/executable/elf/testdata"
            source_directory = repo_root / package
            source_directory.mkdir(parents=True)
            (source_directory / "elementwise_mul_library.c").touch()
            converter = SimpleNamespace(body="")
            functions = bazel_to_cmake_converter.BuildFileFunctions(
                converter=converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=package,
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

        self.assertIn(
            'iree_package_target_name(_FILEGROUP_TARGET "::device_headers")',
            converter.body,
        )
        self.assertIn("add_custom_target(${_FILEGROUP_TARGET}", converter.body)
        self.assertIn(
            "iree_register_generated_compile_input(${_FILEGROUP_TARGET}\n"
            "  OUTPUTS\n"
            '    "${CMAKE_CURRENT_BINARY_DIR}/device_headers.stamp"',
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
            build_dir="synthetic/example",
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

    def test_py_test_preserves_unlocated_generated_data(self):
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.iree_spirv_asm_module(name="generated_data", src="input.spvasm")
        functions.iree_py_test(
            name="generated_data_test",
            srcs=["test.py"],
            data=[":generated_data"],
        )
        self.assertIn(
            'DATA\n    "${CMAKE_CURRENT_BINARY_DIR}/generated_data.spv"', converter.body
        )

    def test_py_test_preserves_configured_generated_arguments_and_data(self):
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.iree_spirv_asm_module(name="generated_data", src="input.spvasm")
        functions.iree_py_test(
            name="configured_data_test",
            srcs=["test.py"],
            args=["--common"]
            + functions.select(
                {
                    "@platforms//os:linux": ["--input=$(rootpath :generated_data)"],
                    "//conditions:default": ["--no-input"],
                }
            ),
            data=functions.select(
                {
                    "@platforms//os:linux": [":generated_data"],
                    "//conditions:default": [],
                }
            ),
        )
        self.assertIn(
            'if(CMAKE_SYSTEM_NAME STREQUAL "Linux")\n'
            "  list(APPEND _configured_data_test_platform_args "
            '"--input={{${CMAKE_CURRENT_BINARY_DIR}/generated_data.spv}}")\n'
            "else()\n"
            '  list(APPEND _configured_data_test_platform_args "--no-input")\n'
            "endif()",
            converter.body,
        )
        self.assertIn(
            'if(CMAKE_SYSTEM_NAME STREQUAL "Linux")\n'
            "  list(APPEND _configured_data_test_platform_data "
            '"${CMAKE_CURRENT_BINARY_DIR}/generated_data.spv")\n'
            "endif()",
            converter.body,
        )
        self.assertIn(
            'ARGS\n    "--common"\n    ${_configured_data_test_platform_args}',
            converter.body,
        )
        self.assertIn(
            "DATA\n    ${_configured_data_test_platform_data}", converter.body
        )

    def test_private_executable_test_can_skip_cmake(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/example",
        )

        functions.iree_executable_test(
            name="test",
            src=":runner",
            visibility=["//visibility:private"],
            tags=["skip-bazel_to_cmake"],
        )

        self.assertEqual(converter.body, "")

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

    def test_build_file_loads_cross_project_requirement_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            alpha = repo_root / "alpha/requirements"
            alpha.mkdir(parents=True)
            (alpha / "defs.bzl").write_text(
                """
FEATURE = build_requirement(
    id = "alpha.feature",
    label = Label("//alpha/requirements:feature"),
    enabled_by = Label("//alpha/config:feature"),
    cmake_condition = "ENABLE_ALPHA",
)
""",
                encoding="utf-8",
            )
            beta = repo_root / "beta/requirements"
            beta.mkdir(parents=True)
            (beta / "defs.bzl").write_text(
                'load("//alpha/requirements:defs.bzl", IMPORTED_FEATURE = "FEATURE")\n',
                encoding="utf-8",
            )
            cmake = bazel_to_cmake_converter.convert_build_file(
                """
load("//beta/requirements:defs.bzl", LOCAL_FEATURE = "IMPORTED_FEATURE")

cc_library(
    name = "guarded",
    srcs = ["guarded.cc"],
    target_compatible_with = [LOCAL_FEATURE],
)
""",
                SimpleNamespace(PROJECTS=[], REPO_MAP={}),
                str(repo_root / "beta"),
                repo_root=str(repo_root),
            )

        self.assertIn("if(ENABLE_ALPHA)\niree_cc_library(", cmake)

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

    def test_executable_test_preserves_resource_lock(self):
        for resource_group in (None, "shared-device", "shared-storage"):
            with self.subTest(resource_group=resource_group):
                converter = SimpleNamespace(body="")
                functions = bazel_to_cmake_converter.BuildFileFunctions(
                    converter=converter,
                    targets=bazel_to_cmake_targets.TargetConverter(
                        repo_map={"@hrx": ""}
                    ),
                    build_dir="",
                )
                functions.iree_executable_test(
                    name="invocation",
                    src="//tools:runner",
                    resource_group=resource_group,
                )
                if resource_group is None:
                    self.assertNotIn("RESOURCE_GROUP", converter.body)
                else:
                    self.assertIn(
                        "RESOURCE_GROUP\n    " + resource_group, converter.body
                    )

    def test_native_test_rejects_multiple_files_in_single_file_locations(self):
        for kind in ("location", "rootpath", "execpath"):
            for field in ("args", "env"):
                with self.subTest(kind=kind, field=field):
                    converter = SimpleNamespace(body="")
                    functions = bazel_to_cmake_converter.BuildFileFunctions(
                        converter=converter,
                        targets=bazel_to_cmake_targets.TargetConverter(
                            repo_map={"@hrx": ""}
                        ),
                        build_dir="/repo/pkg",
                        repo_root="/repo",
                    )
                    functions.filegroup(name="inputs", srcs=["first.txt", "second.txt"])
                    location = f"$({kind} :inputs)"
                    value = [location] if field == "args" else {"INPUT": location}
                    with self.assertRaisesRegex(ValueError, "single-file location"):
                        functions.native_test(
                            name="location_test",
                            src="//tools:runner",
                            **{field: value},
                        )

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

        self.assertIn('"{{${PROJECT_SOURCE_DIR}/pkg/input.txt}}"', converter.body)
        self.assertIn(
            '"--flag={{${PROJECT_SOURCE_DIR}/pkg/nested/input.bin}}"',
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

        self.assertIn('"{{${PROJECT_SOURCE_DIR}/pkg/input.txt}}"', converter.body)
        self.assertIn(
            '"--flag={{${PROJECT_SOURCE_DIR}/pkg/nested/input.bin}}"',
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

    def test_execution_test_suite_maps_size_to_timeout(self):
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
            size="small",
        )

        self.assertIn("  TIMEOUT\n    60\n", converter.body)

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
    data = glob(["*.txt"]),
)
""",
            repo_cfg,
            str(repo_root / "build_tools/testing/test"),
            repo_root=str(repo_root),
        )

        self.assertIn("file(GLOB _GLOB_X_TXT LIST_DIRECTORIES false", cmake)
        self.assertIn('    "${_GLOB_X_TXT}"', cmake)
        self.assertNotIn("::${_GLOB_X_TXT}", cmake)

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

    def test_cc_test_preserves_platform_link_options_and_locations(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.cc_test(
            name="native_test",
            srcs=["native_test.cc"],
            linkopts=["-Wl,--wrap=open"]
            + functions.select(
                {
                    "@platforms//os:linux": [
                        "-Wl,--version-script=$(location :exports.map)",
                    ],
                    "//conditions:default": [],
                }
            ),
        )
        self.assertIn("  LINKOPTS\n", converter.body)
        self.assertIn('"-Wl,--wrap=open"', converter.body)
        self.assertIn('CMAKE_SYSTEM_NAME STREQUAL "Linux"', converter.body)
        self.assertIn(
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map",
            converter.body,
        )
        self.assertNotIn("$(location", converter.body)

    def test_cc_library_emits_dependency_suppressions(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        functions.cc_library(
            name="libvulkan",
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

    def test_loom_artifact_execution_requires_compiler_project(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_config.ProjectTargetConverter(
                repo_map={"@hrx": ""},
                projects=[],
                target_mappings={
                    "//loom/src/loom/tools/loom-compile": ["loom::tools::loom-compile"],
                },
            ),
            build_dir="/repo/experimental/xdna/cts",
            repo_root="/repo",
        )

        functions.iree_execution_test_suite(
            name="integer_shifts_test",
            manifests=["integer_shifts.test.json"],
            tools={"loom-compile": "//loom/src/loom/tools/loom-compile"},
            target_compatible_with=functions.select(
                {
                    "//loom/config/target:xdna_artifacts": [],
                    "//conditions:default": ["@platforms//:incompatible"],
                }
            ),
        )

        self.assertIn(
            "if(LOOM_BUILD AND LOOM_TARGET_ARCH_XDNA AND LOOM_EMIT_XDNA)",
            converter.body,
        )
        self.assertIn("loom-compile=", converter.body)

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
