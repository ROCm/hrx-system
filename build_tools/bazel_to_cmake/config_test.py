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
    def test_configured_selection_uses_effective_native_values(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
            configuration={
                "CMAKE_SYSTEM_NAME": "Windows",
                "CMAKE_C_COMPILER_ID": "Clang",
                "MSVC": "ON",
                "IREE_ENABLE_RUNTIME_TRACING": "ON",
                "IREE_TRACING_MODE": "2",
                "IREE_TRACING_PROVIDER": "console",
            },
        )
        self.assertEqual(
            functions.select(
                {
                    "//build_tools/bazel:cc_compiler_clang": ["-fexceptions"],
                    "//build_tools/bazel:cc_compiler_clang_cl": ["/EHsc"],
                    "//conditions:default": [],
                }
            ),
            ["/EHsc"],
        )
        providers = {
            "//runtime/config/base/tracing:provider_console": ":console",
            "//runtime/config/base/tracing:provider_tracy": ":tracy",
            "//conditions:default": ":disabled",
        }
        self.assertEqual(functions.select(providers), ":console")
        functions._configuration["IREE_ENABLE_RUNTIME_TRACING"] = "OFF"
        self.assertEqual(functions.select(providers), ":disabled")
        with self.assertRaisesRegex(ValueError, "Ambiguous native selection"):
            functions.select(
                {
                    "//build_tools/bazel:iree_is_windows": ["windows.c"],
                    "@platforms//os:windows": ["other.c"],
                }
            )
        with self.assertRaisesRegex(ValueError, "No matching native selection"):
            functions.select({"@platforms//os:linux": ["linux.c"]})
        self.assertEqual(
            functions.select(
                {
                    "//build_tools/bazel:iree_is_windows": ["windows.c"],
                    "@platforms//os:windows": ["windows.c"],
                }
            ),
            ["windows.c"],
        )

    def test_cc_library_compiles_linkstamp_as_a_native_source(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.cc_library(
            name="identity",
            hdrs=["identity.h"],
            linkstamp="identity.c",
        )
        self.assertIn('  SRCS\n    "identity.c"\n', converter.body)

    def test_cc_test_tokenizes_arguments_after_location_expansion(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.filegroup(name="inputs", srcs=["first.txt", "second.txt"])
        self.assertEqual(
            functions._convert_cc_test_args(
                [
                    "'--payload=two words' --enabled",
                    "--input=$(location :first.txt)",
                    "$(locations :inputs)",
                    "'$(locations :inputs)'",
                ]
            ),
            [
                "--payload=two words",
                "--enabled",
                "--input={{${PROJECT_SOURCE_DIR}/pkg/first.txt}}",
                "{{${PROJECT_SOURCE_DIR}/pkg/first.txt}}",
                "{{${PROJECT_SOURCE_DIR}/pkg/second.txt}}",
                "{{${PROJECT_SOURCE_DIR}/pkg/first.txt}} {{${PROJECT_SOURCE_DIR}/pkg/second.txt}}",
            ],
        )

    def test_target_compatibility_preserves_default_exclusions(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        condition = functions.select(
            {
                "@platforms//os:windows": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }
        )
        self.assertEqual(
            functions._target_compatible_condition(condition),
            'NOT ((CMAKE_SYSTEM_NAME STREQUAL "Windows"))',
        )

    def test_compile_options_preserve_language_and_platform(self):
        for rule in ("cc_library", "cc_binary", "cc_test", "iree_cc_fuzz"):
            with self.subTest(rule=rule):
                output = bazel_to_cmake_converter.convert_build_file(
                    f"""
{rule}(
    name = "mixed",
    srcs = ["first.c", "second.cc"],
    copts = ["-Wall"],
    conlyopts = ["-std=c11"],
    cxxopts = select({{
        "@platforms//os:windows": ["/EHsc"],
        "//conditions:default": ["-fexceptions"],
    }}),
)
""",
                    SimpleNamespace(),
                    "/repo/pkg",
                    repo_root="/repo",
                )
                self.assertIn('if(CMAKE_SYSTEM_NAME STREQUAL "Windows")', output)
                self.assertIn('"$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"', output)
                self.assertIn('"$<$<COMPILE_LANGUAGE:CXX>:-fexceptions>"', output)
                self.assertIn(
                    '  COPTS\n    "-Wall"\n    "$<$<COMPILE_LANGUAGE:C>:-std=c11>"\n'
                    "    ${_mixed_CXX_platform_copts}\n",
                    output,
                )

    def test_cc_library_group_preserves_interface_dependencies(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.iree_cc_library_group(name="aggregate", deps=[":first", ":second"])
        self.assertEqual(
            converter.body,
            "iree_cc_library(\n  NAME\n    aggregate\n  DEPS\n    ::first\n    ::second\n  PUBLIC\n)\n\n",
        )

    def test_project_conditions_reach_other_project_consumers(self):
        repo_cfg = SimpleNamespace(
            REPO_MAP={"@hrx": ""},
            CONDITION_MAPPINGS={"//config:enabled": "REPO_ENABLED"},
            PROJECTS=[
                bazel_to_cmake_config.ProjectConfig(
                    name="producer",
                    package_prefixes=["producer"],
                    condition_mappings={
                        "//producer/config:enabled": "PRODUCER_ENABLED",
                    },
                ),
                bazel_to_cmake_config.ProjectConfig(
                    name="consumer", package_prefixes=["consumer"]
                ),
            ],
        )
        output = bazel_to_cmake_converter.convert_build_file(
            """
cc_library(
    name = "consumer",
    srcs = select({
        "//producer/config:enabled": ["enabled.c"],
        "//conditions:default": ["disabled.c"],
    }),
    target_compatible_with = select({
        "//config:enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
""",
            repo_cfg,
            "/repo/consumer",
            repo_root="/repo",
        )
        self.assertIn("if(PRODUCER_ENABLED)", output)
        self.assertIn('"enabled.c"', output)
        self.assertIn('"disabled.c"', output)
        self.assertIn("if(REPO_ENABLED)", output)

    def test_gpu_image_targets_use_their_own_driver_selection(self):
        cases = (
            ("amdgpu", "IREE_HAL_AMDGPU_TARGETS", "IREE_HAL_AMD_GPU_TARGETS"),
            ("amd/gpu", "IREE_HAL_AMD_GPU_TARGETS", "IREE_HAL_AMDGPU_TARGETS"),
        )
        for driver, variable, other_variable in cases:
            with self.subTest(driver=driver):
                converter = SimpleNamespace(body="")
                functions = bazel_to_cmake_converter.BuildFileFunctions(
                    converter=converter,
                    targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
                    build_dir=f"/repo/runtime/src/iree/hal/drivers/{driver}/image",
                    repo_root="/repo",
                )
                functions._iree_amdgpu_binary_variants_embed_data(
                    name="hip_images",
                    target="amdgcn-amd-amdhsa",
                    srcs=["image.hip"],
                    target_selectors_flag=f"//runtime/src/iree/hal/drivers/{driver}:targets",
                    source_format="hip",
                    testonly=True,
                )
                self.assertIn("TARGETS\n    ${" + variable + "}", converter.body)
                self.assertNotIn("${" + other_variable + "}", converter.body)

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

    def test_package_boundary_has_no_cmake_target(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})

        cmake = bazel_to_cmake_converter.convert_build_file(
            """
load("//build_tools/bazel:package_boundary.bzl", "iree_package_boundary")

iree_package_boundary(
    allowed_descendant_paths = ["include/**"],
    reason = "Preserve installed include paths.",
)
""",
            repo_cfg,
            str(repo_root / "synthetic"),
            repo_root=str(repo_root),
        )

        self.assertEqual(cmake.count("iree_add_all_subdirs()"), 1)
        self.assertNotIn("package_boundary_policy", cmake)

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

    def test_loaded_objc_library_preserves_platform_and_arc_policy(self):
        repo_root = Path(__file__).resolve().parents[2]
        repo_cfg = SimpleNamespace(PROJECTS=[], REPO_MAP={"@hrx": ""})
        build = """
load("@rules_cc//cc:objc_library.bzl", "objc_library")

objc_library(
    name = "native_activity",
    srcs = ["automatic.m"],
    non_arc_srcs = ["activity.m"],
    target_compatible_with = ["@platforms//os:macos"],
    tags = ["skip-bazel_to_cmake"],
)
"""
        cmake = bazel_to_cmake_converter.convert_build_file(
            build,
            repo_cfg,
            str(repo_root / "synthetic"),
            repo_root=str(repo_root),
        )
        self.assertNotIn("native_activity", cmake)
        cmake = bazel_to_cmake_converter.convert_build_file(
            build.replace('"skip-bazel_to_cmake"', ""),
            repo_cfg,
            str(repo_root / "synthetic"),
            repo_root=str(repo_root),
        )
        self.assertIn(
            'if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")\nenable_language(OBJC)', cmake
        )
        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/synthetic/automatic.m"\n  PROPERTIES COMPILE_OPTIONS "-fobjc-arc"',
            cmake,
        )
        self.assertIn(
            '"${PROJECT_SOURCE_DIR}/synthetic/activity.m"\n  PROPERTIES COMPILE_OPTIONS "-fno-objc-arc"',
            cmake,
        )
        self.assertIn("iree_cc_library(\n  NAME\n    native_activity", cmake)

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
        for family, condition in (
            ("rdna", "AMDF_FAMILY_RDNA"),
            ("cdna", "AMDF_FAMILY_CDNA"),
            ("xdna", "AMDF_FAMILY_XDNA"),
            ("gpu", "AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA"),
        ):
            with self.subTest(family=family):
                label = f"//libamdf/config/family:{family}"
                self.assertEqual(functions._convert_select_condition(label), condition)
                self.assertEqual(
                    functions._target_compatible_condition([label]),
                    f"({condition})" if family == "gpu" else condition,
                )

    def test_target_compatible_with_composes_selects_and_requirements(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        target_compatible_with = [
            "//runtime/config/hal:driver_webgpu",
            "//runtime/config/hal:driver_webgpu",
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

    def test_target_compatible_with_x86(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )

        for architecture in ("x86_32", "x86_64"):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    functions._target_compatible_condition(
                        [f"@platforms//cpu:{architecture}"]
                    ),
                    f'IREE_ARCH STREQUAL "{architecture}"',
                )

    def test_cc_source_selection_preserves_architecture_and_fallback(self):
        for rule in ("cc_library", "cc_binary", "cc_test", "iree_cc_fuzz"):
            for mixed in (False, True):
                with self.subTest(rule=rule, mixed=mixed):
                    converter = SimpleNamespace(body="")
                    functions = bazel_to_cmake_converter.BuildFileFunctions(
                        converter=converter,
                        targets=bazel_to_cmake_targets.TargetConverter(
                            repo_map={"@hrx": ""}
                        ),
                        build_dir="",
                    )
                    srcs = functions.select(
                        {
                            "@platforms//cpu:x86_32": [":host_x86.c"],
                            "@platforms//cpu:x86_64": [":host_x86.c"],
                            "//conditions:default": [":host_generic.c"],
                        }
                    )
                    if mixed:
                        srcs = [":shared.c"] + srcs
                    getattr(functions, rule)(name="host", srcs=srcs)
                    prefix, target = converter.body.split("iree_cc_", 1)
                    self.assertEqual(
                        prefix,
                        'set(_host_platform_srcs "")\n'
                        'if(IREE_ARCH STREQUAL "x86_32")\n'
                        '  list(APPEND _host_platform_srcs "host_x86.c")\n'
                        'elseif(IREE_ARCH STREQUAL "x86_64")\n'
                        '  list(APPEND _host_platform_srcs "host_x86.c")\n'
                        "else()\n"
                        '  list(APPEND _host_platform_srcs "host_generic.c")\n'
                        "endif()\n",
                    )
                    self.assertIn("    ${_host_platform_srcs}\n", target)
                    self.assertEqual('    "shared.c"\n' in target, mixed)

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

    def test_tracing_provider_compatibility(self):
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=SimpleNamespace(body=""),
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="",
        )
        for provider in ("console", "tracy"):
            with self.subTest(provider=provider):
                compatibility = functions.select(
                    {
                        f"//runtime/config/base/tracing:provider_{provider}": [],
                        "//conditions:default": ["@platforms//:incompatible"],
                    }
                )
                self.assertEqual(
                    functions._target_compatible_condition(compatibility),
                    "IREE_ENABLE_RUNTIME_TRACING AND IREE_TRACING_MODE GREATER 0 "
                    f'AND IREE_TRACING_PROVIDER STREQUAL "{provider}"',
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
                    "//libamdf/config:enabled_setting",
                    "//libamdf/config/family:gpu",
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
            "-Wl,--version-script=${_IREE_PACKAGE_SOURCE_DIR}/amdhip64.map",
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
            '    "${IREE_BINARY_DIR}/runtime/src/example/device_headers.stamp"',
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

    def test_py_test_preserves_resource_group(self):
        repo_root = Path(__file__).resolve().parents[2]
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="build_tools/bazel_to_cmake",
            repo_root=str(repo_root),
        )

        functions.iree_py_test(
            name="serialized_test",
            srcs=["config_test.py"],
            deps=[],
            resource_group="shared-device",
        )

        self.assertIn("RESOURCE_GROUP\n    shared-device", converter.body)

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
        functions.iree_spirv_asm_module(
            name="generated_data", src="input.spvasm", target_compatible_with=[]
        )
        functions.iree_py_test(
            name="generated_data_test",
            srcs=["test.py"],
            data=[":generated_data"],
        )
        self.assertIn(
            'DATA\n    "${IREE_BINARY_DIR}/pkg/generated_data.spv"', converter.body
        )

    def test_py_test_preserves_configured_generated_arguments_and_data(self):
        converter = SimpleNamespace(body="")
        functions = _PythonBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.iree_spirv_asm_module(
            name="generated_data", src="input.spvasm", target_compatible_with=[]
        )
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
            '"--input={{${IREE_BINARY_DIR}/pkg/generated_data.spv}}")\n'
            "else()\n"
            '  list(APPEND _configured_data_test_platform_args "--no-input")\n'
            "endif()",
            converter.body,
        )
        self.assertIn(
            'if(CMAKE_SYSTEM_NAME STREQUAL "Linux")\n'
            "  list(APPEND _configured_data_test_platform_data "
            '"${IREE_BINARY_DIR}/pkg/generated_data.spv")\n'
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

    def test_generated_files_resolve_output_arguments_and_locations(self):
        converter = bazel_to_cmake_converter.Converter()
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        functions.iree_generated_files(
            name="tables_gen",
            tool=":generator",
            outs=["tables/values.h", "tables/values.c"],
            output_args={"values.h": "--header={path}", "tables/values.c": "--source"},
            use_param_file=True,
        )
        content = converter.convert()
        self.assertIn('"--header=${IREE_BINARY_DIR}/pkg/tables/values.h"', content)
        self.assertIn(
            '"--source"\n    "${IREE_BINARY_DIR}/pkg/tables/values.c"',
            content,
        )
        self.assertEqual(
            functions._cmake_location_paths(":tables_gen"),
            [
                "${IREE_BINARY_DIR}/pkg/tables/values.h",
                "${IREE_BINARY_DIR}/pkg/tables/values.c",
            ],
        )
        self.assertEqual(
            functions._cmake_location_paths(":tables/values.h"),
            ["${IREE_BINARY_DIR}/pkg/tables/values.h"],
        )

        for outputs, arguments in (
            (["values.h"], {"missing.h": "--header"}),
            (["a/values.h", "b/values.h"], {"values.h": "--header"}),
            (["a/values.h"], {"a/values.h": "--header", "values.h": "--duplicate"}),
            (["values.c", "values.h"], {"values.h": "--header"}),
        ):
            with self.subTest(outputs=outputs, arguments=arguments):
                with self.assertRaisesRegex(ValueError, "output_args"):
                    functions.iree_generated_files(
                        name="invalid",
                        tool=":generator",
                        outs=outputs,
                        output_args=arguments,
                    )

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
                SimpleNamespace(
                    PROJECTS=[],
                    REPO_MAP={},
                    CONDITION_MAPPINGS={"//alpha/config:feature": "ENABLE_ALPHA"},
                ),
                str(repo_root / "beta"),
                repo_root=str(repo_root),
            )

        self.assertIn("if(ENABLE_ALPHA)\niree_cc_library(", cmake)

    def test_requirement_policy_excludes_matching_subtree(self):
        broad_requirement = bazel_to_cmake_requirements.build_requirement(
            id="synthetic.broad",
            label="//synthetic:requires_broad",
            enabled_by="//synthetic:enable_broad",
        )
        host_requirement = bazel_to_cmake_requirements.build_requirement(
            id="synthetic.host",
            label="//synthetic:requires_host",
            enabled_by="//synthetic:enable_host",
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
        converter = bazel_to_cmake_converter.Converter()
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

        self.assertIn('if(IREE_ARCH STREQUAL "wasm_32")', converter.convert())
        self.assertIn("add_test(", converter.convert())
        self.assertIn("endif()", converter.convert())

    def test_executable_test_preserves_resource_lock(self):
        for resource_group in (None, "shared-device", "shared-storage"):
            with self.subTest(resource_group=resource_group):
                converter = bazel_to_cmake_converter.Converter()
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
                    self.assertNotIn("RESOURCE_GROUP", converter.convert())
                else:
                    self.assertIn(
                        'RESOURCE_LOCK "' + resource_group + '"', converter.convert()
                    )

    def test_native_test_rejects_multiple_files_in_single_file_locations(self):
        for kind in ("location", "rootpath", "execpath"):
            for field in ("args", "env"):
                with self.subTest(kind=kind, field=field):
                    converter = bazel_to_cmake_converter.Converter()
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
                        converter.convert()

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
        self.assertIn(
            'list(TRANSFORM _GLOB_X_TXT PREPEND "${_IREE_PACKAGE_SOURCE_DIR}/"'
            " OUTPUT_VARIABLE _GLOB_X_TXT_ABSOLUTE)",
            cmake,
        )
        self.assertIn('    "${_GLOB_X_TXT_ABSOLUTE}"', cmake)
        self.assertNotIn("::${_GLOB_X_TXT_ABSOLUTE}", cmake)

    def test_glob_data_preserves_source_paths_and_exclusions(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/pkg",
            repo_root="/repo",
        )
        sources = functions.glob(["testdata/*.txt"], exclude=["testdata/skip.txt"])
        variable = sources[0][2:-1]
        functions.cc_test(name="direct", srcs=sources, data=sources)
        functions.filegroup(name="fixtures", srcs=sources)
        functions.cc_test(name="grouped", data=[":fixtures"])

        self.assertIn(f'  SRCS\n    "${{{variable}}}"', converter.body)
        self.assertEqual(
            converter.body.count(f'  DATA\n    "${{{variable}_ABSOLUTE}}"'), 2
        )
        exclusion = converter.body.index(f"list(REMOVE_ITEM {variable} ")
        conversion = converter.body.index(
            f'list(TRANSFORM {variable} PREPEND "${{_IREE_PACKAGE_SOURCE_DIR}}/"'
            f" OUTPUT_VARIABLE {variable}_ABSOLUTE)"
        )
        self.assertLess(exclusion, conversion)

    def test_native_test_omits_unresolved_external_location_env(self):
        converter = bazel_to_cmake_converter.Converter()
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/runtime/src/iree/fixture",
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

        self.assertNotIn("LLVM_OBJDUMP=", converter.convert())
        self.assertIn('REQUIRED_FILES ""', converter.convert())
        self.assertNotIn("@wasi_sdk", converter.convert())
        self.assertNotIn("TARGET_FILE:pkg_@wasi_sdk", converter.convert())

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
            "-Wl,--version-script=${_IREE_PACKAGE_SOURCE_DIR}/exports.map",
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
