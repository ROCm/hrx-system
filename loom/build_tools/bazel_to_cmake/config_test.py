# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for Loom-owned Bazel-to-CMake configuration and rule projections."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "build_tools/bazel_to_cmake"))

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_targets
from loom_binary import LoomBinaryBuildFileFunctions


class _LoomBinaryBuildFileFunctions(
    LoomBinaryBuildFileFunctions, bazel_to_cmake_converter.BuildFileFunctions
):
    pass


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = bazel_to_cmake_config.include_project(
            REPO_ROOT / ".bazel_to_cmake.cfg.py", "loom/.bazel_to_cmake.cfg.py"
        )

    def test_loom_conditions_reach_non_loom_build_files(self):
        repo_cfg = SimpleNamespace(PROJECTS=[self.project], REPO_MAP={"@hrx": ""})
        for label, condition in (
            ("//loom/config/import:cxx", "LOOM_IMPORT_CXX"),
            (
                "//loom/config/target:xdna_artifacts",
                "LOOM_BUILD AND LOOM_TARGET_ARCH_XDNA AND LOOM_EMIT_XDNA",
            ),
            (
                "//loom/config/target/amdgpu:descriptor_set_gfx11_generic_core",
                "LOOM_TARGET_AMDGPU_DESCRIPTOR_SET_GFX11_GENERIC_CORE",
            ),
        ):
            with self.subTest(label=label):
                output = bazel_to_cmake_converter.convert_build_file(
                    f"""
cc_library(
    name = "consumer",
    srcs = ["consumer.c"],
    target_compatible_with = select({{
        "{label}": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }}),
)
""",
                    repo_cfg,
                    "/repo/consumer",
                    repo_root="/repo",
                )
                self.assertIn(f"if({condition})", output)

    def test_loom_image_provider_selection_survives_conversion(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/loom/src/loom/serve/media",
            repo_root="/repo",
            condition_mappings=self.project.condition_mappings,
        )
        functions.cc_library(
            name="image_codecs",
            srcs=["image_codecs.c"],
            deps=functions.select(
                {
                    "//loom/config/serve:png": ["//third_party:libpng"],
                    "//conditions:default": [],
                }
            ),
        )
        self.assertIn("if(LOOM_SERVE_IMAGE_CODEC_PNG)", converter.body)
        self.assertIn("iree::third_party::libpng", converter.body)

    def test_loom_audio_decoder_selection_survives_conversion(self):
        converter = SimpleNamespace(body="")
        functions = bazel_to_cmake_converter.BuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/loom/src/loom/serve/media",
            repo_root="/repo",
            condition_mappings=self.project.condition_mappings,
        )
        functions.cc_library(
            name="audio_decoder",
            srcs=functions.select(
                {
                    "//loom/config/serve:audio_decoding": ["audio_decoder.c"],
                    "//conditions:default": ["audio_decoder_disabled.c"],
                }
            ),
            deps=functions.select(
                {
                    "//loom/config/serve:audio_decoding": ["//third_party:miniaudio"],
                    "//conditions:default": [],
                }
            ),
        )
        self.assertIn("if(LOOM_SERVE_AUDIO_DECODING)", converter.body)
        self.assertIn("iree::third_party::miniaudio", converter.body)
        self.assertIn('"audio_decoder.c"', converter.body)
        self.assertIn('"audio_decoder_disabled.c"', converter.body)
        for codec in ("wav", "mp3", "flac"):
            converter.body = ""
            functions.cc_library(
                name="selection",
                defines=functions.select(
                    {
                        "//loom/config/serve:" + codec: ["ENABLED=1"],
                        "//conditions:default": ["ENABLED=0"],
                    }
                ),
            )
            self.assertIn(
                "if(LOOM_SERVE_AUDIO_CODEC_" + codec.upper() + ")", converter.body
            )

    def test_loom_kernel_projects_profile_roots_configs_and_library_edges(self):
        converter = SimpleNamespace(body="")
        functions = _LoomBinaryBuildFileFunctions(
            converter=converter,
            targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
            build_dir="/repo/programs",
            repo_root="/repo",
            condition_mappings=self.project.condition_mappings,
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
            condition_mappings=self.project.condition_mappings,
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
            PROJECTS=[self.project],
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

    def test_shared_bindings_preserve_configuration_and_helper_ownership(self):
        repo_cfg = SimpleNamespace(PROJECTS=[self.project], REPO_MAP={"@hrx": ""})
        output = bazel_to_cmake_converter.convert_build_file(
            """
loomc_shared_libraries(
    name="loomc_shared",
    components=[":base"] + select({
        "//loom/config/target:amdgpu_artifacts": ["//loom/binding/c/target/amdgpu"],
        "//conditions:default": [],
    }),
    runtime_deps=["//runtime/src/iree/base"],
)
cc_library(name="helper", tags=["LINKABLE_MORE_THAN_ONCE"])
loom_generated_cc_library(name="catalog", generator=":generate", tags=["LINKABLE_MORE_THAN_ONCE"])
""",
            repo_cfg,
            str(REPO_ROOT / "loom/binding/c"),
            repo_root=str(REPO_ROOT),
        )
        self.assertIn(
            "if(LOOM_BUILD AND LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU)", output
        )
        self.assertIn("loom::binding::c::target::amdgpu", output)
        self.assertIn("  RUNTIME_DEPS\n    iree::base", output)
        self.assertEqual(output.count("  LINKABLE_MORE_THAN_ONCE\n"), 2)

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
            build_dir="/repo/consumer",
            repo_root="/repo",
            condition_mappings=self.project.condition_mappings,
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


if __name__ == "__main__":
    unittest.main()
