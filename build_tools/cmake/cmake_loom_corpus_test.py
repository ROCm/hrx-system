# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import bazel_to_cmake_config
import bazel_to_cmake_requirements
import bazel_to_cmake_targets

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    configure_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/loom_corpus"


class CMakeLoomCorpusTest(unittest.TestCase):
    def test_profile_requirements_control_registered_execution_tests(self):
        project = bazel_to_cmake_config.include_project(
            str(REPO_ROOT / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        with tempfile.TemporaryDirectory(prefix="cmake loom corpus ") as temporary:
            source, build = Path(temporary) / "source", Path(temporary) / "build"
            shutil.copytree(FIXTURE, source)
            source_converter = SimpleNamespace(body="")
            source_functions = project.build_file_functions(
                converter=source_converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(source),
                repo_root=str(REPO_ROOT),
            )
            target_converter = SimpleNamespace(body="")
            target_functions = project.build_file_functions(
                converter=target_converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(source),
                repo_root=str(REPO_ROOT),
            )
            manifest = source_functions.loom_corpus_manifest(
                "examples", "//fixture", ["identity.loom"]
            )
            source_functions.loom_corpus_sources("sources", manifest)
            profiles = []
            for name, expression in (
                ("unconditional", None),
                ("single", "FEATURE_ENABLED"),
                (
                    "compound",
                    "FEATURE_ENABLED AND (FEATURE_DISABLED OR FEATURE_ENABLED)",
                ),
                ("disabled", "FEATURE_DISABLED"),
                (
                    "compound_disabled",
                    "FEATURE_ENABLED AND (FEATURE_DISABLED OR FEATURE_MISSING)",
                ),
                ("literal_false", "FALSE"),
            ):
                requirements = []
                if expression is not None:
                    requirements.append(
                        bazel_to_cmake_requirements.build_requirement(
                            name, "//config:" + name, name, expression
                        )
                    )
                target_functions.loom_target_profile(
                    name,
                    "amdgpu",
                    "gfx90a",
                    target_compatible_with=[]
                    if expression is None
                    else [
                        bazel_to_cmake_requirements.CMakeRequirementCondition(
                            expression
                        )
                    ],
                )
                profiles.append(
                    target_functions.loom_execution_profile(
                        name,
                        "amdgpu",
                        "hardware",
                        "amdgpu",
                        build_requirements=requirements,
                    )
                )
            target_functions.loom_corpus_test(
                "execution",
                target_functions.loom_corpus_catalog([manifest]),
                execution_profiles=profiles,
            )
            converted_source = source / "converted_source.cmake"
            converted_source.write_text(source_converter.body)
            converted_target = source / "converted_target.cmake"
            converted_target.write_text(target_converter.body)
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    configure_project(
                        source,
                        build,
                        f"-DLOOM_CONVERTED_SOURCE_RULES={converted_source}",
                        f"-DLOOM_CONVERTED_TARGET_RULES={converted_target}",
                        f"-DFEATURE_ENABLED={'ON' if enabled else 'OFF'}",
                        "-DFEATURE_DISABLED=OFF",
                    )
                    tests = json.loads(test_project(build, "--show-only=json-v1"))[
                        "tests"
                    ]
                    expected_profiles = {"unconditional"}
                    if enabled:
                        expected_profiles.update(("single", "compound"))
                    self.assertEqual(
                        set(
                            (build / "available_profiles.txt").read_text().splitlines()
                        ),
                        expected_profiles,
                    )
                    self.assertEqual(
                        {test["name"] for test in tests},
                        {
                            f"fixture/target/examples_identity_test_execute_{name}_test"
                            for name in expected_profiles
                        },
                    )
                    self.assertEqual(
                        (build / "module_targets.txt").read_text().split(";"),
                        [
                            "fixture_source_examples_identity",
                            "fixture_source_examples_identity_library",
                        ],
                    )
                    for test in tests:
                        labels = next(
                            property["value"]
                            for property in test["properties"]
                            if property["name"] == "LABELS"
                        )
                        self.assertIn("loom-corpus", labels)


if __name__ == "__main__":
    unittest.main()
