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
from conditions import all_of, any_of

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    build_project,
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
            temporary_root = Path(temporary)
            source, build = temporary_root / "fixture", temporary_root / "build"
            shutil.copytree(FIXTURE, source)
            shutil.copy2(
                REPO_ROOT / "build_tools/cmake/testdata/loom_binary/tool.c",
                source / "tool.c",
            )
            source_converter = SimpleNamespace(body="")
            source_functions = project.build_file_functions(
                converter=source_converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(REPO_ROOT / "fixture"),
                repo_root=str(REPO_ROOT),
            )
            nested_source_converter = SimpleNamespace(body="")
            nested_source_functions = project.build_file_functions(
                converter=nested_source_converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(REPO_ROOT / "fixture/nested"),
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
                "examples",
                "//fixture",
                ["nested/fixture.loom", "identity.loom"],
                configs={
                    "identity.loom": {
                        "identity.empty": "",
                        "identity.value": "42",
                    },
                    "nested/fixture.loom": {"nested.value": "7"},
                },
            )
            source_functions.loom_corpus_sources("sources", manifest)
            nested_source_functions.loom_corpus_sources("sources", manifest)
            profiles = []
            for name, predicate in (
                ("unconditional", None),
                ("single", "FEATURE_ENABLED"),
                (
                    "compound",
                    all_of(
                        "FEATURE_ENABLED", any_of("FEATURE_DISABLED", "FEATURE_ENABLED")
                    ),
                ),
                ("disabled", "FEATURE_DISABLED"),
                (
                    "compound_disabled",
                    all_of(
                        "FEATURE_ENABLED", any_of("FEATURE_DISABLED", "FEATURE_MISSING")
                    ),
                ),
                ("literal_false", "FALSE"),
            ):
                requirements = []
                if predicate is not None:
                    target_functions._condition_mappings["//config:" + name] = predicate
                    requirements.append(
                        bazel_to_cmake_requirements.build_requirement(
                            name, "//config:" + name, "//config:" + name
                        )
                    )
                target_functions.loom_target_profile(
                    name,
                    "amdgpu",
                    "gfx90a",
                    target_compatible_with=[]
                    if predicate is None
                    else ["//config:" + name],
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
            converted_nested_source = source / "converted_nested_source.cmake"
            converted_nested_source.write_text(nested_source_converter.body)
            converted_target = source / "converted_target.cmake"
            converted_target.write_text(target_converter.body)
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    configure_project(
                        source,
                        build,
                        f"-DLOOM_CONVERTED_SOURCE_RULES={converted_source}",
                        f"-DLOOM_CONVERTED_NESTED_SOURCE_RULES={converted_nested_source}",
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
                        }
                        | {
                            f"fixture/target/examples_nested_fixture_test_execute_{name}_test"
                            for name in expected_profiles
                        },
                    )
                    self.assertEqual(
                        (build / "module_targets.txt").read_text().split(";"),
                        [
                            "fixture_source_examples_identity",
                            "fixture_source_examples_identity_library",
                            "fixture_source_nested_nested_examples_nested_fixture",
                            "fixture_source_nested_nested_examples_nested_fixture_library",
                            "fixture_target_examples_identity_subjects",
                            "fixture_target_examples_nested_fixture_subjects",
                        ],
                    )
                    for test in tests:
                        expected_config = (
                            "--config=identity.value=42"
                            if "examples_identity" in test["name"]
                            else "--config=nested.value=7"
                        )
                        self.assertIn(expected_config, test["command"])
                        if "examples_identity" in test["name"]:
                            self.assertIn("--config=identity.empty=", test["command"])
                        labels = next(
                            property["value"]
                            for property in test["properties"]
                            if property["name"] == "LABELS"
                        )
                        self.assertIn("loom-corpus", labels)

            artifact_line, report_line = (
                (build / "artifacts.txt").read_text().splitlines()
            )
            artifacts = [Path(path) for path in artifact_line.split(";")]
            reports = [Path(path) for path in report_line.split(";")]
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(len(reports), 2)
            self.assertIn("examples_nested_fixture", str(artifacts[0]))
            self.assertIn("examples_identity", str(artifacts[1]))
            artifact = next(
                path for path in artifacts if "examples_identity" in str(path)
            )
            nested_artifact = next(
                path for path in artifacts if "examples_nested_fixture" in str(path)
            )
            report = next(path for path in reports if "examples_identity" in str(path))
            build_project(build, "fixture_target_compile")
            self.assertIn("func.def @identity", artifact.read_text())
            self.assertIn("func.def @nested", nested_artifact.read_text())
            timestamp = artifact.stat().st_mtime_ns
            build_project(build, "fixture_target_compile")
            self.assertEqual(artifact.stat().st_mtime_ns, timestamp)

            subject = source / "identity.loom"
            subject.write_text(subject.read_text() + "// changed subject\n")
            build_project(build, "fixture_target_compile")
            self.assertIn("changed subject", artifact.read_text())
            tool_source = source / "tool.c"
            tool_source.write_text(
                tool_source.read_text().replace(
                    "int failed = 0;",
                    'int failed = fputs("changed tool\\n", output) < 0;',
                )
            )
            build_project(build, "fixture_target_compile")
            self.assertIn("changed tool\n", artifact.read_text())
            report.unlink()
            build_project(build, "fixture_target_compile")
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
