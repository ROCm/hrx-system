# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    build_project,
    configure_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/test_metadata"


class CMakeTestMetadataTest(unittest.TestCase):
    def test_escaped_names_keep_ordered_unique_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = root / "extension.cmake"
            name = 'quoted "name" \\ path\nwith\ttab'
            extension.write_text(
                f"set(_NAME [=[{name}]=])\n"
                'add_test(NAME "${_NAME}" COMMAND "${CMAKE_COMMAND}" -E true)\n'
                'iree_register_test_build_targets("${_NAME}"\n'
                "  TARGETS tool_backed_root host_root tool_backed_root)\n"
                'add_test(NAME OFF COMMAND "${CMAKE_COMMAND}" -E true)\n'
                "iree_register_test_build_targets(OFF)\n"
            )
            build = root / "build"
            configure_project(
                FIXTURE, build, f"-DIREE_TEST_METADATA_EXTENSION_FILE={extension}"
            )
            catalog = json.loads((build / "iree_ctest_build_targets.json").read_text())
            self.assertEqual(catalog["tests"][name], ["tool_backed_root", "host_root"])
            self.assertEqual(catalog["tests"]["OFF"], [])
            tests = json.loads(test_project(build, "--show-only=json-v1"))["tests"]
            self.assertIn(name, {test["name"] for test in tests})

    def test_dependency_suppressions_merge_and_track_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("leak:first\n")
            second.write_text("leak:second")
            extension = root / "sanitizers.cmake"
            extension.write_text(f"""
include("${{IREE_REPO_ROOT}}/build_tools/sanitizer/iree_sanitizer_suppressions.cmake")
set(IREE_SANITIZER_SUPPRESSION_FIRST_LSAN "{first.as_posix()}")
set(IREE_SANITIZER_SUPPRESSION_SECOND_LSAN "{second.as_posix()}")
set(IREE_SANITIZER_SUPPRESSION_FIRST_TSAN "{first.as_posix()}")
set(IREE_SANITIZER_SUPPRESSION_SECOND_TSAN "{second.as_posix()}")
iree_target_sanitizer_suppressions(host_root DEPS runtime_alias second_runtime)
add_library(first_runtime INTERFACE)
add_library(runtime_alias ALIAS first_runtime)
iree_target_sanitizer_suppressions(first_runtime SUPPRESSIONS lsan first tsan first)
add_library(second_runtime INTERFACE)
iree_target_sanitizer_suppressions(second_runtime DEPS first_runtime SUPPRESSIONS lsan second tsan second)
iree_target_sanitizer_suppressions(tool_backed_root)
iree_finalize_sanitizer_suppressions()
set_property(TEST host PROPERTY ENVIRONMENT "$<TARGET_PROPERTY:host_root,IREE_SANITIZER_ENVIRONMENT>")
set_property(TEST tool-backed PROPERTY ENVIRONMENT "$<TARGET_PROPERTY:tool_backed_root,IREE_SANITIZER_ENVIRONMENT>")
""")
            build_dir = root / "build"
            configure_project(
                FIXTURE, build_dir, f"-DIREE_TEST_METADATA_EXTENSION_FILE={extension}"
            )
            test_model = json.loads(test_project(build_dir, "--show-only=json-v1"))
            tests = {test["name"]: test for test in test_model["tests"]}
            environment = next(
                prop["value"]
                for prop in tests["host"]["properties"]
                if prop["name"] == "ENVIRONMENT"
            )
            self.assertEqual(len(environment), 2)
            lsan = next(
                value for value in environment if value.startswith("LSAN_OPTIONS=")
            )
            merged = Path(
                lsan.removeprefix('LSAN_OPTIONS=suppressions="').removesuffix(
                    '":allow_addr2line=1'
                )
            )
            self.assertEqual(merged.read_text().split(), ["leak:first", "leak:second"])
            tsan = next(
                value for value in environment if value.startswith("TSAN_OPTIONS=")
            )
            tsan_merged = Path(
                tsan.removeprefix('TSAN_OPTIONS=suppressions="').removesuffix('"')
            )
            self.assertNotEqual(merged, tsan_merged)
            self.assertEqual(tsan_merged.read_text(), merged.read_text())
            self.assertFalse(
                any(
                    prop["name"] == "ENVIRONMENT" and prop["value"]
                    for prop in tests["tool-backed"]["properties"]
                )
            )
            # A build must regenerate the merged policy after an input edit.
            second.write_text("leak:changed")
            build_project(build_dir, "host_root")
            self.assertEqual(merged.read_text().split(), ["leak:first", "leak:changed"])

    def test_rejects_invalid_build_metadata(self):
        cases = (
            ("", "missing IREE_BUILD_TARGETS metadata: OFF"),
            (
                "iree_register_test_build_targets(OFF TARGETS missing)",
                "missing IREE_BUILD_TARGETS target: missing",
            ),
            (
                "add_library(actual INTERFACE)\nadd_library(invalid ALIAS actual)\niree_register_test_build_targets(OFF TARGETS invalid)",
                "non-buildable alias",
            ),
            (
                "add_executable(invalid IMPORTED)\niree_register_test_build_targets(OFF TARGETS invalid)",
                "non-buildable imported",
            ),
            (
                "add_library(invalid INTERFACE)\niree_register_test_build_targets(OFF TARGETS invalid)",
                "non-buildable interface library",
            ),
            (
                "iree_register_test_build_targets(OFF)\niree_register_test_build_targets(OFF)",
                "duplicate IREE_BUILD_TARGETS metadata: OFF",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = root / "extension.cmake"
            for declarations, diagnostic in cases:
                with self.subTest(diagnostic=diagnostic):
                    extension.write_text(
                        'add_test(NAME OFF COMMAND "${CMAKE_COMMAND}" -E true)\n'
                        + declarations
                    )
                    output = configure_project(
                        FIXTURE,
                        root / "build",
                        f"-DIREE_TEST_METADATA_EXTENSION_FILE={extension}",
                        expect_failure=True,
                    )
                    self.assertIn(diagnostic, " ".join(output.split()))


if __name__ == "__main__":
    unittest.main()
