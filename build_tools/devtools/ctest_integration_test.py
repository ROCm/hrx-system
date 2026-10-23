# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.cmake.test_environment import (
    CMAKE_COMMAND,
    CONFIGURATION,
    CTEST_COMMAND,
    REPO_ROOT,
    configure_project,
    test_project,
)
from build_tools.devtools import ctest as ctest_dev

FIXTURES = REPO_ROOT / "build_tools/cmake/testdata"


class CTestIntegrationTest(unittest.TestCase):
    def test_shared_closure_is_fresh_and_visited_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, build = root / "source with spaces", root / "build with spaces"
            shutil.copytree(FIXTURES / "selected_build", source)
            configure_project(source, build)
            step = ctest_dev.CTestBuildAndRunStep(
                cmake=CMAKE_COMMAND,
                ctest=CTEST_COMMAND,
                build_dir=build,
                arguments=["-R", "^(left|right)$", "-C", CONFIGURATION],
                cwd=REPO_ROOT,
                env={**os.environ, "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
            )
            make_temporary = root / "make $files = # '"
            make_temporary.mkdir()
            unselected_temporary_directory = (
                build / "test_tmpdir" / "unselected_test_tmpdir"
            )
            unselected_temporary_directory.mkdir(parents=True)
            unselected_sentinel = unselected_temporary_directory / "preserved"
            unselected_sentinel.write_text("unselected\n")

            def check(build_count):
                # Exercise the real Make/shell boundary, including cleanup.
                with mock.patch.object(tempfile, "tempdir", str(make_temporary)):
                    self.assertEqual(step.run(), 0)
                self.assertEqual(list(make_temporary.iterdir()), [])
                self.assertEqual(unselected_sentinel.read_text(), "unselected\n")
                generator = os.environ["IREE_TEST_CMAKE_GENERATOR"]
                if generator == "Unix Makefiles" or generator.startswith("Ninja"):
                    self.assertEqual(
                        (build / "visits.txt").read_text().splitlines(),
                        ["visit"] * build_count,
                    )
                for name in ("left", "right"):
                    self.assertEqual(
                        (build / f"{name}.ran").read_text().splitlines(),
                        ["run"] * build_count,
                    )

            check(1)
            with mock.patch.dict(
                os.environ,
                {ctest_dev.SKIP_GLOBAL_TMPDIR_CLEANUP_ENV: "1"},
            ):
                tests = json.loads(
                    test_project(build, "-R", "^left$", "--show-only=json-v1")
                )["tests"]
            self.assertEqual(unselected_sentinel.read_text(), "unselected\n")
            executable = Path(tests[0]["command"][0])
            timestamp = executable.stat().st_mtime_ns
            check(2)
            self.assertEqual(executable.stat().st_mtime_ns, timestamp)
            # Runtime expected values make stale compilation fail the actual test.
            (source / "value.txt").write_text("5\n")
            (source / "expected.txt").write_text("6\n")
            check(3)
            shared = source / "shared.c"
            shared.write_text(shared.read_text().replace("+ 1", "+ 2"))
            (source / "expected.txt").write_text("7\n")
            check(4)
            executable.unlink()
            check(5)

            # A failed build must not run the previously built executables.
            shared.write_text("#error selected build must stop\n")
            with mock.patch.object(tempfile, "tempdir", str(make_temporary)):
                self.assertNotEqual(step.run(), 0)
            self.assertEqual(list(make_temporary.iterdir()), [])
            for name in ("left", "right"):
                self.assertEqual(
                    (build / f"{name}.ran").read_text().splitlines(), ["run"] * 5
                )

    def test_selection_builds_only_its_refreshed_closure_and_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build, extension = root / "build", root / "extension.cmake"
            extension.write_text("")
            configure_project(
                FIXTURES / "test_metadata",
                build,
                f"-DIREE_TEST_METADATA_EXTENSION_FILE={extension}",
            )

            def select(pattern):
                step = ctest_dev.CTestBuildAndRunStep(
                    cmake=CMAKE_COMMAND,
                    ctest=CTEST_COMMAND,
                    build_dir=build,
                    arguments=["-R", pattern, "-C", CONFIGURATION],
                    cwd=REPO_ROOT,
                )
                self.assertEqual(step.run(), 0)

            select("^source-only$")
            self.assertFalse((build / "host.built").exists())
            select("^host$")
            self.assertTrue((build / "host.built").is_file())
            self.assertFalse((build / "benchmark.built").exists())
            select("^fixture-required$")
            for name in ("fixture-setup", "fixture-required", "fixture-cleanup"):
                self.assertTrue((build / f"{name}.built").is_file())
            select("^tool-backed$")
            self.assertTrue((build / "tool.built").is_file())
            extension.write_text("add_metadata_test(host-late host_late_root)\n")
            select("^host")
            self.assertTrue((build / "host-late.built").is_file())


if __name__ == "__main__":
    unittest.main()
