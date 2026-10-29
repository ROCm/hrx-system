# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    build_project,
    configure_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/python_dependencies"


class CMakePythonDependenciesTest(unittest.TestCase):
    def test_shared_sources_stay_fresh_and_preserve_import_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            shutil.copytree(FIXTURE, source)
            configure_project(source, build)
            outputs = [build / f"output_{index}.txt" for index in range(3)]

            def check(expected):
                build_project(build, "iree_generated_compile_inputs")
                for output in outputs:
                    self.assertEqual(output.read_text(), expected)
                test_project(build)

            check("left: initial\n")
            timestamps = [path.stat().st_mtime_ns for path in outputs]
            check("left: initial\n")
            self.assertEqual([path.stat().st_mtime_ns for path in outputs], timestamps)
            value = source / "shared/value.py"
            value.write_text(value.read_text().replace('"initial"', '"changed"'))
            check("left: changed\n")
            configure_project(source, build, "-DIREE_TEST_REVERSE_DEPS=ON")
            check("right: changed\n")

    def test_shared_graph_work_scales_with_declarations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command_counts = []
            for depth in (4, 8):
                trace = root / f"trace-{depth}.jsonl"
                configure_project(
                    FIXTURE,
                    root / f"build-{depth}",
                    f"-DIREE_TEST_DIAMOND_DEPTH={depth}",
                    "--trace-format=json-v1",
                    f"--trace-source={REPO_ROOT}/build_tools/cmake/iree_python.cmake",
                    f"--trace-redirect={trace}",
                )
                with trace.open() as stream:
                    command_counts.append(sum(1 for _ in stream))
            # Doubling graph depth must not approach the 16x growth in paths.
            self.assertLess(command_counts[1], 3 * command_counts[0], command_counts)

    def test_missing_transitive_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = configure_project(
                FIXTURE,
                Path(temporary),
                "-DIREE_TEST_MISSING_DEP=ON",
                expect_failure=True,
            )
            self.assertIn("fixture::missing was not found", " ".join(output.split()))


if __name__ == "__main__":
    unittest.main()
