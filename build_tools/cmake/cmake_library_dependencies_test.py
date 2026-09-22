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
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
    run_command,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/library_dependencies"


class CMakeLibraryDependenciesTest(unittest.TestCase):
    def test_generated_inputs_and_transitive_links_stay_fresh(self):
        with tempfile.TemporaryDirectory(prefix="cmake library fixture ") as temporary:
            root = Path(temporary)
            source, build = (
                root / "source.headers with spaces",
                root / "build with spaces",
            )
            shutil.copytree(FIXTURE, source)
            configure_project(source, build)
            leaf, *programs = [
                Path(path)
                for path in (build / f"paths-{CONFIGURATION}.txt")
                .read_text()
                .splitlines()
            ]
            build_project(build, "iree_fixture_library")
            self.assertFalse(
                leaf.exists(), "Building an archive needs no dependent archive"
            )

            def check(expected):
                build_project(build, "iree_fixture_probe", "iree_fixture_unified_probe")
                for program in programs:
                    self.assertEqual(run_command(str(program)), f"{expected}\n")

            check(40)
            timestamps = [path.stat().st_mtime_ns for path in programs]
            check(40)
            self.assertEqual([path.stat().st_mtime_ns for path in programs], timestamps)
            configure_project(source, build)
            (source / "value.txt").write_text("11\n")
            check(48)
            leaf_source = source / "leaf.hpp.c"
            leaf_source.write_text(
                leaf_source.read_text().replace("return 5;", "return 6;")
            )
            check(49)
            for name in ("generated.h", "generated.c"):
                (build / "producer" / name).unlink()
            check(49)


if __name__ == "__main__":
    unittest.main()
