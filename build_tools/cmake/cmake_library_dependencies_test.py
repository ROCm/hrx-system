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
        for always_link in (False, True):
            with self.subTest(always_link=always_link):
                self.check_library_composition(always_link)

    def check_library_composition(self, always_link):
        with tempfile.TemporaryDirectory(prefix="cmake library fixture ") as temporary:
            root = Path(temporary)
            source, build = (
                root / "source.headers with spaces",
                root / "build with spaces",
            )
            shutil.copytree(FIXTURE, source)
            configure_project(
                source, build, f"-DFIXTURE_ALWAYS_LINK={'ON' if always_link else 'OFF'}"
            )
            library_target, leaf_path, *program_paths = (
                (build / f"paths-{CONFIGURATION}.txt").read_text().splitlines()
            )
            leaf = Path(leaf_path)
            programs = [Path(path) for path in program_paths]
            build_project(build, library_target)
            self.assertFalse(
                leaf.exists(), "Building an archive needs no dependent archive"
            )

            def check(expected):
                build_project(
                    build,
                    "iree_fixture_probe",
                    "iree_fixture_unified_probe",
                    "iree_fixture_collected_probe",
                    "iree_fixture_source_less_test_with_a_long_target_name_matching_real_cts_binaries",
                )
                for program in programs:
                    self.assertEqual(run_command(str(program)), f"{expected}\n")

            check(42)
            timestamps = [path.stat().st_mtime_ns for path in programs]
            check(42)
            self.assertEqual([path.stat().st_mtime_ns for path in programs], timestamps)
            configure_project(source, build)
            (source / "value.txt").write_text("11\n")
            check(50)
            leaf_source = source / "leaf.hpp.c"
            leaf_source.write_text(
                leaf_source.read_text().replace("return 5;", "return 6;")
            )
            check(51)
            for name in ("generated.h", "generated.c"):
                (build / "producer" / name).unlink()
            check(51)
            external_source = source / "external.c"
            external_source.write_text(
                external_source.read_text().replace("return 2;", "return 3;")
            )
            check(52)
            (build / f"external-{CONFIGURATION}.o").unlink()
            check(52)


if __name__ == "__main__":
    unittest.main()
