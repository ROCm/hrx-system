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
    install_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/installed_tests"


class CMakeInstalledTestsTest(unittest.TestCase):
    def test_shared_artifacts_install_at_every_referenced_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, build, prefix = (
                root / name for name in ("source", "build", "install")
            )
            shutil.copytree(FIXTURE, source)
            shutil.copytree(
                FIXTURE.parent / "data_dependencies", root / "data_dependencies"
            )
            shutil.copytree(FIXTURE / "python", root / "python")
            configure_project(source, build)
            build_project(build)
            install_project(build, prefix)
            manifest = (
                (build / "install_manifest_FixtureTests.txt").read_text().splitlines()
            )
            self.assertEqual(len(manifest), len(set(manifest)))
            relocated = root / "relocated install"
            prefix.rename(relocated)
            source.rename(root / "retired source")
            build.rename(root / "retired build")
            (root / "python").rename(root / "retired python")
            test_project(relocated / "share/tests")

    def test_deferred_artifact_must_exist_at_its_recorded_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, (option, diagnostic) in enumerate(
                (
                    ("-DIREE_TEST_DECLARE_TOOL=OFF", "was never created: first::tool"),
                    (
                        "-DIREE_TEST_TOOL_OUTPUT_NAME=renamed",
                        "after a test had already recorded",
                    ),
                )
            ):
                with self.subTest(option=option):
                    output = configure_project(
                        FIXTURE,
                        Path(temporary) / str(index),
                        option,
                        expect_failure=True,
                    )
                    self.assertIn(diagnostic, " ".join(output.split()))


if __name__ == "__main__":
    unittest.main()
