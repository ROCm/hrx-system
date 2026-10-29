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

FIXTURES = REPO_ROOT / "build_tools/cmake/testdata"


class CMakeDataDependenciesTest(unittest.TestCase):
    def test_shared_staged_data_stays_fresh(self):
        with tempfile.TemporaryDirectory(prefix="cmake staged data ") as temporary:
            root = Path(temporary)
            source, build = root / "source with spaces", root / "build with spaces"
            shutil.copytree(FIXTURES / "data_staging", source)
            configure_project(source, build)
            build_project(build)
            paths = [build / name for name in ("shared.txt", "a/b.txt", "a_b.txt")]
            timestamps = [path.stat().st_mtime_ns for path in paths]
            build_project(build)
            self.assertEqual([path.stat().st_mtime_ns for path in paths], timestamps)

            (source / "shared.txt").write_text("updated shared payload\n")
            build_project(build)
            self.assertEqual(paths[0].read_text(), "updated shared payload\n")
            self.assertEqual(
                [path.stat().st_mtime_ns for path in paths[1:]], timestamps[1:]
            )
            # A newer input with identical contents must not leave its copy stale.
            (source / "shared.txt").touch()
            build_project(build)
            self.assertGreaterEqual(
                paths[0].stat().st_mtime_ns, (source / "shared.txt").stat().st_mtime_ns
            )
            timestamps = [path.stat().st_mtime_ns for path in paths]
            build_project(build)
            self.assertEqual([path.stat().st_mtime_ns for path in paths], timestamps)
            paths[1].unlink()
            build_project(build)
            for path in paths:
                self.assertEqual(
                    path.read_bytes(), (source / path.relative_to(build)).read_bytes()
                )
            test_project(build)

    def test_file_and_target_data_build_their_producers(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, (target, reference, output_directory) in enumerate(
                (
                    ("data_consumer-NOTFOUND", "fixture::tool", "build"),
                    ("data_consumer-NOTFOUND", "fixture::tool", "source"),
                    ("later_data_consumer", "fixture::tool", "source"),
                    ("tool_path_consumer", "fixture::tool", "build"),
                    ("tool_path_consumer", "data_tool-NOTFOUND", "build"),
                )
            ):
                with self.subTest(
                    target=target,
                    reference=reference,
                    output_directory=output_directory,
                ):
                    root = Path(temporary) / str(index)
                    source, build = root / "source", root / "build"
                    shutil.copytree(FIXTURES / "data_dependencies", source)
                    generated = root / output_directory / "generated.txt"
                    configure_project(
                        source,
                        build,
                        f"-DIREE_TEST_TOOL_REFERENCE={reference}",
                        f"-DIREE_TEST_GENERATED_DATA={generated}",
                    )
                    build_project(build, target)
                    self.assertTrue((build / "tool-built.marker").is_file())
                    if target != "tool_path_consumer":
                        for path in (build / "fixture.txt", generated):
                            self.assertEqual(path.read_text(), "fixture data\n")
                        # Existing outputs retain their producer across configure.
                        configure_project(source, build)
                        (source / "fixture.txt").write_text("updated fixture\n")
                        build_project(build, target)
                        for path in (build / "fixture.txt", generated):
                            self.assertEqual(path.read_text(), "updated fixture\n")
                        generated.unlink()
                        build_project(build, target)
                        self.assertEqual(generated.read_text(), "updated fixture\n")

    def test_invalid_dependencies_report_the_owner(self):
        for option, message in (
            (
                "-DIREE_TEST_DECLARE_TOOL=OFF",
                "data_consumer-NOTFOUND depends on missing target: fixture::tool",
            ),
            (
                "-DIREE_TEST_DUPLICATE_PRODUCER=ON",
                "has multiple producers: generated_data-NOTFOUND and duplicate_data",
            ),
        ):
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = configure_project(
                    FIXTURES / "data_dependencies",
                    Path(temporary),
                    option,
                    expect_failure=True,
                )
                self.assertIn(message, " ".join(output.split()))


if __name__ == "__main__":
    unittest.main()
