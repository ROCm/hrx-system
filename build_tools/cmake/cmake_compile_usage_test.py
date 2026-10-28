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
    CMAKE_COMMAND,
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
    run_command,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/compile_usage"


class CMakeCompileUsageTest(unittest.TestCase):
    def test_native_requirements_and_relocated_consumer(self):
        with tempfile.TemporaryDirectory(prefix="cmake compile usage ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            shutil.copytree(FIXTURE, source)
            caller_hook = source / "caller.cmake"
            caller_hook.write_text(
                'set_property(DIRECTORY PROPERTY CALLER_PROJECT "${PROJECT_NAME}")\n'
            )

            def check():
                build_project(build, "probe", "unified_probe")
                for name in ("probe", "unified_probe"):
                    executable = Path(
                        (build / f"{name}-{CONFIGURATION}.txt").read_text()
                    )
                    self.assertEqual(run_command(str(executable)), "42\n")

            # Compare evaluated compiler commands, not serialized target lists.
            # Reuse the native build for boundary variants and incremental work.
            version = tuple(
                map(
                    int,
                    run_command(CMAKE_COMMAND, "--version").split()[2].split(".")[:2],
                )
            )
            variants = ["", "SCOPED", "NO_PYTHON"]
            if version >= (3, 30):
                variants.extend(("CUSTOM", "SCOPED_CUSTOM"))
            for variant in variants:
                arguments = [
                    f"-DCMAKE_PROJECT_INCLUDE={caller_hook.as_posix()}",
                    "-DFIXTURE_REQUIRE_CALLER=ON",
                    f"-DFIXTURE_SCOPED={'ON' if 'SCOPED' in variant else 'OFF'}",
                    f"-DFIXTURE_CUSTOM={'ON' if 'CUSTOM' in variant else 'OFF'}",
                ]
                if variant == "NO_PYTHON":
                    arguments.append("-DPython3_EXECUTABLE=")
                configure_project(source, build, "-DFIXTURE_RESOLVE=OFF", *arguments)
                commands = build / "compile_commands.json"
                baseline = commands.read_bytes() if commands.exists() else None
                configure_project(source, build, "-DFIXTURE_RESOLVE=ON", *arguments)
                if baseline is not None:
                    self.assertEqual(commands.read_bytes(), baseline)
                check()

            configure_project(
                source, build, "-DFIXTURE_SCOPED=OFF", "-DFIXTURE_CUSTOM=OFF"
            )
            check()
            if version >= (3, 30):
                # Ensure the lifecycle checks actually traversed resolved edges.
                self.assertIn(
                    "$<LINK_ONLY:iree::fixture::leaf>",
                    (build / ".iree/compile_usage.cmake").read_text(),
                )
            prefix = root / "stage"
            run_command(
                CMAKE_COMMAND,
                "--install",
                str(build),
                "--config",
                CONFIGURATION,
                "--prefix",
                str(prefix),
                "--component",
                "FixtureLibraries",
            )
            relocated = root / "relocated"
            prefix.rename(relocated)
            consumer, consumer_build = root / "consumer", root / "consumer build"
            consumer.mkdir()
            shutil.copyfile(source / "main.c", consumer / "main.c")
            (
                consumer / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(usage_consumer C)
include("${FIXTURE_PREFIX}/lib/cmake/fixture/IREETargets-fixture.cmake")
add_executable(probe main.c)
target_compile_definitions(probe PRIVATE INSTALLED_CONSUMER=1)
target_link_libraries(probe PRIVATE iree_fixture_unified)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-$<CONFIG>.txt" CONTENT "$<TARGET_FILE:probe>")
""")
            source.rename(root / "source hidden")
            build.rename(root / "build hidden")
            configure_project(
                consumer, consumer_build, f"-DFIXTURE_PREFIX={relocated.as_posix()}"
            )
            build_project(consumer_build, "probe")
            executable = Path(
                (consumer_build / f"probe-{CONFIGURATION}.txt").read_text()
            )
            self.assertEqual(run_command(str(executable)), "42\n")

    def test_enclosing_project_can_change_requirements(self):
        with tempfile.TemporaryDirectory(prefix="cmake usage subproject ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            shutil.copytree(FIXTURE, source / "child")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(usage_parent C CXX)
set(FIXTURE_RESOLVE ON)
add_subdirectory(child)
target_compile_definitions(iree_fixture_leaf INTERFACE PARENT_VALUE=61)
target_compile_definitions(probe PRIVATE REQUIRE_PARENT=1)
""")
            configure_project(source, build)
            build_project(build, "probe")
            executable = Path((build / f"child/probe-{CONFIGURATION}.txt").read_text())
            self.assertEqual(run_command(str(executable)), "42\n")


if __name__ == "__main__":
    unittest.main()
