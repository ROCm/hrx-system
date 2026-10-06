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

import bazel_to_cmake_converter
import bazel_to_cmake_targets

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    build_project,
    configure_project,
    install_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/test_arguments"


class FixtureBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _should_emit_python_target(self):
        return True


class CMakeTestArgumentsTest(unittest.TestCase):
    def test_filegroups_keep_package_identity_for_data_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.26)\n"
                "project(filegroup_packages NONE)\n"
                'set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")\n'
                "set(IREE_PACKAGE_ROOT_PREFIX fixture)\n"
                'include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")\n'
                "add_custom_target(consumer)\n"
                "iree_add_data_dependencies(NAME consumer\n"
                "  DATA fixture::first::inputs fixture::second::inputs)\n"
                "add_subdirectory(first)\n"
                "add_subdirectory(second)\n"
                "iree_finalize_target_dependencies()\n"
            )
            for name in ("first", "second"):
                package = source / name
                package.mkdir()
                (package / "input.loom").write_text(name)
                converter = SimpleNamespace(body="")
                functions = FixtureBuildFileFunctions(
                    converter=converter,
                    targets=bazel_to_cmake_targets.TargetConverter(
                        repo_map={"@hrx": ""}
                    ),
                    build_dir=str(package),
                    repo_root=str(source),
                )
                functions.filegroup(name="inputs", srcs=["input.loom"])
                (package / "CMakeLists.txt").write_text(converter.body)
            configure_project(source, build)
            build_project(build, "consumer")
            stamps = [build / name / "inputs.stamp" for name in ("first", "second")]
            timestamps = [stamp.stat().st_mtime_ns for stamp in stamps]
            (source / "first/input.loom").write_text("changed")
            build_project(build, "consumer")
            self.assertGreater(stamps[0].stat().st_mtime_ns, timestamps[0])
            self.assertEqual(stamps[1].stat().st_mtime_ns, timestamps[1])

    def test_converted_files_and_arguments_survive_generation_and_relocation(self):
        with tempfile.TemporaryDirectory(prefix="cmake test arguments ") as temporary:
            root = Path(temporary)
            source, build = root / "source with spaces", root / "build with spaces"
            shutil.copytree(FIXTURE, source)
            runner_directory = source / "build_tools/testing"
            runner_directory.mkdir(parents=True)
            for name in ("execution.py", "execution_main.py"):
                shutil.copy2(REPO_ROOT / "build_tools/testing" / name, runner_directory)

            # Run actual converter output through the native/C/Python rules.
            package = source / "converted"
            package.mkdir()
            converter = SimpleNamespace(body="")
            functions = FixtureBuildFileFunctions(
                converter=converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(package),
                repo_root=str(source),
            )
            functions.filegroup(
                name="inputs", srcs=["//:fixture.txt", "//:fixture.txt.more"]
            )
            functions.iree_py_test(
                name="converted_python",
                srcs=["//:reader.py", "//:reader_helper.py"],
                main="reader.py",
                args=["$(location //tools:runner)", "$(locations :inputs)"],
                data=[":inputs", "//:first input.txt", "//tools:runner"],
                package_dirs=["${PROJECT_SOURCE_DIR}"],
            )
            suppression = {"lsan": "//:lsan_suppressions_fixture.txt"}
            functions.cc_test(
                name="compiled_reader",
                srcs=["//:reader.cc"],
                args=["--wrapped=[$(location //:fixture.txt)]", "--check-suppression"],
                env={"TEST_INPUT": "$(rootpath //:fixture.txt.more)"},
                sanitizer_suppressions=suppression,
            )
            functions.native_test(
                name="environment_reader",
                src="//tools:runner",
                args=["--check-suppression", "--input=$(location //:fixture.txt)"],
                env={"TEST_INPUT": "$(location //:fixture.txt)"},
                data=[":inputs", "//tools:runner"],
                sanitizer_suppressions=suppression,
            )
            (package / "CMakeLists.txt").write_text(converter.body)
            configure_project(source, build)
            build_project(build)
            test_project(build)
            tests = json.loads(test_project(build, "--show-only=json-v1"))["tests"]
            converted = next(
                test
                for test in tests
                if test["name"] == "fixture/converted/converted_python"
            )
            self.assertEqual(
                [Path(argument) for argument in converted["command"][-2:]],
                [source / "fixture.txt", source / "fixture.txt.more"],
            )

            generated = [build / "generated.py", build / "generated.txt"]
            timestamps = [path.stat().st_mtime_ns for path in generated]
            build_project(build)
            self.assertEqual(
                [path.stat().st_mtime_ns for path in generated], timestamps
            )
            with (source / "reader.py").open("a") as output:
                output.write('\nprint("updated Python source")\n')
            with (source / "fixture.txt").open("a") as output:
                output.write("updated input\n")
            build_project(build)
            for output, input_name in zip(generated, ("reader.py", "fixture.txt")):
                self.assertEqual(
                    output.read_bytes(), (source / input_name).read_bytes()
                )
                output.unlink()
            build_project(build)
            self.assertIn("updated Python source", test_project(build, "--verbose"))

            prefix = root / "install"
            install_project(build, prefix)
            relocated = root / "relocated install"
            prefix.rename(relocated)
            source.rename(root / "retired source")
            build.rename(root / "retired build")
            test_project(relocated / "share/tests")

    def test_malformed_locators_fail_at_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, argument in enumerate(("--input={{}}", "--input={{missing")):
                with self.subTest(argument=argument):
                    output = configure_project(
                        FIXTURE,
                        Path(temporary) / str(index),
                        f"-DIREE_TEST_INVALID_ARGUMENT={argument}",
                        expect_failure=True,
                    )
                    self.assertIn("Invalid file locator", output)


if __name__ == "__main__":
    unittest.main()
