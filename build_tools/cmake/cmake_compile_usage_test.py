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
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/compile_usage"


def copy_source(source):
    shutil.copytree(FIXTURE, source)
    for template in source.rglob("BUILD.bazel.in"):
        template.rename(template.with_suffix(""))


class CMakeCompileUsageTest(unittest.TestCase):
    def test_executable_includes_reach_source_and_generated_headers(self):
        with tempfile.TemporaryDirectory(
            prefix="cmake executable includes "
        ) as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            package = source / "package"
            include = source / "include files"
            package.mkdir(parents=True)
            include.mkdir()
            (
                source / ".bazel_to_cmake.cfg.py"
            ).write_text("""from bazel_to_cmake_targets import TargetConverter
BAZEL_TO_CMAKE_REPO_ROOT = True
DEFAULT_ROOT_DIRS = ["package"]
class CustomTargetConverter(TargetConverter):
    def _convert_unmatched_target(self, target):
        return ["iree::fixture::" + self._convert_to_cmake_path(target)]
""")
            (include / "source_value.h").write_text("#define SOURCE_VALUE 13\n")
            (package / "probe.cc").write_text("""#include "source_value.h"
#include "generated_value.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(FUZZ_PROBE)
extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, size_t) {
  if (SOURCE_VALUE + GENERATED_VALUE != 42) std::abort();
  return 0;
}
#else
int main(int argc, char** argv) {
  if (argc != EXPECTED_ARGUMENTS || std::strcmp(argv[argc - 1], "--probe")) return 1;
  std::printf("%d\\n", SOURCE_VALUE + GENERATED_VALUE);
  return SOURCE_VALUE + GENERATED_VALUE == 42 ? 0 : 1;
}
#endif
""")
            (package / "BUILD.bazel").write_text("""cc_binary(
    name="binary", srcs=["probe.cc"], includes=["../include files"],
    defines=["EXPECTED_ARGUMENTS=2"], testonly=True)
cc_test(
    name="unit", srcs=["probe.cc"], includes=["../include files"],
    defines=["EXPECTED_ARGUMENTS=2"], args=["--probe"])
iree_cc_benchmark(
    name="benchmark", srcs=["probe.cc"], includes=["../include files"],
    defines=["EXPECTED_ARGUMENTS=4"], args=["--probe"])
iree_cc_fuzz(
    name="fuzz", srcs=["probe.cc"], includes=["../include files"],
    defines=["FUZZ_PROBE"])
""")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(executable_includes C CXX)
enable_testing()
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_CXX_STANDARD 20)
set(IREE_CXX_STANDARD 20)
set(IREE_BUILD_TESTS ON)
set(IREE_BUILD_BENCHMARKS ON)
set(IREE_ENABLE_FUZZING ON)
foreach(module macros copts install_support configure_testing native_test
    cc_binary cc_test cc_binary_benchmark cc_fuzz add_all_subdirs generated_cmake)
  include("${IREE_REPO_ROOT}/build_tools/cmake/iree_${module}.cmake")
endforeach()
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX iree/fixture)
file(WRITE "${IREE_BINARY_DIR}/include files/generated_value.h"
  "#define GENERATED_VALUE 29\\n")
iree_generate_cmake()
iree_add_subdirectory(package)
iree_finalize_target_dependencies()
iree_finalize_test_build_targets()
foreach(name binary fuzz)
  file(GENERATE OUTPUT "${IREE_BINARY_DIR}/${name}-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:iree_fixture_package_${name}>")
endforeach()
# Always check the fuzz rule's usage requirements, even on toolchains without
# libFuzzer. Where available, build and execute its real callback too.
file(GENERATE OUTPUT "${IREE_BINARY_DIR}/fuzz-includes-$<CONFIG>.txt"
  CONTENT "$<JOIN:$<TARGET_PROPERTY:iree_fixture_package_fuzz,INCLUDE_DIRECTORIES>,\\n>")
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer")
check_cxx_source_compiles(
  "#include <stddef.h>\nextern \\"C\\" int LLVMFuzzerTestOneInput(const unsigned char*, size_t) { return 0; }"
  HAS_LIBFUZZER)
file(WRITE "${IREE_BINARY_DIR}/has-libfuzzer.txt" "${HAS_LIBFUZZER}")
""")
            configure_project(source, build)
            build_project(build)
            test_project(build)
            executable = Path((build / f"binary-{CONFIGURATION}.txt").read_text())
            self.assertEqual(run_command(str(executable), "--probe"), "42\n")
            fuzz_includes = {
                Path(path).resolve()
                for path in (build / f"fuzz-includes-{CONFIGURATION}.txt")
                .read_text()
                .splitlines()
            }
            self.assertIn(include.resolve(), fuzz_includes)
            self.assertIn((build / "include files").resolve(), fuzz_includes)
            if (build / "has-libfuzzer.txt").read_text():
                build_project(build, "iree_fixture_package_fuzz")
                executable = Path((build / f"fuzz-{CONFIGURATION}.txt").read_text())
                run_command(str(executable), "-runs=1")

    def test_wasm_companions_reach_the_bundler_through_generated_libraries(self):
        with tempfile.TemporaryDirectory(prefix="cmake wasm companions ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            package = source / "package"
            package.mkdir(parents=True)
            shutil.copyfile(
                FIXTURE / ".bazel_to_cmake.cfg.py",
                source / ".bazel_to_cmake.cfg.py",
            )
            (package / "BUILD.bazel").write_text(
                'cc_library(name="middle", deps=["//native:leaf"])\n'
                'cc_library(name="branch", deps=[":middle"])\n'
            )
            (source / "companion.mjs").write_text(
                "export function createImports(context) { return {}; }\n"
            )
            (source / "main.mjs").write_text("createWasmImports({});\n")
            (source / "module.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(wasm_companions NONE)
set(CMAKE_SYSTEM_PROCESSOR wasm32)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${IREE_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX iree/fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_add_all_subdirs.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_generated_cmake.cmake")
include("${IREE_REPO_ROOT}/runtime/build_tools/cmake/iree_wasm_library.cmake")
iree_generate_cmake()
iree_add_subdirectory(package)
# Providers arrive after the source-owned consumers, as with native imports.
iree_cc_library(NAME leaf DEPS ::companion)
iree_wasm_cc_library(NAME companion MODULE fixture SRCS companion.mjs)
add_executable(module IMPORTED)
set_property(TARGET module PROPERTY IMPORTED_LOCATION "${IREE_SOURCE_DIR}/module.wasm")
_iree_wasm_setup_bundler(WASM_TARGET module
  OUTPUT "${IREE_BINARY_DIR}/module.mjs" MAIN main.mjs
  DEPS iree::fixture::branch)
""")
            configure_project(source, build)
            self.assertEqual(
                (build / "module.mjs.modules").read_text().splitlines(),
                [f"fixture:{(source / 'companion.mjs').as_posix()}"],
            )

    def test_native_requirements_and_relocated_consumer(self):
        with tempfile.TemporaryDirectory(prefix="cmake compile usage ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            copy_source(source)
            caller_hook = source / "caller.cmake"
            caller_hook.write_text(
                'set_property(DIRECTORY PROPERTY CALLER_PROJECT "${PROJECT_NAME}")\n'
            )

            def check():
                build_project(build, "probe")
                executable = Path((build / f"probe-{CONFIGURATION}.txt").read_text())
                self.assertEqual(run_command(str(executable)), "42\n")

            # Exercise native imports through the resolved source-owned chain.
            version = tuple(
                map(
                    int,
                    run_command(CMAKE_COMMAND, "--version").split()[2].split(".")[:2],
                )
            )
            variants = ["", "SCOPED"]
            if version >= (3, 30):
                variants.extend(("CUSTOM", "SCOPED_CUSTOM"))
            for variant in variants:
                arguments = [
                    "-Werror=dev",
                    f"-DCMAKE_PROJECT_INCLUDE={caller_hook.as_posix()}",
                    "-DFIXTURE_REQUIRE_CALLER=ON",
                    f"-DFIXTURE_SCOPED={'ON' if 'SCOPED' in variant else 'OFF'}",
                    f"-DFIXTURE_CUSTOM={'ON' if 'CUSTOM' in variant else 'OFF'}",
                ]
                configure_project(source, build, *arguments)
                check()

            configure_project(
                source, build, "-DFIXTURE_SCOPED=OFF", "-DFIXTURE_CUSTOM=OFF"
            )
            check()
            consumer, consumer_build = root / "consumer", root / "consumer build"
            consumer.mkdir()
            shutil.copyfile(source / "main.c", consumer / "main.c")
            (
                consumer / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(usage_consumer C)
include("${FIXTURE_TARGETS}")
add_executable(probe main.c)
target_compile_definitions(probe PRIVATE INSTALLED_CONSUMER=1)
target_link_libraries(probe PRIVATE iree_fixture_branch)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-$<CONFIG>.txt" CONTENT "$<TARGET_FILE:probe>")
""")

            def check_consumer(targets):
                configure_project(
                    consumer, consumer_build, f"-DFIXTURE_TARGETS={targets.as_posix()}"
                )
                build_project(consumer_build, "probe")
                executable = Path(
                    (consumer_build / f"probe-{CONFIGURATION}.txt").read_text()
                )
                self.assertEqual(run_command(str(executable)), "42\n")

            check_consumer(build / "IREETargets-fixture.cmake")
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
            source.rename(root / "source hidden")
            build.rename(root / "build hidden")
            check_consumer(relocated / "lib/cmake/fixture/IREETargets-fixture.cmake")

    def test_enclosing_project_can_change_requirements(self):
        with tempfile.TemporaryDirectory(prefix="cmake usage subproject ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            copy_source(source / "child")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26...4.2)
project(usage_parent C CXX)
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
