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

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
    run_command,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/generated_packages"


class CMakeGeneratedPackagesTest(unittest.TestCase):
    def test_setup_defaults_and_live_native_interfaces(self):
        with tempfile.TemporaryDirectory(prefix="cmake native defaults ") as temporary:
            source, build = Path(temporary) / "source", Path(temporary) / "build"
            source.mkdir()
            shutil.copyfile(
                FIXTURE / ".bazel_to_cmake.cfg.py", source / ".bazel_to_cmake.cfg.py"
            )

            def declarations(name, value):
                return (
                    f'cc_library(name="{name}", srcs=["value.c"], '
                    f'defines=["EXPECTED_VALUE={value}"])\n'
                    f'cc_binary(name="{name}_probe", srcs=["main.c"], '
                    f'defines=["EXPECTED_VALUE={value}"], '
                    f'deps=[":{name}", "//external:usage"])\n'
                )

            for name, value in (("first", 11), ("second", 11), ("inline", None)):
                package = source / "package" / name
                package.mkdir(parents=True)
                (package / "value.c").write_text(
                    "#if NATIVE_VALUE != EXPECTED_VALUE || IMPLICIT_VALUE != EXPECTED_VALUE * 10 || LATE_VALUE != 5\n"
                    "#error Native declaration binding changed\n#endif\n"
                    "#ifdef OPTION_VALUE\n#error Interface options affected their provider\n#endif\n"
                    "int binding_value(void) { return NATIVE_VALUE; }\n"
                )
                (package / "main.c").write_text(
                    "#if NATIVE_VALUE != EXPECTED_VALUE || IMPLICIT_VALUE != EXPECTED_VALUE * 10 || LATE_VALUE != 5 || LITERAL_VALUE != 31\n"
                    "#error Native declaration binding changed\n#endif\n"
                    "#if OPTION_VALUE != EXPECTED_VALUE\n#error Native compiler options changed\n#endif\n"
                    "int binding_value(void);\n"
                    "int main(void) { return binding_value() != EXPECTED_VALUE; }\n"
                )
                content = ""
                if value is not None:
                    content = declarations("value", value)
                else:
                    # A disabled declaration cannot own a shared binding that
                    # is also needed by a later enabled declaration.
                    content = (
                        'iree_cmake_extra_content(inline=True, content="if(FALSE)")\n'
                        'cc_library(name="disabled", deps=["//external:usage"])\n'
                        'iree_cmake_extra_content(inline=True, content="endif()")\n'
                    )
                    for provider in ("first", "second"):
                        content += declarations(provider, 11)
                (package / "BUILD.bazel").write_text(content)
            wrapper = source / "package" / "wrapper"
            wrapper.mkdir()
            (wrapper / "BUILD.bazel").write_text(
                'cc_library(name="wrapper", deps=["//package/first:value"])\n'
            )
            (source / "public.c").write_text(
                "#if NATIVE_VALUE != 11 || IMPLICIT_VALUE != 110 || OPTION_VALUE != 11 || LATE_VALUE != 5\n"
                "#error Native public interface changed\n#endif\n"
                "int binding_value(void);\n"
                "int main(void) { return binding_value() != 11; }\n"
            )
            consumer = source / "package" / "consumer"
            consumer.mkdir()
            shutil.copyfile(source / "public.c", consumer / "main.c")
            (consumer / "BUILD.bazel").write_text(
                'cc_binary(name="probe", srcs=["main.c"], deps=["//package/wrapper"])\n'
            )
            (source / "launcher.py").write_text(
                "import subprocess, sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1] == 'parent configuration'\n"
                "Path(__file__).with_suffix('.called').mkdir(exist_ok=True)\n"
                "sys.exit(subprocess.call(sys.argv[2:]))\n"
            )
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.26...4.2)
project(native_defaults C CXX)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${IREE_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
add_library(parent_configuration INTERFACE)
set_property(TARGET parent_configuration PROPERTY LINK_LAUNCHER "${Python3_EXECUTABLE};-B;${CMAKE_CURRENT_SOURCE_DIR}/launcher.py;parent configuration")
set(CMAKE_C_LINKER_LAUNCHER "$<TARGET_PROPERTY:parent_configuration,LINK_LAUNCHER>")
foreach(helper iree_macros iree_copts iree_install_support iree_cc_library iree_cc_binary iree_configure_testing iree_python iree_add_all_subdirs iree_generated_cmake)
  include("${IREE_REPO_ROOT}/build_tools/cmake/${helper}.cmake")
endforeach()
add_library(native_first INTERFACE IMPORTED)
set_property(TARGET native_first PROPERTY INTERFACE_COMPILE_DEFINITIONS NATIVE_VALUE=11)
add_library(implicit_first INTERFACE IMPORTED)
set_property(TARGET implicit_first PROPERTY INTERFACE_COMPILE_DEFINITIONS IMPLICIT_VALUE=110)
add_library(authored_usage INTERFACE IMPORTED)
set_property(TARGET authored_usage PROPERTY INTERFACE_COMPILE_DEFINITIONS LITERAL_VALUE=31)
set(IREE_IMPLICIT_DEFS_CC_DEPS implicit_first)
set(IREE_DEFAULT_LINK_LIBRARIES native_first)
set(IREE_INTERFACE_COPTS -DOPTION_VALUE=11)
iree_generate_cmake()
iree_add_subdirectory(package/first)
iree_add_subdirectory(package/second)
iree_add_subdirectory(package/inline)
target_compile_definitions(native_first INTERFACE LATE_VALUE=5)
iree_add_subdirectory(package/wrapper)
add_executable(public_probe public.c)
target_link_libraries(public_probe PRIVATE fixture::package::wrapper)
iree_add_subdirectory(package/consumer)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/paths-$<CONFIG>.txt" CONTENT
  "$<TARGET_FILE:fixture_package_first_value_probe>\n$<TARGET_FILE:fixture_package_second_value_probe>\n$<TARGET_FILE:fixture_package_inline_first_probe>\n$<TARGET_FILE:fixture_package_inline_second_probe>\n$<TARGET_FILE:public_probe>\n$<TARGET_FILE:fixture_package_consumer_probe>\n")
"""
            )
            configure_project(source, build)
            build_project(build)
            self.assertTrue((source / "launcher.called").is_dir())
            for program in (
                (build / f"paths-{CONFIGURATION}.txt").read_text().splitlines()
            ):
                run_command(program)

    def test_logical_package_generators_work_with_makefiles(self):
        make = shutil.which("gmake") or shutil.which("make")
        if make is None:
            self.skipTest("GNU Make is not available")
        with tempfile.TemporaryDirectory(
            prefix="cmake logical generator "
        ) as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            source.mkdir()
            input_file = source / "input.txt"
            input_file.write_text("first")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26)
project(logical_generator NONE)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${IREE_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_generated_files.cmake")
set(IREE_PACKAGE_SOURCE_DIR "${IREE_SOURCE_DIR}/logical")
set(IREE_PACKAGE_BINARY_DIR "${IREE_BINARY_DIR}/logical")
file(MAKE_DIRECTORY "${IREE_PACKAGE_BINARY_DIR}")
add_executable(copy_tool IMPORTED)
set_property(TARGET copy_tool PROPERTY IMPORTED_LOCATION "${CMAKE_COMMAND}")
iree_generated_files(NAME copy GENERATOR copy_tool
  OUTPUTS "${IREE_PACKAGE_BINARY_DIR}/output.txt"
  INPUTS "${IREE_SOURCE_DIR}/input.txt"
  ARGS -E copy_if_different "${IREE_SOURCE_DIR}/input.txt"
    "${IREE_PACKAGE_BINARY_DIR}/output.txt")
""")
            configure_project(
                source, build, "-G", "Unix Makefiles", f"-DCMAKE_MAKE_PROGRAM={make}"
            )
            output = build / "logical/output.txt"
            build_project(build, "fixture_logical_copy")
            self.assertEqual(output.read_text(), "first")
            timestamp = output.stat().st_mtime_ns
            build_project(build, "fixture_logical_copy")
            self.assertEqual(output.stat().st_mtime_ns, timestamp)
            input_file.write_text("second")
            build_project(build, "fixture_logical_copy")
            self.assertEqual(output.read_text(), "second")
            output.unlink()
            build_project(build, "fixture_logical_copy")
            self.assertEqual(output.read_text(), "second")

    def test_embedded_packages_regenerate_without_source_outputs(self):
        with tempfile.TemporaryDirectory(
            prefix="cmake generated packages "
        ) as temporary:
            root = Path(temporary)
            source = root / "source"
            # Stress Windows object paths while leaving room for the generator's
            # configuration directory and a complete object-name digest.
            build = root / "build directory for an embedded CI workspace"
            shutil.copytree(FIXTURE, source)
            for template in source.rglob("BUILD.bazel.in"):
                template.rename(template.with_suffix(""))
            for name in ("leaf", "data", "native"):
                (source / f"{name}.txt").write_text(f"leak:{name}\n")
            # Python sees declarations in packages that native configuration
            # disables. Their link inputs must not require absent targets.
            disabled = source / "package/disabled"
            disabled.mkdir()
            (disabled / "BUILD.bazel").write_text(
                'iree_cmake_extra_content(content="return()")\n'
                'cc_library(name="value", srcs=["//package:registration.cc"])\n'
                'cc_binary(name="probe", srcs=["//package:main.c"], deps=[":value"])\n'
            )
            # Exercise add_subdirectory embedding with independent source and
            # binary roots, including a nested authored project and local import.
            host = source / "host"
            host.mkdir()
            (host / "fallback.c").write_text("int fixture_value(void) { return -1; }\n")
            (host / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.26)\n"
                "project(embedding C CXX)\n"
                "add_library(native_fallback STATIC fallback.c)\n"
                'add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." owned)\n'
                "target_link_libraries(native_suppression_provider INTERFACE native_fallback)\n"
                'if(NOT PROJECT_NAME STREQUAL "embedding")\n'
                '  message(FATAL_ERROR "Embedded project scope leaked")\n'
                "endif()\n"
            )
            configure_project(host, build)
            owned = build / "owned"
            generated = owned / ".iree/cmake/package/CMakeLists.txt"
            manifest = owned / ".iree/cmake/inputs.cmake"
            programs = [
                Path(path)
                for path in (owned / f"paths-{CONFIGURATION}.txt")
                .read_text()
                .splitlines()
            ]
            archive = programs.pop()

            def check(value):
                build_project(
                    build,
                    "fixture_package_probe",
                    "fixture_package_authored_child_probe",
                    "fixture_package_tool",
                )
                self.assertEqual(run_command(str(programs[0])), f"{value}\n")
                self.assertEqual(run_command(str(programs[1])), "23\n")
                self.assertEqual(run_command(str(programs[2])), f"{value}\n")
                self.assertFalse((source / "package/CMakeLists.txt").exists())

            check(7)
            # Source libraries precede native defaults regardless of whether
            # Ninja places the link inputs in a response file.
            if os.environ["IREE_TEST_CMAKE_GENERATOR"].startswith("Ninja"):
                configure_project(host, build, "-DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON")
                check(7)
                configure_project(host, build, "-UCMAKE_NINJA_FORCE_RESPONSE_FILE")
                check(7)
            environment = (owned / f"environment-{CONFIGURATION}.txt").read_text()
            suppression = Path(
                environment.removeprefix('LSAN_OPTIONS=suppressions="').removesuffix(
                    '":allow_addr2line=1'
                )
            )
            self.assertEqual(
                suppression.read_text().split(),
                ["leak:data", "leak:leaf", "leak:native"],
            )
            (source / "data.txt").write_text("leak:changed\n")
            check(7)
            self.assertEqual(
                suppression.read_text().split(),
                ["leak:changed", "leak:leaf", "leak:native"],
            )
            paths = [generated, manifest, *programs]
            timestamps = [path.stat().st_mtime_ns for path in paths]
            check(7)
            configure_project(host, build)
            check(7)
            self.assertEqual([path.stat().st_mtime_ns for path in paths], timestamps)
            generator = source / "package/producer/generator.py"
            generator.write_text(generator.read_text() + "\n")
            check(7)
            timestamps = [path.stat().st_mtime_ns for path in paths]
            check(7)
            self.assertEqual([path.stat().st_mtime_ns for path in paths], timestamps)
            # Cleaning build outputs preserves configure_file outputs, which
            # have no build-time producer that could recreate them.
            build_project(build, "clean")
            check(7)

            settings = source / "package/settings.bzl"
            settings.write_text(
                settings.read_text().replace("BASE_VALUE=4", "BASE_VALUE=6")
            )
            check(9)
            declaration = source / "package/BUILD.bazel"
            declaration.write_text(
                declaration.read_text().replace("ADJUSTMENT=0", "ADJUSTMENT=5")
            )
            check(14)
            header = source / "package/generated.h.in"
            header.write_text(
                header.read_text().replace("HEADER_VALUE 3", "HEADER_VALUE 4")
            )
            check(15)
            (owned / "package/generated.h").unlink()
            check(15)
            archive.unlink()
            check(15)
            link_inputs = (owned / f"link-inputs-{CONFIGURATION}.txt").read_text()
            for name in link_inputs.split(";"):
                if name.endswith(".rsp"):
                    response = Path(name)
                    response.unlink()
                    check(15)
                    self.assertTrue(response.is_file())

            extra = source / "package/value_extra.c"
            extra.write_text("int fixture_extra(void) { return 42; }\n")
            check(15)
            commands = json.loads((build / "compile_commands.json").read_text())
            self.assertIn(
                extra.as_posix(), {Path(row["file"]).as_posix() for row in commands}
            )

            # A leaf package needs no BUILD or CMake boilerplate in its parents.
            added = source / "package/group/nested/added"
            added.mkdir(parents=True)
            (added / "BUILD.bazel").write_text(
                'cc_binary(name="probe", srcs=["main.c"])\n'
                'iree_cmake_extra_content(content="""\n'
                'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/path-$<CONFIG>.txt"\n'
                '  CONTENT "$<TARGET_FILE:fixture_package_group_nested_added_probe>")\n'
                '""", inline=True)\n'
            )
            (added / "main.c").write_text(
                '#include <stdio.h>\nint main(void) { puts("37"); return 0; }\n'
            )
            check(15)
            added_program = Path(
                (
                    owned / f"package/group/nested/added/path-{CONFIGURATION}.txt"
                ).read_text()
            )
            build_project(build, "fixture_package_group_nested_added_probe")
            self.assertEqual(run_command(str(added_program)), "37\n")
            added_outputs = [
                owned / ".iree/cmake/package" / directory / "CMakeLists.txt"
                for directory in ("group", "group/nested", "group/nested/added")
            ]
            for path in added_outputs:
                self.assertTrue(path.is_file())
            added_outputs[0].unlink()
            check(15)
            self.assertTrue(added_outputs[0].is_file())
            (added / "BUILD.bazel").unlink()
            check(15)
            for path in added_outputs:
                self.assertFalse(path.exists())

            # Native regeneration repairs missing generated lists; it must not
            # require a separate dev.py or source updater invocation.
            generated.unlink()
            check(15)
            self.assertTrue(generated.is_file())
            manifest.unlink()
            check(15)
            self.assertTrue(manifest.is_file())
            self.assertEqual(
                {path.relative_to(source) for path in source.rglob("CMakeLists.txt")},
                {
                    Path("CMakeLists.txt"),
                    Path("host/CMakeLists.txt"),
                    Path("package/authored/CMakeLists.txt"),
                },
            )
            configure_project(
                host,
                build,
                f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build / 'archive outputs'}",
            )
            check(15)
            relocated_archive = Path(
                (owned / f"paths-{CONFIGURATION}.txt").read_text().splitlines()[-1]
            )
            self.assertTrue(relocated_archive.is_relative_to(build / "archive outputs"))
            relocated_archive.unlink()
            check(15)

            # A native directory may change artifact locations after bootstrap.
            # Final links consume the actual target files, not predicted names.
            declaration.write_text(
                declaration.read_text() + '\niree_cmake_extra_content(content="""\n'
                'set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/native\'s, archives")\n'
                f'set(CMAKE_{CONFIGURATION.upper()}_POSTFIX "_native")\n'
                '""")\n'
            )
            configure_project(host, build, "-UCMAKE_ARCHIVE_OUTPUT_DIRECTORY")
            archive.unlink()
            check(15)
            scoped_archive = Path(
                (owned / f"paths-{CONFIGURATION}.txt").read_text().splitlines()[-1]
            )
            self.assertTrue(
                scoped_archive.is_relative_to(owned / "package/native's, archives")
            )
            self.assertTrue(scoped_archive.stem.endswith("_native"))
            scoped_archive.unlink()
            check(15)
            link_inputs = (owned / f"link-inputs-{CONFIGURATION}.txt").read_text()
            for name in link_inputs.split(";"):
                if name.endswith(".rsp"):
                    response = Path(name)
                    response.unlink()
                    check(15)
                    self.assertTrue(response.is_file())

            # Explicit native directory options keep their build semantics even
            # for a package that ordinarily uses a generated function scope.
            authored = source / "package/authored/CMakeLists.txt"
            authored.write_text(
                authored.read_text().replace(
                    "iree_add_subdirectory(child)",
                    "iree_add_subdirectory(child EXCLUDE_FROM_ALL)",
                )
            )
            configure_project(host, build)
            programs[1].unlink()
            build_project(build)
            self.assertFalse(programs[1].exists())
            check(15)

            # Embedding the same authored fragment in BUILD preserves its
            # private imported interface and the consumers in sibling packages.
            (authored.parent / "BUILD.bazel").write_text(
                f"iree_cmake_extra_content(content={authored.read_text()!r})\n"
            )
            authored.unlink()
            check(15)


if __name__ == "__main__":
    unittest.main()
