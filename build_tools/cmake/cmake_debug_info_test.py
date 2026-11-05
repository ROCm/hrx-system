# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises native compiler/linker debug compression through CMake policy."""

import platform
import struct
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    build_project,
    configure_project,
    run_command,
)


def compressed_section_count(path):
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] not in (1, 2) or data[5] not in (1, 2):
        raise AssertionError(f"expected ELF: {path}")
    order = "<" if data[5] == 1 else ">"
    word = "I" if data[4] == 1 else "Q"
    offset = struct.unpack_from(order + word, data, 32 if data[4] == 1 else 40)[0]
    stride, count = struct.unpack_from(order + "HH", data, 46 if data[4] == 1 else 58)
    return sum(
        bool(struct.unpack_from(order + word, data, offset + i * stride + 8)[0] & 0x800)
        for i in range(count)
    )


@unittest.skipUnless(platform.system() == "Linux", "Linux debug compression policy")
class CMakeDebugInfoTest(unittest.TestCase):
    def test_compile_and_link_compression_and_opt_out(self):
        with tempfile.TemporaryDirectory(prefix="cmake debug info ") as temporary:
            root = Path(temporary)
            source, build = root / "source", root / "build"
            source.mkdir()
            for extension in ("c", "cc"):
                (source / f"main.{extension}").write_text(
                    "int main(void) { volatile int value = 42; return value != 42; }\n"
                )
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26)
project(debug_compression C CXX)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_setup_toolchain.cmake")
iree_setup_toolchain()
add_executable(probe_c main.c)
add_executable(probe_cc main.cc)
# Retain debug information independently of the outer test configuration.
target_compile_options(probe_c PRIVATE -g)
target_compile_options(probe_cc PRIVATE -g)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/outputs-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:probe_c>\n$<TARGET_FILE:probe_cc>\n")
""")
            for enabled in (True, False):
                configure_project(
                    source,
                    build,
                    "-DIREE_ENABLE_COMPRESSED_DEBUG_INFO="
                    + ("ON" if enabled else "OFF"),
                )
                build_project(build, "probe_c", "probe_cc")
                outputs = (
                    (build / f"outputs-{CONFIGURATION}.txt").read_text().splitlines()
                )
                self.assertEqual(len(outputs), 2)
                for output in outputs:
                    run_command(output)
                    self.assertEqual(
                        compressed_section_count(Path(output)) > 0, enabled
                    )
                objects = list(build.rglob("main.*.o"))
                self.assertEqual(len(objects), 2)
                for object_file in objects:
                    self.assertEqual(compressed_section_count(object_file) > 0, enabled)


if __name__ == "__main__":
    unittest.main()
