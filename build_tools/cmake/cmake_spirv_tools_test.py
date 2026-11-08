# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    build_project,
    configure_project,
    test_project,
)


class CMakeSpirvToolsTest(unittest.TestCase):
    def test_msan_stays_on_the_subject_and_out_of_external_tools(self):
        with tempfile.TemporaryDirectory(prefix="cmake spirv tools ") as temporary:
            source = Path(temporary) / "source"
            build = Path(temporary) / "build"
            tools = source / "tools"
            tools.mkdir(parents=True)
            (source / "headers").mkdir()
            # Substitute the upstream dependency, retaining its executable
            # family and a C library to check both compiler language settings.
            (tools / "tool.cc").write_text("""#if __has_feature(memory_sanitizer)
#error External tools must not inherit the subject's MSAN configuration
#endif
extern "C" int library_value(void);
int main() { return library_value(); }
""")
            (tools / "library.c").write_text("""#if __has_feature(memory_sanitizer)
#error External tool libraries must share the tool configuration
#endif
int library_value(void) { return 0; }
""")
            (tools / "CMakeLists.txt").write_text("""add_library(tool_library library.c)
foreach(tool spirv-as spirv-dis spirv-val)
  add_executable(${tool} tool.cc)
  target_link_libraries(${tool} PRIVATE tool_library)
endforeach()
""")
            (source / "subject.c").write_text("""#if !__has_feature(memory_sanitizer)
#error The subject must retain MSAN instrumentation
#endif
int main(void) { return 0; }
""")
            (
                source / "CMakeLists.txt"
            ).write_text("""cmake_minimum_required(VERSION 3.26)
project(spirv_tool_configuration C CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  file(WRITE "${CMAKE_BINARY_DIR}/unsupported.txt" "MSAN requires Linux Clang")
  return()
endif()
enable_testing()
list(APPEND CMAKE_MODULE_PATH "${IREE_REPO_ROOT}/build_tools/cmake")
include("${IREE_REPO_ROOT}/build_tools/third_party/spirv_tools/spirv_tools.cmake")
set(IREE_DEPENDENCY_MODE pinned)
set(IREE_ENABLE_MSAN ON)
string(APPEND CMAKE_C_FLAGS " -fsanitize=memory")
string(APPEND CMAKE_CXX_FLAGS " -fsanitize=memory")
foreach(dependency spirv_headers spirv_tools)
  string(TOUPPER "${dependency}" dependency_upper)
  set(IREE_DEP_${dependency_upper}_URLS "https://unused.invalid/source.tar.gz")
  set(IREE_DEP_${dependency_upper}_SHA256
    "0000000000000000000000000000000000000000000000000000000000000000")
  set(IREE_DEP_${dependency_upper}_PATCHES "")
  set(IREE_DEP_${dependency_upper}_PATCH_ARGS "")
endforeach()
set(FETCHCONTENT_SOURCE_DIR_SPIRV_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/headers")
set(FETCHCONTENT_SOURCE_DIR_SPIRV_TOOLS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
iree_configure_spirv_tools()
# Compile the instrumented subject without requiring an MSAN runtime package
# in the outer CMake toolchain. The external tools must link and execute.
add_library(subject OBJECT subject.c)
foreach(tool spirv_as spirv_dis spirv_val)
  add_test(NAME ${tool} COMMAND iree::third_party::${tool})
endforeach()
""")
            configure_project(source, build)
            if (build / "unsupported.txt").exists():
                self.skipTest((build / "unsupported.txt").read_text())
            build_project(build)
            test_project(build)


if __name__ == "__main__":
    unittest.main()
