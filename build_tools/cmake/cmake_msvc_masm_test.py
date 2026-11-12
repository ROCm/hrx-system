# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import configure_project


class CMakeMsvcMasmTest(unittest.TestCase):
    def test_empty_configured_compiler_resolves_ml64(self):
        with tempfile.TemporaryDirectory(prefix="cmake msvc masm ") as temporary:
            root = Path(temporary)
            source, build, tools = root / "source", root / "build", root / "tools"
            source.mkdir()
            tools.mkdir()

            compiler_name = "ml64.exe" if os.name == "nt" else "ml64"
            compiler = tools / compiler_name
            compiler.write_text("")
            compiler.chmod(0o755)
            (source / "bridge.asm").write_text("; MASM fixture\n")
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.26)
project(msvc_masm_resolution C)

function(iree_package_source_dir output)
  set(${output} "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

function(iree_package_binary_dir output)
  set(${output} "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
endfunction()

function(iree_cc_library)
  cmake_parse_arguments(_RULE "PUBLIC" "NAME" "SRCS" ${ARGN})
  add_library(${_RULE_NAME} STATIC ${_RULE_SRCS})
endfunction()

function(iree_package_target_name output local_name)
  string(REGEX REPLACE "^::" "" target_name "${local_name}")
  set(${output} "${target_name}" PARENT_SCOPE)
endfunction()

include("${IREE_REPO_ROOT}/build_tools/cmake/iree_msvc_masm_library.cmake")
iree_msvc_masm_library(NAME bridge SRC bridge.asm)
"""
            )

            configure_project(
                source,
                build,
                "-DCMAKE_ASM_MASM_COMPILER=",
                f"-DCMAKE_PROGRAM_PATH={tools.as_posix()}",
            )

            generated_build_files = [
                build / "build.ninja",
                *build.rglob("*.vcxproj"),
                *build.rglob("build.make"),
            ]
            generated_commands = [
                path.read_text(errors="replace")
                for path in generated_build_files
                if path.is_file()
            ]
            self.assertTrue(generated_commands, "CMake generated no known build files")
            self.assertTrue(
                any(
                    compiler_name in command and "bridge.obj" in command
                    for command in generated_commands
                ),
                "the MASM build edge did not invoke the resolved ml64 compiler",
            )


if __name__ == "__main__":
    unittest.main()
