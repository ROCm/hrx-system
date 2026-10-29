# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)
project(iree_cmake_python_dependencies_test NONE)

set(IREE_BUILD_TESTS ON)
set(IREE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_python.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")

option(IREE_TEST_REVERSE_DEPS "Reverse import precedence" OFF)
option(IREE_TEST_MISSING_DEP "Reference a missing Python library" OFF)
set(IREE_TEST_DIAMOND_DEPTH 4 CACHE STRING "Shared dependency depth")
set(_ORDERED_DEPS ::left ::right)
if(IREE_TEST_REVERSE_DEPS)
  list(REVERSE _ORDERED_DEPS)
endif()
if(IREE_TEST_MISSING_DEP)
  list(APPEND _ORDERED_DEPS ::missing)
endif()

# The tests are registered before their transitive parent-owned libraries.
add_subdirectory(tests)

# Forward declarations and several target spellings must reach the same library.
iree_py_library(
  NAME generator
  MAIN generator.py
  SRCS generator.py
  IMPORTS .
  DEPS ${_ORDERED_DEPS} ::diamond_left_${IREE_TEST_DIAMOND_DEPTH}
)
iree_py_library(NAME empty)
iree_py_library(NAME shared SRCS shared/value.py IMPORTS shared)
iree_py_library(NAME left SRCS left/choice.py IMPORTS left DEPS ::empty)
iree_py_library(NAME right SRCS right/choice.py IMPORTS right DEPS ::empty)

# Each level doubles the paths through a graph whose declarations grow linearly.
set(_DEPS ::shared ::empty)
foreach(_INDEX RANGE 0 ${IREE_TEST_DIAMOND_DEPTH})
  iree_py_library(NAME diamond_left_${_INDEX} DEPS ${_DEPS})
  iree_py_library(NAME diamond_right_${_INDEX} DEPS ${_DEPS})
  set(_DEPS ::diamond_left_${_INDEX} ::diamond_right_${_INDEX})
endforeach()

set(_INDEX 0)
foreach(_GENERATOR ::generator fixture::generator fixture_generator)
  iree_py_generated_files(
    NAME output_${_INDEX}
    GENERATOR "${_GENERATOR}"
    OUTPUTS "output_${_INDEX}.txt"
    OUTPUT_FLAGS --output
  )
  math(EXPR _INDEX "${_INDEX} + 1")
endforeach()

# Finalizers outside a package consume fully qualified names without access to
# the declaration's directory context, including when a closure is already held.
unset(IREE_PACKAGE_ROOT_DIR)
unset(IREE_PACKAGE_ROOT_PREFIX)
iree_py_library_collect_sources(_SOURCES fixture::generator)
iree_py_library_collect_package_dirs(_PACKAGE_DIRS fixture_generator)
iree_finalize_target_dependencies()
iree_finalize_python_test_environments()
