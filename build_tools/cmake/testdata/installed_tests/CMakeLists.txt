# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_installed_tests_test C)

set(IREE_BUILD_TESTS ON)
set(HRX_INSTALL_TESTS ON)
set(HRX_INSTALL_TESTS_DIR share/tests)
set(HRX_INSTALL_TESTS_COMPONENT FixtureTests)
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX fixture)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/libhrx/build_tools/cmake/hrx_installed_tests.cmake")

# Installed paths may end in CMake's false-value suffix.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fixture-NOTFOUND" "fixture data\n")

# Repeated consumers share files, trees, and forward-declared aliases. A source
# may also be needed in more than one installed package directory.
foreach(_INDEX RANGE 0 1)
  hrx_installed_tests_install_source_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/fixture.txt" "other/fixture.txt" _OTHER_FILE)
  hrx_installed_tests_install_source_tree(
    "${CMAKE_CURRENT_SOURCE_DIR}/tree" "other/tree" _OTHER_TREE)
  hrx_register_installed_test(
    NAME "deferred/${_INDEX}"
    TARGET first::tool
    DATA fixture.txt tree "${CMAKE_CURRENT_BINARY_DIR}/fixture-NOTFOUND"
    REQUIRED_FILES "${_OTHER_FILE}" "${_OTHER_TREE}/nested.txt"
  )
endforeach()
hrx_register_installed_test(NAME second-alias TARGET second::tool)

option(IREE_TEST_DECLARE_TOOL "Declare the deferred tool" ON)
set(IREE_TEST_TOOL_OUTPUT_NAME tool CACHE STRING "Tool output filename")
if(IREE_TEST_DECLARE_TOOL)
  add_executable(tool ../data_dependencies/tool.c)
  set_target_properties(tool PROPERTIES OUTPUT_NAME "${IREE_TEST_TOOL_OUTPUT_NAME}")
  add_executable(first::tool ALIAS tool)
  add_executable(second::tool ALIAS tool)
  add_executable(third::tool ALIAS tool)
  hrx_register_installed_test(NAME existing-target TARGET tool)
  hrx_register_installed_test(NAME existing-alias TARGET third::tool)

  add_library(library MODULE ../data_dependencies/tool.c)
  set_target_properties(library PROPERTIES PREFIX "")
  add_library(first::library ALIAS library)
  add_library(second::library ALIAS library)
  hrx_register_installed_test(
    NAME library-aliases
    TARGET tool
    DATA first::library second::library
  )

  add_library(shared SHARED shared.c)
  set_target_properties(shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_executable(shared_consumer shared_consumer.c)
  target_link_libraries(shared_consumer PRIVATE shared)
  hrx_register_installed_test(
    NAME shared-library-consumer
    TARGET shared_consumer
    DATA shared
  )
endif()

# External Python dependencies can appear in several package namespaces and
# through overlapping import roots. Only identical mappings may be shared.
function(register_python_sources)
  iree_package_path(_PACKAGE)
  set(_ROOT "${PROJECT_SOURCE_DIR}/../python")
  set(_LIBRARY "${_ROOT}/library")
  foreach(_ORDER forward reverse)
    set(_PACKAGE_DIRS "${_ROOT}" "${_LIBRARY}")
    if(_ORDER STREQUAL reverse)
      list(REVERSE _PACKAGE_DIRS)
    endif()
    foreach(_INDEX RANGE 0 1)
      hrx_register_installed_python_test(
        NAME "${_PACKAGE}/${_ORDER}/${_INDEX}"
        SRC "${_LIBRARY}/probe.py"
        SOURCES "${_LIBRARY}/probe.py" "${_LIBRARY}/payload.py"
        PACKAGE_DIRS ${_PACKAGE_DIRS})
    endforeach()
  endforeach()
endfunction()

foreach(_PREFIX first second)
  set(IREE_PACKAGE_ROOT_PREFIX "${_PREFIX}")
  register_python_sources()
endforeach()
add_subdirectory(python)

hrx_create_installed_tests()
