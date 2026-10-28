# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)
project(iree_cmake_library_dependencies_test C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/fixture")
set(IREE_BUILD_TESTS ON)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_configure_testing.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_binary.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_test.cmake")

# External usage requirements are not implementations owned by a shared image.
# Cover imported targets, aliases, and opaque expressions alongside platform
# libraries such as libm, which main.c calls directly.
add_library(external_imported INTERFACE IMPORTED)
add_library(external_usage INTERFACE)
add_library(external::usage ALIAS external_usage)
list(APPEND IREE_DEFAULT_LINK_LIBRARIES
  external_imported external::usage "$<1:${IREE_DEFAULT_LINK_LIBRARIES}>")

# An existing source wins over a same-named file in the binary directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/library.c" "#error Binary shadow source\n")
foreach(_EXTENSION hpp inc inl)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/private.${_EXTENSION}" "#pragma once\n")
endforeach()
# This custom command has no registered producer target. The header library
# owns its build rule and orders all of its consumers after generation.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/owned.h"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/owned.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/owned.h"
  DEPENDS owned.h.in VERBATIM)
iree_cc_library(NAME headers
  SRCS inactive/generated.h
    "${CMAKE_CURRENT_BINARY_DIR}/private.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/private.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/private.inl"
  HDRS owned.h
  TEXTUAL_HDRS producer/generated.h)
iree_cc_library(NAME leaf
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/leaf.hpp.c" DEFINES TRANSITIVE_VALUE=3)
# A false condition excludes the entire list; active system includes keep their
# order. The inactive header catches either a leaked path or reordered includes.
set(_LIBRARY_OPTIONS)
if(FIXTURE_ALWAYS_LINK)
  list(APPEND _LIBRARY_OPTIONS ALWAYSLINK)
endif()
iree_cc_library(NAME library
  ${_LIBRARY_OPTIONS}
  SRCS library.c producer/generated.c DEPS ::leaf ::headers
  INCLUDES "$<0:${CMAKE_CURRENT_BINARY_DIR}/unused;${CMAKE_CURRENT_SOURCE_DIR}/inactive>"
  SYSTEM_INCLUDES "$<1:${CMAKE_CURRENT_BINARY_DIR}/producer;${CMAKE_CURRENT_SOURCE_DIR}/inactive>")

# A custom-command object models the Windows ELF loader's MASM bridge. Native
# archives, unified libraries, and shared-tool composition must all retain it.
add_library(external OBJECT external.c)
set(_EXTERNAL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/external-$<CONFIG>.o")
add_custom_command(OUTPUT "${_EXTERNAL_OBJECT}"
  COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_OBJECTS:external>" "${_EXTERNAL_OBJECT}"
  DEPENDS external "$<TARGET_OBJECTS:external>"
  VERBATIM)
iree_cc_library_add_external_objects(iree_fixture_library "${_EXTERNAL_OBJECT}")

iree_cc_library(NAME test_main SRCS main.c DEPS ::library)
iree_cc_binary(NAME probe DEPS ::test_main)
iree_cc_unified_library(NAME unified ROOT ::library)
iree_cc_binary(NAME unified_probe SRCS main.c DEPS ::unified)
include("${IREE_REPO_ROOT}/loom/binding/c/cmake/loomc_shared.cmake")
_loomc_collect(iree_fixture_leaf runtime
  _OBJECTS _LIBRARIES _runtime_owned _OPTIONS)
add_library(loomc_runtime SHARED ${_OBJECTS})
set_target_properties(loomc_runtime PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(loomc_runtime PRIVATE ${_LIBRARIES})
target_link_options(loomc_runtime PRIVATE ${_OPTIONS})
foreach(_DEPENDENCY IN LISTS IREE_DEFAULT_LINK_LIBRARIES)
  _loomc_resolve(_EXTERNAL "${_DEPENDENCY}")
  if(_EXTERNAL IN_LIST _runtime_owned)
    message(FATAL_ERROR "Runtime claimed external requirement: ${_EXTERNAL}")
  endif()
endforeach()

_loomc_collect(iree_fixture_library compiler
  _OBJECTS _LIBRARIES _compiler_owned _OPTIONS)
add_library(collected SHARED ${_OBJECTS})
add_library(loomc_shared ALIAS collected)
set_target_properties(collected PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(collected PRIVATE ${_LIBRARIES})
target_link_options(collected PRIVATE ${_OPTIONS})
iree_cc_binary(NAME collected_probe SRCS main.c DEPS collected loomc_runtime)

# Runtime-only tools keep compiler-owned objects local, even when a compiler
# export exists. Fully shared tools substitute both implementation owners.
# Replace the final link libraries just as shared-tool composition does, without
# reintroducing default link requirements through iree_cc_binary.
set(_compiler_exports iree_fixture_library)
foreach(_MODE runtime_tool tool)
  _loomc_collect("iree_fixture_library;iree_fixture_leaf;${IREE_DEFAULT_LINK_LIBRARIES}" ${_MODE}
    _OBJECTS _LIBRARIES _OWNED _OPTIONS)
  foreach(_DEPENDENCY IN LISTS IREE_DEFAULT_LINK_LIBRARIES)
    _loomc_resolve(_EXTERNAL "${_DEPENDENCY}")
    if(NOT _EXTERNAL IN_LIST _LIBRARIES)
      message(FATAL_ERROR "${_MODE} lost external requirement: ${_EXTERNAL}")
    endif()
  endforeach()
  iree_cc_binary(NAME ${_MODE}_probe SRCS main.c DEPS ${_LIBRARIES})
  set_property(TARGET iree_fixture_${_MODE}_probe PROPERTY LINK_LIBRARIES
    "${_LIBRARIES}")
  target_sources(iree_fixture_${_MODE}_probe PRIVATE ${_OBJECTS})
  target_link_options(iree_fixture_${_MODE}_probe PRIVATE ${_OPTIONS})
endforeach()

iree_cc_test(
  NAME source_less_test_with_a_long_target_name_matching_real_cts_binaries
  DEPS ::test_main
)

# Private link requirements stay link-only when forwarded to the unified target.
add_library(private_usage INTERFACE)
target_compile_definitions(private_usage INTERFACE PRIVATE_VALUE=7)
get_target_property(_LIBRARY_OBJECT_TARGET iree_fixture_library IREE_CC_OBJECT_TARGET)
target_link_libraries(${_LIBRARY_OBJECT_TARGET} PRIVATE private_usage)
target_compile_definitions(${_LIBRARY_OBJECT_TARGET} PRIVATE LATE_VALUE=11)

# Independent forward producers in another directory must reach both the
# interface header's consumer and the library's generated source.
add_subdirectory(producer)
iree_finalize_target_dependencies()
cmake_language(DEFER CALL iree_finalize_compile_usage)

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/paths-$<CONFIG>.txt"
  CONTENT "${_LIBRARY_OBJECT_TARGET}\n$<TARGET_FILE:iree_fixture_leaf>\n$<TARGET_FILE:iree_fixture_probe>\n$<TARGET_FILE:iree_fixture_unified_probe>\n$<TARGET_FILE:iree_fixture_collected_probe>\n$<TARGET_FILE:iree_fixture_runtime_tool_probe>\n$<TARGET_FILE:iree_fixture_tool_probe>\n$<TARGET_FILE:iree_fixture_source_less_test_with_a_long_target_name_matching_real_cts_binaries>\n")
