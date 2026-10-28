# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)
project(iree_compile_usage_test C CXX)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_CXX_STANDARD 20)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX iree/fixture
  DEFAULT_EXPORT_SET fixture DEFAULT_INSTALL_COMPONENT FixtureLibraries
  IMPLICIT_DEFS_TARGET fixture_defs)

# The static chain is exported and must remain usable without this source tree.
iree_cc_library(NAME leaf SRCS leaf.c HDRS leaf.h extra/config.h system/system_value.h
  DEFINES LEAF_VALUE=41)
target_include_directories(iree_fixture_leaf INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extra>"
  "$<INSTALL_INTERFACE:include/extra>")
target_include_directories(iree_fixture_leaf SYSTEM INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/system>"
  "$<INSTALL_INTERFACE:include/system>")
target_compile_definitions(iree_fixture_leaf INTERFACE
  "$<$<CONFIG:Release>:PROFILE_VALUE=7>"
  "$<$<COMPILE_LANGUAGE:C>:LANGUAGE_VALUE=19>")
iree_cc_library(NAME middle DEPS ::leaf)
iree_cc_library(NAME branch SRCS branch.c HDRS branch.h DEPS ::middle)
iree_cc_unified_library(NAME unified ROOT ::branch)
iree_generate_export_targets(EXPORT_SET fixture COMPONENT FixtureLibraries
  INSTALL_DESTINATION lib/cmake/fixture)

# These local-only boundaries exercise native requirements inherited through
# resolved libraries without changing the installed library's dependencies.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT OFF)
iree_cc_library(NAME cycle_a DEFINES CYCLE_A=3 DEPS ::cycle_b)
iree_cc_library(NAME cycle_b DEFINES CYCLE_B=5 DEPS ::cycle_a)
iree_cc_library(NAME position)
set_property(TARGET iree_fixture_position PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE TRUE)
add_library(native_provider INTERFACE IMPORTED)
set_property(TARGET native_provider PROPERTY INTERFACE_COMPILE_DEFINITIONS NATIVE_VALUE=13)
set_property(TARGET native_provider PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/extra")
iree_cc_library(NAME boundary DEPS native_provider ::position)
target_compile_definitions(iree_fixture_boundary INTERFACE
  "$<TARGET_PROPERTY:native_provider,INTERFACE_COMPILE_DEFINITIONS>")
iree_cc_library(NAME requirements DEPS ::boundary ::cycle_a)
add_executable(probe main.c)
target_link_libraries(probe PRIVATE iree_fixture_branch iree_fixture_requirements)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-$<CONFIG>.txt" CONTENT "$<TARGET_FILE:probe>")
add_executable(unified_probe main.c)
target_link_libraries(unified_probe PRIVATE iree_fixture_unified iree_fixture_requirements)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/unified_probe-$<CONFIG>.txt" CONTENT "$<TARGET_FILE:unified_probe>")

if(FIXTURE_SCOPED)
  add_subdirectory(scoped)
  target_link_libraries(probe PRIVATE iree_fixture_scoped)
  target_compile_definitions(probe PRIVATE REQUIRE_SCOPED=1)
endif()
if(FIXTURE_CUSTOM)
  if(NOT FIXTURE_SCOPED)
    set_property(TARGET probe PROPERTY TRANSITIVE_COMPILE_PROPERTIES TOKEN)
  endif()
  set_property(TARGET iree_fixture_leaf PROPERTY INTERFACE_TOKEN 29)
  target_compile_definitions(probe PRIVATE "CUSTOM_VALUE=$<TARGET_PROPERTY:probe,TOKEN>")
endif()

# This arrives after the library declarations, like parent-directory edits.
function(late_requirements)
  target_compile_definitions(iree_fixture_leaf INTERFACE LATE_VALUE=23)
endfunction()
cmake_language(DEFER CALL late_requirements)
if(FIXTURE_RESOLVE)
  cmake_language(DEFER CALL iree_finalize_compile_usage)
endif()
