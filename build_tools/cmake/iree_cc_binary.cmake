# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/../sanitizer/iree_sanitizer_suppressions.cmake")

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# TESTONLY: for testing; not installed and not compiled when tests are disabled
# HOSTONLY: host only; compile using host toolchain when cross-compiling
# COVERAGE: whether to enable code coverage during compilation/linking.
# SETUP_INSTALL_RPATH: Sets an install RPATH which assumes the standard
#   directory layout (to be used if linking against installed shared libs).
# INSTALL_COMPONENT: CMake install component (Defaults to "IREETool-${_RULE_NAME}").
# Note:
# iree_cc_binary will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# iree_base_foo with an alias to ${PACKAGE_NS}::${NAME}.
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_binary(
#   NAME
#     awesome_tool
#   SRCS
#     "awesome-tool-main.cc"
#   DEPS
#     iree::awesome
# )
function(iree_cc_binary)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL;HOSTONLY;TESTONLY;SETUP_INSTALL_RPATH"
    "NAME;INSTALL_COMPONENT;COVERAGE"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  if("${_PACKAGE_NAME}" STREQUAL "")
    set(_NAME "${_RULE_NAME}")
  else()
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  endif()

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  : iree_cc_binary(${_NAME})")
  endif()

  add_executable(${_NAME} "")

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    # Alias the iree_package_name binary to iree::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    if(_DEBUG_IREE_PACKAGE_NAME)
      message(STATUS "  + alias ${_PACKAGE_NS}::${_RULE_NAME}")
    endif()
    iree_add_alias_executable(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

    # If the binary name matches the package then treat it as a default. For
    # example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
    # likely to be common for binaries, but is consistent with the behavior for
    # libraries and in Bazel.
    iree_package_dir(_PACKAGE_DIR)
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      iree_add_alias_executable(${_PACKAGE_NS} ${_NAME})
    endif()
  endif()

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  if(_RULE_SRCS)
    iree_resolve_source_files(_RULE_SRCS ${_RULE_SRCS})
    iree_set_source_object_names(${_RULE_SRCS})
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
    foreach(_SOURCE IN LISTS _RULE_SRCS)
      iree_generated_output_add_consumer("${_SOURCE}" "${_NAME}")
    endforeach()
  else()
    iree_add_empty_object(${_NAME})
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_COPTS}
      ${IREE_INTERFACE_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  # Enable coverage if requested.
  if(${_RULE_COVERAGE})
    target_compile_options(${_NAME}
      PRIVATE
        "-fprofile-instr-generate"
        "-fcoverage-mapping"
    )
    target_link_options(${_NAME}
      PRIVATE
        "-fprofile-instr-generate"
        "-fcoverage-mapping"
    )
    set_property(GLOBAL APPEND PROPERTY IREE_RUNTIME_COVERAGE_TARGETS "${_NAME}=$<TARGET_FILE:${_NAME}>")
  endif()

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
      ${IREE_DEFAULT_LINK_LIBRARIES}
      ${IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA}
    OUT_TARGET_DATA _DATA_TARGETS)
  # Retain the declared link graph for final shared-library composition. The
  # generated LINK_LIBRARIES also contains object and usage-only expressions.
  set_property(TARGET ${_NAME} PROPERTY IREE_CC_DEPS "${_RULE_DEPS};${IREE_DEFAULT_LINK_LIBRARIES}")
  set_property(TARGET ${_NAME} PROPERTY IREE_CC_LINKOPTS "${_RULE_LINKOPTS}")
  iree_target_sanitizer_suppressions("${_NAME}"
    DEPS ${_RULE_DEPS} ${_DATA_TARGETS})

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  if(_RULE_EXCLUDE_FROM_ALL)
    set_property(TARGET ${_NAME} PROPERTY EXCLUDE_FROM_ALL ON)
  endif()
  if(NOT _RULE_TESTONLY)
    set(_INSTALL_COMPONENT "${_RULE_INSTALL_COMPONENT}")
    if(NOT _INSTALL_COMPONENT)
      set(_INSTALL_COMPONENT "IREETool-${_RULE_NAME}")
    endif()
    set(_INSTALL_OPTIONS)
    if(_RULE_EXCLUDE_FROM_ALL)
      list(APPEND _INSTALL_OPTIONS EXCLUDE_FROM_ALL)
    endif()
    install(TARGETS ${_NAME}
      RENAME ${_RULE_NAME}
      COMPONENT ${_INSTALL_COMPONENT}
      RUNTIME DESTINATION bin
      BUNDLE DESTINATION bin
      ${_INSTALL_OPTIONS})
  endif()

  # Setup RPATH if on a Unix-like system. We have two use cases that we are
  # handling here:
  #   1. Install tree layouts like bin/ and lib/ directories that are
  #      peers.
  #   2. Single directory bundles (language bindings do this) where the
  #      shared library is placed next to the consumer.
  #
  # The common solution is to use an RPATH of the origin and the
  # lib/ directory that may be a peer of the origin. Distributions
  # outside of this setup will need to do their own manipulation.
  if(_RULE_SETUP_INSTALL_RPATH)
    if(APPLE OR UNIX)
      set(_origin_prefix "\$ORIGIN")
      if(APPLE)
        set(_origin_prefix "@loader_path")
      endif()
      # See: https://cmake.org/cmake/help/latest/module/GNUInstallDirs.html
      # Assume relative path as a sibling of the lib dir.
      set(_lib_dir "${CMAKE_INSTALL_LIBDIR}")
      if (NOT _lib_dir)
        set(_lib_dir "lib")
      endif()
      set(_install_rpath "${_origin_prefix}" "${_origin_prefix}/../${_lib_dir}")
      if(_lib_dir)
        cmake_path(IS_ABSOLUTE _lib_dir _is_abs_libdir)
        if(_is_abs_libdir)
          # Use the libdir verbatim.
          set(_install_rpath "${_origin_prefix}" "${_lib_dir}")
        endif()
      endif()
      set_target_properties(${_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH OFF
        INSTALL_RPATH "${_install_rpath}"
      )
    endif()
  endif()

  # Set up Info.plist properties when building macOS/iOS app bundles.
  get_target_property(APPLE_BUNDLE ${_NAME} MACOSX_BUNDLE)
  if (APPLE_BUNDLE)
    set_target_properties(${_NAME} PROPERTIES
      MACOSX_BUNDLE_BUNDLE_NAME "${_RULE_NAME}"
      MACOSX_BUNDLE_GUI_IDENTIFIER "dev.iree.${_RULE_NAME}"
      MACOSX_BUNDLE_COPYRIGHT "Copyright © 2023 The IREE Authors"
      # These are just placeholder version numbers until we define proper
      # version scheme and support.
      MACOSX_BUNDLE_BUNDLE_VERSION 0.1
      MACOSX_BUNDLE_SHORT_VERSION_STRING 0.1
      MACOSX_BUNDLE_LONG_VERSION_STRING 0.1)
  endif()
endfunction()
