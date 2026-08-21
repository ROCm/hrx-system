# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Missing CMake Variables
#-------------------------------------------------------------------------------

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(IREE_HOST_SCRIPT_EXT "bat")
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(IREE_HOST_EXECUTABLE_SUFFIX ".exe")
else()
  set(IREE_HOST_SCRIPT_EXT "sh")
  set(IREE_HOST_EXECUTABLE_SUFFIX "")
endif()

#-------------------------------------------------------------------------------
# IREE_ARCH: identifies the target CPU architecture. May be empty when this is
# ill-defined, such as multi-architecture builds.
# This should be kept consistent with the C preprocessor token IREE_ARCH defined
# in target_platform.h.
#-------------------------------------------------------------------------------

# First, get the raw CMake architecture name, not yet normalized. Even that is
# non-trivial: it usually is CMAKE_SYSTEM_PROCESSOR, but on some platforms, we
# have to read other variables instead.
if(CMAKE_OSX_ARCHITECTURES)
  # Borrowing from:
  # https://boringssl.googlesource.com/boringssl/+/c5f0e58e653d2d9afa8facc090ce09f8aaa3fa0d/CMakeLists.txt#43
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L40
  list(LENGTH CMAKE_OSX_ARCHITECTURES NUM_ARCHES)
  if(${NUM_ARCHES} EQUAL 1)
    # Only one arch in CMAKE_OSX_ARCHITECTURES, use that.
    set(_IREE_UNNORMALIZED_ARCH "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  # Leaving _IREE_UNNORMALIZED_ARCH empty disables arch code paths. We will
  # issue a performance warning about that below.
elseif(CMAKE_GENERATOR MATCHES "^Visual Studio " AND CMAKE_GENERATOR_PLATFORM)
  # Borrowing from:
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L50
  set(_IREE_UNNORMALIZED_ARCH "${CMAKE_GENERATOR_PLATFORM}")
else()
  set(_IREE_UNNORMALIZED_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

string(TOLOWER "${_IREE_UNNORMALIZED_ARCH}" _IREE_UNNORMALIZED_ARCH_LOWERCASE)

# Normalize _IREE_UNNORMALIZED_ARCH into IREE_ARCH.
if(EMSCRIPTEN)
  # TODO: figure what to do about the wasm target, which masquerades as x86.
  # This is the one case where the IREE_ARCH CMake variable is currently
  # inconsistent with the IREE_ARCH C preprocessor token.
  set(IREE_ARCH "")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "wasm32")
  set(IREE_ARCH "wasm_32")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "wasm64")
  set(IREE_ARCH "wasm_64")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "aarch64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64e") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64ec"))
  set(IREE_ARCH "arm_64")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^armv[5-8]"))
  set(IREE_ARCH "arm_32")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86_64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "amd64") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x64"))
  set(IREE_ARCH "x86_64")
elseif((_IREE_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^i[3-7]86$") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86") OR
        (_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "win32"))
  set(IREE_ARCH "x86_32")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv64")
  set(IREE_ARCH "riscv_64")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv32")
  set(IREE_ARCH "riscv_32")
elseif(_IREE_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "")
  set(IREE_ARCH "")
  message(WARNING "Performance advisory: architecture-specific code paths "
    "disabled because no target architecture was specified or we didn't know "
    "which CMake variable to read. Some relevant CMake variables:\n"
    "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}\n"
    "CMAKE_GENERATOR=${CMAKE_GENERATOR}\n"
    "CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}\n"
    "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}\n"
    )
else()
  set(IREE_ARCH "")
  message(SEND_ERROR "Unrecognized target architecture ${_IREE_UNNORMALIZED_ARCH_LOWERCASE}")
endif()

#-------------------------------------------------------------------------------
# General utilities
#-------------------------------------------------------------------------------

if(NOT TARGET iree_generated_compile_inputs)
  add_custom_target(iree_generated_compile_inputs)
endif()
set_property(GLOBAL PROPERTY IREE_TARGET_DEPENDENCIES "")

# Records a build dependency to resolve after all repository targets have been
# declared. CMake subdirectory traversal does not encode the Bazel package
# dependency graph, so producers and consumers may be declared in either order.
#
# Parameters:
#   TARGET: CMake target that should receive the dependency.
#   DEPENDENCY: CMake target required before TARGET is complete.
function(iree_register_target_dependency)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;DEPENDENCY"
    ""
    ${ARGN}
  )

  if(NOT _RULE_TARGET OR NOT _RULE_DEPENDENCY)
    message(FATAL_ERROR
      "iree_register_target_dependency requires TARGET and DEPENDENCY")
  endif()

  set_property(GLOBAL APPEND PROPERTY IREE_TARGET_DEPENDENCIES
    "${_RULE_TARGET}|${_RULE_DEPENDENCY}")
endfunction()

function(_iree_resolve_target OUTPUT_TARGET_NAME TARGET_NAME)
  set(_TARGET_NAME "${TARGET_NAME}")
  if(TARGET "${_TARGET_NAME}")
    get_target_property(_ALIASED_TARGET "${_TARGET_NAME}" ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_TARGET_NAME "${_ALIASED_TARGET}")
    endif()
  elseif("${_TARGET_NAME}" MATCHES "::")
    string(REPLACE "::" "_" _CANDIDATE_TARGET_NAME "${_TARGET_NAME}")
    if(TARGET "${_CANDIDATE_TARGET_NAME}")
      set(_TARGET_NAME "${_CANDIDATE_TARGET_NAME}")
    endif()
  endif()

  if(NOT TARGET "${_TARGET_NAME}")
    set(_TARGET_NAME "")
  endif()
  set(${OUTPUT_TARGET_NAME} "${_TARGET_NAME}" PARENT_SCOPE)
endfunction()

function(iree_finalize_target_dependencies)
  get_property(_TARGET_DEPENDENCIES GLOBAL PROPERTY IREE_TARGET_DEPENDENCIES)
  list(REMOVE_DUPLICATES _TARGET_DEPENDENCIES)
  foreach(_ENTRY IN LISTS _TARGET_DEPENDENCIES)
    if(NOT _ENTRY MATCHES "^([^|]+)[|](.+)$")
      message(FATAL_ERROR
        "IREE target dependency entry is malformed: ${_ENTRY}")
    endif()
    set(_UNRESOLVED_TARGET_NAME "${CMAKE_MATCH_1}")
    set(_UNRESOLVED_DEPENDENCY_TARGET_NAME "${CMAKE_MATCH_2}")
    _iree_resolve_target(_TARGET_NAME "${_UNRESOLVED_TARGET_NAME}")
    if(NOT _TARGET_NAME)
      message(FATAL_ERROR
        "IREE target dependency consumer does not exist: "
        "${_UNRESOLVED_TARGET_NAME}")
    endif()
    _iree_resolve_target(
      _DEPENDENCY_TARGET_NAME
      "${_UNRESOLVED_DEPENDENCY_TARGET_NAME}"
    )
    if(NOT _DEPENDENCY_TARGET_NAME)
      message(FATAL_ERROR
        "IREE target ${_UNRESOLVED_TARGET_NAME} depends on missing target: "
        "${_UNRESOLVED_DEPENDENCY_TARGET_NAME}")
    endif()
    get_target_property(_DEPENDENCY_IMPORTED
      "${_DEPENDENCY_TARGET_NAME}"
      IMPORTED)
    if(NOT _DEPENDENCY_IMPORTED)
      add_dependencies("${_TARGET_NAME}" "${_DEPENDENCY_TARGET_NAME}")
    endif()
  endforeach()
endfunction()

# Connects a generated output to a target that consumes it.
#
# Producers and consumers may be declared in either order because CMake
# subdirectory traversal does not encode the Bazel package dependency graph.
function(iree_generated_output_add_consumer INPUT_PATH CONSUMER_TARGET)
  if(NOT TARGET "${CONSUMER_TARGET}")
    message(FATAL_ERROR
      "Generated output consumer ${CONSUMER_TARGET} was not found")
  endif()
  if("${INPUT_PATH}" MATCHES "^\\$<")
    return()
  endif()
  if(NOT IS_ABSOLUTE "${INPUT_PATH}" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_PATH}")
    return()
  endif()

  get_filename_component(_INPUT_PATH "${INPUT_PATH}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  cmake_path(NORMAL_PATH _INPUT_PATH)
  string(FIND "${_INPUT_PATH}/" "${IREE_BINARY_DIR}/" _BINARY_PATH_INDEX)
  if(NOT _BINARY_PATH_INDEX EQUAL 0 AND EXISTS "${_INPUT_PATH}")
    return()
  endif()

  string(SHA256 _INPUT_KEY "${_INPUT_PATH}")
  get_property(_PRODUCER_TARGET GLOBAL
    PROPERTY "IREE_GENERATED_OUTPUT_PRODUCER_${_INPUT_KEY}")
  if(_PRODUCER_TARGET)
    if(NOT "${_PRODUCER_TARGET}" STREQUAL "${CONSUMER_TARGET}")
      add_dependencies("${CONSUMER_TARGET}" "${_PRODUCER_TARGET}")
    endif()
  else()
    set_property(GLOBAL APPEND
      PROPERTY "IREE_GENERATED_OUTPUT_CONSUMERS_${_INPUT_KEY}"
      "${CONSUMER_TARGET}")
  endif()
endfunction()

# Registers the paths a target generates for cross-target dependency edges.
function(iree_register_generated_output_producer TARGET_NAME)
  cmake_parse_arguments(_RULE "" "" "OUTPUTS" ${ARGN})
  if(NOT TARGET "${TARGET_NAME}")
    message(FATAL_ERROR
      "Generated output producer ${TARGET_NAME} was not found")
  endif()
  foreach(_OUTPUT IN LISTS _RULE_OUTPUTS)
    get_filename_component(_OUTPUT_PATH "${_OUTPUT}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    cmake_path(NORMAL_PATH _OUTPUT_PATH)
    string(SHA256 _OUTPUT_KEY "${_OUTPUT_PATH}")
    get_property(_EXISTING_PRODUCER_TARGET GLOBAL
      PROPERTY "IREE_GENERATED_OUTPUT_PRODUCER_${_OUTPUT_KEY}")
    if(_EXISTING_PRODUCER_TARGET AND
       NOT "${_EXISTING_PRODUCER_TARGET}" STREQUAL "${TARGET_NAME}")
      message(FATAL_ERROR
        "Generated output ${_OUTPUT_PATH} has multiple producers: "
        "${_EXISTING_PRODUCER_TARGET} and ${TARGET_NAME}")
    endif()
    set_property(GLOBAL
      PROPERTY "IREE_GENERATED_OUTPUT_PRODUCER_${_OUTPUT_KEY}"
      "${TARGET_NAME}")

    get_property(_CONSUMER_TARGETS GLOBAL
      PROPERTY "IREE_GENERATED_OUTPUT_CONSUMERS_${_OUTPUT_KEY}")
    foreach(_CONSUMER_TARGET IN LISTS _CONSUMER_TARGETS)
      if(NOT "${TARGET_NAME}" STREQUAL "${_CONSUMER_TARGET}")
        add_dependencies("${_CONSUMER_TARGET}" "${TARGET_NAME}")
      endif()
    endforeach()
    set_property(GLOBAL
      PROPERTY "IREE_GENERATED_OUTPUT_CONSUMERS_${_OUTPUT_KEY}" "")
  endforeach()
endfunction()

# Registers a target that materializes generated C/C++ compile inputs.
#
# Compile-database consumers such as clang-tidy read compile_commands.json
# without asking CMake to build the source target first. Generated headers and
# sources therefore need a stable aggregate target that prepares the filesystem
# view before external analysis starts.
function(iree_register_generated_compile_input TARGET_NAME)
  cmake_parse_arguments(_RULE "" "" "OUTPUTS" ${ARGN})
  if(NOT TARGET "${TARGET_NAME}")
    message(FATAL_ERROR
      "Generated compile input target ${TARGET_NAME} was not found")
  endif()
  add_dependencies(iree_generated_compile_inputs "${TARGET_NAME}")
  iree_register_generated_output_producer("${TARGET_NAME}"
    OUTPUTS ${_RULE_OUTPUTS}
  )
endfunction()

# iree_to_bool
#
# Sets `variable` to `ON` if `value` is true and `OFF` otherwise.
function(iree_to_bool VARIABLE VALUE)
  if(VALUE)
    set(${VARIABLE} "ON" PARENT_SCOPE)
  else()
    set(${VARIABLE} "OFF" PARENT_SCOPE)
  endif()
endfunction()

# iree_append_list_to_string
#
# Joins ${ARGN} together as a string separated by " " and appends it to
# ${VARIABLE}.
function(iree_append_list_to_string VARIABLE)
  if(NOT "${ARGN}" STREQUAL "")
    string(JOIN " " _ARGN_STR ${ARGN})
    set(${VARIABLE} "${${VARIABLE}} ${_ARGN_STR}" PARENT_SCOPE)
  endif()
endfunction()


#-------------------------------------------------------------------------------
# Packages and Paths
#-------------------------------------------------------------------------------

# Performs a variety of setup tasks for a directory that forms the root
# of a C source tree. Many of our build macros use directory-scoped CMake
# variables to drive default behavior, and this sets those up in a consolidated
# way.
# Arguments:
# DEFAULT_EXPORT_SET:
#   The export set to use by default. If linking any static libraries
#   that are to be installed as dev libraries, then this is mandatory
#   (or each target must specify it). Typically this must be set in
#   runtime directories.
# DEFAULT_INSTALL_COMPONENT:
#   The default install component for libraries.
# IMPLICIT_DEFS_TARGET:
#   A new target to create and export/install which includes the current
#   source and binary directory as an include dir. It will be added to
#   every cc library created after this in this directory tree.
# IMPLICIT_DEFS_INSTALL_COMPONENT:
#   Install component for *just* the implicit defs target.
# IMPLICIT_DEFS_EXPORT_SET:
#   Export set for *just* the implicit defs target.
# PACKAGE_ROOT_PREFIX:
#   Explicitly set the package root prefix (as something like "iree::foobar").
#   Default is empty.
# See runtime/src/CMakeLists.txt for typical usage in a directory tree that
# installs static dev libraries.
function(iree_setup_c_src_root)
  cmake_parse_arguments(
    _RULE
    ""
    "PACKAGE_ROOT_PREFIX;DEFAULT_EXPORT_SET;DEFAULT_INSTALL_COMPONENT;IMPLICIT_DEFS_TARGET;IMPLICIT_DEFS_INSTALL_COMPONENT;IMPLICIT_DEFS_EXPORT_SET"
    ""
    ${ARGN}
  )

  # Make C++ library package names start here unless if told not to.
  set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)
  set(IREE_PACKAGE_ROOT_PREFIX "${_RULE_PACKAGE_ROOT_PREFIX}" PARENT_SCOPE)

  # Instruct install support that headers are installable from this root
  # directory.
  set(IREE_HDRS_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)

  # Export and install by default.
  if(_RULE_DEFAULT_EXPORT_SET)
    set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET
      "${_RULE_DEFAULT_EXPORT_SET}"
      PARENT_SCOPE)
  endif()
  if(_RULE_DEFAULT_INSTALL_COMPONENT)
    set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT
      "${_RULE_DEFAULT_INSTALL_COMPONENT}"
      PARENT_SCOPE)
  endif()

  # Create an implicit defs target that adds this include directory.
  if(_RULE_IMPLICIT_DEFS_TARGET)
    add_library(${_RULE_IMPLICIT_DEFS_TARGET} INTERFACE)
    target_include_directories(${_RULE_IMPLICIT_DEFS_TARGET}
      INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
    if(NOT _RULE_IMPLICIT_DEFS_INSTALL_COMPONENT)
      set(_RULE_IMPLICIT_DEFS_INSTALL_COMPONENT ${_RULE_DEFAULT_INSTALL_COMPONENT})
    endif()
    if(NOT _RULE_IMPLICIT_DEFS_EXPORT_SET)
      set(_RULE_IMPLICIT_DEFS_EXPORT_SET ${_RULE_DEFAULT_EXPORT_SET})
    endif()
    iree_install_targets(
      TARGETS ${_RULE_IMPLICIT_DEFS_TARGET}
      COMPONENT "${_RULE_IMPLICIT_DEFS_INSTALL_COMPONENT}"
      EXPORT_SET "${_RULE_IMPLICIT_DEFS_EXPORT_SET}"
    )

    # Include this target in all cc_libraries.
    set(IREE_IMPLICIT_DEFS_CC_DEPS ${_RULE_IMPLICIT_DEFS_TARGET} PARENT_SCOPE)
  endif()
endfunction()

# Sets ${PACKAGE_NS} to the IREE-root relative package name in C++ namespace
# format (::).
#
# Examples:
#   runtime/src/iree/base/CMakeLists.txt -> iree::base
#   libhrx/src/libhrx/CMakeLists.txt -> libhrx::src::libhrx
#   tests/e2e/CMakeLists.txt -> iree::tests::e2e
function(iree_package_ns PACKAGE_NS)
  if(DEFINED IREE_PACKAGE_ROOT_DIR)
    # If an enclosing package root dir is set, then the package is just the
    # relative part after that.
    cmake_path(RELATIVE_PATH CMAKE_CURRENT_LIST_DIR
      BASE_DIRECTORY "${IREE_PACKAGE_ROOT_DIR}"
      OUTPUT_VARIABLE _PACKAGE)
    if(_PACKAGE STREQUAL ".")
      set(_PACKAGE "")
    endif()
    if(IREE_PACKAGE_ROOT_PREFIX)
      if("${_PACKAGE}" STREQUAL "")
        set(_PACKAGE "${IREE_PACKAGE_ROOT_PREFIX}")
      else()
        set(_PACKAGE "${IREE_PACKAGE_ROOT_PREFIX}/${_PACKAGE}")
      endif()
    endif()
  else()
    # Get the relative path of the current dir (i.e. runtime/src/iree/vm).
    string(REPLACE ${IREE_ROOT_DIR} "" _IREE_RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR})
    string(SUBSTRING ${_IREE_RELATIVE_PATH} 1 -1 _IREE_RELATIVE_PATH)

    if(NOT ${CMAKE_CURRENT_LIST_DIR} MATCHES "^${IREE_ROOT_DIR}/.*")
      # Function is being called from outside IREE. Use the source-relative path.
      # Please check the README.md to see the potential risk.
      string(REPLACE ${PROJECT_SOURCE_DIR} "" _SOURCE_RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR})
      string(SUBSTRING ${_SOURCE_RELATIVE_PATH} 1 -1 _SOURCE_RELATIVE_PATH)
      set(_PACKAGE "${_SOURCE_RELATIVE_PATH}")

    # If changing the directory/package mapping rules, please also implement
    # the corresponding rule in:
    #   build_tools/bazel_to_cmake/bazel_to_cmake_targets.py
    # Some sub-trees form their own roots for package purposes. Rewrite them.
    else()
      message(SEND_ERROR "iree_package_ns(): Could not determine package for ${CMAKE_CURRENT_LIST_DIR}")
      set(_PACKAGE "iree/unknown")
    endif()
  endif()

  string(REPLACE "/" "::" _PACKAGE_NS "${_PACKAGE}")

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "iree_package_ns(): map ${CMAKE_CURRENT_LIST_DIR} -> ${_PACKAGE_NS}")
  endif()

  set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the IREE-root relative package name.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree_base
function(iree_package_name PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "_" _PACKAGE_NAME "${_PACKAGE_NS}")
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Resolves a package-relative or aliased target name to the concrete CMake
# target name used by rules that cannot consume aliases directly.
function(iree_package_target_name OUTPUT_TARGET_NAME TARGET_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_TARGET_NAME "${TARGET_NAME}")
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _TARGET_NAME "${_TARGET_NAME}")

  if(TARGET "${_TARGET_NAME}")
    get_target_property(_ALIASED_TARGET "${_TARGET_NAME}" ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_TARGET_NAME "${_ALIASED_TARGET}")
    endif()
  elseif("${_TARGET_NAME}" MATCHES "::")
    string(REPLACE "::" "_" _TARGET_NAME "${_TARGET_NAME}")
  endif()

  set(${OUTPUT_TARGET_NAME} "${_TARGET_NAME}" PARENT_SCOPE)
endfunction()

# Resolves a list of package-relative or aliased target names to concrete CMake
# target names.
function(iree_package_target_names OUTPUT_TARGET_NAMES)
  set(_TARGET_NAMES)
  foreach(_TARGET_NAME ${ARGN})
    iree_package_target_name(_RESOLVED_TARGET_NAME "${_TARGET_NAME}")
    list(APPEND _TARGET_NAMES "${_RESOLVED_TARGET_NAME}")
  endforeach()
  set(${OUTPUT_TARGET_NAMES} "${_TARGET_NAMES}" PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the IREE-root relative package path.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree/base
function(iree_package_path PACKAGE_PATH)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_DIR} to the directory name of the current package.
#
# Example when called from iree/base/CMakeLists.txt:
#   base
function(iree_package_dir PACKAGE_DIR)
  iree_package_ns(_PACKAGE_NS)
  string(FIND "${_PACKAGE_NS}" "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# select()-like Evaluation
#-------------------------------------------------------------------------------

# Appends ${OPTS} with a list of values based on the current compiler.
#
# Example:
#   iree_select_compiler_opts(COPTS
#     CLANG
#       "-Wno-foo"
#       "-Wno-bar"
#     CLANG_CL
#       "/W3"
#     GCC
#       "-Wsome-old-flag"
#     MSVC
#       "/W3"
#   )
#
# Note that variables are allowed, making it possible to share options between
# different compiler targets.
function(iree_select_compiler_opts OPTS)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _IREE_SELECTS
    ""
    ""
    "ALL;CLANG;CLANG_GTE_10;CLANG_GTE_12;CLANG_CL;MSVC;GCC;GCC_GTE_13;CLANG_OR_GCC;MSVC_OR_CLANG_CL"
  )
  # OPTS is a variable containing the *name* of the variable being populated, so
  # we need to dereference it twice.
  set(_OPTS "${${OPTS}}")
  list(APPEND _OPTS "${_IREE_SELECTS_ALL}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND _OPTS "${_IREE_SELECTS_GCC}")
    list(APPEND _OPTS "${_IREE_SELECTS_CLANG_OR_GCC}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
      list(APPEND _OPTS ${_IREE_SELECTS_GCC_GTE_13})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(MSVC)
      list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_CL})
    else()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG})
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        list(APPEND _OPTS ${_IREE_SELECTS_CLANG_GTE_10})
      endif()
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        list(APPEND _OPTS ${_IREE_SELECTS_CLANG_GTE_12})
      endif()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_OR_GCC})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC})
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
  else()
    message(ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER}")
    list(APPEND _OPTS "")
  endif()
  set(${OPTS} ${_OPTS} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Data dependencies
#-------------------------------------------------------------------------------

# Adds 'data' dependencies to a target.
#
# Parameters:
# NAME: name of the target to add data dependencies to
# DATA: List of targets and/or files in the source tree (relative to the
# project root).
# OUT_FILE_DATA: Optional output variable receiving file-shaped DATA entries.
# OUT_TARGET_DATA: Optional output variable receiving target-shaped DATA
#     entries as CMake target names.
function(iree_add_data_dependencies)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT_FILE_DATA;OUT_TARGET_DATA"
    "DATA"
    ${ARGN}
  )

  if(NOT _RULE_DATA)
    if(_RULE_OUT_FILE_DATA)
      set(${_RULE_OUT_FILE_DATA} "" PARENT_SCOPE)
    endif()
    if(_RULE_OUT_TARGET_DATA)
      set(${_RULE_OUT_TARGET_DATA} "" PARENT_SCOPE)
    endif()
    return()
  endif()

  set(_FILE_DATA)
  set(_TARGET_DATA)
  foreach(_DATA_LABEL ${_RULE_DATA})
    set(_DATA_TARGET_NAME "${_DATA_LABEL}")
    if(_DATA_TARGET_NAME MATCHES "^::")
      iree_package_ns(_DATA_PACKAGE_NS)
      string(REGEX REPLACE "^::" "${_DATA_PACKAGE_NS}::"
             _DATA_TARGET_NAME "${_DATA_TARGET_NAME}")
    endif()

    if(TARGET "${_DATA_TARGET_NAME}" OR
       "${_DATA_TARGET_NAME}" MATCHES "::")
      list(APPEND _TARGET_DATA "${_DATA_TARGET_NAME}")
      iree_register_target_dependency(
        TARGET "${_RULE_NAME}"
        DEPENDENCY "${_DATA_TARGET_NAME}"
      )
      continue()
    endif()

    if(IS_ABSOLUTE "${_DATA_LABEL}")
      list(APPEND _FILE_DATA "${_DATA_LABEL}")
      iree_generated_output_add_consumer(
        "${_DATA_LABEL}"
        "${_RULE_NAME}"
      )
      continue()
    endif()

    # Some Bazel data edges refer to generated files in the current package
    # rather than source-tree files. The CMake custom commands that produce
    # those files typically expose a package-prefixed target named after the
    # output stem (for example, data `foo.so` produced by target
    # `iree_package_foo`). If such a target exists in this directory, depend
    # on it and leave the file in its generated location.
    get_filename_component(_DATA_STEM "${_DATA_LABEL}" NAME_WE)
    get_property(_DATA_DIR_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    set(_DATA_GENERATED_TARGET)
    foreach(_DATA_DIR_TARGET ${_DATA_DIR_TARGETS})
      if(_DATA_DIR_TARGET MATCHES "_${_DATA_STEM}$")
        set(_DATA_GENERATED_TARGET ${_DATA_DIR_TARGET})
        break()
      endif()
    endforeach()
    if(_DATA_GENERATED_TARGET)
      list(APPEND _FILE_DATA "${_DATA_LABEL}")
      iree_register_target_dependency(
        TARGET "${_RULE_NAME}"
        DEPENDENCY "${_DATA_GENERATED_TARGET}"
      )
      continue()
    endif()

    # Not a target, assume to be a file instead.
    list(APPEND _FILE_DATA "${_DATA_LABEL}")
    set(_FILE_PATH ${_DATA_LABEL})

    # Create a target which copies the data file into the build directory.
    # If this file is included in multiple rules, only create the target once.
    string(REPLACE "::" "_" _DATA_TARGET ${_DATA_LABEL})
    string(REPLACE "/" "_" _DATA_TARGET ${_DATA_TARGET})
    if(NOT TARGET ${_DATA_TARGET})
      set(_INPUT_PATH "${PROJECT_SOURCE_DIR}/${_FILE_PATH}")
      set(_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${_FILE_PATH}")
      add_custom_target(${_DATA_TARGET}
        COMMAND ${CMAKE_COMMAND} -E copy ${_INPUT_PATH} ${_OUTPUT_PATH}
      )
    endif()

    iree_register_target_dependency(
      TARGET "${_RULE_NAME}"
      DEPENDENCY "${_DATA_TARGET}"
    )
  endforeach()

  if(_RULE_OUT_FILE_DATA)
    set(${_RULE_OUT_FILE_DATA} "${_FILE_DATA}" PARENT_SCOPE)
  endif()
  if(_RULE_OUT_TARGET_DATA)
    set(${_RULE_OUT_TARGET_DATA} "${_TARGET_DATA}" PARENT_SCOPE)
  endif()
endfunction()

#-------------------------------------------------------------------------------
# iree_make_empty_file
#-------------------------------------------------------------------------------

# Creates an empty file by copying an in-tree empty file. Unlike `file(WRITE)`
# or `file(TOUCH)`, this does not update the timestamp every time CMake is run,
# avoiding unnecessary rebuilds when the empty file is used as a rule input.
function(iree_make_empty_file _FILENAME)
  configure_file("${PROJECT_SOURCE_DIR}/build_tools/cmake/empty_file" "${_FILENAME}" COPYONLY)
endfunction()

#-------------------------------------------------------------------------------
# Emscripten
#-------------------------------------------------------------------------------

# A global counter to guarantee unique names for js library files.
set(_LINK_JS_COUNTER 1)

# Links a JavaScript library to a target using --js-library=file.js.
#
# This function is only supported when running under Emscripten (emcmake).
# This implementation is forked from `em_add_tracked_link_flag()` in
# https://github.com/emscripten-core/emscripten/blob/main/cmake/Modules/Platform/Emscripten.cmake
# with changes to be compatible with IREE project style and CMake conventions.
#
# Parameters:
# TARGET: Name of the target to link against
# SRCS: List of JavaScript source files to link
function(iree_link_js_library)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET"
    "SRCS"
    ${ARGN}
  )

  # Convert from aliased, possibly package-relative, names to target names.
  iree_package_ns(_PACKAGE_NS)
  string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _RULE_TARGET ${_RULE_TARGET})
  string(REPLACE "::" "_" _RULE_TARGET ${_RULE_TARGET})

  foreach(_SRC_FILE ${_RULE_SRCS})
    # If the JS file is changed, we want to relink dependent binaries, but
    # unfortunately it is not possible to make a link step depend directly on a
    # source file. Instead, we must make a dummy no-op build target on that
    # source file, and make the original target depend on that dummy target.

    # Sanitate the source .js filename to a good dummy filename.
    get_filename_component(_JS_NAME "${_SRC_FILE}" NAME)
    string(REGEX REPLACE "[/:\\\\.\ ]" "_" _DUMMY_JS_TARGET ${_JS_NAME})
    set(_DUMMY_LIB_NAME ${_RULE_TARGET}_${_LINK_JS_COUNTER}_${_DUMMY_JS_TARGET})
    set(_DUMMY_C_NAME "${CMAKE_BINARY_DIR}/${_DUMMY_JS_TARGET}_tracker.c")

    # Create a new static library target that with a single dummy .c file.
    add_library(${_DUMMY_LIB_NAME} STATIC ${_DUMMY_C_NAME})
    # Make the dummy .c file depend on the .js file we are linking, so that if
    # the .js file is edited, the dummy .c file, and hence the static library
    # will be rebuild (no-op). This causes the main application to be
    # relinked, which is what we want. This approach was recommended by
    # http://www.cmake.org/pipermail/cmake/2010-May/037206.html
    add_custom_command(
      OUTPUT ${_DUMMY_C_NAME}
      COMMAND ${CMAKE_COMMAND} -E touch ${_DUMMY_C_NAME}
      DEPENDS ${_SRC_FILE}
    )
    target_link_libraries(${_RULE_TARGET}
      PUBLIC
        ${_DUMMY_LIB_NAME}
    )

    iree_install_targets(
      TARGETS ${_DUMMY_LIB_NAME}
    )

    # Link the js-library to the target.
    # When a linked library starts with a "-" cmake will just add it to the
    # linker command line as it is. The advantage of doing it this way is
    # that the js-library will also be automatically linked to targets that
    # depend on this target.
    get_filename_component(_SRC_ABSOLUTE_PATH "${_SRC_FILE}" ABSOLUTE)
    target_link_libraries(${_RULE_TARGET}
      PUBLIC
        "--js-library \"${_SRC_ABSOLUTE_PATH}\""
    )

    math(EXPR _LINK_JS_COUNTER "${_LINK_JS_COUNTER} + 1")
  endforeach()
endfunction()

#-------------------------------------------------------------------------------
# Tool symlinks
#-------------------------------------------------------------------------------

# iree_symlink_tool
#
# Adds a command to TARGET which aliases a tool from elsewhere
# (FROM_TOOL_TARGET_NAME) to a local file name (TO_EXE_NAME) in the current
# binary directory. Windows uses a copy because file symlinks require
# Developer Mode or elevated privileges.
#
# Parameters:
#   TARGET: Local target to which to add the symlink command (i.e. an
#     iree_py_library, etc).
#   FROM_TOOL_TARGET: Target of the tool executable that is the source of the
#     link.
#   TO_EXE_NAME: The executable name to output in the current binary dir.
function(iree_symlink_tool)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;FROM_TOOL_TARGET;TO_EXE_NAME"
    ""
    ${ARGN}
  )

  # Transform TARGET
  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_TARGET}")
  set(_FROM_TOOL_TARGET ${_RULE_FROM_TOOL_TARGET})
  set(_TO_TOOL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
  get_filename_component(_TO_TOOL_DIR "${_TO_TOOL_PATH}" DIRECTORY)
  if(WIN32)
    set(_TO_TOOL_MATERIALIZATION_COMMAND copy_if_different)
  else()
    set(_TO_TOOL_MATERIALIZATION_COMMAND create_symlink)
  endif()

  add_custom_command(
    TARGET "${_TARGET}"
    POST_BUILD
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory "${_TO_TOOL_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E ${_TO_TOOL_MATERIALIZATION_COMMAND}
        "$<TARGET_FILE:${_FROM_TOOL_TARGET}>"
        "${_TO_TOOL_PATH}"
  )
endfunction()


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

# iree_check_defined
#
# A lightweight way to check that all the given variables are defined. Useful
# in cases like checking that a function has been passed all required arguments.
# Doesn't give usage-specific error messages, but still significantly better
# than no error checking.
# Variable names should be passed directly without quoting or dereferencing.
# Example:
#   iree_check_defined(_SOME_VAR _AND_ANOTHER_VAR)
macro(iree_check_defined)
  foreach(_VAR ${ARGN})
    if(NOT DEFINED "${_VAR}")
      message(SEND_ERROR "${_VAR} is not defined")
    endif()
  endforeach()
endmacro()

# iree_validate_required_arguments
#
# Validates that no arguments went unparsed or were given no values and that all
# required arguments have values. Expects to be called after
# cmake_parse_arguments and verifies that the variables it creates have been
# populated as appropriate.
function(iree_validate_required_arguments
         PREFIX
         REQUIRED_ONE_VALUE_KEYWORDS
         REQUIRED_MULTI_VALUE_KEYWORDS)
  if(DEFINED ${PREFIX}_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unparsed argument(s): '${${PREFIX}_UNPARSED_ARGUMENTS}'")
  endif()
  if(DEFINED ${PREFIX}_KEYWORDS_MISSING_VALUES)
    message(SEND_ERROR
            "No values for field(s) '${${PREFIX}_KEYWORDS_MISSING_VALUES}'")
  endif()

  foreach(_KEYWORD IN LISTS REQUIRED_ONE_VALUE_KEYWORDS REQUIRED_MULTI_VALUE_KEYWORDS)
    if(NOT DEFINED ${PREFIX}_${_KEYWORD})
      message(SEND_ERROR "Missing required argument ${_KEYWORD}")
    endif()
  endforeach()
endfunction()
