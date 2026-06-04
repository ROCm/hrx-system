# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

set(_IREE_DEPENDENCY_MODES pinned package auto)

function(iree_dependency_mode out_var)
  if(NOT DEFINED IREE_DEPENDENCY_MODE OR "${IREE_DEPENDENCY_MODE}" STREQUAL "")
    set(IREE_DEPENDENCY_MODE "pinned" CACHE STRING
      "How source dependencies are resolved: pinned, package, or auto." FORCE)
    set_property(CACHE IREE_DEPENDENCY_MODE PROPERTY STRINGS
      ${_IREE_DEPENDENCY_MODES})
  endif()

  string(TOLOWER "${IREE_DEPENDENCY_MODE}" _mode)
  if(NOT _mode IN_LIST _IREE_DEPENDENCY_MODES)
    message(FATAL_ERROR
      "IREE_DEPENDENCY_MODE must be one of pinned, package, or auto; got "
      "'${IREE_DEPENDENCY_MODE}'")
  endif()

  set(${out_var} "${_mode}" PARENT_SCOPE)
endfunction()

function(iree_dependency_package_discovery_allowed out_var)
  iree_dependency_mode(_mode)
  if(_mode STREQUAL "package" OR _mode STREQUAL "auto")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(iree_dependency_pinned_source_allowed out_var)
  iree_dependency_mode(_mode)
  if(_mode STREQUAL "pinned" OR _mode STREQUAL "auto")
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(iree_dependency_require_pinned_source_allowed dep_name)
  iree_dependency_pinned_source_allowed(_pinned_source_allowed)
  if(NOT _pinned_source_allowed)
    iree_dependency_mode(_mode)
    message(FATAL_ERROR
      "${dep_name} was not found through package discovery and "
      "IREE_DEPENDENCY_MODE=${_mode} forbids pinned FetchContent sources. "
      "Provide the package through CMAKE_PREFIX_PATH or use "
      "IREE_DEPENDENCY_MODE=pinned or auto.")
  endif()
endfunction()

function(_iree_dependency_variable_name out_var dep_name property_name)
  string(TOUPPER "${dep_name}" _dependency_identifier)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _dependency_identifier
    "${_dependency_identifier}")
  set(${out_var} "IREE_DEP_${_dependency_identifier}_${property_name}"
    PARENT_SCOPE)
endfunction()

function(iree_get_locked_dependency_property out_var dep_name property_name)
  _iree_dependency_variable_name(_variable_name "${dep_name}" "${property_name}")
  if(NOT DEFINED ${_variable_name})
    message(FATAL_ERROR
      "${dep_name} is missing ${property_name} in MODULE.cmake.lock. "
      "Regenerate the lock with build_tools/bazel_to_cmake/deps.py.")
  endif()
  set(${out_var} "${${_variable_name}}" PARENT_SCOPE)
endfunction()

function(iree_declare_locked_fetch_content dep_name)
  cmake_parse_arguments(PARSE_ARGV 1 _ARGS "" "FETCH_NAME" "")
  if(_ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "iree_declare_locked_fetch_content(${dep_name}) received unexpected "
      "arguments: ${_ARGS_UNPARSED_ARGUMENTS}")
  endif()
  if(_ARGS_FETCH_NAME)
    set(_fetch_name "${_ARGS_FETCH_NAME}")
  else()
    set(_fetch_name "${dep_name}")
  endif()

  iree_get_locked_dependency_property(_urls "${dep_name}" "URLS")
  iree_get_locked_dependency_property(_sha256 "${dep_name}" "SHA256")
  if(NOT _urls)
    message(FATAL_ERROR "${dep_name} has no locked source URLs")
  endif()
  if(NOT _sha256)
    message(FATAL_ERROR "${dep_name} has no locked SHA256")
  endif()

  FetchContent_Declare(
    ${_fetch_name}
    URL ${_urls}
    URL_HASH SHA256=${_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
  )
endfunction()

function(iree_populate_locked_fetch_content dep_name out_source_dir)
  iree_declare_locked_fetch_content(${dep_name})
  FetchContent_GetProperties(${dep_name})
  if(NOT ${dep_name}_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(${dep_name})
    if(POLICY CMP0169)
      cmake_policy(POP)
    endif()
  endif()
  set(${out_source_dir} "${${dep_name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

function(iree_add_alias_interface alias_name)
  set(_deps ${ARGN})
  if(TARGET ${alias_name})
    return()
  endif()
  string(MAKE_C_IDENTIFIER "${alias_name}" _target_name)
  set(_target_name "iree_${_target_name}")
  add_library(${_target_name} INTERFACE)
  target_link_libraries(${_target_name} INTERFACE ${_deps})
  add_library(${alias_name} ALIAS ${_target_name})
endfunction()
