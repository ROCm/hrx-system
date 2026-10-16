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

# Expands GitHub download URLs with checksum-identical independent routes.
# Keep these routes synchronized with build_tools/bazel/downloader.cfg.
function(_iree_dependency_expand_download_urls out_var)
  set(_expanded_urls)
  foreach(_url IN LISTS ARGN)
    list(APPEND _expanded_urls "${_url}")

    # GitHub source archives normally redirect to codeload. A direct request
    # avoids routing a second attempt through github.com.
    if(_url MATCHES
       "^https://github\\.com/([^/]+/[^/]+)/archive/(.+)\\.tar\\.gz$")
      list(APPEND _expanded_urls
        "https://codeload.github.com/${CMAKE_MATCH_1}/tar.gz/${CMAKE_MATCH_2}")
    endif()

    # These release assets are byte-identical to their source tag archives.
    if(_url MATCHES
       "^https://github\\.com/(bazel-contrib/rules_python|bazelbuild/rules_cc|bazelbuild/rules_shell|bazelbuild/rules_testing|google/googletest)/releases/download/([^/]+)/[^/]+\\.tar\\.gz$")
      list(APPEND _expanded_urls
        "https://codeload.github.com/${CMAKE_MATCH_1}/tar.gz/refs/tags/${CMAKE_MATCH_2}")
    endif()

    # Bazel's public mirror carries these exact release archives.
    if(_url MATCHES
       "^https://(github\\.com/(bazelbuild/(bazel-skylib|platforms|rules_shell|rules_testing)|google/googletest|facebook/zstd)/releases/download/[^/]+/[^/]+\\.tar\\.gz)$")
      list(APPEND _expanded_urls
        "https://mirror.bazel.build/${CMAKE_MATCH_1}")
    endif()

    # Kernel.org is the independent canonical mirror for Torvalds' Linux tree.
    if(_url MATCHES
       "^https://raw\\.githubusercontent\\.com/torvalds/linux/([^/]+)/(.*)$")
      list(APPEND _expanded_urls
        "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/${CMAKE_MATCH_2}?id=${CMAKE_MATCH_1}")
    endif()

    # jsDelivr mirrors the remaining pinned raw-file repositories.
    if(_url MATCHES
       "^https://raw\\.githubusercontent\\.com/(amd/xdna-driver|ROCm/rocm-systems)/([^/]+)/(.*)$")
      list(APPEND _expanded_urls
        "https://cdn.jsdelivr.net/gh/${CMAKE_MATCH_1}@${CMAKE_MATCH_2}/${CMAKE_MATCH_3}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _expanded_urls)
  set(${out_var} "${_expanded_urls}" PARENT_SCOPE)
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

function(_iree_dependency_resolve_patch out_var patch_label)
  if(NOT patch_label MATCHES "^//([^:]*):([^:]+)$")
    message(FATAL_ERROR
      "Locked dependency patch '${patch_label}' is not a root-repository label")
  endif()
  set(_patch_package "${CMAKE_MATCH_1}")
  set(_patch_filename "${CMAKE_MATCH_2}")
  if(_patch_package MATCHES "(^|/)\\.\\.(/|$)" OR
     _patch_filename MATCHES "(^|/)\\.\\.(/|$)")
    message(FATAL_ERROR
      "Locked dependency patch '${patch_label}' escapes the repository root")
  endif()
  set(_patch_path
    "${IREE_ROOT_DIR}/${_patch_package}/${_patch_filename}")
  cmake_path(NORMAL_PATH _patch_path)
  if(NOT EXISTS "${_patch_path}")
    message(FATAL_ERROR
      "Locked dependency patch '${patch_label}' does not exist at "
      "${_patch_path}")
  endif()
  set(${out_var} "${_patch_path}" PARENT_SCOPE)
endfunction()

function(iree_declare_locked_fetch_content dep_name)
  cmake_parse_arguments(PARSE_ARGV 1 _ARGS "" "FETCH_NAME;SOURCE_SUBDIR" "")
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
  iree_get_locked_dependency_property(_patches "${dep_name}" "PATCHES")
  iree_get_locked_dependency_property(_patch_args "${dep_name}" "PATCH_ARGS")
  if(NOT _urls)
    message(FATAL_ERROR "${dep_name} has no locked source URLs")
  endif()
  if(NOT _sha256)
    message(FATAL_ERROR "${dep_name} has no locked SHA256")
  endif()
  _iree_dependency_expand_download_urls(_urls ${_urls})

  set(_source_subdir_args)
  if(_ARGS_SOURCE_SUBDIR)
    list(APPEND _source_subdir_args SOURCE_SUBDIR "${_ARGS_SOURCE_SUBDIR}")
  endif()
  set(_patch_command)
  if(_patches)
    if(NOT _patch_args)
      message(FATAL_ERROR
        "${dep_name} has locked patches without explicit patch arguments")
    endif()
    find_program(_iree_dependency_git_executable NAMES git)
    if(NOT _iree_dependency_git_executable)
      message(FATAL_ERROR
        "${dep_name} has locked patches but Git is not available")
    endif()
    set(_patch_contents)
    set(_patch_fingerprints)
    foreach(_patch IN LISTS _patches)
      _iree_dependency_resolve_patch(_patch_file "${_patch}")
      file(READ "${_patch_file}" _patch_content)
      string(APPEND _patch_contents "${_patch_content}\n")
      file(SHA256 "${_patch_file}" _patch_sha256)
      list(APPEND _patch_fingerprints "${_patch}:${_patch_sha256}")
    endforeach()
    set(_patch_driver "${IREE_ROOT_DIR}/build_tools/cmake/iree_apply_dependency_patches.cmake")
    file(SHA256 "${_patch_driver}" _patch_driver_sha256)
    list(APPEND _patch_fingerprints "${_patch_driver}:${_patch_driver_sha256}")
    # Git checks and commits each patch-file argument independently. One input
    # carries intermediate file states across the ordered stack and makes the
    # complete patch set atomic, including files created by earlier patches.
    set(_patch_file
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_fetch_name}-patch.diff")
    # Keep Git's patch format LF-delimited on every host. Expand the complete
    # value once so CMake syntax within the patch itself remains literal.
    file(CONFIGURE OUTPUT "${_patch_file}" CONTENT "@_patch_contents@"
      @ONLY NEWLINE_STYLE UNIX)
    # FetchContent expands command lists, including semicolons inside quoted
    # -D arguments. Keep list-valued patch parameters in the invoked script.
    set(_patch_parameters
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_fetch_name}-patch.cmake")
    file(CONFIGURE OUTPUT "${_patch_parameters}" CONTENT [=[
set(IREE_PATCH_ARGS [==[@_patch_args@]==])
set(IREE_PATCH_FILE [==[@_patch_file@]==])
include([==[@_patch_driver@]==])
]=] @ONLY)
    file(SHA256 "${_patch_parameters}" _patch_parameters_sha256)
    list(APPEND _patch_fingerprints "parameters:${_patch_parameters_sha256}")
    list(JOIN _patch_fingerprints ";" _patch_fingerprint)
    string(SHA256 _patch_set_sha256 "${_patch_fingerprint}")
    list(APPEND _patch_command
      PATCH_COMMAND
      "${CMAKE_COMMAND}"
      "-DIREE_PATCH_SET_SHA256=${_patch_set_sha256}"
      "-DIREE_PATCH_GIT_EXECUTABLE=${_iree_dependency_git_executable}"
      "-DIREE_PATCH_SOURCE_DIR=<SOURCE_DIR>"
      -P "${_patch_parameters}"
    )
  endif()
  FetchContent_Declare(
    ${_fetch_name}
    URL ${_urls}
    URL_HASH SHA256=${_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
    ${_source_subdir_args}
    ${_patch_command}
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

function(iree_populate_locked_file dep_name out_file)
  iree_dependency_require_pinned_source_allowed("${dep_name}")
  iree_get_locked_dependency_property(_kind "${dep_name}" "KIND")
  iree_get_locked_dependency_property(_urls "${dep_name}" "URLS")
  iree_get_locked_dependency_property(_sha256 "${dep_name}" "SHA256")
  iree_get_locked_dependency_property(
    _downloaded_file_path
    "${dep_name}"
    "DOWNLOADED_FILE_PATH"
  )
  if(NOT _kind STREQUAL "http_file")
    message(FATAL_ERROR
      "${dep_name} is a ${_kind} dependency; expected http_file")
  endif()
  if(NOT _urls)
    message(FATAL_ERROR "${dep_name} has no locked source URLs")
  endif()
  if(NOT _sha256)
    message(FATAL_ERROR "${dep_name} has no locked SHA256")
  endif()
  _iree_dependency_expand_download_urls(_urls ${_urls})
  if(NOT _downloaded_file_path OR
     IS_ABSOLUTE "${_downloaded_file_path}" OR
     _downloaded_file_path MATCHES "^[A-Za-z]:" OR
     _downloaded_file_path MATCHES "\\\\" OR
     _downloaded_file_path MATCHES "[/\\\\]$" OR
     _downloaded_file_path MATCHES "(^|[/\\\\])\\.\\.([/\\\\]|$)")
    message(FATAL_ERROR
      "${dep_name} has invalid DOWNLOADED_FILE_PATH "
      "'${_downloaded_file_path}'")
  endif()

  set(_download_root "${CMAKE_BINARY_DIR}/_deps/${dep_name}-src")
  set(_download_file "${_download_root}/file/${_downloaded_file_path}")
  cmake_path(GET _download_file PARENT_PATH _download_directory)
  file(MAKE_DIRECTORY "${_download_directory}")

  if(EXISTS "${_download_file}")
    file(SHA256 "${_download_file}" _download_sha256)
    if("${_download_sha256}" STREQUAL "${_sha256}")
      set(${out_file} "${_download_file}" PARENT_SCOPE)
      return()
    endif()
    file(REMOVE "${_download_file}")
  endif()

  set(_download_errors)
  set(_download_candidate "${_download_file}.download")
  foreach(_url IN LISTS _urls)
    file(REMOVE "${_download_candidate}")
    file(DOWNLOAD
      "${_url}"
      "${_download_candidate}"
      STATUS _download_status
      TLS_VERIFY ON
    )
    list(GET _download_status 0 _download_code)
    if(_download_code EQUAL 0)
      file(SHA256 "${_download_candidate}" _download_sha256)
      if("${_download_sha256}" STREQUAL "${_sha256}")
        file(RENAME "${_download_candidate}" "${_download_file}")
        set(${out_file} "${_download_file}" PARENT_SCOPE)
        return()
      endif()
      list(APPEND _download_errors
        "${_url}: SHA256 ${_download_sha256} does not match ${_sha256}")
    else()
      list(GET _download_status 1 _download_message)
      list(APPEND _download_errors "${_url}: ${_download_message}")
    endif()
  endforeach()
  file(REMOVE "${_download_candidate}")

  list(JOIN _download_errors "\n  " _download_error_text)
  message(FATAL_ERROR
    "Failed to download ${dep_name}:\n  ${_download_error_text}")
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
  if(COMMAND iree_record_target_alias)
    iree_record_target_alias(${alias_name} ${_target_name})
  endif()
endfunction()
