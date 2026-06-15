# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom target-low generated table build helpers.
#
# These helpers mirror loom/build_tools/bazel/build_defs.bzl for generated
# CMake.
# Target packages declare descriptor shards and target-info tables; this file
# owns the CMake mechanics for running generators into the binary tree and
# wrapping the outputs in loom_cc_library targets.

include(FetchContent)

function(loom_low_descriptor_data_archive)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SOURCE_DIR;SHA256"
    "URLS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires NAME")
  endif()
  if(NOT _RULE_SOURCE_DIR)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires SOURCE_DIR")
  endif()
  if(NOT _RULE_URLS)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires URLS")
  endif()
  if(NOT _RULE_SHA256)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires SHA256")
  endif()

  FetchContent_Declare("${_RULE_NAME}"
    URL
      ${_RULE_URLS}
    URL_HASH
      "SHA256=${_RULE_SHA256}"
    DOWNLOAD_NO_PROGRESS
      TRUE
    DOWNLOAD_EXTRACT_TIMESTAMP
      FALSE
    SOURCE_DIR
      "${_RULE_SOURCE_DIR}"
  )
  FetchContent_MakeAvailable("${_RULE_NAME}")
  if(NOT TARGET "${_RULE_NAME}")
    add_custom_target("${_RULE_NAME}")
  endif()
endfunction()

function(loom_target_table_cc_library)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL;HEADER_ONLY;TESTONLY"
    "NAME;GENERATOR;SOURCE;HEADER"
    "ARGS;INPUTS;DEPS;GENERATED_HDR_FLAGS;GENERATED_HDRS;IDS_DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_target_table_cc_library requires NAME")
  endif()
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_SOURCE)
    set(_RULE_SOURCE "${_RULE_NAME}.c")
  endif()
  if(NOT _RULE_HEADER)
    set(_RULE_HEADER "${_RULE_NAME}.h")
  endif()
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  else()
    set(_TESTONLY_ARG)
  endif()

  set(_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SOURCE}")
  set(_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_HEADER}")
  list(LENGTH _RULE_GENERATED_HDR_FLAGS _GENERATED_HDR_FLAG_COUNT)
  list(LENGTH _RULE_GENERATED_HDRS _GENERATED_HDR_COUNT)
  if(NOT _GENERATED_HDR_FLAG_COUNT EQUAL _GENERATED_HDR_COUNT)
    message(FATAL_ERROR
      "loom_target_table_cc_library generated header flags and headers must match")
  endif()
  set(_GENERATED_HDRS)
  set(_GENERATED_HDR_OUTPUT_ARGS)
  if(_GENERATED_HDR_COUNT GREATER 0)
    math(EXPR _GENERATED_HDR_LAST_INDEX "${_GENERATED_HDR_COUNT} - 1")
    foreach(_INDEX RANGE ${_GENERATED_HDR_LAST_INDEX})
      list(GET _RULE_GENERATED_HDR_FLAGS ${_INDEX} _GENERATED_HDR_FLAG)
      list(GET _RULE_GENERATED_HDRS ${_INDEX} _GENERATED_HDR)
      set(_GENERATED_HDR "${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_HDR}")
      list(APPEND _GENERATED_HDRS "${_GENERATED_HDR}")
      list(APPEND _GENERATED_HDR_OUTPUT_ARGS
        "${_GENERATED_HDR_FLAG}=${_GENERATED_HDR}")
    endforeach()
  endif()

  iree_package_name(_PACKAGE_NAME)
  if(_RULE_HEADER_ONLY)
    if(_RULE_GENERATOR)
      message(FATAL_ERROR
        "loom_target_table_cc_library HEADER_ONLY target cannot set GENERATOR")
    endif()
    if(_RULE_ARGS OR _RULE_INPUTS OR _RULE_GENERATED_HDRS)
      message(FATAL_ERROR
        "loom_target_table_cc_library HEADER_ONLY target cannot generate outputs")
    endif()
    set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
    add_custom_target("${_GEN_TARGET}"
      DEPENDS
        "${_HEADER}"
    )
    iree_register_generated_compile_input("${_GEN_TARGET}")
    loom_cc_library(
      NAME
        ${_RULE_NAME}
      HDRS
        "${_HEADER}"
      DEPS
        ${_RULE_DEPS}
      ${_TESTONLY_ARG}
      PUBLIC
    )
    add_dependencies(
      "${_PACKAGE_NAME}_${_RULE_NAME}"
      "${_GEN_TARGET}"
    )
    return()
  endif()

  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "loom_target_table_cc_library requires GENERATOR")
  endif()

  iree_py_library_main(_GENERATOR "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")

  add_custom_command(
    OUTPUT
      "${_SOURCE}"
      "${_HEADER}"
      ${_GENERATED_HDRS}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_GENERATOR}"
      ${_RULE_ARGS}
      "--source=${_SOURCE}"
      "--header=${_HEADER}"
      ${_GENERATED_HDR_OUTPUT_ARGS}
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} target table"
    VERBATIM
  )

  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      "${_SOURCE}"
      "${_HEADER}"
      ${_GENERATED_HDRS}
  )
  iree_register_generated_compile_input("${_GEN_TARGET}")

  loom_cc_library(
    NAME
      ${_RULE_NAME}
    HDRS
      "${_HEADER}"
    SRCS
      "${_SOURCE}"
    DEPS
      ${_RULE_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )

  if(_RULE_EXCLUDE_FROM_ALL)
    iree_cc_library_exclude_from_all(${_RULE_NAME} TRUE)
  endif()
endfunction()

function(loom_low_descriptor_cc_library)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL;HEADER_ONLY;TESTONLY"
    "NAME;HEADER"
    "DEPS;IDS_DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_low_descriptor_cc_library requires NAME")
  endif()
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  loom_target_table_cc_library(${ARGN})

  if(NOT _RULE_HEADER)
    set(_RULE_HEADER "${_RULE_NAME}.h")
  endif()
  if(NOT _RULE_IDS_DEPS)
    set(_RULE_IDS_DEPS ${_RULE_DEPS})
  endif()
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  else()
    set(_TESTONLY_ARG)
  endif()

  set(_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_HEADER}")
  iree_package_name(_PACKAGE_NAME)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  loom_cc_library(
    NAME
      "${_RULE_NAME}_ids"
    HDRS
      "${_HEADER}"
    DEPS
      ${_RULE_IDS_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )
  add_dependencies(
    "${_PACKAGE_NAME}_${_RULE_NAME}_ids"
    "${_GEN_TARGET}"
  )
endfunction()

function(loom_low_descriptor_exclude_from_all)
  cmake_parse_arguments(
    _RULE
    ""
    ""
    "CC_LIBRARIES;TARGETS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  foreach(_CC_LIBRARY IN LISTS _RULE_CC_LIBRARIES)
    set(_NAME "${_PACKAGE_NAME}_${_CC_LIBRARY}")
    if(NOT TARGET "${_NAME}")
      message(FATAL_ERROR
        "Cannot exclude missing low descriptor library ${_CC_LIBRARY} from all")
    endif()
    iree_cc_library_exclude_from_all(${_CC_LIBRARY} TRUE)
  endforeach()

  foreach(_TARGET IN LISTS _RULE_TARGETS)
    set(_NAME "${_PACKAGE_NAME}_${_TARGET}")
    if(TARGET "${_NAME}")
      set_property(TARGET "${_NAME}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    elseif(IREE_BUILD_TESTS)
      message(FATAL_ERROR
        "Cannot exclude missing low descriptor target ${_TARGET} from all")
    endif()
  endforeach()
endfunction()
