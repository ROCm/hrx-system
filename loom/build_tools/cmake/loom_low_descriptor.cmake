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
    "HEADER_ONLY;TESTONLY"
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
    iree_generated_output_add_consumer("${_HEADER}" "${_GEN_TARGET}")
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

  iree_py_library_entrypoint(_GENERATOR_ENTRYPOINT "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
  _loom_python_command_prefix(_PYTHON_COMMAND_PREFIX "${_RULE_GENERATOR}")

  set(_OUTPUTS
    "${_SOURCE}"
    "${_HEADER}"
    ${_GENERATED_HDRS}
  )
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  set(_GEN_STAMP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_GEN_TARGET}.stamp")
  add_custom_command(
    OUTPUT
      "${_GEN_STAMP}"
    BYPRODUCTS
      ${_OUTPUTS}
    COMMAND
      ${_PYTHON_COMMAND_PREFIX}
      "${Python3_EXECUTABLE}"
      ${_GENERATOR_ENTRYPOINT}
      ${_RULE_ARGS}
      "--source=${_SOURCE}"
      "--header=${_HEADER}"
      ${_GENERATED_HDR_OUTPUT_ARGS}
    COMMAND
      "${CMAKE_COMMAND}" -E touch "${_GEN_STAMP}"
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} target table"
    VERBATIM
  )

  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      "${_GEN_STAMP}"
      ${_OUTPUTS}
  )
  iree_register_generated_compile_input("${_GEN_TARGET}"
    OUTPUTS
      ${_OUTPUTS}
  )

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
endfunction()

function(loom_target_contract_cc_libraries)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME"
    "CONTRACT_DEPS;LOWER_RULE_DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_target_contract_cc_libraries requires NAME")
  endif()
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  else()
    set(_TESTONLY_ARG)
  endif()

  loom_cc_library(
    NAME
      "${_RULE_NAME}"
    HDRS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.h"
    SRCS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.c"
    DEPS
      ${_RULE_CONTRACT_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )
  loom_cc_library(
    NAME
      "${_RULE_NAME}_lower_rules"
    HDRS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_lower_rules.h"
    SRCS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_lower_rules.c"
    DEPS
      ${_RULE_LOWER_RULE_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )
endfunction()

function(loom_target_contract_table_cc_libraries)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;GENERATOR"
    "ARGS;INPUTS;CONTRACT_DEPS;LOWER_RULE_DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR
      "loom_target_contract_table_cc_libraries requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR
      "loom_target_contract_table_cc_libraries requires GENERATOR")
  endif()
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  else()
    set(_TESTONLY_ARG)
  endif()

  set(_CONTRACT_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.c")
  set(_CONTRACT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.h")
  set(_LOWER_RULE_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_lower_rules.c")
  set(_LOWER_RULE_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_lower_rules.h")

  iree_py_library_entrypoint(_GENERATOR_ENTRYPOINT "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
  _loom_python_command_prefix(_PYTHON_COMMAND_PREFIX "${_RULE_GENERATOR}")

  set(_OUTPUTS
    "${_CONTRACT_SOURCE}"
    "${_CONTRACT_HEADER}"
    "${_LOWER_RULE_SOURCE}"
    "${_LOWER_RULE_HEADER}"
  )
  iree_package_name(_PACKAGE_NAME)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  set(_GEN_STAMP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_GEN_TARGET}.stamp")
  add_custom_command(
    OUTPUT
      "${_GEN_STAMP}"
    BYPRODUCTS
      ${_OUTPUTS}
    COMMAND
      ${_PYTHON_COMMAND_PREFIX}
      "${Python3_EXECUTABLE}"
      ${_GENERATOR_ENTRYPOINT}
      ${_RULE_ARGS}
      "--contract-source=${_CONTRACT_SOURCE}"
      "--contract-header=${_CONTRACT_HEADER}"
      "--lower-rule-source=${_LOWER_RULE_SOURCE}"
      "--lower-rule-header=${_LOWER_RULE_HEADER}"
    COMMAND
      "${CMAKE_COMMAND}" -E touch "${_GEN_STAMP}"
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} contract table family"
    VERBATIM
  )

  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      "${_GEN_STAMP}"
      ${_OUTPUTS}
  )
  iree_register_generated_compile_input("${_GEN_TARGET}"
    OUTPUTS
      ${_OUTPUTS}
  )

  loom_target_contract_cc_libraries(
    NAME
      "${_RULE_NAME}"
    CONTRACT_DEPS
      ${_RULE_CONTRACT_DEPS}
    LOWER_RULE_DEPS
      ${_RULE_LOWER_RULE_DEPS}
    ${_TESTONLY_ARG}
  )
endfunction()

function(loom_target_contract_file_family)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;COMMENT"
    "FRAGMENTS;ARGS;INPUTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_target_contract_file_family requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR
      "loom_target_contract_file_family requires GENERATOR")
  endif()
  list(LENGTH _RULE_FRAGMENTS _FRAGMENT_COUNT)
  if(_FRAGMENT_COUNT LESS 2)
    message(FATAL_ERROR
      "loom_target_contract_file_family requires at least two fragments")
  endif()

  set(_OUTPUTS)
  set(_OUTPUT_FLAGS)
  set(_GENERATOR_ARGS ${_RULE_ARGS})
  foreach(_FRAGMENT_SPEC IN LISTS _RULE_FRAGMENTS)
    if(NOT _FRAGMENT_SPEC MATCHES "^([^=]+)=(.+)$")
      message(FATAL_ERROR
        "invalid contract fragment specification: ${_FRAGMENT_SPEC}")
    endif()
    set(_STEM "${CMAKE_MATCH_1}")
    set(_FRAGMENT_KEY "${CMAKE_MATCH_2}")
    list(APPEND _GENERATOR_ARGS "--contract-fragment=${_FRAGMENT_KEY}")
    list(APPEND _OUTPUTS
      "${_STEM}.c"
      "${_STEM}.h"
      "${_STEM}_lower_rules.c"
      "${_STEM}_lower_rules.h"
    )
    list(APPEND _OUTPUT_FLAGS
      "--contract-source"
      "--contract-header"
      "--lower-rule-source"
      "--lower-rule-header"
    )
  endforeach()

  _loom_generated_files(
    NAME "${_RULE_NAME}"
    GENERATOR "${_RULE_GENERATOR}"
    OUTPUTS ${_OUTPUTS}
    OUTPUT_FLAGS ${_OUTPUT_FLAGS}
    ARGS ${_GENERATOR_ARGS}
    INPUTS ${_RULE_INPUTS}
    COMMENT "${_RULE_COMMENT}"
  )
endfunction()

function(loom_low_descriptor_cc_library)
  cmake_parse_arguments(
    _RULE
    "HEADER_ONLY;TESTONLY"
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
