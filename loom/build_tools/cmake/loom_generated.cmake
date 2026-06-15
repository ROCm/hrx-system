# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom generated C build helpers.
#
# These helpers mirror loom/build_tools/bazel/build_defs.bzl for generated
# CMake.
# Loom packages keep source-of-truth tables in Python and generate compact C
# data into the build tree.

function(_loom_generated_python_command_prefix OUTPUT_PREFIX GENERATOR)
  iree_py_library_collect_package_dirs(_GENERATOR_PACKAGE_DIRS "${GENERATOR}")
  list(APPEND _GENERATOR_PACKAGE_DIRS "$ENV{PYTHONPATH}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(JOIN _GENERATOR_PACKAGE_DIRS "\\;" _GENERATOR_PYTHONPATH)
  else()
    list(JOIN _GENERATOR_PACKAGE_DIRS ":" _GENERATOR_PYTHONPATH)
  endif()
  set(${OUTPUT_PREFIX}
    "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_GENERATOR_PYTHONPATH}"
    PARENT_SCOPE
  )
endfunction()

function(loom_generated_textual_header)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;OUTPUT;OUTPUT_FLAG;COMMENT"
    "ARGS;INPUTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_generated_textual_header requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "loom_generated_textual_header requires GENERATOR")
  endif()
  if(NOT _RULE_OUTPUT)
    message(FATAL_ERROR "loom_generated_textual_header requires OUTPUT")
  endif()
  if(NOT _RULE_OUTPUT_FLAG)
    message(FATAL_ERROR "loom_generated_textual_header requires OUTPUT_FLAG")
  endif()

  set(_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_OUTPUT}")
  if(NOT _RULE_COMMENT)
    set(_RULE_COMMENT "Generating ${_RULE_NAME}")
  endif()

  iree_py_library_main(_GENERATOR "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
  _loom_generated_python_command_prefix(_PYTHON_COMMAND_PREFIX "${_RULE_GENERATOR}")

  add_custom_command(
    OUTPUT
      "${_OUTPUT}"
    COMMAND
      ${_PYTHON_COMMAND_PREFIX}
      "${Python3_EXECUTABLE}"
      "${_GENERATOR}"
      ${_RULE_ARGS}
      "${_RULE_OUTPUT_FLAG}=${_OUTPUT}"
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "${_RULE_COMMENT}"
    VERBATIM
  )
  set_source_files_properties(
    "${_RULE_OUTPUT}"
    "${_OUTPUT}"
    PROPERTIES GENERATED TRUE
  )

  iree_package_name(_PACKAGE_NAME)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      "${_OUTPUT}"
  )
  iree_register_generated_compile_input("${_GEN_TARGET}")
endfunction()

function(loom_generated_cc_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;GENERATOR;SOURCE"
    "SRCS;GENERATED_SRC_FLAGS;GENERATED_SRCS;HDRS;GENERATED_HDR_FLAGS;GENERATED_HDRS;ARGS;INPUTS;EXTRA_OUTPUT_FLAGS;EXTRA_OUTPUTS;DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_generated_cc_library requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "loom_generated_cc_library requires GENERATOR")
  endif()
  if(NOT _RULE_SOURCE AND NOT _RULE_GENERATED_SRCS)
    set(_RULE_SOURCE "${_RULE_NAME}.c")
  endif()

  list(LENGTH _RULE_GENERATED_SRC_FLAGS _GENERATED_SRC_FLAG_COUNT)
  list(LENGTH _RULE_GENERATED_SRCS _GENERATED_SRC_COUNT)
  if(NOT _GENERATED_SRC_FLAG_COUNT EQUAL _GENERATED_SRC_COUNT)
    message(FATAL_ERROR
      "loom_generated_cc_library requires paired GENERATED_SRC_FLAGS and GENERATED_SRCS")
  endif()

  list(LENGTH _RULE_GENERATED_HDR_FLAGS _GENERATED_HDR_FLAG_COUNT)
  list(LENGTH _RULE_GENERATED_HDRS _GENERATED_HDR_COUNT)
  if(NOT _GENERATED_HDR_FLAG_COUNT EQUAL _GENERATED_HDR_COUNT)
    message(FATAL_ERROR
      "loom_generated_cc_library requires paired GENERATED_HDR_FLAGS and GENERATED_HDRS")
  endif()

  list(LENGTH _RULE_EXTRA_OUTPUT_FLAGS _EXTRA_OUTPUT_FLAG_COUNT)
  list(LENGTH _RULE_EXTRA_OUTPUTS _EXTRA_OUTPUT_COUNT)
  if(NOT _EXTRA_OUTPUT_FLAG_COUNT EQUAL _EXTRA_OUTPUT_COUNT)
    message(FATAL_ERROR
      "loom_generated_cc_library requires paired EXTRA_OUTPUT_FLAGS and EXTRA_OUTPUTS")
  endif()

  set(_OUTPUTS)
  set(_OUTPUT_ARGS)
  set(_GENERATED_SRCS)
  if(_RULE_SOURCE)
    set(_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SOURCE}")
    list(APPEND _OUTPUTS "${_SOURCE}")
    list(APPEND _OUTPUT_ARGS "--source=${_SOURCE}")
    list(APPEND _GENERATED_SRCS "${_SOURCE}")
  endif()
  if(_GENERATED_SRC_COUNT GREATER 0)
    math(EXPR _GENERATED_SRC_LAST "${_GENERATED_SRC_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_GENERATED_SRC_LAST})
      list(GET _RULE_GENERATED_SRC_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_GENERATED_SRCS ${_INDEX} _OUTPUT)
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
      list(APPEND _OUTPUTS "${_OUTPUT_PATH}")
      list(APPEND _OUTPUT_ARGS "${_OUTPUT_FLAG}=${_OUTPUT_PATH}")
      list(APPEND _GENERATED_SRCS "${_OUTPUT_PATH}")
    endforeach()
  endif()
  if(NOT _GENERATED_SRCS)
    message(FATAL_ERROR "loom_generated_cc_library requires at least one generated source")
  endif()

  set(_GENERATED_HDRS)
  if(_GENERATED_HDR_COUNT GREATER 0)
    math(EXPR _GENERATED_HDR_LAST "${_GENERATED_HDR_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_GENERATED_HDR_LAST})
      list(GET _RULE_GENERATED_HDR_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_GENERATED_HDRS ${_INDEX} _OUTPUT)
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
      list(APPEND _OUTPUTS "${_OUTPUT_PATH}")
      list(APPEND _OUTPUT_ARGS "${_OUTPUT_FLAG}=${_OUTPUT_PATH}")
      list(APPEND _GENERATED_HDRS "${_OUTPUT_PATH}")
    endforeach()
  endif()
  if(_EXTRA_OUTPUT_COUNT GREATER 0)
    math(EXPR _EXTRA_OUTPUT_LAST "${_EXTRA_OUTPUT_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_EXTRA_OUTPUT_LAST})
      list(GET _RULE_EXTRA_OUTPUT_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_EXTRA_OUTPUTS ${_INDEX} _OUTPUT)
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
      list(APPEND _OUTPUTS "${_OUTPUT_PATH}")
      list(APPEND _OUTPUT_ARGS "${_OUTPUT_FLAG}=${_OUTPUT_PATH}")
    endforeach()
  endif()

  iree_py_library_main(_GENERATOR "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
  _loom_generated_python_command_prefix(_PYTHON_COMMAND_PREFIX "${_RULE_GENERATOR}")

  add_custom_command(
    OUTPUT
      ${_OUTPUTS}
    COMMAND
      ${_PYTHON_COMMAND_PREFIX}
      "${Python3_EXECUTABLE}"
      "${_GENERATOR}"
      ${_RULE_ARGS}
      ${_OUTPUT_ARGS}
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} C table"
    VERBATIM
  )
  set_source_files_properties(
    ${_OUTPUTS}
    PROPERTIES GENERATED TRUE
  )

  iree_package_name(_PACKAGE_NAME)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}_gen")
  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      ${_OUTPUTS}
  )
  iree_register_generated_compile_input("${_GEN_TARGET}")
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  else()
    set(_TESTONLY_ARG)
  endif()

  loom_cc_library(
    NAME
      ${_RULE_NAME}
    HDRS
      ${_RULE_HDRS}
      ${_GENERATED_HDRS}
    SRCS
      ${_RULE_SRCS}
      ${_GENERATED_SRCS}
    DEPS
      ${_RULE_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )
endfunction()
