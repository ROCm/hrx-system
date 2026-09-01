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
#
# Generated commands use the shared content-stable Python projection helper so
# every file in an output family comes from one generator process.

# Low-descriptor archive generation constructs a multi-target command graph and
# uses these command/target primitives directly.
function(_loom_python_command_prefix OUTPUT_PREFIX GENERATOR)
  iree_py_library_collect_package_dirs(_GENERATOR_PACKAGE_DIRS "${GENERATOR}")
  list(APPEND _GENERATOR_PACKAGE_DIRS "$ENV{PYTHONPATH}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(JOIN _GENERATOR_PACKAGE_DIRS "\\;" _GENERATOR_PYTHONPATH)
  else()
    list(JOIN _GENERATOR_PACKAGE_DIRS ":" _GENERATOR_PYTHONPATH)
  endif()
  set(${OUTPUT_PREFIX}
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${_GENERATOR_PYTHONPATH}"
    "PYTHONDONTWRITEBYTECODE=1"
    PARENT_SCOPE
  )
endfunction()

function(_loom_add_generated_target TARGET_NAME STAMP_PATH)
  add_custom_target("${TARGET_NAME}"
    DEPENDS
      "${STAMP_PATH}"
  )
endfunction()

function(_loom_generated_files)
  iree_py_generated_files(${ARGN})
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

  _loom_generated_files(
    NAME "${_RULE_NAME}"
    GENERATOR "${_RULE_GENERATOR}"
    OUTPUTS "${_RULE_OUTPUT}"
    OUTPUT_FLAGS "${_RULE_OUTPUT_FLAG}"
    ARGS ${_RULE_ARGS}
    INPUTS ${_RULE_INPUTS}
    COMMENT "${_RULE_COMMENT}"
  )
endfunction()

function(loom_generated_file_family)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;COMMENT"
    "OUTPUTS;OUTPUT_FLAGS;ARGS;INPUTS"
    ${ARGN}
  )

  list(LENGTH _RULE_OUTPUTS _OUTPUT_COUNT)
  if(_OUTPUT_COUNT LESS 2)
    message(FATAL_ERROR
      "loom_generated_file_family requires at least two outputs")
  endif()

  _loom_generated_files(
    NAME "${_RULE_NAME}"
    GENERATOR "${_RULE_GENERATOR}"
    OUTPUTS ${_RULE_OUTPUTS}
    OUTPUT_FLAGS ${_RULE_OUTPUT_FLAGS}
    ARGS ${_RULE_ARGS}
    INPUTS ${_RULE_INPUTS}
    COMMENT "${_RULE_COMMENT}"
  )
endfunction()

function(loom_generated_cc_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;GENERATOR;SOURCE"
    "SRCS;TEXTUAL_HDRS;GENERATED_SRC_FLAGS;GENERATED_SRCS;HDRS;GENERATED_HDR_FLAGS;GENERATED_HDRS;ARGS;INPUTS;EXTRA_OUTPUT_FLAGS;EXTRA_OUTPUTS;DEPS"
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
  set(_OUTPUT_FLAGS)
  set(_GENERATED_SRCS)
  if(_RULE_SOURCE)
    set(_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SOURCE}")
    list(APPEND _OUTPUTS "${_RULE_SOURCE}")
    list(APPEND _OUTPUT_FLAGS "--source")
    list(APPEND _GENERATED_SRCS "${_SOURCE}")
  endif()
  if(_GENERATED_SRC_COUNT GREATER 0)
    math(EXPR _GENERATED_SRC_LAST "${_GENERATED_SRC_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_GENERATED_SRC_LAST})
      list(GET _RULE_GENERATED_SRC_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_GENERATED_SRCS ${_INDEX} _OUTPUT)
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
      list(APPEND _OUTPUTS "${_OUTPUT}")
      list(APPEND _OUTPUT_FLAGS "${_OUTPUT_FLAG}")
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
      list(APPEND _OUTPUTS "${_OUTPUT}")
      list(APPEND _OUTPUT_FLAGS "${_OUTPUT_FLAG}")
      list(APPEND _GENERATED_HDRS "${_OUTPUT_PATH}")
    endforeach()
  endif()
  if(_EXTRA_OUTPUT_COUNT GREATER 0)
    math(EXPR _EXTRA_OUTPUT_LAST "${_EXTRA_OUTPUT_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_EXTRA_OUTPUT_LAST})
      list(GET _RULE_EXTRA_OUTPUT_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_EXTRA_OUTPUTS ${_INDEX} _OUTPUT)
      list(APPEND _OUTPUTS "${_OUTPUT}")
      list(APPEND _OUTPUT_FLAGS "${_OUTPUT_FLAG}")
    endforeach()
  endif()

  iree_py_generated_files(
    NAME "${_RULE_NAME}_gen"
    GENERATOR "${_RULE_GENERATOR}"
    OUTPUTS ${_OUTPUTS}
    OUTPUT_FLAGS ${_OUTPUT_FLAGS}
    ARGS ${_RULE_ARGS}
    INPUTS ${_RULE_INPUTS}
    COMMENT "Generating ${_RULE_NAME} C table"
  )
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
    TEXTUAL_HDRS
      ${_RULE_TEXTUAL_HDRS}
    DEPS
      ${_RULE_DEPS}
    ${_TESTONLY_ARG}
    PUBLIC
  )
endfunction()
