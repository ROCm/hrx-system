# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Defines one libamdf CTS executable in the public artifact directory. The
# installed library and its private runtime companions share this layout, so
# every linkage mode exercises the same sibling-library discovery contract.
function(amdf_cts_binary)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    ""
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_cts_binary requires NAME.")
  endif()

  iree_cc_binary(${ARGN})

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties(${_TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${_TARGET_NAME}"
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:amdf>"
  )
endfunction()

# Compiles one authored behavior for its available physical target profiles,
# then embeds those products with a small declaration-only header.
function(amdf_cts_gpu_kernel_set)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME;ENTRY_POINT;NAMESPACE;INPUT_FORMAT"
    "SRCS;DATA;INPUTOPTS;TARGETS" ${ARGN})
  if(_RULE_UNPARSED_ARGUMENTS OR NOT _RULE_NAME OR NOT _RULE_SRCS OR
     NOT _RULE_ENTRY_POINT OR NOT _RULE_NAMESPACE OR NOT _RULE_TARGETS)
    message(FATAL_ERROR "Incomplete GPU CTS kernel set declaration")
  endif()
  set(_INPUTS)
  set(_SELECTORS)
  foreach(_SELECTOR IN LISTS _RULE_TARGETS)
    iree_package_target_name(_PROFILE "::${_SELECTOR}")
    get_target_property(_AVAILABLE "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
    if(NOT _AVAILABLE)
      continue()
    endif()
    set(_PRODUCT "${_RULE_NAME}_${_SELECTOR}")
    loom_kernel_binary(
      NAME "${_PRODUCT}"
      TARGET "::${_SELECTOR}"
      OUTPUT "${_PRODUCT}.hsaco"
      SRCS ${_RULE_SRCS}
      DATA ${_RULE_DATA}
      INPUT_FORMAT "${_RULE_INPUT_FORMAT}"
      INPUTOPTS ${_RULE_INPUTOPTS}
      ROOTS "@${_RULE_ENTRY_POINT}"
      TESTONLY
    )
    list(APPEND _INPUTS "${CMAKE_CURRENT_BINARY_DIR}/${_PRODUCT}.hsaco")
    list(APPEND _SELECTORS "${_SELECTOR}")
  endforeach()
  amdf_cts_embed_gpu_kernel_set(
    NAME "${_RULE_NAME}_embed"
    HEADER "${_RULE_NAME}.h"
    IMPLEMENTATION "${_RULE_NAME}.cc"
    ENTRY_POINT "${_RULE_ENTRY_POINT}"
    NAMESPACE "${_RULE_NAMESPACE}"
    INPUTS ${_INPUTS}
    SELECTORS ${_SELECTORS}
  )
  iree_cc_library(
    NAME "${_RULE_NAME}"
    HDRS "${_RULE_NAME}.h"
    SRCS "${_RULE_NAME}.cc"
    DEPS amdf::headers libamdf::cts::gpu::kernels::kernel
    TESTONLY
    PUBLIC
  )
endfunction()

# Embeds all physical compiler variants of one CTS behavior into one object.
function(amdf_cts_embed_gpu_kernel_set)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE ""
    "NAME;HEADER;IMPLEMENTATION;ENTRY_POINT;NAMESPACE" "INPUTS;SELECTORS" ${ARGN})
  if(_RULE_UNPARSED_ARGUMENTS OR NOT _RULE_NAME OR NOT _RULE_INPUTS OR
     NOT _RULE_SELECTORS OR NOT _RULE_HEADER OR NOT _RULE_IMPLEMENTATION OR
     NOT _RULE_ENTRY_POINT OR NOT _RULE_NAMESPACE)
    message(FATAL_ERROR "Incomplete GPU CTS kernel set declaration")
  endif()
  list(LENGTH _RULE_INPUTS _INPUT_COUNT)
  list(LENGTH _RULE_SELECTORS _SELECTOR_COUNT)
  if(NOT _INPUT_COUNT EQUAL _SELECTOR_COUNT)
    message(FATAL_ERROR "GPU CTS input and selector counts must match")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "GPU CTS image extraction requires a host Python interpreter")
  endif()
  set(_VARIANT_ARGS)
  foreach(_INPUT _SELECTOR IN ZIP_LISTS _RULE_INPUTS _RULE_SELECTORS)
    list(APPEND _VARIANT_ARGS --variant "${_SELECTOR}=${_INPUT}")
  endforeach()
  set(_SCRIPT "${PROJECT_SOURCE_DIR}/libamdf/cts/gpu/kernels/embed.py")
  set(_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_HEADER}")
  set(_IMPLEMENTATION "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_IMPLEMENTATION}")
  add_custom_command(
    OUTPUT "${_HEADER}" "${_IMPLEMENTATION}"
    COMMAND "${Python3_EXECUTABLE}" "${_SCRIPT}" ${_VARIANT_ARGS}
      --output "${_HEADER}" --implementation "${_IMPLEMENTATION}"
      --symbol "${_RULE_ENTRY_POINT}" --namespace "${_RULE_NAMESPACE}"
    DEPENDS "${_SCRIPT}" ${_RULE_INPUTS}
    COMMENT "Embedding GPU kernel variants ${_RULE_NAME}"
    VERBATIM
  )
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET}" DEPENDS "${_HEADER}" "${_IMPLEMENTATION}")
  foreach(_INPUT IN LISTS _RULE_INPUTS)
    iree_generated_output_add_consumer("${_INPUT}" "${_TARGET}")
  endforeach()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS "${_HEADER}" "${_IMPLEMENTATION}")
endfunction()
