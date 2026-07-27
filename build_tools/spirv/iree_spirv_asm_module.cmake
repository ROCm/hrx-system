# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include(iree_c_embed_data)

# iree_spirv_asm_module()
#
# Builds a SPIR-V binary module from SPIR-V assembly.
#
# Parameters:
# PACKAGE: Name of the package (overrides actual path).
# NAME: Name of target.
# SRC: Source .spvasm file to assemble into a SPIR-V module.
# OUT: Optional output module file name. Defaults to ${NAME}.spv.
# TARGET_ENV: Optional spirv-as --target-env value.
# SPIRV_AS_ARGS: Additional arguments passed to spirv-as.
# ASSEMBLE_TOOL: Assembler tool target or executable path to invoke. Defaults
#     to iree::third_party::spirv_as.
# C_IDENTIFIER: Identifier to use for generated C embed code. If omitted then
#     no C embed code will be generated.
# DEPS: Library dependencies to add to the generated embed cc library.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
# PUBLIC: Add this so that generated libraries are exported under ${PACKAGE}::.
function(iree_spirv_asm_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "PACKAGE;NAME;SRC;OUT;TARGET_ENV;ASSEMBLE_TOOL;C_IDENTIFIER"
    "SPIRV_AS_ARGS;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_spirv_asm_module requires NAME")
  endif()
  if(NOT _RULE_SRC)
    message(FATAL_ERROR "iree_spirv_asm_module requires SRC")
  endif()

  if(DEFINED _RULE_ASSEMBLE_TOOL)
    set(_ASSEMBLE_TOOL "${_RULE_ASSEMBLE_TOOL}")
  else()
    if(NOT COMMAND iree_configure_spirv_tools)
      include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/spirv_tools/spirv_tools.cmake")
    endif()
    iree_configure_spirv_tools()
    set(_ASSEMBLE_TOOL iree::third_party::spirv_as)
  endif()

  if(TARGET "${_ASSEMBLE_TOOL}")
    set(_ASSEMBLE_TOOL_COMMAND "$<TARGET_FILE:${_ASSEMBLE_TOOL}>")
  else()
    set(_ASSEMBLE_TOOL_COMMAND "${_ASSEMBLE_TOOL}")
  endif()

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.spv")
  endif()

  if(IS_ABSOLUTE "${_RULE_SRC}")
    set(_SRC_PATH "${_RULE_SRC}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
    set(_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
  elseif(EXISTS "${IREE_SOURCE_DIR}/${_RULE_SRC}")
    set(_SRC_PATH "${IREE_SOURCE_DIR}/${_RULE_SRC}")
  else()
    set(_SRC_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRC}")
  endif()

  set(_ASSEMBLE_ARGS)
  if(DEFINED _RULE_TARGET_ENV)
    list(APPEND _ASSEMBLE_ARGS "--target-env=${_RULE_TARGET_ENV}")
  endif()
  list(APPEND _ASSEMBLE_ARGS ${_RULE_SPIRV_AS_ARGS})

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      ${_ASSEMBLE_TOOL_COMMAND}
      ${_ASSEMBLE_ARGS}
      "-o"
      "${_OUT}"
      "${_SRC_PATH}"
    DEPENDS
      "${_SRC_PATH}"
      ${_ASSEMBLE_TOOL}
    COMMENT
      "Assembling SPIR-V module ${_RULE_NAME}"
    VERBATIM
  )

  if(_RULE_PACKAGE)
    string(REPLACE "::" "_" _PACKAGE_NAME "${_RULE_PACKAGE}")
  else()
    iree_package_name(_PACKAGE_NAME)
  endif()
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS
      "${_OUT}"
  )

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

  if(_RULE_C_IDENTIFIER)
    iree_c_embed_data(
      PACKAGE
        "${_RULE_PACKAGE}"
      NAME
        "${_RULE_NAME}_c"
      SRCS
        "${_OUT}"
      C_FILE_OUTPUT
        "${_RULE_NAME}_c.c"
      H_FILE_OUTPUT
        "${_RULE_NAME}_c.h"
      IDENTIFIER
        "${_RULE_C_IDENTIFIER}"
      FLATTEN
      ${_PUBLIC_ARG}
      ${_TESTONLY_ARG}
      DEPS
        ${_RULE_DEPS}
    )
  endif()
endfunction()
