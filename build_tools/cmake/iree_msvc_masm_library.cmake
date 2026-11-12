# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)

# MASM receives only assembler flags, independent of C/C++ compiler options.
function(iree_msvc_masm_library)
  cmake_parse_arguments(_RULE "" "NAME;SRC" "" ${ARGN})
  iree_package_source_dir(_SOURCE_DIR)
  cmake_path(ABSOLUTE_PATH _RULE_SRC BASE_DIRECTORY "${_SOURCE_DIR}")
  iree_package_binary_dir(_IREE_PACKAGE_BINARY_DIR)
  set(_OBJECT "${_IREE_PACKAGE_BINARY_DIR}/${_RULE_NAME}.obj")

  set(_MASM_COMPILER "${CMAKE_ASM_MASM_COMPILER}")
  if(NOT _MASM_COMPILER)
    find_program(_MASM_COMPILER NAMES ml64 REQUIRED)
  endif()
  cmake_path(GET _MASM_COMPILER FILENAME _MASM_COMPILER_NAME)
  if(_MASM_COMPILER_NAME MATCHES "^llvm-ml")
    set(_MASM_OPTIONS -m64 /c)
  else()
    set(_MASM_OPTIONS /nologo /Zi /c)
  endif()
  add_custom_command(
    OUTPUT "${_OBJECT}"
    COMMAND "${_MASM_COMPILER}" ${_MASM_OPTIONS}
      /Fo "${_OBJECT}" "${_RULE_SRC}"
    DEPENDS "${_RULE_SRC}"
    COMMENT "Assembling ${_RULE_NAME}"
    VERBATIM
  )
  set_source_files_properties("${_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE)
  # The bridge is linked into its caller even in a component-DSO build.
  set(BUILD_SHARED_LIBS OFF)
  iree_cc_library(NAME "${_RULE_NAME}" SRCS "${_OBJECT}" PUBLIC)
  iree_package_target_name(_TARGET "::${_RULE_NAME}")
  set_target_properties(${_TARGET} PROPERTIES
    LINKER_LANGUAGE C
    IREE_CC_OBJECTS "${_OBJECT}")
endfunction()
