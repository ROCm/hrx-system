# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/selectors.cmake")

# Builds an LLVM shared library for AMDGPU from input files via clang.
#
# Parameters:
# NAME: Name of the target.
# OUT: Output file name.
# TARGET: LLVM `-target` flag.
# ARCH: LLVM `-march` flag.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: headers that should invalidate device compilation but are not
#                compiled as translation units or exposed as interface headers.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
# MINIMIZE: apply post-link symbol-table minimization. Only valid for opaque
#           code-object data blobs whose kernels are not looked up by name.
function(iree_amdgpu_binary)
  cmake_parse_arguments(
    _RULE
    "MINIMIZE"
    "NAME;OUT;TARGET;ARCH"
    "SRCS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.so")
  endif()

  set(_COPTS
    # C configuration.
    "-x" "c"
    "-Xclang" "-finclude-default-header"
    "-std=c23"
    "-nogpulib"
    "-fno-short-wchar"

    # Target architecture/machine.
    "-target" "${_RULE_TARGET}"
    "-march=${_RULE_ARCH}"
    "-fgpu-rdc"  # NOTE: may not be required for all targets

    # Header paths for builtins and our own includes.
    "-isystem" "${IREE_CLANG_BUILTIN_HEADERS_PATH}"
    "-I${IREE_SOURCE_DIR}/runtime/src"
    "-I${IREE_BINARY_DIR}/runtime/src"

    # Avoid warnings about things we do that are not compatible across compilers
    # but are fine because we're only ever compiling with clang.
    "-Wno-gnu-pointer-arith"

    # Optimized.
    "-fno-ident"
    "-fvisibility=hidden"
    "-O3"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"
  )

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_SRC_FRAGMENT "${_SRC}")
    string(REPLACE "\\" "_" _BITCODE_SRC_FRAGMENT "${_BITCODE_SRC_FRAGMENT}")
    string(REPLACE "/" "_" _BITCODE_SRC_FRAGMENT "${_BITCODE_SRC_FRAGMENT}")
    string(REPLACE ":" "_" _BITCODE_SRC_FRAGMENT "${_BITCODE_SRC_FRAGMENT}")
    string(REPLACE "." "_" _BITCODE_SRC_FRAGMENT "${_BITCODE_SRC_FRAGMENT}")
    set(_BITCODE_FILE "${_RULE_NAME}_${_BITCODE_SRC_FRAGMENT}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${IREE_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_CLANG_BINARY}"
        "${_BITCODE_SRC_PATH}"
        "${_RULE_INTERNAL_HDRS}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  set(_ARCHIVE_FILE "${_RULE_NAME}.a")
  add_custom_command(
    OUTPUT
      ${_ARCHIVE_FILE}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_ARCHIVE_FILE}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Archiving bitcode to ${_ARCHIVE_FILE}"
    VERBATIM
  )

  set(_LINKED_FILE "${_RULE_NAME}.bc")
  add_custom_command(
    OUTPUT
      ${_LINKED_FILE}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      "-internalize"
      "-only-needed"
      "${_ARCHIVE_FILE}"
      "-o" "${_LINKED_FILE}"
    DEPENDS
      "${IREE_LLVM_LINK_BINARY}"
      "${_ARCHIVE_FILE}"
    COMMENT
      "Linking bitcode to ${_LINKED_FILE}"
    VERBATIM
  )

  set(_LINK_OUT "${_OUT}")
  set(_LINKOPTS ${_RULE_LINKOPTS})
  if(_RULE_MINIMIZE)
    if(NOT IREE_LLVM_OBJCOPY_BINARY)
      message(FATAL_ERROR
        "iree_amdgpu_binary(MINIMIZE) requires IREE_LLVM_OBJCOPY_BINARY")
    endif()
    set(_LINK_OUT "${_RULE_NAME}.linked.so")
    set(_VERSION_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.local.version")
    file(WRITE "${_VERSION_SCRIPT}" "{\n  local:\n    *;\n};\n")
    list(APPEND _LINKOPTS "--version-script=${_VERSION_SCRIPT}")
  endif()

  get_filename_component(_LINK_OUT_DIR "${_LINK_OUT}" DIRECTORY)
  set(_LINK_OUT_MAKE_DIRECTORY_COMMAND)
  if(_LINK_OUT_DIR)
    set(_LINK_OUT_MAKE_DIRECTORY_COMMAND
      COMMAND ${CMAKE_COMMAND} "-E" "make_directory" "${_LINK_OUT_DIR}"
    )
  endif()
  add_custom_command(
    OUTPUT
      "${_LINK_OUT}"
    ${_LINK_OUT_MAKE_DIRECTORY_COMMAND}
    COMMAND
      ${IREE_LLD_BINARY}
      "-flavor" "gnu"
      "-m" "elf64_amdgpu"
      "--build-id=none"
      "--no-undefined"
      "-shared"
      "-plugin-opt=mcpu=${_RULE_ARCH}"
      "-plugin-opt=O3"
      "--lto-CGO3"
      "--no-whole-archive"
      "--gc-sections"
      "--strip-debug"
      "--discard-all"
      "--discard-locals"
      ${_LINKOPTS}
      "${_LINKED_FILE}"
      "-o" "${_LINK_OUT}"
    DEPENDS
      "${_LINKED_FILE}"
      "${IREE_LLD_BINARY}"
      "${IREE_LLD_TARGET}"
    COMMENT
      "Compiling binary to ${_LINK_OUT}"
    VERBATIM
  )
  if(_RULE_MINIMIZE)
    get_filename_component(_OUT_DIR "${_OUT}" DIRECTORY)
    set(_OUT_MAKE_DIRECTORY_COMMAND)
    if(_OUT_DIR)
      set(_OUT_MAKE_DIRECTORY_COMMAND
        COMMAND ${CMAKE_COMMAND} "-E" "make_directory" "${_OUT_DIR}"
      )
    endif()
    add_custom_command(
      OUTPUT
        "${_OUT}"
      ${_OUT_MAKE_DIRECTORY_COMMAND}
      COMMAND
        ${IREE_LLVM_OBJCOPY_BINARY}
        "-R" ".comment"
        "-R" ".AMDGPU.gpr_maximums"
        "--discard-all"
        "-N" "_DYNAMIC"
        "${_LINK_OUT}"
        "${_OUT}"
      DEPENDS
        "${_LINK_OUT}"
        "${IREE_LLVM_OBJCOPY_BINARY}"
      COMMENT
        "Minimizing AMDGPU binary to ${_OUT}"
      VERBATIM
    )
  endif()

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

# Builds one AMDGPU binary per selected code-object target.
#
# Parameters:
# NAME: Name of the aggregate target.
# TARGET: LLVM `-target` flag.
# TARGETS: AMDGPU target selectors to expand to code-object targets.
# BINARY_NAME_PREFIX: Optional prefix for per-target binary names.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: headers that should invalidate device compilation.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
# OUTPUTS_OUT: Optional variable receiving generated output file names relative
#              to the current binary directory.
# OUTPUT_PATHS_OUT: Optional variable receiving absolute generated output paths.
# TARGETS_OUT: Optional variable receiving generated CMake target names.
# MINIMIZE: apply post-link symbol-table minimization.
function(iree_amdgpu_binary_variants)
  cmake_parse_arguments(
    _RULE
    "MINIMIZE"
    "NAME;TARGET;BINARY_NAME_PREFIX;OUTPUTS_OUT;OUTPUT_PATHS_OUT;TARGETS_OUT"
    "TARGETS;SRCS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_binary_variants requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_binary_variants requires TARGET")
  endif()

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_BINARY_NAME_PREFIX)
    set(_BINARY_NAME_PREFIX "${_RULE_BINARY_NAME_PREFIX}")
  else()
    set(_BINARY_NAME_PREFIX "${_RULE_NAME}")
  endif()

  iree_amdgpu_expand_target_selectors(
    _CODE_OBJECT_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}"
    ${_RULE_TARGETS}
  )

  set(_VARIANT_OUTPUTS)
  set(_VARIANT_OUTPUT_PATHS)
  set(_VARIANT_TARGETS)
  foreach(_CODE_OBJECT_TARGET ${_CODE_OBJECT_TARGETS})
    iree_amdgpu_target_label_fragment(
      _TARGET_FRAGMENT
      "${_CODE_OBJECT_TARGET}"
    )
    set(_VARIANT_NAME "${_BINARY_NAME_PREFIX}_${_TARGET_FRAGMENT}")
    set(_VARIANT_OUTPUT "${_VARIANT_NAME}.so")

    set(_MINIMIZE)
    if(_RULE_MINIMIZE)
      set(_MINIMIZE MINIMIZE)
    endif()
    iree_amdgpu_binary(
      NAME
        "${_VARIANT_NAME}"
      OUT
        "${_VARIANT_OUTPUT}"
      TARGET
        "${_RULE_TARGET}"
      ARCH
        "${_CODE_OBJECT_TARGET}"
      SRCS
        ${_RULE_SRCS}
      INTERNAL_HDRS
        ${_RULE_INTERNAL_HDRS}
      COPTS
        ${_RULE_COPTS}
      LINKOPTS
        ${_RULE_LINKOPTS}
      ${_MINIMIZE}
    )

    list(APPEND _VARIANT_OUTPUTS "${_VARIANT_OUTPUT}")
    list(APPEND _VARIANT_OUTPUT_PATHS
      "${CMAKE_CURRENT_BINARY_DIR}/${_VARIANT_OUTPUT}")
    list(APPEND _VARIANT_TARGETS "${_PACKAGE_NAME}_${_VARIANT_NAME}")
  endforeach()

  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}")
  if(_VARIANT_TARGETS)
    add_dependencies("${_PACKAGE_NAME}_${_RULE_NAME}" ${_VARIANT_TARGETS})
  endif()

  if(DEFINED _RULE_OUTPUTS_OUT)
    set(${_RULE_OUTPUTS_OUT} "${_VARIANT_OUTPUTS}" PARENT_SCOPE)
  endif()
  if(DEFINED _RULE_OUTPUT_PATHS_OUT)
    set(${_RULE_OUTPUT_PATHS_OUT} "${_VARIANT_OUTPUT_PATHS}" PARENT_SCOPE)
  endif()
  if(DEFINED _RULE_TARGETS_OUT)
    set(${_RULE_TARGETS_OUT} "${_VARIANT_TARGETS}" PARENT_SCOPE)
  endif()
endfunction()

# Builds selected AMDGPU binaries and embeds them into a C library.
#
# Parameters:
# NAME: Name of the generated C embed-data library.
# TARGET: LLVM `-target` flag.
# TARGETS: AMDGPU target selectors to expand to code-object targets.
# BINARY_NAME_PREFIX: Optional prefix for per-target binary names.
# C_FILE_OUTPUT: Generated C implementation filename. Defaults to NAME.c.
# H_FILE_OUTPUT: Generated C header filename. Defaults to NAME.h.
# IDENTIFIER: C identifier prefix. Defaults to NAME.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: headers that should invalidate device compilation.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
# DEPS: dependencies for the generated C embed-data library.
# INCLUDES: include paths for the generated C embed-data library.
# FLATTEN: drop directory components from table-of-contents names.
# PUBLIC: expose the generated C embed-data library publicly.
# TESTONLY: only build the generated library when tests are enabled.
# MINIMIZE: apply post-link symbol-table minimization.
function(iree_amdgpu_binary_variants_embed_data)
  cmake_parse_arguments(
    _RULE
    "FLATTEN;PUBLIC;TESTONLY;MINIMIZE"
    "NAME;TARGET;BINARY_NAME_PREFIX;C_FILE_OUTPUT;H_FILE_OUTPUT;IDENTIFIER"
    "TARGETS;SRCS;INTERNAL_HDRS;COPTS;LINKOPTS;DEPS;INCLUDES"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_binary_variants_embed_data requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_binary_variants_embed_data requires TARGET")
  endif()

  if(DEFINED _RULE_C_FILE_OUTPUT)
    set(_C_FILE_OUTPUT "${_RULE_C_FILE_OUTPUT}")
  else()
    set(_C_FILE_OUTPUT "${_RULE_NAME}.c")
  endif()
  if(DEFINED _RULE_H_FILE_OUTPUT)
    set(_H_FILE_OUTPUT "${_RULE_H_FILE_OUTPUT}")
  else()
    set(_H_FILE_OUTPUT "${_RULE_NAME}.h")
  endif()

  set(_BINARY_VARIANTS_NAME "${_RULE_NAME}_binaries")

  set(_BINARY_NAME_PREFIX_ARG)
  if(DEFINED _RULE_BINARY_NAME_PREFIX)
    set(_BINARY_NAME_PREFIX_ARG BINARY_NAME_PREFIX "${_RULE_BINARY_NAME_PREFIX}")
  endif()
  set(_MINIMIZE_ARG)
  if(_RULE_MINIMIZE)
    set(_MINIMIZE_ARG MINIMIZE)
  endif()
  iree_amdgpu_binary_variants(
    NAME
      "${_BINARY_VARIANTS_NAME}"
    TARGET
      "${_RULE_TARGET}"
    TARGETS
      ${_RULE_TARGETS}
    ${_BINARY_NAME_PREFIX_ARG}
    OUTPUT_PATHS_OUT
      _VARIANT_OUTPUT_PATHS
    SRCS
      ${_RULE_SRCS}
    INTERNAL_HDRS
      ${_RULE_INTERNAL_HDRS}
    COPTS
      ${_RULE_COPTS}
    LINKOPTS
      ${_RULE_LINKOPTS}
    ${_MINIMIZE_ARG}
  )

  set(_IDENTIFIER_ARG)
  if(DEFINED _RULE_IDENTIFIER)
    set(_IDENTIFIER_ARG IDENTIFIER "${_RULE_IDENTIFIER}")
  endif()
  set(_FLATTEN_ARG)
  if(_RULE_FLATTEN)
    set(_FLATTEN_ARG FLATTEN)
  endif()
  set(_PUBLIC_ARG)
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG PUBLIC)
  endif()
  set(_TESTONLY_ARG)
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  endif()
  iree_c_embed_data(
    NAME
      "${_RULE_NAME}"
    SRCS
      ${_VARIANT_OUTPUT_PATHS}
    C_FILE_OUTPUT
      "${_C_FILE_OUTPUT}"
    H_FILE_OUTPUT
      "${_H_FILE_OUTPUT}"
    ${_IDENTIFIER_ARG}
    DEPS
      ${_RULE_DEPS}
    INCLUDES
      ${_RULE_INCLUDES}
    ${_FLATTEN_ARG}
    ${_PUBLIC_ARG}
    ${_TESTONLY_ARG}
  )
endfunction()
