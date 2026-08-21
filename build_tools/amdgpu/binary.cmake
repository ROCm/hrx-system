# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/selectors.cmake")

function(_iree_amdgpu_bitcode_copts out_var target arch)
  set(_COPTS
    # C configuration.
    "-x" "c"
    "-Xclang" "-finclude-default-header"
    "-std=c23"
    "-nogpulib"
    "-fno-short-wchar"

    # Target architecture/machine.
    "-target" "${target}"
    "-march=${arch}"
    "-fgpu-rdc"  # NOTE: may not be required for all targets

    # Header paths for builtins and our own includes.
    "-isystem"
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_RESOURCE_INCLUDE}"
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
  list(APPEND _COPTS ${ARGN})
  set(${out_var} "${_COPTS}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_abs_binary_path out_var path)
  if(IS_ABSOLUTE "${path}")
    set(${out_var} "${path}" PARENT_SCOPE)
  else()
    set(${out_var} "${CMAKE_CURRENT_BINARY_DIR}/${path}" PARENT_SCOPE)
  endif()
endfunction()

function(_iree_amdgpu_add_generated_input_dependencies TARGET_NAME)
  if(NOT TARGET "${TARGET_NAME}")
    return()
  endif()
  set(_CONSUMER_TARGETS "${TARGET_NAME}")
  if(TARGET "${TARGET_NAME}.objects")
    list(APPEND _CONSUMER_TARGETS "${TARGET_NAME}.objects")
  endif()
  foreach(_INPUT IN LISTS ARGN)
    foreach(_CONSUMER_TARGET IN LISTS _CONSUMER_TARGETS)
      iree_generated_output_add_consumer(
        "${_INPUT}" "${_CONSUMER_TARGET}")
    endforeach()
  endforeach()
endfunction()

function(_iree_amdgpu_resolve_bitcode_deps out_paths out_targets)
  set(_DEP_PATHS)
  set(_DEP_TARGETS)
  foreach(_DEP_REF ${ARGN})
    iree_package_target_name(_DEP "${_DEP_REF}")
    if(TARGET "${_DEP}")
      get_target_property(_DEP_PATH
        "${_DEP}" IREE_AMDGPU_BITCODE_ARCHIVE_OUTPUT)
      if(NOT _DEP_PATH)
        message(FATAL_ERROR
          "iree_amdgpu_binary DEPS target '${_DEP}' is not an "
          "iree_amdgpu_library target")
      endif()
      list(APPEND _DEP_PATHS "${_DEP_PATH}")
      list(APPEND _DEP_TARGETS "${_DEP}")
    elseif("${_DEP_REF}" MATCHES "::")
      iree_package_ns(_PACKAGE_NS)
      string(REGEX REPLACE "^::" "${_PACKAGE_NS}::" _DEP_NS "${_DEP_REF}")
      string(FIND "${_DEP_NS}" "::" _DEP_TARGET_OFFSET REVERSE)
      if(_DEP_TARGET_OFFSET EQUAL -1)
        message(FATAL_ERROR
          "iree_amdgpu_binary DEPS target '${_DEP_REF}' is malformed")
      endif()
      math(EXPR _DEP_NAME_OFFSET "${_DEP_TARGET_OFFSET} + 2")
      string(SUBSTRING "${_DEP_NS}" 0 "${_DEP_TARGET_OFFSET}" _DEP_PACKAGE_NS)
      string(SUBSTRING "${_DEP_NS}" "${_DEP_NAME_OFFSET}" -1 _DEP_NAME)
      string(REPLACE "::" "/" _DEP_PACKAGE_PATH "${_DEP_PACKAGE_NS}")
      if(NOT EXISTS "${IREE_SOURCE_DIR}/${_DEP_PACKAGE_PATH}/CMakeLists.txt")
        message(FATAL_ERROR
          "iree_amdgpu_binary DEPS target '${_DEP_REF}' references unknown "
          "package '${_DEP_PACKAGE_PATH}'")
      endif()
      list(APPEND _DEP_PATHS
        "${IREE_BINARY_DIR}/${_DEP_PACKAGE_PATH}/${_DEP_NAME}.a")
    else()
      list(APPEND _DEP_PATHS "${_DEP_REF}")
    endif()
  endforeach()
  set(${out_paths} "${_DEP_PATHS}" PARENT_SCOPE)
  set(${out_targets} "${_DEP_TARGETS}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_code_object_target_deps out_var code_object_target)
  iree_amdgpu_target_label_fragment(
    _CODE_OBJECT_TARGET_FRAGMENT
    "${code_object_target}"
  )
  set(_TARGET_DEPS)
  foreach(_DEP ${ARGN})
    string(REPLACE
      "{AMDGPU_CODE_OBJECT_TARGET}"
      "${code_object_target}"
      _TARGET_DEP
      "${_DEP}"
    )
    string(REPLACE
      "{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}"
      "${_CODE_OBJECT_TARGET_FRAGMENT}"
      _TARGET_DEP
      "${_TARGET_DEP}"
    )
    list(APPEND _TARGET_DEPS "${_TARGET_DEP}")
  endforeach()
  set(${out_var} "${_TARGET_DEPS}" PARENT_SCOPE)
endfunction()

# Builds an LLVM bitcode archive for AMDGPU from input files via clang.
#
# Parameters:
# NAME: Name of the target.
# OUT: Output archive file name.
# TARGET: LLVM `-target` flag.
# ARCH: LLVM `-march` flag.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: headers that should invalidate device compilation but are not
#                compiled as translation units or exposed as interface headers.
# COPTS: additional flags to pass to clang.
function(iree_amdgpu_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;OUT;TARGET;ARCH"
    "SRCS;INTERNAL_HDRS;COPTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_library requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_library requires TARGET")
  endif()
  if(NOT _RULE_ARCH)
    message(FATAL_ERROR "iree_amdgpu_library requires ARCH")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR "iree_amdgpu_library requires SRCS")
  endif()
  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE)
    message(FATAL_ERROR
      "iree_amdgpu_library requires an AMDGPU device toolchain")
  endif()

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.a")
  endif()

  _iree_amdgpu_bitcode_copts(
    _COPTS
    "${_RULE_TARGET}"
    "${_RULE_ARCH}"
    ${_RULE_COPTS}
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
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
        "${_BITCODE_SRC_PATH}"
        "${_RULE_INTERNAL_HDRS}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

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
      ${CMAKE_COMMAND} "-E" "rm" "-f" "${_OUT}"
    COMMAND
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_AR_BINARY}"
      "rc"
      "${_OUT}"
      ${_BITCODE_FILES}
    DEPENDS
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_AR_BINARY}"
      ${_BITCODE_FILES}
    COMMENT
      "Archiving bitcode to ${_OUT}"
    VERBATIM
  )

  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET_NAME}"
    DEPENDS "${_OUT}"
  )
  _iree_amdgpu_add_generated_input_dependencies(
    "${_TARGET_NAME}" ${_RULE_INTERNAL_HDRS})
  _iree_amdgpu_abs_binary_path(_OUT_PATH "${_OUT}")
  iree_register_generated_output_producer("${_TARGET_NAME}"
    OUTPUTS "${_OUT_PATH}"
  )
  set_property(TARGET "${_TARGET_NAME}"
    PROPERTY IREE_AMDGPU_BITCODE_ARCHIVE_OUTPUT "${_OUT_PATH}")
endfunction()

# Builds an LLVM shared library for AMDGPU from input files via clang.
#
# Parameters:
# NAME: Name of the target.
# OUT: Output file name.
# TARGET: LLVM `-target` flag.
# ARCH: LLVM `-march` flag.
# SRCS: source files to pass to clang.
# DEPS: bitcode archive dependencies to link with `llvm-link -only-needed`.
#       Entries may use `{AMDGPU_CODE_OBJECT_TARGET}` or
#       `{AMDGPU_CODE_OBJECT_TARGET_FRAGMENT}` placeholders to refer to the
#       code-object target being generated.
# INTERNAL_HDRS: headers that should invalidate device compilation but are not
#                compiled as translation units or exposed as interface headers.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
# INTERNALIZE: whether to internalize linked dependency symbols after lazy
#              archive extraction. Defaults ON. Set OFF when dependencies
#              provide executable ABI symbols such as HAL globals.
# MINIMIZE: apply post-link symbol-table minimization. Only valid for opaque
#           code-object data blobs whose kernels are not looked up by name.
function(iree_amdgpu_binary)
  cmake_parse_arguments(
    _RULE
    "MINIMIZE"
    "NAME;OUT;TARGET;ARCH;INTERNALIZE"
    "SRCS;DEPS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_binary requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_binary requires TARGET")
  endif()
  if(NOT _RULE_ARCH)
    message(FATAL_ERROR "iree_amdgpu_binary requires ARCH")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR "iree_amdgpu_binary requires SRCS")
  endif()
  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE)
    message(FATAL_ERROR
      "iree_amdgpu_binary requires an AMDGPU device toolchain")
  endif()

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.so")
  endif()

  _iree_amdgpu_bitcode_copts(
    _COPTS
    "${_RULE_TARGET}"
    "${_RULE_ARCH}"
    ${_RULE_COPTS}
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
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
        "${_BITCODE_SRC_PATH}"
        "${_RULE_INTERNAL_HDRS}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  set(_SOURCE_BITCODE_FILE "${_RULE_NAME}.srcs.bc")
  add_custom_command(
    OUTPUT
      ${_SOURCE_BITCODE_FILE}
    COMMAND
      ${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_SOURCE_BITCODE_FILE}"
    DEPENDS
      ${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Linking source bitcode to ${_SOURCE_BITCODE_FILE}"
    VERBATIM
  )

  _iree_amdgpu_resolve_bitcode_deps(
    _BITCODE_DEP_PATHS
    _BITCODE_DEP_TARGETS
    ${_RULE_DEPS}
  )

  set(_INTERNALIZE ON)
  if(DEFINED _RULE_INTERNALIZE)
    set(_INTERNALIZE "${_RULE_INTERNALIZE}")
  endif()
  set(_INTERNALIZE_ARGS)
  if(_INTERNALIZE)
    list(APPEND _INTERNALIZE_ARGS "-internalize")
  endif()

  set(_LINKED_FILE "${_RULE_NAME}.bc")
  add_custom_command(
    OUTPUT
      ${_LINKED_FILE}
    COMMAND
      ${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY}
      ${_INTERNALIZE_ARGS}
      "-only-needed"
      "${_SOURCE_BITCODE_FILE}"
      ${_BITCODE_DEP_PATHS}
      "-o" "${_LINKED_FILE}"
    DEPENDS
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_LINK_BINARY}"
      "${_SOURCE_BITCODE_FILE}"
      ${_BITCODE_DEP_PATHS}
      ${_BITCODE_DEP_TARGETS}
    COMMENT
      "Linking bitcode to ${_LINKED_FILE}"
    VERBATIM
  )

  set(_LINK_OUT "${_OUT}")
  set(_LINKOPTS ${_RULE_LINKOPTS})
  if(_RULE_MINIMIZE)
    if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_OBJCOPY_BINARY)
      message(FATAL_ERROR
        "iree_amdgpu_binary(MINIMIZE) requires llvm-objcopy in the AMDGPU "
        "device toolchain")
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
      ${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLD_BINARY}
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
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLD_BINARY}"
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
        ${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_OBJCOPY_BINARY}
        "-R" ".comment"
        "-R" ".AMDGPU.gpr_maximums"
        "--discard-all"
        "-N" "_DYNAMIC"
        "${_LINK_OUT}"
        "${_OUT}"
      DEPENDS
        "${_LINK_OUT}"
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_LLVM_OBJCOPY_BINARY}"
      COMMENT
        "Minimizing AMDGPU binary to ${_OUT}"
      VERBATIM
    )
  endif()

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET_NAME}"
    DEPENDS "${_OUT}"
  )
  _iree_amdgpu_add_generated_input_dependencies(
    "${_TARGET_NAME}" ${_RULE_INTERNAL_HDRS} ${_BITCODE_DEP_PATHS})
  _iree_amdgpu_abs_binary_path(_OUT_PATH "${_OUT}")
  iree_register_generated_output_producer("${_TARGET_NAME}"
    OUTPUTS "${_OUT_PATH}"
  )
endfunction()

# Builds one AMDGPU bitcode archive per selected code-object target.
#
# Parameters:
# NAME: Name of the aggregate target.
# TARGET: LLVM `-target` flag.
# TARGETS: AMDGPU target selectors to expand to code-object targets.
# LIBRARY_NAME_PREFIX: Optional prefix for per-target archive names.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: headers that should invalidate device compilation.
# COPTS: additional flags to pass to clang.
# OUTPUTS_OUT: Optional variable receiving generated output file names relative
#              to the current binary directory.
# OUTPUT_PATHS_OUT: Optional variable receiving absolute generated output paths.
# TARGETS_OUT: Optional variable receiving generated CMake target names.
function(iree_amdgpu_library_variants)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TARGET;LIBRARY_NAME_PREFIX;OUTPUTS_OUT;OUTPUT_PATHS_OUT;TARGETS_OUT"
    "TARGETS;SRCS;INTERNAL_HDRS;COPTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_library_variants requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_library_variants requires TARGET")
  endif()

  iree_package_name(_PACKAGE_NAME)

  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE)
    add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}")
    if(DEFINED _RULE_OUTPUTS_OUT)
      set(${_RULE_OUTPUTS_OUT} "" PARENT_SCOPE)
    endif()
    if(DEFINED _RULE_OUTPUT_PATHS_OUT)
      set(${_RULE_OUTPUT_PATHS_OUT} "" PARENT_SCOPE)
    endif()
    if(DEFINED _RULE_TARGETS_OUT)
      set(${_RULE_TARGETS_OUT} "" PARENT_SCOPE)
    endif()
    return()
  endif()

  if(DEFINED _RULE_LIBRARY_NAME_PREFIX)
    set(_LIBRARY_NAME_PREFIX "${_RULE_LIBRARY_NAME_PREFIX}")
  else()
    set(_LIBRARY_NAME_PREFIX "${_RULE_NAME}")
  endif()

  iree_amdgpu_expand_target_selectors(
    _CODE_OBJECT_TARGETS
    "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}"
    ${_RULE_TARGETS}
  )

  set(_VARIANT_OUTPUTS)
  set(_VARIANT_OUTPUT_PATHS)
  set(_VARIANT_TARGETS)
  set(_TESTONLY_ARG)
  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG TESTONLY)
  endif()
  foreach(_CODE_OBJECT_TARGET ${_CODE_OBJECT_TARGETS})
    iree_amdgpu_target_label_fragment(
      _TARGET_FRAGMENT
      "${_CODE_OBJECT_TARGET}"
    )
    set(_VARIANT_NAME "${_LIBRARY_NAME_PREFIX}_${_TARGET_FRAGMENT}")
    set(_VARIANT_OUTPUT "${_VARIANT_NAME}.a")
    iree_amdgpu_library(
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
      ${_TESTONLY_ARG}
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

# Builds one device-library-linked HIP code object.
function(_iree_amdgpu_hip_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;TARGET;ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  if(NOT _RULE_TARGET STREQUAL "amdgcn-amd-amdhsa")
    message(FATAL_ERROR
      "HIP device sources require TARGET amdgcn-amd-amdhsa")
  endif()
  list(LENGTH _RULE_SRCS _SOURCE_COUNT)
  if(NOT _SOURCE_COUNT EQUAL 1)
    message(FATAL_ERROR "HIP device sources require exactly one source")
  endif()
  if(NOT IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_HIP_AVAILABLE)
    message(FATAL_ERROR
      "HIP device sources require ROCm device libraries and "
      "clang-offload-bundler")
  endif()

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.so")
  endif()
  get_filename_component(_SOURCE_PATH "${_RULE_SRCS}" REALPATH)
  set(_OFFLOAD_BUNDLE "${_RULE_NAME}.offload_bundle")
  file(GLOB _DEVICE_LIBRARIES CONFIGURE_DEPENDS
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_ROCM_DEVICE_LIB_PATH}/*.bc")

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
      "--cuda-device-only"
      "-x" "hip"
      "-nogpuinc"
      "--rocm-device-lib-path=${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_ROCM_DEVICE_LIB_PATH}"
      "--offload-arch=${_RULE_ARCH}"
      "-fno-gpu-rdc"
      "-fno-ident"
      "-O3"
      ${_RULE_COPTS}
      "${_SOURCE_PATH}"
      "-o" "${_OFFLOAD_BUNDLE}"
    COMMAND
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_OFFLOAD_BUNDLER_BINARY}"
      "--unbundle"
      "--type=o"
      "--targets=hipv4-amdgcn-amd-amdhsa--${_RULE_ARCH}"
      "--input=${_OFFLOAD_BUNDLE}"
      "--output=${_OUT}"
    DEPENDS
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_BINARY}"
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_RESOLVED_CLANG_OFFLOAD_BUNDLER_BINARY}"
      "${_SOURCE_PATH}"
      ${_DEVICE_LIBRARIES}
    COMMENT
      "Compiling HIP code object ${_RULE_NAME} for ${_RULE_ARCH}"
    VERBATIM
  )

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET_NAME}" DEPENDS "${_OUT}")
  _iree_amdgpu_abs_binary_path(_OUT_PATH "${_OUT}")
  iree_register_generated_output_producer("${_TARGET_NAME}"
    OUTPUTS "${_OUT_PATH}"
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
# DEPS: bitcode archive dependencies to link with `llvm-link -only-needed`.
# INTERNAL_HDRS: headers that should invalidate device compilation.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
# OUTPUTS_OUT: Optional variable receiving generated output file names relative
#              to the current binary directory.
# OUTPUT_PATHS_OUT: Optional variable receiving absolute generated output paths.
# TARGETS_OUT: Optional variable receiving generated CMake target names.
# INTERNALIZE: whether to internalize linked dependency symbols after lazy
#              archive extraction. Defaults ON.
# MINIMIZE: apply post-link symbol-table minimization.
# SOURCE_FORMAT: source compilation pipeline. freestanding_c uses the HAL
#                device build; hip links the ROCm device libraries.
function(iree_amdgpu_binary_variants)
  cmake_parse_arguments(
    _RULE
    "MINIMIZE"
    "NAME;TARGET;BINARY_NAME_PREFIX;OUTPUTS_OUT;OUTPUT_PATHS_OUT;TARGETS_OUT;INTERNALIZE;SOURCE_FORMAT"
    "TARGETS;SRCS;DEPS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "iree_amdgpu_binary_variants requires NAME")
  endif()
  if(NOT _RULE_TARGET)
    message(FATAL_ERROR "iree_amdgpu_binary_variants requires TARGET")
  endif()
  if(NOT DEFINED _RULE_SOURCE_FORMAT)
    set(_RULE_SOURCE_FORMAT "freestanding_c")
  elseif(NOT _RULE_SOURCE_FORMAT STREQUAL "freestanding_c" AND
         NOT _RULE_SOURCE_FORMAT STREQUAL "hip")
    message(FATAL_ERROR
      "Unsupported AMDGPU binary SOURCE_FORMAT: ${_RULE_SOURCE_FORMAT}")
  endif()
  if(_RULE_SOURCE_FORMAT STREQUAL "hip" AND
     (_RULE_DEPS OR _RULE_INTERNAL_HDRS OR _RULE_LINKOPTS OR
      _RULE_MINIMIZE OR DEFINED _RULE_INTERNALIZE))
    message(FATAL_ERROR
      "HIP device sources do not support DEPS, INTERNAL_HDRS, LINKOPTS, "
      "MINIMIZE, or INTERNALIZE")
  endif()

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_BINARY_NAME_PREFIX)
    set(_BINARY_NAME_PREFIX "${_RULE_BINARY_NAME_PREFIX}")
  else()
    set(_BINARY_NAME_PREFIX "${_RULE_NAME}")
  endif()

  set(_DEVICE_SOURCE_AVAILABLE
    "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE}")
  if(_RULE_SOURCE_FORMAT STREQUAL "hip")
    set(_DEVICE_SOURCE_AVAILABLE
      "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_HIP_AVAILABLE}")
  endif()

  set(_CODE_OBJECT_TARGETS)
  if(_DEVICE_SOURCE_AVAILABLE)
    iree_amdgpu_expand_target_selectors(
      _CODE_OBJECT_TARGETS
      "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}"
      ${_RULE_TARGETS}
    )
  endif()

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
    _iree_amdgpu_code_object_target_deps(
      _VARIANT_DEPS
      "${_CODE_OBJECT_TARGET}"
      ${_RULE_DEPS}
    )

    if(_RULE_SOURCE_FORMAT STREQUAL "hip")
      _iree_amdgpu_hip_binary(
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
        COPTS
          ${_RULE_COPTS}
      )
    else()
      set(_MINIMIZE)
      if(_RULE_MINIMIZE)
        set(_MINIMIZE MINIMIZE)
      endif()
      set(_INTERNALIZE_ARG)
      if(DEFINED _RULE_INTERNALIZE)
        set(_INTERNALIZE_ARG INTERNALIZE "${_RULE_INTERNALIZE}")
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
        DEPS
          ${_VARIANT_DEPS}
        INTERNAL_HDRS
          ${_RULE_INTERNAL_HDRS}
        COPTS
          ${_RULE_COPTS}
        LINKOPTS
          ${_RULE_LINKOPTS}
        ${_MINIMIZE}
        ${_INTERNALIZE_ARG}
      )
    endif()

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
# DEPS: bitcode archive dependencies to link into each generated binary.
# INCLUDES: include paths for the generated C embed-data library.
# FLATTEN: drop directory components from table-of-contents names.
# PUBLIC: expose the generated C embed-data library publicly.
# TESTONLY: only build the generated library when tests are enabled.
# MINIMIZE: apply post-link symbol-table minimization.
# INTERNALIZE: whether to internalize linked dependency symbols after lazy
#              archive extraction. Defaults ON.
# SOURCE_FORMAT: source compilation pipeline. See
#                iree_amdgpu_binary_variants.
function(iree_amdgpu_binary_variants_embed_data)
  cmake_parse_arguments(
    _RULE
    "FLATTEN;PUBLIC;TESTONLY;MINIMIZE"
    "NAME;TARGET;BINARY_NAME_PREFIX;C_FILE_OUTPUT;H_FILE_OUTPUT;IDENTIFIER;INTERNALIZE;SOURCE_FORMAT"
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
  set(_INTERNALIZE_ARG)
  if(DEFINED _RULE_INTERNALIZE)
    set(_INTERNALIZE_ARG INTERNALIZE "${_RULE_INTERNALIZE}")
  endif()
  set(_SOURCE_FORMAT_ARG)
  if(DEFINED _RULE_SOURCE_FORMAT)
    set(_SOURCE_FORMAT_ARG SOURCE_FORMAT "${_RULE_SOURCE_FORMAT}")
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
    DEPS
      ${_RULE_DEPS}
    ${_SOURCE_FORMAT_ARG}
    ${_MINIMIZE_ARG}
    ${_INTERNALIZE_ARG}
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
    INCLUDES
      ${_RULE_INCLUDES}
    ${_FLATTEN_ARG}
    ${_PUBLIC_ARG}
    ${_TESTONLY_ARG}
  )
endfunction()
