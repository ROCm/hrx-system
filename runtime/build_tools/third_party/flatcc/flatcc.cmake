# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
include(iree_third_party_helpers)

function(_iree_fetch_flatcc_if_needed out_source_dir)
  find_package(flatcc CONFIG QUIET)
  if(TARGET flatcc::runtime AND TARGET flatcc::parsing AND TARGET iree-flatcc-cli)
    set(${out_source_dir} "" PARENT_SCOPE)
    return()
  endif()

  iree_fetch_content_assert_allowed("flatcc")
  FetchContent_Declare(
    flatcc
    URL https://github.com/dvidelabs/flatcc/archive/9362cd00f0007d8cbee7bff86e90fb4b6b227ff3.tar.gz
    URL_HASH SHA256=f77f842e996f5bbfa25305a7b38b40d1325fb44154f2bf9880c58356ece92c62
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(flatcc)
  if(NOT flatcc_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(flatcc)
    if(POLICY CMP0169)
      cmake_policy(POP)
    endif()
  endif()
  set(${out_source_dir} "${flatcc_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

function(iree_configure_flatcc)
  _iree_fetch_flatcc_if_needed(_flatcc_source_dir)
  if(TARGET flatcc::runtime AND TARGET flatcc::parsing AND TARGET iree-flatcc-cli)
    return()
  endif()
  if(NOT _flatcc_source_dir)
    message(FATAL_ERROR
      "flatcc package was found but did not provide flatcc::runtime, "
      "flatcc::parsing, and iree-flatcc-cli")
  endif()

  set(IREE_FLATCC_SOURCE_DIR "${_flatcc_source_dir}" CACHE PATH
    "flatcc source directory used by runtime schema generation." FORCE)
  set(IREE_FLATCC_INCLUDE_DIR "${IREE_FLATCC_SOURCE_DIR}/include" CACHE PATH
    "flatcc include directory used by generated schema targets." FORCE)

  add_library(flatcc_parsing STATIC
    "${IREE_FLATCC_SOURCE_DIR}/config/config.h"
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/verifier.c"
  )
  target_include_directories(flatcc_parsing SYSTEM PUBLIC
    "${IREE_FLATCC_SOURCE_DIR}/include"
  )
  add_library(flatcc::parsing ALIAS flatcc_parsing)

  add_library(flatcc_runtime STATIC
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/builder.c"
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/emitter.c"
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/json_parser.c"
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/json_printer.c"
    "${IREE_FLATCC_SOURCE_DIR}/src/runtime/refmap.c"
  )
  target_include_directories(flatcc_runtime SYSTEM PUBLIC
    "${IREE_FLATCC_SOURCE_DIR}/include"
  )
  target_link_libraries(flatcc_runtime PUBLIC flatcc::parsing)
  add_library(flatcc::runtime ALIAS flatcc_runtime)

  if(IREE_ENABLE_POSIX)
    add_executable(iree-flatcc-cli
      "${IREE_FLATCC_SOURCE_DIR}/src/cli/flatcc_cli.c"
      "${IREE_FLATCC_SOURCE_DIR}/external/hash/cmetrohash64.c"
      "${IREE_FLATCC_SOURCE_DIR}/external/hash/str_set.c"
      "${IREE_FLATCC_SOURCE_DIR}/external/hash/ptr_set.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/hash_tables/symbol_table.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/hash_tables/scope_table.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/hash_tables/name_table.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/hash_tables/schema_table.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/hash_tables/value_set.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/fileio.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/parser.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/semantics.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/coerce.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_schema.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/flatcc.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_reader.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_sort.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_builder.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_verifier.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_sorter.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_json_parser.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/compiler/codegen_c_json_printer.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/runtime/builder.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/runtime/emitter.c"
      "${IREE_FLATCC_SOURCE_DIR}/src/runtime/refmap.c"
    )
    target_include_directories(iree-flatcc-cli SYSTEM PUBLIC
      "${IREE_FLATCC_SOURCE_DIR}/external"
      "${IREE_FLATCC_SOURCE_DIR}/include"
      "${IREE_FLATCC_SOURCE_DIR}/config"
    )
    set_target_properties(iree-flatcc-cli PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
    # This executable is a host-side generator invoked during the build. Do not
    # MSAN-instrument it unless the host C runtime and flatcc tool dependencies
    # are also instrumented; otherwise the build fails before runtime/test code
    # gets compiled.
    if(IREE_ENABLE_MSAN)
      target_compile_options(iree-flatcc-cli PRIVATE "-fno-sanitize=memory")
      target_link_options(iree-flatcc-cli PRIVATE "-fno-sanitize=memory")
    endif()
  elseif(IREE_HOST_BIN_DIR)
    iree_import_binary(NAME iree-flatcc-cli)
  else()
    message(FATAL_ERROR
      "flatcc schema generation requires POSIX support or IREE_HOST_BIN_DIR")
  endif()

  if(NOT TARGET flatcc)
    add_executable(flatcc ALIAS iree-flatcc-cli)
  endif()
endfunction()
