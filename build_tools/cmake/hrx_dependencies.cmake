# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

function(_hrx_assert_fetch_allowed dep_name)
  if(HRX_HERMETIC_BUILD)
    message(FATAL_ERROR
      "${dep_name} was not found via package discovery and HRX_HERMETIC_BUILD=ON "
      "forbids FetchContent. Provide the package via CMAKE_PREFIX_PATH or an "
      "equivalent CMake package path.")
  endif()
endfunction()

function(_hrx_alias_interface alias_name)
  set(_deps ${ARGN})
  if(TARGET ${alias_name})
    return()
  endif()
  string(MAKE_C_IDENTIFIER "${alias_name}" _target_name)
  set(_target_name "hrx_${_target_name}")
  add_library(${_target_name} INTERFACE)
  target_link_libraries(${_target_name} INTERFACE ${_deps})
  add_library(${alias_name} ALIAS ${_target_name})
endfunction()

function(_hrx_configure_googletest)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  find_package(GTest CONFIG QUIET)
  if(NOT TARGET GTest::gtest)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::gtest)
    _hrx_assert_fetch_allowed("googletest")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
  endif()

  if(TARGET gtest)
    # FetchContent's googletest build already defines the traditional target.
  elseif(TARGET GTest::gtest)
    _hrx_alias_interface(gtest GTest::gtest)
  else()
    message(FATAL_ERROR "googletest did not provide a gtest target")
  endif()

  if(TARGET gtest_main)
    # Provided by FetchContent.
  elseif(TARGET GTest::gtest_main)
    _hrx_alias_interface(gtest_main GTest::gtest_main)
  endif()

  if(TARGET gmock)
    # Provided by FetchContent.
  elseif(TARGET GTest::gmock)
    _hrx_alias_interface(gmock GTest::gmock)
  else()
    # Some packaged GTest installs omit gmock. IREE's test helpers link gmock
    # unconditionally, so provide a compatibility target that forwards to gtest.
    _hrx_alias_interface(gmock gtest)
  endif()

  if(TARGET gmock_main)
    # Provided by FetchContent.
  elseif(TARGET GTest::gmock_main)
    _hrx_alias_interface(gmock_main GTest::gmock_main)
  endif()
endfunction()

function(_hrx_configure_benchmark)
  if(NOT IREE_ENABLE_THREADING OR NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  find_package(benchmark CONFIG QUIET)
  if(NOT TARGET benchmark AND NOT TARGET benchmark::benchmark)
    _hrx_assert_fetch_allowed("google benchmark")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.5
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(google_benchmark)
  endif()

  if(TARGET benchmark)
    return()
  endif()
  if(TARGET benchmark::benchmark)
    _hrx_alias_interface(benchmark benchmark::benchmark)
    return()
  endif()
  message(FATAL_ERROR "google benchmark did not provide a benchmark target")
endfunction()

function(_hrx_configure_catch2)
  if(NOT LIBHRX_BUILD_CTS)
    return()
  endif()

  find_package(Catch2 3 CONFIG QUIET)
  if(NOT TARGET Catch2::Catch2)
    _hrx_assert_fetch_allowed("Catch2")
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.8.1
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(catch2)
  endif()

  if(NOT TARGET Catch2::Catch2)
    message(FATAL_ERROR "Catch2 did not provide a Catch2::Catch2 target")
  endif()
endfunction()

function(_hrx_fetch_flatcc_if_needed out_source_dir)
  find_package(flatcc CONFIG QUIET)
  if(TARGET flatcc::runtime AND TARGET flatcc::parsing AND TARGET iree-flatcc-cli)
    set(${out_source_dir} "" PARENT_SCOPE)
    return()
  endif()

  _hrx_assert_fetch_allowed("flatcc")
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

function(_hrx_configure_flatcc)
  _hrx_fetch_flatcc_if_needed(_flatcc_source_dir)
  if(TARGET flatcc::runtime AND TARGET flatcc::parsing AND TARGET iree-flatcc-cli)
    return()
  endif()
  if(NOT _flatcc_source_dir)
    message(FATAL_ERROR
      "flatcc package was found but did not provide flatcc::runtime, "
      "flatcc::parsing, and iree-flatcc-cli")
  endif()

  set(IREE_FLATCC_SOURCE_DIR "${_flatcc_source_dir}" CACHE PATH
    "flatcc source directory used by HRX's runtime schema generation." FORCE)
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

function(_hrx_configure_libbacktrace)
  if(NOT IREE_ENABLE_LIBBACKTRACE)
    set(IREE_LIBBACKTRACE_TARGET "" CACHE INTERNAL
      "libbacktrace target (empty when disabled)" FORCE)
    return()
  endif()
  _hrx_assert_fetch_allowed("libbacktrace")
  add_subdirectory(
    "${CMAKE_SOURCE_DIR}/build_tools/third_party/libbacktrace"
    "${CMAKE_BINARY_DIR}/build_tools/third_party/libbacktrace"
    EXCLUDE_FROM_ALL
  )
endfunction()

function(_hrx_configure_amdgpu_toolchain)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "IREE_HAL_DRIVER_AMDGPU=ON requires the configured C compiler to be "
      "Clang-family and capable of targeting amdgcn-amd-amdhsa. Configure "
      "with ROCm/LLVM clang instead of relying on a separate ROCm path knob.")
  endif()

  set(IREE_CLANG_BINARY "${CMAKE_C_COMPILER}" CACHE FILEPATH
    "Clang used by HRX/IREE AMDGPU device binary builds." FORCE)

  set(_probe_src "${CMAKE_BINARY_DIR}/CMakeFiles/hrx-amdgcn-probe.c")
  set(_probe_obj "${CMAKE_BINARY_DIR}/CMakeFiles/hrx-amdgcn-probe.bc")
  file(WRITE "${_probe_src}" "void hrx_amdgcn_probe(void) {}\n")
  execute_process(
    COMMAND "${IREE_CLANG_BINARY}"
      -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib
      -x c -std=c11 -c -emit-llvm "${_probe_src}" -o "${_probe_obj}"
    RESULT_VARIABLE _probe_result
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
  )
  if(NOT _probe_result EQUAL 0)
    message(FATAL_ERROR
      "Configured C compiler cannot compile a minimal amdgcn-amd-amdhsa object.\n"
      "Compiler: ${IREE_CLANG_BINARY}\n"
      "stderr:\n${_probe_stderr}")
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-link
    OUTPUT_VARIABLE _llvm_link_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_llvm_link_from_clang}")
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_from_clang}" CACHE FILEPATH
      "llvm-link used by HRX/IREE AMDGPU device binary builds." FORCE)
  else()
    find_program(_llvm_link_fallback llvm-link REQUIRED)
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_fallback}" CACHE FILEPATH
      "llvm-link used by HRX/IREE AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=ld.lld
    OUTPUT_VARIABLE _lld_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_lld_from_clang}")
    set(IREE_LLD_BINARY "${_lld_from_clang}" CACHE FILEPATH
      "lld used by HRX/IREE AMDGPU device binary builds." FORCE)
  else()
    find_program(_lld_fallback NAMES ld.lld lld REQUIRED)
    set(IREE_LLD_BINARY "${_lld_fallback}" CACHE FILEPATH
      "lld used by HRX/IREE AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-objcopy
    OUTPUT_VARIABLE _objcopy_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(EXISTS "${_objcopy_from_clang}")
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_from_clang}" CACHE FILEPATH
      "llvm-objcopy used by HRX/IREE AMDGPU device binary builds." FORCE)
  else()
    find_program(_objcopy_fallback llvm-objcopy REQUIRED)
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_fallback}" CACHE FILEPATH
      "llvm-objcopy used by HRX/IREE AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-resource-dir
    OUTPUT_VARIABLE _clang_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _clang_resource_dir OR NOT EXISTS "${_clang_resource_dir}/include")
    message(FATAL_ERROR
      "Could not determine clang resource include directory from ${IREE_CLANG_BINARY}")
  endif()
  set(IREE_CLANG_BUILTIN_HEADERS_PATH "${_clang_resource_dir}/include" CACHE PATH
    "Clang resource include directory used by AMDGPU device binary builds." FORCE)
endfunction()

function(_hrx_configure_rocm_headers)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()

  find_package(hsa-runtime64 CONFIG REQUIRED)
  if(NOT TARGET hsa_runtime::headers)
    get_target_property(_hsa_runtime_include_dirs
      hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _hsa_runtime_include_dirs)
      message(FATAL_ERROR
        "hsa-runtime64::hsa-runtime64 does not publish include directories")
    endif()
    add_library(hsa_runtime_headers INTERFACE)
    target_include_directories(hsa_runtime_headers SYSTEM INTERFACE
      ${_hsa_runtime_include_dirs})
    add_library(hsa_runtime::headers ALIAS hsa_runtime_headers)
  endif()

  # TODO(upstream ROCm packaging): replace this compatibility target with:
  #
  #   find_package(aqlprofile-sdk CONFIG REQUIRED)
  #
  # and link against the target exported by that package. As of the ROCm nightly
  # used for this bring-up, the SDK headers are installed under
  # include/aqlprofile-sdk (including the generated version.h), but there is no
  # CMake package config dedicated to the headers.
  #
  # The temporary assumption, confirmed for the local ROCm tree, is that the HSA
  # runtime package's include directories are broad enough to make
  # aqlprofile-sdk/aql_profile_v2.h and aqlprofile-sdk/version.h visible. This
  # is intentionally represented as a separate aqlprofile-sdk::headers target so
  # generated CMake can model the real dependency shape now. Once ROCm publishes
  # the package, only this shim should change.
  if(NOT TARGET aqlprofile-sdk::headers)
    add_library(aqlprofile_sdk_headers INTERFACE)
    target_link_libraries(aqlprofile_sdk_headers INTERFACE hsa_runtime::headers)
    add_library(aqlprofile-sdk::headers ALIAS aqlprofile_sdk_headers)
  endif()
endfunction()

function(hrx_configure_dependencies)
  _hrx_configure_amdgpu_toolchain()
  _hrx_configure_rocm_headers()
  _hrx_configure_libbacktrace()
  _hrx_configure_googletest()
  _hrx_configure_benchmark()
  _hrx_configure_catch2()
  _hrx_configure_flatcc()
endfunction()
