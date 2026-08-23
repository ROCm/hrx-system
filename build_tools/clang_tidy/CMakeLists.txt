# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)

project(IREEClangTidy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

set(IREE_CLANG_TIDY_SOURCES
  iree/ExtentChecks.cc
  iree/IreeTidyModule.cc
  iree/LifecycleChecks.cc
  iree/RecursionCheck.cc
  iree/RefCountChecks.cc
  iree/SmokeCheck.cc
  iree/StatusChecks.cc
  iree/StyleChecks.cc
  iree/TraceChecks.cc
)

if(WIN32)
  add_executable(IREEClangTidy
    iree_clang_tidy_main.cc
    ${IREE_CLANG_TIDY_SOURCES}
  )
  set(IREE_CLANG_TIDY_TARGET IREEClangTidy)
  set(IREE_CLANG_TIDY_TEST_EXECUTABLE "$<TARGET_FILE:IREEClangTidy>")
  set(IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT)
  file(GLOB IREE_CLANG_TIDY_LLVM_STATIC_LIBRARIES
    "${LLVM_LIBRARY_DIRS}/*.lib"
  )
  list(FILTER IREE_CLANG_TIDY_LLVM_STATIC_LIBRARIES EXCLUDE REGEX
    "[/\\\\](LLVM-C|LTO|Remarks|libclang)\\.lib$"
  )
  target_link_libraries(IREEClangTidy
    PRIVATE
      ${IREE_CLANG_TIDY_LLVM_STATIC_LIBRARIES}
      advapi32
      ntdll
      ole32
      psapi
      shell32
      uuid
      version
      ws2_32
  )
  target_compile_definitions(IREEClangTidy PRIVATE CLANG_BUILD_STATIC)
  set_property(TARGET IREEClangTidy PROPERTY
    MSVC_RUNTIME_LIBRARY MultiThreaded
  )
else()
  add_library(IREEClangTidyPlugin MODULE ${IREE_CLANG_TIDY_SOURCES})
  set(IREE_CLANG_TIDY_TARGET IREEClangTidyPlugin)
  find_program(CLANG_TIDY_EXECUTABLE
    NAMES
      clang-tidy
      clang-tidy-${LLVM_VERSION_MAJOR}
    HINTS
      "${LLVM_TOOLS_BINARY_DIR}"
    REQUIRED
  )
  set(IREE_CLANG_TIDY_TEST_EXECUTABLE "${CLANG_TIDY_EXECUTABLE}")
  set(IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
  )
endif()

target_include_directories(${IREE_CLANG_TIDY_TARGET}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)
target_compile_definitions(${IREE_CLANG_TIDY_TARGET}
  PRIVATE
    ${LLVM_DEFINITIONS}
)

if(NOT LLVM_ENABLE_EH)
  if(MSVC)
    target_compile_definitions(${IREE_CLANG_TIDY_TARGET}
      PRIVATE _HAS_EXCEPTIONS=0
    )
    target_compile_options(${IREE_CLANG_TIDY_TARGET} PRIVATE /EHs-c-)
  else()
    target_compile_options(${IREE_CLANG_TIDY_TARGET} PRIVATE -fno-exceptions)
  endif()
endif()
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(${IREE_CLANG_TIDY_TARGET} PRIVATE /GR-)
  else()
    target_compile_options(${IREE_CLANG_TIDY_TARGET} PRIVATE -fno-rtti)
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

enable_testing()
add_test(
  NAME iree-clang-tidy-extent-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/extent_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-plugin-smoke
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-status-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/status_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-lifecycle-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/lifecycle_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-recursion-check
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/recursion_check_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-recursion-cross-tu
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/recursion_cross_tu_driver_test.py"
    "--aggregator=${CMAKE_CURRENT_SOURCE_DIR}/aggregate_recursion_summaries.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-refcount-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/refcount_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-trace-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/trace_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)

add_test(
  NAME iree-clang-tidy-style-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/style_checks_test.py"
    "--clang-tidy=${IREE_CLANG_TIDY_TEST_EXECUTABLE}"
    ${IREE_CLANG_TIDY_TEST_PLUGIN_ARGUMENT}
)
