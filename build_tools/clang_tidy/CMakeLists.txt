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

add_library(IREEClangTidyPlugin MODULE
  iree/IreeTidyModule.cc
  iree/LifecycleChecks.cc
  iree/RefCountChecks.cc
  iree/SmokeCheck.cc
  iree/StatusChecks.cc
  iree/StyleChecks.cc
  iree/TraceChecks.cc
)

target_include_directories(IREEClangTidyPlugin
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)
target_compile_definitions(IREEClangTidyPlugin
  PRIVATE
    ${LLVM_DEFINITIONS}
)

if(NOT LLVM_ENABLE_EH)
  if(MSVC)
    target_compile_options(IREEClangTidyPlugin PRIVATE /EHs-c-)
  else()
    target_compile_options(IREEClangTidyPlugin PRIVATE -fno-exceptions)
  endif()
endif()
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(IREEClangTidyPlugin PRIVATE /GR-)
  else()
    target_compile_options(IREEClangTidyPlugin PRIVATE -fno-rtti)
  endif()
endif()

find_program(CLANG_TIDY_EXECUTABLE
  NAMES
    clang-tidy
    clang-tidy-${LLVM_VERSION_MAJOR}
  HINTS
    "${LLVM_TOOLS_BINARY_DIR}"
  REQUIRED
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

enable_testing()
add_test(
  NAME iree-clang-tidy-plugin-smoke
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)

add_test(
  NAME iree-clang-tidy-status-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/status_checks_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)

add_test(
  NAME iree-clang-tidy-lifecycle-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/lifecycle_checks_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)

add_test(
  NAME iree-clang-tidy-refcount-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/refcount_checks_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)

add_test(
  NAME iree-clang-tidy-trace-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/trace_checks_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)

add_test(
  NAME iree-clang-tidy-style-checks
  COMMAND
    Python3::Interpreter
    "${CMAKE_CURRENT_SOURCE_DIR}/test/style_checks_test.py"
    "--clang-tidy=${CLANG_TIDY_EXECUTABLE}"
    "--plugin=$<TARGET_FILE:IREEClangTidyPlugin>"
)
