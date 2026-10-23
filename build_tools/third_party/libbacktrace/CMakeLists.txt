# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build libbacktrace from source for Linux stack traces.
# Uses FetchContent to pull upstream and provides our own CMake build.
#
# libbacktrace (https://github.com/ianlancetaylor/libbacktrace) provides
# complete stack trace functionality including unwinding and symbol resolution
# with file:line:function information.

# Only build libbacktrace on Linux when enabled.
# Other platforms have their own stack trace implementations:
# - Windows: dbghelp
# - macOS: backtrace+dladdr
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT IREE_ENABLE_LIBBACKTRACE)
  # Set empty target variable for consumers to use in DEPS.
  set(IREE_LIBBACKTRACE_TARGET "" CACHE INTERNAL "libbacktrace target (empty when disabled)")
  return()
endif()

# Set target variable for consumers to use in DEPS.
set(IREE_LIBBACKTRACE_TARGET "iree::third_party::libbacktrace" CACHE INTERNAL "libbacktrace target")

include(CheckSymbolExists)
include(iree_third_party_helpers)

# pthreads is required for pthread_once used in status_stack_trace.c.
find_package(Threads REQUIRED)

# Suppress deprecation warning for FetchContent_Populate (CMake 4.x).
# We use Populate because libbacktrace has no CMakeLists.txt.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

iree_declare_locked_fetch_content(libbacktrace FETCH_NAME libbacktrace_src)

# Just populate, don't configure (upstream has no CMake).
FetchContent_GetProperties(libbacktrace_src)
if(NOT libbacktrace_src_POPULATED)
  FetchContent_Populate(libbacktrace_src)
endif()
set(LIBBACKTRACE_SOURCE_DIR "${libbacktrace_src_SOURCE_DIR}")

# Detect dl_iterate_phdr availability at configure time.
# This is critical for cross-compilation: the check uses the cross-toolchain
# so it correctly detects target capabilities, not host.
# - glibc: always has it
# - musl: has it (with some quirks on older versions)
# - uclibc-ng: depends on configuration
# - newlib/other minimal libc: usually not
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(dl_iterate_phdr "link.h" LIBBACKTRACE_HAVE_DL_ITERATE_PHDR)
unset(CMAKE_REQUIRED_DEFINITIONS)

if(LIBBACKTRACE_HAVE_DL_ITERATE_PHDR)
  set(LIBBACKTRACE_DL_ITERATE_PHDR_DEF "#define HAVE_DL_ITERATE_PHDR 1")
  set(LIBBACKTRACE_LINK_H_DEF "#define HAVE_LINK_H 1")
else()
  message(STATUS "libbacktrace: dl_iterate_phdr not found, shared library symbols won't be resolved")
  set(LIBBACKTRACE_DL_ITERATE_PHDR_DEF "/* dl_iterate_phdr not available */")
  set(LIBBACKTRACE_LINK_H_DEF "/* link.h not available */")
endif()

# Generate config.h with detected features.
# Uses @VAR@ substitution for cross-compilation-safe detection.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${LIBBACKTRACE_SOURCE_DIR}/config.h
  @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/backtrace-supported.h
  ${LIBBACKTRACE_SOURCE_DIR}/backtrace-supported.h
  COPYONLY
)
# Copy our wrapper header that re-exports backtrace.h.
# This provides a consistent include path for both CMake and Bazel builds,
# allowing status_stack_trace.c to use <iree_libbacktrace.h>.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/iree_libbacktrace.h
  ${LIBBACKTRACE_SOURCE_DIR}/iree_libbacktrace.h
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/symbol_prefix.h
  ${LIBBACKTRACE_SOURCE_DIR}/symbol_prefix.h
  COPYONLY
)

# Linux has ELF, mmap and unwind support. Select those implementations instead
# of placing competing definitions and fallback stubs in the same archive.
set(LIBBACKTRACE_SOURCES
  # Core files (always needed).
  ${LIBBACKTRACE_SOURCE_DIR}/atomic.c
  ${LIBBACKTRACE_SOURCE_DIR}/dwarf.c
  ${LIBBACKTRACE_SOURCE_DIR}/fileline.c
  ${LIBBACKTRACE_SOURCE_DIR}/posix.c
  ${LIBBACKTRACE_SOURCE_DIR}/print.c
  ${LIBBACKTRACE_SOURCE_DIR}/sort.c
  ${LIBBACKTRACE_SOURCE_DIR}/state.c
  ${LIBBACKTRACE_SOURCE_DIR}/backtrace.c
  ${LIBBACKTRACE_SOURCE_DIR}/simple.c
  ${LIBBACKTRACE_SOURCE_DIR}/elf.c
  ${LIBBACKTRACE_SOURCE_DIR}/mmap.c
  ${LIBBACKTRACE_SOURCE_DIR}/mmapio.c
)

add_library(iree_libbacktrace_impl STATIC ${LIBBACKTRACE_SOURCES})
target_include_directories(iree_libbacktrace_impl
  PUBLIC $<BUILD_INTERFACE:${LIBBACKTRACE_SOURCE_DIR}>
)
target_compile_definitions(iree_libbacktrace_impl
  PRIVATE _GNU_SOURCE  # Required for dl_iterate_phdr on glibc.
)
# Suppress warnings in third-party code.
target_compile_options(iree_libbacktrace_impl
  PRIVATE
    -Wno-unused-parameter
    -Wno-cast-function-type
)

# Create the private wrapper as an IMPORTED INTERFACE library.
# IMPORTED libraries don't get tracked for export, which is what we want
# since libbacktrace is only used internally during build.
# Threads::Threads is needed for pthread_once used in status_stack_trace.c.
# CMAKE_DL_LIBS is needed for dladdr used in status_stack_trace.c.
# On modern glibc (2.34+), dladdr is in libc, but older systems need libdl.
add_library(iree_libbacktrace INTERFACE IMPORTED GLOBAL)
target_link_libraries(iree_libbacktrace
  INTERFACE
    iree_libbacktrace_impl
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
target_include_directories(iree_libbacktrace
  INTERFACE $<BUILD_INTERFACE:${LIBBACKTRACE_SOURCE_DIR}>
)
target_compile_definitions(iree_libbacktrace
  INTERFACE IREE_HAVE_LIBBACKTRACE=1 IREE_LIBBACKTRACE_PRIVATE=1
)
iree_add_alias_interface(iree::third_party::libbacktrace iree_libbacktrace)

iree_setup_c_src_root(PACKAGE_ROOT_PREFIX iree/third_party/libbacktrace)

iree_cc_test(
  NAME
    symbol_isolation_test
  SRCS
    "symbol_isolation_test.cc"
  DEPS
    iree::base::base
    iree::testing::gtest
    iree::testing::gtest_main
)
