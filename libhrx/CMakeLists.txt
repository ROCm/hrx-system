# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(libhrx VERSION ${HRX_VERSION} LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LIBHRX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBHRX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT TARGET iree_vm_vm)
  message(FATAL_ERROR
    "libhrx must be added after the reduced IREE runtime has been configured")
endif()

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX libhrx
)

add_library(libhrx_defs INTERFACE)
target_include_directories(libhrx_defs INTERFACE
  "$<BUILD_INTERFACE:${LIBHRX_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${IREE_ROOT_DIR}/runtime/src>"
  "$<BUILD_INTERFACE:${IREE_BINARY_DIR}/runtime/src>"
)
include_directories(
  "${IREE_ROOT_DIR}/runtime/src"
  "${IREE_BINARY_DIR}/runtime/src"
)

function(hrx_cc_test)
  set(_libhrx_saved_iree_build_tests "${IREE_BUILD_TESTS}")
  set(IREE_BUILD_TESTS ON)
  iree_cc_test(${ARGN})
  set(IREE_BUILD_TESTS "${_libhrx_saved_iree_build_tests}")
endfunction()

add_subdirectory(src/libhrx)
add_subdirectory(src/streaming)
add_subdirectory(src/binding/hip)
add_subdirectory(tools)

if(LIBHRX_BUILD_CUDA_BINDING)
  add_subdirectory(src/binding/cuda)
endif()

if(LIBHRX_BUILD_CTS)
  enable_testing()
  add_subdirectory(cts)
endif()

if(LIBHRX_BUILD_PASSTHROUGH)
  add_subdirectory(src/passthrough)
endif()

set(_hrx_cmake_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/hrx")
set(_hrx_build_package_dir "${LIBHRX_BINARY_DIR}/cmake/hrx")

install(TARGETS libhrx_src_libhrx_hrx EXPORT hrx-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    NAMELINK_COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hrx")
install(FILES
  include/hrx_runtime.h
  include/hrx_runtime_cxx.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hrx"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
install(EXPORT hrx-targets
  NAMESPACE hrx::
  DESTINATION "${_hrx_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")

configure_package_config_file(
  cmake/hrx-config.cmake.in
  "${_hrx_build_package_dir}/hrx-config.cmake"
  INSTALL_DESTINATION "${_hrx_cmake_package_dir}")
write_basic_package_version_file(
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
export(EXPORT hrx-targets
  NAMESPACE hrx::
  FILE "${_hrx_build_package_dir}/hrx-targets.cmake")

install(FILES
  "${_hrx_build_package_dir}/hrx-config.cmake"
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  DESTINATION "${_hrx_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
