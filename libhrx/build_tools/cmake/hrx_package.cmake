# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakePackageConfigHelpers)

if(BUILD_SHARED_LIBS)
  # Component installs cannot form a standalone public SDK.
  # Component DSOs are an in-tree build shape, not the public HRX package.
  install(CODE [[
    message(FATAL_ERROR
      "The public HRX package requires BUILD_SHARED_LIBS=OFF. "
      "Shared runtime components are supported for in-tree development only.")
  ]] COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
endif()

set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
  OUTPUT_NAME hrx
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  EXPORT_NAME hrx
  POSITION_INDEPENDENT_CODE ON
)
if(APPLE)
  set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
    C_VISIBILITY_PRESET hidden
  )
  set(_hrx_export_file "${LIBHRX_SOURCE_DIR}/exports/hrx_exports_macos.txt")
  target_link_options(libhrx_src_libhrx_hrx PRIVATE
    "-Wl,-exported_symbols_list,${_hrx_export_file}"
  )
  set_property(TARGET libhrx_src_libhrx_hrx APPEND PROPERTY
    LINK_DEPENDS "${_hrx_export_file}"
  )
elseif(UNIX)
  set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    C_VISIBILITY_PRESET hidden
  )
  set(_hrx_export_file "${LIBHRX_SOURCE_DIR}/exports/hrx_exports.lds")
  target_link_options(libhrx_src_libhrx_hrx PRIVATE
    "-Wl,--version-script=${_hrx_export_file}"
  )
  set_property(TARGET libhrx_src_libhrx_hrx APPEND PROPERTY
    LINK_DEPENDS "${_hrx_export_file}"
  )
endif()
set_property(TARGET libhrx_src_libhrx_hrx PROPERTY
  INTERFACE_LINK_LIBRARIES ""
)
if(NOT TARGET hrx)
  add_custom_target(hrx DEPENDS libhrx_src_libhrx_hrx)
endif()
if(NOT TARGET hrx::hrx)
  iree_add_alias_library(hrx::hrx libhrx_src_libhrx_hrx)
endif()

set(_hrx_cmake_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/hrx")
set(_hrx_build_package_dir "${LIBHRX_BINARY_DIR}/cmake/hrx")

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/hrx-config.cmake.in"
  "${_hrx_build_package_dir}/hrx-config.cmake"
  INSTALL_DESTINATION "${_hrx_cmake_package_dir}")
write_basic_package_version_file(
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

# The config records the build mode even when no package is exported. This
# rejects stale exports after reconfiguring a static build to share components.
if(BUILD_SHARED_LIBS)
  return()
endif()

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
  "${LIBHRX_SOURCE_DIR}/include/hrx_base.h"
  "${LIBHRX_SOURCE_DIR}/include/hrx_device.h"
  "${LIBHRX_SOURCE_DIR}/include/hrx_memory.h"
  "${LIBHRX_SOURCE_DIR}/include/hrx_runtime.h"
  "${LIBHRX_SOURCE_DIR}/include/hrx_runtime_cxx.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hrx"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
install(EXPORT hrx-targets
  NAMESPACE hrx::
  DESTINATION "${_hrx_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")

export(EXPORT hrx-targets
  NAMESPACE hrx::
  FILE "${_hrx_build_package_dir}/hrx-targets.cmake")

install(FILES
  "${_hrx_build_package_dir}/hrx-config.cmake"
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  DESTINATION "${_hrx_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
