# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

function(amdf_finalize_project)
  if(NOT TARGET amdf)
    return()
  endif()
  set(_PACKAGE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/amdf")
  set(_PACKAGE_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/amdf-config.cmake")
  set(_PACKAGE_VERSION "${CMAKE_CURRENT_BINARY_DIR}/amdf-config-version.cmake")

  configure_package_config_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/amdf-config.cmake.in"
    "${_PACKAGE_CONFIG}"
    INSTALL_DESTINATION "${_PACKAGE_INSTALL_DIR}"
  )
  write_basic_package_version_file(
    "${_PACKAGE_VERSION}"
    VERSION "${AMDF_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  install(
    EXPORT amdf-targets
    FILE amdf-targets.cmake
    NAMESPACE amdf::
    DESTINATION "${_PACKAGE_INSTALL_DIR}"
    COMPONENT AMDF
  )
  install(
    FILES "${_PACKAGE_CONFIG}" "${_PACKAGE_VERSION}"
    DESTINATION "${_PACKAGE_INSTALL_DIR}"
    COMPONENT AMDF
  )

  export(
    EXPORT amdf-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/amdf-targets.cmake"
    NAMESPACE amdf::
  )

endfunction()
