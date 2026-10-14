# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)
include(ExternalProject)

iree_add_alias_interface(iree::third_party::libjpeg_turbo)

function(_iree_configure_libjpeg_turbo_facade target_name)
  get_target_property(_facade_target
    iree::third_party::libjpeg_turbo ALIASED_TARGET)
  target_link_libraries(${_facade_target} INTERFACE ${target_name})
  set_property(TARGET ${_facade_target}
    PROPERTY IREE_LIBJPEG_TURBO_CONFIGURED TRUE)
endfunction()

function(iree_configure_libjpeg_turbo)
  get_target_property(_facade_target
    iree::third_party::libjpeg_turbo ALIASED_TARGET)
  get_target_property(_already_configured
    ${_facade_target} IREE_LIBJPEG_TURBO_CONFIGURED)
  if(_already_configured)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(libjpeg-turbo CONFIG QUIET)
    foreach(_target_name
        libjpeg-turbo::jpeg-static
        JPEG::JPEG)
      if(TARGET ${_target_name})
        _iree_configure_libjpeg_turbo_facade(${_target_name})
        return()
      endif()
    endforeach()
  endif()

  iree_dependency_require_pinned_source_allowed("libjpeg-turbo")
  # Populate without add_subdirectory: upstream intentionally requires an
  # isolated top-level configuration and names ExternalProject as its embedding
  # contract.
  iree_declare_locked_fetch_content(libjpeg_turbo
    SOURCE_SUBDIR iree-external-project-only)
  FetchContent_MakeAvailable(libjpeg_turbo)

  set(_binary_dir "${libjpeg_turbo_BINARY_DIR}/project")
  set(_archive_dir "${_binary_dir}/lib")
  if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    set(_archive_name "jpeg-static${CMAKE_STATIC_LIBRARY_SUFFIX}")
  else()
    set(_archive_name
      "${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")
  endif()
  set(_archive "${_archive_dir}/${_archive_name}")
  ExternalProject_Add(iree_libjpeg_turbo_project
    SOURCE_DIR "${libjpeg_turbo_SOURCE_DIR}"
    BINARY_DIR "${_binary_dir}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_COMMAND
      "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
      --target jpeg-static
    BUILD_BYPRODUCTS "${_archive}"
    CMAKE_ARGS
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
      "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
      "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
      "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
      "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
      "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
      "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
      "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${_archive_dir}"
      "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG=${_archive_dir}"
      "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE=${_archive_dir}"
      "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO=${_archive_dir}"
      "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL=${_archive_dir}"
      "-DENABLE_SHARED=OFF"
      "-DENABLE_STATIC=ON"
      "-DWITH_JAVA=OFF"
      "-DWITH_TURBOJPEG=OFF"
      "-DWITH_TOOLS=OFF"
      "-DWITH_TESTS=OFF"
      "-DWITH_FUZZ=OFF"
  )
  add_library(iree_third_party_libjpeg_turbo_source STATIC IMPORTED GLOBAL)
  set_target_properties(iree_third_party_libjpeg_turbo_source PROPERTIES
    IMPORTED_LOCATION "${_archive}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${libjpeg_turbo_SOURCE_DIR}/src;${_binary_dir}"
  )
  add_dependencies(iree_third_party_libjpeg_turbo_source
    iree_libjpeg_turbo_project)
  _iree_configure_libjpeg_turbo_facade(
    iree_third_party_libjpeg_turbo_source)
endfunction()
