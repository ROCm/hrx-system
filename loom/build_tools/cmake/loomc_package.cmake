# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(LOOMC_VERSION "0.1.0" CACHE STRING "Loom C API package version.")
if(NOT LOOMC_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
  message(FATAL_ERROR
    "LOOMC_VERSION must be a semantic version of the form MAJOR.MINOR.PATCH")
endif()
set(LOOMC_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(LOOMC_VERSION_MINOR "${CMAKE_MATCH_2}")
set(LOOMC_VERSION_PATCH "${CMAKE_MATCH_3}")

set(_loomc_cmake_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/loomc")
set(_loomc_build_package_dir "${CMAKE_CURRENT_BINARY_DIR}/cmake/loomc")

function(_loomc_add_shared_component target_name)
  _loomc_resolve(target_name "${target_name}")
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR
      "loomc shared package expected enabled binding target: ${target_name}")
  endif()
  get_target_property(_object_target "${target_name}" IREE_CC_OBJECT_TARGET)
  if(NOT _object_target)
    message(FATAL_ERROR
      "loomc shared package expected a compilation owner: ${target_name}")
  endif()
  get_target_property(_objects "${target_name}" IREE_CC_OBJECTS)
  list(APPEND _loomc_component_targets "${target_name}")
  list(APPEND _loomc_object_sources ${_objects})
  list(APPEND _loomc_object_targets "${_object_target}")
  set(_loomc_component_targets "${_loomc_component_targets}" PARENT_SCOPE)
  set(_loomc_object_sources "${_loomc_object_sources}" PARENT_SCOPE)
  set(_loomc_object_targets "${_loomc_object_targets}" PARENT_SCOPE)
endfunction()

function(_loomc_collect_shared_component_deps target_name)
  get_property(_deps TARGET "${target_name}" PROPERTY IREE_CC_DEPS)
  get_property(_options TARGET "${target_name}" PROPERTY IREE_CC_LINKOPTS)
  foreach(_dep IN LISTS _deps)
    _loomc_resolve(_dep_target "${_dep}")
    if(_dep_target IN_LIST _loomc_component_targets)
      continue()
    endif()
    list(APPEND _loomc_direct_deps "${_dep}")
  endforeach()
  list(APPEND _loomc_link_options ${_options})
  set(_loomc_direct_deps "${_loomc_direct_deps}" PARENT_SCOPE)
  set(_loomc_link_options "${_loomc_link_options}" PARENT_SCOPE)
endfunction()

if(NOT TARGET loom_binding_c_loomc)
  message(FATAL_ERROR "loomc shared package requires loom_binding_c_loomc")
endif()

set(_loomc_component_targets)
set(_loomc_object_sources)
set(_loomc_object_targets)
foreach(_loomc_component IN LISTS _LOOMC_COMPONENTS)
  _loomc_add_shared_component("${_loomc_component}")
endforeach()

foreach(_loomc_object_target IN LISTS _loomc_object_targets)
  target_compile_definitions("${_loomc_object_target}" PRIVATE
    LOOMC_BUILDING_SHARED_LIBRARY)
endforeach()

# Link the shared library from the selected public binding objects directly.
# Their immediate non-selected deps provide the ordinary link libraries; selected
# components are skipped to avoid ALWAYSLINK interface deps re-adding objects.
set(_loomc_direct_deps)
set(_loomc_link_options)
foreach(_loomc_component_target IN LISTS _loomc_component_targets)
  _loomc_collect_shared_component_deps("${_loomc_component_target}")
endforeach()
if(_loomc_direct_deps)
  list(REMOVE_DUPLICATES _loomc_direct_deps)
endif()

add_library(loomc_shared SHARED ${_loomc_object_sources})
target_link_libraries(loomc_shared PRIVATE ${_loomc_direct_deps})
target_link_options(loomc_shared PRIVATE
  ${IREE_DEFAULT_LINKOPTS} ${_loomc_link_options})
target_include_directories(loomc_shared PUBLIC
  "$<BUILD_INTERFACE:${_IREE_PACKAGE_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_definitions(loomc_shared INTERFACE
  LOOMC_USING_SHARED_LIBRARY)
set_target_properties(loomc_shared PROPERTIES
  OUTPUT_NAME loomc
  EXPORT_NAME loomc
  VERSION ${LOOMC_VERSION}
  SOVERSION ${LOOMC_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON
  LINKER_LANGUAGE CXX
  INTERFACE_LINK_LIBRARIES "")
if(NOT WIN32)
  set_target_properties(loomc_shared PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
endif()
if(APPLE)
  set_target_properties(loomc_shared PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(loomc_shared PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
  set(_loomc_export_file
    "${_IREE_PACKAGE_SOURCE_DIR}/exports/loomc_exports.lds")
  target_link_options(loomc_shared PRIVATE
    "-Wl,--version-script=${_loomc_export_file}")
  set_property(TARGET loomc_shared APPEND PROPERTY
    LINK_DEPENDS "${_loomc_export_file}")
endif()

if(LOOM_LINK_SHARED_TOOLS)
  loomc_configure_shared_tools()
endif()

if(NOT TARGET loomc::loomc)
  add_library(loomc::loomc ALIAS loomc_shared)
endif()

install(TARGETS loomc_shared EXPORT loomc-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    NAMELINK_COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${_IREE_PACKAGE_SOURCE_DIR}/include/loomc"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
  FILES_MATCHING PATTERN "*.h")
install(EXPORT loomc-targets
  NAMESPACE loomc::
  DESTINATION "${_loomc_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/loomc-config.cmake.in"
  "${_loomc_build_package_dir}/loomc-config.cmake"
  INSTALL_DESTINATION "${_loomc_cmake_package_dir}")
write_basic_package_version_file(
  "${_loomc_build_package_dir}/loomc-config-version.cmake"
  VERSION "${LOOMC_VERSION}"
  COMPATIBILITY SameMajorVersion)
export(EXPORT loomc-targets
  NAMESPACE loomc::
  FILE "${_loomc_build_package_dir}/loomc-targets.cmake")

install(FILES
  "${_loomc_build_package_dir}/loomc-config.cmake"
  "${_loomc_build_package_dir}/loomc-config-version.cmake"
  DESTINATION "${_loomc_cmake_package_dir}"
  COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
