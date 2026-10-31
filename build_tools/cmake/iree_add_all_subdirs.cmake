# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# BUILD packages have variable scopes and explicit artifact paths. Authored
# CMake and callers requesting native directory options retain add_subdirectory
# semantics. A macro preserves their parent variable scope.
macro(iree_add_subdirectory DIRECTORY)
  iree_package_source_dir(_IREE_PARENT_SOURCE_DIR)
  iree_package_binary_dir(_IREE_PARENT_BINARY_DIR)
  get_filename_component(_IREE_CHILD_SOURCE_DIR "${DIRECTORY}" ABSOLUTE
    BASE_DIR "${_IREE_PARENT_SOURCE_DIR}")
  file(RELATIVE_PATH _IREE_CHILD_PACKAGE "${IREE_SOURCE_DIR}"
    "${_IREE_CHILD_SOURCE_DIR}")
  set(_IREE_CHILD_BINARY_DIR "${_IREE_PARENT_BINARY_DIR}/${DIRECTORY}")
  if(IREE_GENERATED_CMAKE_DIR AND "${ARGN}" STREQUAL "" AND EXISTS
      "${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/package.cmake")
    include("${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/package.cmake")
    cmake_language(CALL ${_IREE_PACKAGE_FUNCTION} "${_IREE_CHILD_BINARY_DIR}")
  else()
    set(_IREE_RESTORE_PACKAGE_SOURCE FALSE)
    set(_IREE_RESTORE_PACKAGE_BINARY FALSE)
    if(DEFINED IREE_PACKAGE_SOURCE_DIR)
      set(_IREE_RESTORE_PACKAGE_SOURCE TRUE)
    endif()
    if(DEFINED IREE_PACKAGE_BINARY_DIR)
      set(_IREE_RESTORE_PACKAGE_BINARY TRUE)
    endif()
    unset(IREE_PACKAGE_SOURCE_DIR)
    unset(IREE_PACKAGE_BINARY_DIR)
    if(IREE_GENERATED_CMAKE_DIR AND EXISTS
        "${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/CMakeLists.txt")
      add_subdirectory("${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}"
        "${_IREE_CHILD_BINARY_DIR}" ${ARGN})
    else()
      add_subdirectory("${_IREE_CHILD_SOURCE_DIR}"
        "${_IREE_CHILD_BINARY_DIR}" ${ARGN})
    endif()
    if(_IREE_RESTORE_PACKAGE_SOURCE)
      set(IREE_PACKAGE_SOURCE_DIR "${_IREE_PARENT_SOURCE_DIR}")
    endif()
    if(_IREE_RESTORE_PACKAGE_BINARY)
      set(IREE_PACKAGE_BINARY_DIR "${_IREE_PARENT_BINARY_DIR}")
    endif()
  endif()
endmacro()

# Children may be source-authored projects or generated BUILD packages.
macro(iree_add_all_subdirs)
  iree_package_source_dir(_IREE_PARENT_SOURCE_DIR)
  file(GLOB _IREE_CHILDREN RELATIVE "${_IREE_PARENT_SOURCE_DIR}"
    "${_IREE_PARENT_SOURCE_DIR}/*")
  foreach(_IREE_CHILD IN LISTS _IREE_CHILDREN)
    file(RELATIVE_PATH _IREE_CHILD_PACKAGE "${IREE_SOURCE_DIR}"
      "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}")
    if(IS_DIRECTORY "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}" AND
        (EXISTS "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}/CMakeLists.txt" OR
         EXISTS "${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/CMakeLists.txt"))
      iree_add_subdirectory("${_IREE_CHILD}")
    endif()
  endforeach()
endmacro()
