# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# project() includes this file in nested dependency scopes. Their private
# imports are inspectable only there, before the directory finishes processing.
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  get_property(_IREE_USAGE_CAPTURE DIRECTORY PROPERTY IREE_COMPILE_USAGE_CAPTURE)
  if(NOT _IREE_USAGE_CAPTURE)
    set_property(DIRECTORY PROPERTY IREE_COMPILE_USAGE_CAPTURE TRUE)
    cmake_language(DEFER CALL _iree_capture_compile_imports)
  endif()
endif()

include_guard(GLOBAL)

if(Python3_EXECUTABLE AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND
   NOT CMAKE_VERSION VERSION_LESS 3.30)
  list(APPEND CMAKE_PROJECT_INCLUDE "${CMAKE_CURRENT_LIST_FILE}")
endif()

function(_iree_capture_compile_imports)
  # Target-producing callbacks must finish before their imports are recorded.
  cmake_language(DEFER GET_CALL_IDS _PENDING)
  if(_PENDING)
    cmake_language(DEFER CALL _iree_capture_compile_imports)
    return()
  endif()
  get_property(_IMPORTED DIRECTORY PROPERTY IMPORTED_TARGETS)
  set(_SCOPED)
  foreach(_TARGET IN LISTS _IMPORTED)
    get_property(_GLOBAL TARGET "${_TARGET}" PROPERTY IMPORTED_GLOBAL)
    if(NOT _GLOBAL)
      list(APPEND _SCOPED "${_TARGET}")
    endif()
    get_property(_CUSTOM TARGET "${_TARGET}" PROPERTY TRANSITIVE_COMPILE_PROPERTIES)
    if(NOT "${_CUSTOM}" STREQUAL "")
      set_property(GLOBAL PROPERTY IREE_NATIVE_COMPILE_PROPERTIES TRUE)
    endif()
  endforeach()
  set_property(DIRECTORY PROPERTY IREE_COMPILE_USAGE_IMPORTS "${_IMPORTED}")
  set_property(DIRECTORY PROPERTY IREE_COMPILE_USAGE_SCOPED_IMPORTS "${_SCOPED}")
endfunction()

# Resolves ordinary library compile interfaces once, after target construction.
# CMake retains the native link graph and evaluates all generator expressions.
# Enclosing projects can still modify our targets after add_subdirectory returns,
# so their interfaces must remain native. This needs composable project hooks
# and native link-property traversal through LINK_ONLY (CMP0166, CMake 3.30).
function(iree_finalize_compile_usage)
  if(NOT Python3_EXECUTABLE OR
     NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR OR
     CMAKE_VERSION VERSION_LESS 3.30)
    return()
  endif()
  foreach(_POLICY CMP0099 CMP0166)
    cmake_policy(GET "${_POLICY}" _BEHAVIOR)
    if(NOT _BEHAVIOR STREQUAL "NEW")
      return()
    endif()
  endforeach()

  get_property(_NATIVE_ROOTS GLOBAL PROPERTY IREE_NATIVE_COMPILE_INTERFACES)
  foreach(_ROOT IN LISTS _NATIVE_ROOTS)
    if(NOT TARGET "${_ROOT}")
      # A configuration-selected root can only be resolved during generation.
      return()
    endif()
    get_property(_ALIASED TARGET "${_ROOT}" PROPERTY ALIASED_TARGET)
    if(_ALIASED)
      set(_ROOT "${_ALIASED}")
    endif()
    set("_NATIVE_INTERFACE_${_ROOT}" TRUE)
  endforeach()

  set(_COMPILE_PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS
    INTERFACE_COMPILE_FEATURES INTERFACE_PRECOMPILE_HEADERS INTERFACE_SOURCES)
  set(_NATIVE_PROPERTIES
    INTERFACE_POSITION_INDEPENDENT_CODE
    COMPATIBLE_INTERFACE_BOOL COMPATIBLE_INTERFACE_STRING
    COMPATIBLE_INTERFACE_NUMBER_MAX COMPATIBLE_INTERFACE_NUMBER_MIN
    INTERFACE_HEADER_SETS INTERFACE_CXX_MODULE_SETS
    INTERFACE_LINK_LIBRARIES_DIRECT INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE)

  # Private imports whose owning scope did not record them remain opaque.
  # Custom transitive properties can be declared by consumers, independently
  # of the library that supplies them. Both require native graph traversal.
  set(_DIRECTORIES "${CMAKE_SOURCE_DIR}")
  set(_TARGETS)
  get_property(_CUSTOM_PROPERTIES GLOBAL PROPERTY IREE_NATIVE_COMPILE_PROPERTIES)
  if(_CUSTOM_PROPERTIES)
    return()
  endif()
  while(_DIRECTORIES)
    set(_NEXT)
    foreach(_DIRECTORY IN LISTS _DIRECTORIES)
      get_property(_CHILDREN DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)
      get_property(_IMPORTED DIRECTORY "${_DIRECTORY}" PROPERTY IMPORTED_TARGETS)
      get_property(_RECORDED DIRECTORY "${_DIRECTORY}" PROPERTY IREE_COMPILE_USAGE_IMPORTS)
      get_property(_SCOPED DIRECTORY "${_DIRECTORY}" PROPERTY IREE_COMPILE_USAGE_SCOPED_IMPORTS)
      foreach(_TARGET IN LISTS _IMPORTED)
        if(NOT TARGET "${_TARGET}" OR _TARGET IN_LIST _SCOPED)
          if(NOT _TARGET IN_LIST _RECORDED)
            return()
          endif()
          set("_NATIVE_SCOPE_${_DIRECTORY}" TRUE)
        else()
          list(APPEND _TARGETS "${_TARGET}")
        endif()
      endforeach()
      foreach(_CHILD IN LISTS _CHILDREN)
        set("_NATIVE_SCOPE_${_CHILD}" "${_NATIVE_SCOPE_${_DIRECTORY}}")
      endforeach()
      list(APPEND _NEXT ${_CHILDREN})
      get_property(_LOCAL_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
      list(APPEND _TARGETS ${_LOCAL_TARGETS})
    endforeach()
    set(_DIRECTORIES "${_NEXT}")
  endwhile()
  foreach(_TARGET IN LISTS _TARGETS)
    get_property(_CUSTOM_PROPERTIES TARGET "${_TARGET}" PROPERTY TRANSITIVE_COMPILE_PROPERTIES)
    if(NOT "${_CUSTOM_PROPERTIES}" STREQUAL "")
      return()
    endif()
  endforeach()

  set(_INPUT "${CMAKE_BINARY_DIR}/.iree/compile_usage.json")
  set(_OUTPUT "${CMAKE_BINARY_DIR}/.iree/compile_usage.cmake")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.iree")
  file(WRITE "${_INPUT}" "[")
  set(_SEPARATOR "")
  set(_CHUNK "")
  foreach(_TARGET IN LISTS _TARGETS)
    get_property(_OWNED TARGET "${_TARGET}" PROPERTY IREE_CC_DEPS SET)
    if(NOT _OWNED OR _NATIVE_INTERFACE_${_TARGET})
      continue()
    endif()
    get_property(_SOURCE_DIR TARGET "${_TARGET}" PROPERTY SOURCE_DIR)
    if(_NATIVE_SCOPE_${_SOURCE_DIR})
      continue()
    endif()
    get_property(_TYPE TARGET "${_TARGET}" PROPERTY TYPE)
    if(NOT _TYPE MATCHES "^(STATIC|INTERFACE)_LIBRARY$")
      continue()
    endif()
    get_property(_NATIVE TARGET "${_TARGET}" PROPERTY SYSTEM)
    foreach(_PROPERTY IN LISTS _NATIVE_PROPERTIES)
      get_property(_VALUE TARGET "${_TARGET}" PROPERTY "${_PROPERTY}")
      if(NOT "${_VALUE}" STREQUAL "")
        set(_NATIVE TRUE)
      endif()
    endforeach()
    if(_NATIVE)
      continue()
    endif()
    set(_RECORD "${_SEPARATOR}\n{\"target\":\"${_TARGET}\",\"usage\":{")
    set(_PROPERTY_SEPARATOR "")
    foreach(_PROPERTY IN LISTS _COMPILE_PROPERTIES)
      get_property(_VALUE TARGET "${_TARGET}" PROPERTY "${_PROPERTY}")
      if(NOT "${_VALUE}" STREQUAL "")
        _iree_compile_usage_json(_VALUE "${_VALUE}")
      endif()
      string(APPEND _RECORD "${_PROPERTY_SEPARATOR}\"${_PROPERTY}\":\"${_VALUE}\"")
      set(_PROPERTY_SEPARATOR ",")
    endforeach()
    get_property(_LINKS TARGET "${_TARGET}" PROPERTY INTERFACE_LINK_LIBRARIES)
    _iree_compile_usage_json(_LINKS_JSON "${_LINKS}")
    string(APPEND _RECORD "},\"links\":\"${_LINKS_JSON}\",\"providers\":{")
    set(_PROPERTY_SEPARATOR "")
    foreach(_DEPENDENCY IN LISTS _LINKS)
      if(TARGET "${_DEPENDENCY}")
        get_property(_ALIASED TARGET "${_DEPENDENCY}" PROPERTY ALIASED_TARGET)
        if(NOT _ALIASED)
          set(_ALIASED "${_DEPENDENCY}")
        endif()
        string(APPEND _RECORD "${_PROPERTY_SEPARATOR}\"${_DEPENDENCY}\":\"${_ALIASED}\"")
        set(_PROPERTY_SEPARATOR ",")
      endif()
    endforeach()
    # Amortize file opens and filesystem notification/antivirus work without
    # repeatedly copying a string proportional to the complete target graph.
    string(APPEND _CHUNK "${_RECORD}}}")
    string(LENGTH "${_CHUNK}" _CHUNK_LENGTH)
    if(_CHUNK_LENGTH GREATER_EQUAL 65536)
      file(APPEND "${_INPUT}" "${_CHUNK}")
      set(_CHUNK "")
    endif()
    set(_SEPARATOR ",")
  endforeach()
  file(APPEND "${_INPUT}" "${_CHUNK}\n]\n")
  set(_RESOLVER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/resolve_compile_usage.py")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_RESOLVER}")
  execute_process(COMMAND "${Python3_EXECUTABLE}" -B "${_RESOLVER}" "${_INPUT}" "${_OUTPUT}"
    COMMAND_ERROR_IS_FATAL ANY)
  include("${_OUTPUT}")
endfunction()

function(_iree_compile_usage_json output value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  set(${output} "${value}" PARENT_SCOPE)
endfunction()
