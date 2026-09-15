# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(GNUInstallDirs)

# Declares public headers independently of native provider availability. Offline
# consumers and source components may reference them before package traversal.
function(amdf_declare_headers)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    ""
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_declare_headers requires NAME.")
  endif()

  set(_HEADERS_TARGET "${_RULE_NAME}_headers")
  add_library(${_HEADERS_TARGET} INTERFACE)
  add_library(amdf::headers ALIAS ${_HEADERS_TARGET})
  target_include_directories(${_HEADERS_TARGET}
    INTERFACE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  set_target_properties(${_HEADERS_TARGET} PROPERTIES EXPORT_NAME headers)
endfunction()

# Declares provider artifacts before package traversal so tests may reference
# them before package-owned objects are attached by amdf_library.
function(amdf_declare_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    ""
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_declare_library requires NAME.")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include("${PROJECT_SOURCE_DIR}/build_tools/third_party/linux_uapi/linux_uapi.cmake")
    iree_configure_linux_drm_uapi()
    if(AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA)
      iree_configure_linux_amdgpu_uapi()
      iree_configure_linux_kfd_uapi()
    endif()
    if(AMDF_FAMILY_XDNA)
      iree_configure_linux_xdna_uapi()
    endif()
  endif()

  add_library(${_RULE_NAME} SHARED)
  add_library(amdf::amdf ALIAS ${_RULE_NAME})
  add_library(${_RULE_NAME}_static STATIC)
  add_library(amdf::amdf_static ALIAS ${_RULE_NAME}_static)
endfunction()

# Composes package-owned libamdf targets into the public targets declared by
# amdf_declare_library. Internal component archives are implementation details;
# their object files are folded into the public artifacts so installed targets
# never expose the source-tree package graph.
function(amdf_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;WINDOWS_DEF_FILE"
    "COMPONENTS;RUNTIME_DATA"
    ${ARGN}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "amdf_library requires NAME.")
  endif()
  set(_HEADERS_TARGET "${_RULE_NAME}_headers")
  if(NOT _RULE_COMPONENTS)
    message(FATAL_ERROR "amdf_library requires at least one COMPONENT.")
  endif()
  if(NOT TARGET "${_RULE_NAME}" OR
     NOT TARGET "${_RULE_NAME}_static" OR
     NOT TARGET "${_RULE_NAME}_headers")
    message(FATAL_ERROR
      "amdf_library requires a matching amdf_declare_library call.")
  endif()

  # Traverse only libamdf-owned targets. Every owned object is folded into the
  # public artifacts; dependencies outside libamdf remain ordinary link edges.
  set(_PENDING_COMPONENTS ${_RULE_COMPONENTS})
  set(_COMPONENT_TARGETS)
  set(_OBJECT_SOURCES)
  set(_LINK_DEPS)
  while(_PENDING_COMPONENTS)
    list(POP_FRONT _PENDING_COMPONENTS _COMPONENT)
    if(NOT TARGET "${_COMPONENT}")
      message(FATAL_ERROR
        "amdf_library expected enabled component target: ${_COMPONENT}")
    endif()

    get_target_property(_COMPONENT_TARGET "${_COMPONENT}" ALIASED_TARGET)
    if(NOT _COMPONENT_TARGET)
      set(_COMPONENT_TARGET "${_COMPONENT}")
    endif()
    if(_COMPONENT_TARGET IN_LIST _COMPONENT_TARGETS)
      continue()
    endif()
    list(APPEND _COMPONENT_TARGETS "${_COMPONENT_TARGET}")

    if(TARGET "${_COMPONENT_TARGET}.objects")
      # Public declarations opt in to ELF visibility through AMDF_API.
      set_target_properties("${_COMPONENT_TARGET}.objects" PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
      )
      list(APPEND _OBJECT_SOURCES
        "$<TARGET_OBJECTS:${_COMPONENT_TARGET}.objects>")
    endif()

    get_target_property(
      _PRIVATE_DEPS "${_COMPONENT_TARGET}" LINK_LIBRARIES)
    get_target_property(
      _INTERFACE_DEPS "${_COMPONENT_TARGET}" INTERFACE_LINK_LIBRARIES)
    if(_PRIVATE_DEPS STREQUAL "_PRIVATE_DEPS-NOTFOUND")
      set(_PRIVATE_DEPS)
    endif()
    if(_INTERFACE_DEPS STREQUAL "_INTERFACE_DEPS-NOTFOUND")
      set(_INTERFACE_DEPS)
    endif()
    foreach(_DEP IN LISTS _PRIVATE_DEPS _INTERFACE_DEPS)
      if(NOT _DEP OR _DEP MATCHES "^\\$<TARGET_OBJECTS:")
        continue()
      endif()

      set(_DEP_TARGET "${_DEP}")
      if(_DEP_TARGET MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
        set(_DEP_TARGET "${CMAKE_MATCH_1}")
      endif()
      if(TARGET "${_DEP_TARGET}")
        get_target_property(_ALIASED_DEP "${_DEP_TARGET}" ALIASED_TARGET)
        if(_ALIASED_DEP)
          set(_DEP_TARGET "${_ALIASED_DEP}")
        endif()
        if(_DEP_TARGET STREQUAL "${_RULE_NAME}_headers")
          continue()
        endif()
        if(_DEP MATCHES "^libamdf::" OR _DEP_TARGET MATCHES "^libamdf_")
          list(APPEND _PENDING_COMPONENTS "${_DEP_TARGET}")
          continue()
        endif()
      endif()
      list(APPEND _LINK_DEPS "${_DEP}")
    endforeach()
  endwhile()

  if(NOT _OBJECT_SOURCES)
    message(FATAL_ERROR "amdf_library components contain no object code.")
  endif()
  list(REMOVE_DUPLICATES _OBJECT_SOURCES)
  if(_LINK_DEPS)
    list(REMOVE_DUPLICATES _LINK_DEPS)
  endif()

  target_sources(${_RULE_NAME} PRIVATE ${_OBJECT_SOURCES})
  target_link_libraries(${_RULE_NAME}
    PRIVATE
      ${_LINK_DEPS}
  )
  iree_add_data_dependencies(
    NAME ${_RULE_NAME}
    DATA ${_RULE_RUNTIME_DATA}
  )
  target_include_directories(${_RULE_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  target_compile_definitions(${_RULE_NAME}
    INTERFACE
      AMDF_SHARED_LIBRARY=1
  )
  target_link_options(${_RULE_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
  )
  if(WIN32)
    if(NOT _RULE_WINDOWS_DEF_FILE)
      message(FATAL_ERROR "amdf_library requires WINDOWS_DEF_FILE on Windows.")
    endif()
    target_sources(${_RULE_NAME} PRIVATE "${_RULE_WINDOWS_DEF_FILE}")
  endif()
  set_target_properties(${_RULE_NAME} PROPERTIES
    EXPORT_NAME amdf
    OUTPUT_NAME amdf
    SOVERSION "${AMDF_ABI_VERSION}"
    VERSION "${AMDF_VERSION}"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
  )

  set(_STATIC_TARGET "${_RULE_NAME}_static")
  target_sources(${_STATIC_TARGET} PRIVATE ${_OBJECT_SOURCES})
  target_link_libraries(${_STATIC_TARGET}
    PUBLIC
      ${_LINK_DEPS}
  )
  iree_add_data_dependencies(
    NAME ${_STATIC_TARGET}
    DATA ${_RULE_RUNTIME_DATA}
  )
  target_include_directories(${_STATIC_TARGET}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  set_target_properties(${_STATIC_TARGET} PROPERTIES
    EXPORT_NAME amdf_static
    OUTPUT_NAME amdf_static
  )

  install(
    TARGETS ${_RULE_NAME} ${_STATIC_TARGET}
    EXPORT amdf-targets
    COMPONENT AMDF
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
  install(
    TARGETS ${_HEADERS_TARGET}
    EXPORT amdf-targets
    COMPONENT AMDF
  )
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/amdf"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT AMDF
    FILES_MATCHING PATTERN "*.h"
  )
endfunction()
