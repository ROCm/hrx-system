# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/../sanitizer/iree_sanitizer_suppressions.cmake")

# iree_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# PACKAGE: Name of the package (overrides actual path)
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# SANITIZER_SUPPRESSIONS: Sanitizer/name pairs owned by this runtime dependency.
# COPTS: List of private compile options
# CXX_STANDARD: Optional C++ standard version, such as 23.
# CXX_FEATURES: Required private C++ runtime features: exceptions and rtti.
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# SYSTEM_INCLUDES: Include directories that should be used with "SYSTEM" scope,
#   which makes them more tolerant to certain classes of warnings and issues.
# LINKOPTS: List of link options
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
# SHARED: If set, will compile to a shared object.
# WINDOWS_DEF_FILE: If set, will add a windows .def file to a shared library link
# RESOLVED: INCLUDES/DEFINES already contain the configured BUILD closure.
# LINK_LIBRARIES: Native link edges for a resolved source declaration.
# TRANSITIVE_COPTS: Compiler options inherited from source-owned dependencies.
# TRANSITIVE_LINKOPTS: Link options inherited from source-owned dependencies.
# Note:
# By default, iree_cc_library will always create a library named iree_${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# Source libraries record IREE_CC_OBJECT_TARGET as the native target that owns
# compilation and exposes TARGET_OBJECTS. Ordinary libraries own their objects;
# ALWAYSLINK libraries use a separate object target. Header-only libraries have
# no object target. IREE_CC_OBJECTS retains the complete direct object set,
# including objects produced by custom commands outside native compilation.
#
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# iree_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     iree::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# iree_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::package::fantastic_lib
# )

function(iree_cc_library)
  iree_package_source_dir(_IREE_PACKAGE_SOURCE_DIR)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;SHARED;ALWAYSLINK;LINKABLE_MORE_THAN_ONCE;RESOLVED"
    "PACKAGE;NAME;WINDOWS_DEF_FILE;CXX_STANDARD"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;CXX_FEATURES;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES;SYSTEM_INCLUDES;SANITIZER_SUPPRESSIONS;LINK_LIBRARIES;TRANSITIVE_COPTS;TRANSITIVE_LINKOPTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_COMPILE_TARGET ${_NAME})

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  : iree_cc_library(${_NAME})")
  endif()

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  # Configured BUILD declarations carry their resolved compile interface and
  # separate link edges. Authored native libraries retain native propagation.
  if(_RULE_RESOLVED)
    set(_LINK_LIBRARIES ${_RULE_LINK_LIBRARIES})
  else()
    set(_LINK_LIBRARIES ${_RULE_DEPS})
  endif()
  list(APPEND _LINK_LIBRARIES ${IREE_IMPLICIT_DEFS_CC_DEPS})

  # Bazel srcs can name generated files. Point missing sources at the binary
  # tree so target_sources can attach the producer custom command edge.
  # Producers mark their outputs GENERATED; a binary-tree path may instead be
  # a configure_file output with no native build rule.
  set(_RULE_SRC_TARGET_SRCS)
  foreach(_SRC_FILE IN LISTS _RULE_SRCS)
    if(IS_ABSOLUTE "${_SRC_FILE}" OR "${_SRC_FILE}" MATCHES "^\\$<")
      list(APPEND _RULE_SRC_TARGET_SRCS "${_SRC_FILE}")
    elseif(EXISTS "${_IREE_PACKAGE_SOURCE_DIR}/${_SRC_FILE}")
      list(APPEND _RULE_SRC_TARGET_SRCS "${_IREE_PACKAGE_SOURCE_DIR}/${_SRC_FILE}")
    else()
      set(_SRC_BINARY_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_SRC_FILE}")
      list(APPEND _RULE_SRC_TARGET_SRCS "${_SRC_BINARY_FILE}")
    endif()
  endforeach()
  # Header-only libraries have no native compilation target. Match complete
  # suffixes so a parent directory or filename stem cannot hide a source file.
  set(_CC_SRCS "${_RULE_SRC_TARGET_SRCS}")
  list(FILTER _CC_SRCS EXCLUDE REGEX
    "\\.(h|hh|h[+][+]|hm|hpp|hxx|H|HPP|in|txx|inc|inl)$")
  iree_set_source_object_names(${_CC_SRCS})
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  # Bazel hdrs can name generated files. Point missing headers at the binary
  # tree so target_sources can attach the producer custom command edge.
  set(_RULE_HDR_TARGET_SRCS)
  set(_RULE_BINARY_HDR_FILES)
  foreach(_HDR_FILE IN LISTS _RULE_HDRS _RULE_TEXTUAL_HDRS)
    if(IS_ABSOLUTE "${_HDR_FILE}" OR "${_HDR_FILE}" MATCHES "^\\$<")
      list(APPEND _RULE_HDR_TARGET_SRCS "${_HDR_FILE}")
    elseif(EXISTS "${_IREE_PACKAGE_SOURCE_DIR}/${_HDR_FILE}")
      list(APPEND _RULE_HDR_TARGET_SRCS "${_IREE_PACKAGE_SOURCE_DIR}/${_HDR_FILE}")
    else()
      set(_HDR_BINARY_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_HDR_FILE}")
      list(APPEND _RULE_HDR_TARGET_SRCS "${_HDR_BINARY_FILE}")
      list(APPEND _RULE_BINARY_HDR_FILES "${_HDR_BINARY_FILE}")
    endif()
  endforeach()

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  if(NOT _RULE_IS_INTERFACE AND NOT _RULE_ALWAYSLINK)
    # Ordinary libraries own both compilation and linking.
    if(_RULE_SHARED OR BUILD_SHARED_LIBS)
      add_library(${_NAME} SHARED)
      set(_LINKOPTS_SCOPE PRIVATE)
      if(_RULE_WINDOWS_DEF_FILE AND WIN32)
        target_sources(${_NAME} PRIVATE "${_RULE_WINDOWS_DEF_FILE}")
      endif()
    else()
      add_library(${_NAME} STATIC)
      # An archive has no native link step. Its declared link requirements
      # belong to the executable or shared library that consumes its objects.
      set(_LINKOPTS_SCOPE INTERFACE)
      if(_RULE_WINDOWS_DEF_FILE AND WIN32)
        message(SEND_ERROR "If specifying a .def file library must be shared")
      endif()
    endif()

    if(MSVC)
      # The directory already identifies the target. Repeating its qualified
      # name in the compiler PDB filename can exceed MSVC's path limit.
      set_target_properties(${_NAME} PROPERTIES
        COMPILE_PDB_NAME "compile"
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_NAME}.dir"
      )
    endif()

    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRC_TARGET_SRCS}
        ${_RULE_HDR_TARGET_SRCS}
    )

    # Wrap whole lists to preserve conditions spanning multiple directories.
    # This also avoids per-directory expression work in transitive consumers.
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR};${_RULE_INCLUDES}>"
    )
    target_include_directories(${_NAME}
      SYSTEM PUBLIC
        "$<BUILD_INTERFACE:${_RULE_SYSTEM_INCLUDES}>"
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${IREE_DEFAULT_COPTS}
        ${_RULE_COPTS}
        ${_RULE_TRANSITIVE_COPTS}
      INTERFACE
        ${IREE_INTERFACE_COPTS}
        ${_RULE_TRANSITIVE_COPTS}
    )
    target_link_options(${_NAME}
      PRIVATE
        ${IREE_DEFAULT_LINKOPTS}
      ${_LINKOPTS_SCOPE}
        ${_RULE_LINKOPTS}
        ${_RULE_TRANSITIVE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      PUBLIC
        ${_LINK_LIBRARIES}
        ${IREE_THREADS_DEPS}
        ${IREE_DEFAULT_LINK_LIBRARIES}
    )

    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA}
      OUT_TARGET_DATA _DATA_TARGETS)
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    # If in BUILD_SHARED_LIBS mode, then we need to make sure that visibility
    # is not hidden. We default to hidden visibility in the main copts so
    # need to undo it here.
    # TODO: Switch to the CXX_VISIBILITY_PRESET property and fix the global
    # hidden setting to follow suit.
    if(BUILD_SHARED_LIBS AND IREE_SUPPORTS_VISIBILITY_DEFAULT)
      target_compile_options(${_NAME} PRIVATE
        "-fvisibility=default"
      )
    endif()

    # Add all IREE targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set so only
    # set here.
    iree_set_cxx_options(${_NAME} "${_RULE_CXX_STANDARD}" ${_RULE_CXX_FEATURES})
  elseif(NOT _RULE_IS_INTERFACE AND _RULE_ALWAYSLINK)
    # ALWAYSLINK library: OBJECT for compilation, INTERFACE for propagation.
    # The INTERFACE library propagates the complete direct object set, ensuring
    # all objects are always included in the final binary. STATIC archives let
    # the linker strip unreferenced objects (dropping static init registrations);
    # propagating objects directly bypasses the archive and eliminates that
    # problem. This is the CMake equivalent of Bazel's alwayslink = True.
    set(_COMPILE_TARGET ${_NAME}.objects)
    add_library(${_COMPILE_TARGET} OBJECT)
    add_library(${_NAME} INTERFACE)
    add_dependencies(${_NAME} ${_COMPILE_TARGET})

    target_sources(${_COMPILE_TARGET}
      PRIVATE
        ${_RULE_SRC_TARGET_SRCS}
        ${_RULE_HDR_TARGET_SRCS}
    )

    # INTERFACE library propagates objects and link dependencies to consumers.
    target_link_libraries(${_NAME}
      INTERFACE
        "$<GENEX_EVAL:$<TARGET_PROPERTY:${_NAME},IREE_CC_OBJECTS>>"
        ${_LINK_LIBRARIES}
        ${IREE_THREADS_DEPS}
        ${IREE_DEFAULT_LINK_LIBRARIES}
    )
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR};${_RULE_INCLUDES}>"
    )
    target_include_directories(${_NAME}
      SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${_RULE_SYSTEM_INCLUDES}>"
    )
    target_link_options(${_NAME}
      INTERFACE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
        ${_RULE_TRANSITIVE_LINKOPTS}
    )
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )

    # OBJECT library needs compile-related properties for building the sources.
    # Compile options go directly on the OBJECT library (INTERFACE libraries
    # cannot have PRIVATE properties).
    target_compile_options(${_COMPILE_TARGET}
      PRIVATE
        ${IREE_DEFAULT_COPTS}
        ${_RULE_COPTS}
        ${_RULE_TRANSITIVE_COPTS}
    )

    # Forward transitive compile properties from the INTERFACE library's
    # dependency chain to the OBJECT library so sources see all transitive
    # include directories and definitions.
    target_include_directories(${_COMPILE_TARGET}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_include_directories(${_COMPILE_TARGET} SYSTEM
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(${_COMPILE_TARGET}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_COMPILE_DEFINITIONS>
    )
    # Forward deps to the OBJECT library for transitive compile definitions.
    # We forward deps directly rather than $<TARGET_PROPERTY:INTERFACE_LINK_LIBRARIES>
    # because the latter contains $<TARGET_OBJECTS:${_COMPILE_TARGET}> which would
    # create a circular reference.
    target_link_libraries(${_COMPILE_TARGET}
      PUBLIC
        ${_LINK_LIBRARIES}
    )

    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA}
      OUT_TARGET_DATA _DATA_TARGETS)

    if(BUILD_SHARED_LIBS AND IREE_SUPPORTS_VISIBILITY_DEFAULT)
      target_compile_options(${_COMPILE_TARGET} PRIVATE
        "-fvisibility=default"
      )
    endif()

    if(_RULE_PUBLIC)
      set_property(TARGET ${_COMPILE_TARGET} PROPERTY FOLDER ${IREE_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_COMPILE_TARGET} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_COMPILE_TARGET} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
    endif()

    iree_set_cxx_options(${_COMPILE_TARGET} "${_RULE_CXX_STANDARD}" ${_RULE_CXX_FEATURES})
  else()
    # Generating header-only library (no sources, or ALWAYSLINK on header-only
    # which is meaningless since there are no objects).
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR};${_RULE_INCLUDES}>"
    )
    target_include_directories(${_NAME}
      SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${_RULE_SYSTEM_INCLUDES}>"
    )
    target_link_options(${_NAME}
      INTERFACE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
        ${_RULE_TRANSITIVE_LINKOPTS}
    )
    # Keep build-tree headers on their declaring library. INTERFACE_SOURCES
    # would make every transitive consumer rediscover the same source rules.
    # An interface library with private sources owns their custom commands and
    # carries the build dependency without propagating the source list.
    if(_RULE_BINARY_HDR_FILES)
      target_sources(${_NAME}
        PRIVATE
          ${_RULE_BINARY_HDR_FILES}
      )
    endif()
    target_link_libraries(${_NAME}
      INTERFACE
        ${_LINK_LIBRARIES}
    )
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA}
      OUT_TARGET_DATA _DATA_TARGETS)
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "wasm32")
    _iree_cc_library_add_wasm_deps(${_NAME} ${_RULE_DEPS})
  endif()

  if(_RULE_IS_INTERFACE OR _RULE_ALWAYSLINK)
    target_compile_options(${_NAME} INTERFACE ${_RULE_TRANSITIVE_COPTS})
  endif()

  if(NOT _RULE_IS_INTERFACE)
    set_property(TARGET ${_NAME} PROPERTY IREE_CC_OBJECT_TARGET ${_COMPILE_TARGET})
    set_property(TARGET ${_NAME} PROPERTY IREE_CC_OBJECTS
      "$<TARGET_OBJECTS:${_COMPILE_TARGET}>")
  endif()

  # Order compilation after any registered producers of generated sources or
  # headers. Generated files are represented as paths rather than link targets,
  # so the ordinary target dependency graph cannot infer this edge.
  foreach(_GENERATED_INPUT IN LISTS
      _RULE_SRC_TARGET_SRCS
      _RULE_HDR_TARGET_SRCS)
    iree_generated_output_add_consumer(
      "${_GENERATED_INPUT}"
      "${_COMPILE_TARGET}"
    )
  endforeach()

  if(NOT _RULE_TESTONLY)
    iree_install_targets(
      TARGETS ${_NAME}
      HDRS ${_RULE_HDRS} ${_RULE_TEXTUAL_HDRS}
    )
  endif()

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "  + alias ${_PACKAGE_NS}::${_RULE_NAME}")
  endif()
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    # If the library name matches the final component of the package then treat
    # it as a default. For example, foo/bar/ library 'bar' would end up as
    # 'foo::bar'.
    if(_RULE_PACKAGE)
      string(FIND "${_PACKAGE_NS}" "::" _END_OFFSET REVERSE)
      if(_END_OFFSET EQUAL -1)
        set(_PACKAGE_DIR "${_PACKAGE_NS}")
      else()
        math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
        string(SUBSTRING "${_PACKAGE_NS}" ${_END_OFFSET} -1 _PACKAGE_DIR)
      endif()
    else()
      iree_package_dir(_PACKAGE_DIR)
    endif()
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      iree_add_alias_library(${_PACKAGE_NS} ${_NAME})
    endif()
  endif()
  # Retain the declared link graph for final shared-library composition. The
  # generated LINK_LIBRARIES also contains object and usage-only expressions.
  set_property(TARGET ${_NAME} PROPERTY IREE_CC_DEPS "${_RULE_DEPS}")
  if(NOT _RULE_IS_INTERFACE)
    set_property(TARGET ${_NAME} APPEND PROPERTY IREE_CC_DEPS
      ${IREE_THREADS_DEPS} ${IREE_DEFAULT_LINK_LIBRARIES})
  endif()
  set_property(TARGET ${_NAME} PROPERTY IREE_CC_LINKOPTS "${_RULE_LINKOPTS}")
  set_property(TARGET ${_NAME} PROPERTY IREE_CC_LINKABLE_MORE_THAN_ONCE
    "${_RULE_LINKABLE_MORE_THAN_ONCE}")
  iree_target_sanitizer_suppressions("${_NAME}"
    DEPS ${_RULE_DEPS} ${_DATA_TARGETS}
    SUPPRESSIONS ${_RULE_SANITIZER_SUPPRESSIONS})
endfunction()

# Attaches precompiled objects to their source library and retained object set.
# TARGET_OBJECTS includes native compilation outputs but omits external objects
# supplied through target_sources, such as custom-command MASM thunks.
function(iree_cc_library_add_external_objects target)
  get_target_property(_COMPILE_TARGET ${target} IREE_CC_OBJECT_TARGET)
  target_sources(${_COMPILE_TARGET} PRIVATE ${ARGN})
  set_property(TARGET ${target} APPEND PROPERTY IREE_CC_OBJECTS ${ARGN})
endfunction()

# Preserve transitive JS companions for the final WASM bundler. Generator
# expressions allow providers to be declared after their consumers.
function(_iree_cc_library_add_wasm_deps name)
  foreach(_DEP_TARGET IN LISTS ARGN)
    if(_DEP_TARGET MATCHES "^iree::")
      set_property(TARGET ${name} APPEND PROPERTY
        IREE_WASM_JS_ENTRIES
        "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP_TARGET},IREE_WASM_JS_ENTRIES>>")
      set_property(TARGET ${name} APPEND PROPERTY
        IREE_WASM_JS_SRCS
        "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP_TARGET},IREE_WASM_JS_SRCS>>")
    endif()
  endforeach()
endfunction()
