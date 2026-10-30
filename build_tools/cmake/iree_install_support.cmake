# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(_iree_json_escape OUTPUT_VARIABLE VALUE)
  string(REPLACE "\\" "\\\\" _ESCAPED "${VALUE}")
  string(REPLACE "\"" "\\\"" _ESCAPED "${_ESCAPED}")
  set(${OUTPUT_VARIABLE} "${_ESCAPED}" PARENT_SCOPE)
endfunction()

function(iree_record_target_alias to_target from_target)
  set_property(GLOBAL APPEND PROPERTY IREE_TARGET_ALIAS_MAP "${to_target}=${from_target}")
endfunction()

# Writes the names, aliases, and executable paths used by the developer tools.
# Defer this until after target-producing callbacks have run. TARGET_FILE is
# evaluated by CMake for each configuration, including custom output names and
# directories; consumers never reconstruct a platform's executable path.
function(iree_write_target_map)
  get_property(_ALIAS_MAP GLOBAL PROPERTY IREE_TARGET_ALIAS_MAP)
  set(_ALIAS_MAP_PATH "${CMAKE_BINARY_DIR}/.iree/target_aliases.json")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.iree")
  set(_ALIAS_MAP_JSON "{\n")
  if(_ALIAS_MAP)
    list(REMOVE_DUPLICATES _ALIAS_MAP)
    list(SORT _ALIAS_MAP)
    set(_SEPARATOR "")
    foreach(_ALIAS_ENTRY IN LISTS _ALIAS_MAP)
      string(FIND "${_ALIAS_ENTRY}" "=" _SEPARATOR_OFFSET)
      if(_SEPARATOR_OFFSET EQUAL -1)
        continue()
      endif()
      string(SUBSTRING "${_ALIAS_ENTRY}" 0 ${_SEPARATOR_OFFSET} _ALIAS_NAME)
      math(EXPR _TARGET_NAME_OFFSET "${_SEPARATOR_OFFSET} + 1")
      string(SUBSTRING "${_ALIAS_ENTRY}" ${_TARGET_NAME_OFFSET} -1 _TARGET_NAME)
      _iree_json_escape(_JSON_ALIAS_NAME "${_ALIAS_NAME}")
      _iree_json_escape(_JSON_TARGET_NAME "${_TARGET_NAME}")
      string(APPEND _ALIAS_MAP_JSON
        "${_SEPARATOR}  \"${_JSON_ALIAS_NAME}\": \"${_JSON_TARGET_NAME}\"")
      set(_SEPARATOR ",\n")
    endforeach()
    string(APPEND _ALIAS_MAP_JSON "\n")
  endif()
  string(APPEND _ALIAS_MAP_JSON "}\n")
  file(WRITE "${_ALIAS_MAP_PATH}" "${_ALIAS_MAP_JSON}")

  set(_DIRECTORIES "${CMAKE_SOURCE_DIR}")
  set(_TARGET_MAP "")
  while(_DIRECTORIES)
    list(POP_FRONT _DIRECTORIES _DIRECTORY)
    get_property(_CHILDREN DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)
    list(APPEND _DIRECTORIES ${_CHILDREN})
    get_property(_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_TARGET IN LISTS _TARGETS)
      get_target_property(_TYPE "${_TARGET}" TYPE)
      if(_TYPE STREQUAL "EXECUTABLE")
        string(APPEND _TARGET_MAP "${_TARGET}\t$<TARGET_FILE:${_TARGET}>\n")
      elseif(_TYPE STREQUAL "INTERFACE_LIBRARY")
        # Interfaces with sources/header sets own native generation rules;
        # usage-only interfaces have no target in the generated build system.
        get_target_property(_SOURCES "${_TARGET}" SOURCES)
        get_target_property(_HEADER_SETS "${_TARGET}" HEADER_SETS)
        if(_SOURCES OR _HEADER_SETS)
          string(APPEND _TARGET_MAP "${_TARGET}\n")
        endif()
      else()
        string(APPEND _TARGET_MAP "${_TARGET}\n")
      endif()
    endforeach()
  endwhile()
  # One line per native target; executable paths follow a tab. Unlike JSON,
  # this preserves generator-resolved paths without another escaping pass.
  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/.iree/targets-$<CONFIG>.txt"
    CONTENT "${_TARGET_MAP}")

  # Name only the current configurations so stale files from a previous
  # configure cannot contribute targets or executable paths.
  get_property(_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_MULTI_CONFIG)
    set(_TARGET_MAP_FILES ${CMAKE_CONFIGURATION_TYPES})
    list(TRANSFORM _TARGET_MAP_FILES PREPEND "targets-")
    list(TRANSFORM _TARGET_MAP_FILES APPEND ".txt")
  else()
    set(_TARGET_MAP_FILES "targets-${CMAKE_BUILD_TYPE}.txt")
  endif()
  list(JOIN _TARGET_MAP_FILES "\n" _TARGET_MAP_FILES_CONTENT)
  file(WRITE "${CMAKE_BINARY_DIR}/.iree/target_configurations.txt"
    "${_TARGET_MAP_FILES_CONTENT}\n")
endfunction()

function(iree_add_alias_library to_target from_target)
  add_library(${to_target} ALIAS ${from_target})
  iree_record_target_alias(${to_target} ${from_target})
  # Yes: Leading-lowercase property names are load bearing and the recommended
  # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
  # We have to export the aliases on the target, and the when we generate
  # IREEConfig.cmake, generate code to re-establish the alias on import.
  # Yeah, this is completely sane.
  set_property(TARGET ${from_target} APPEND PROPERTY iree_ALIAS_TO ${to_target})
endfunction()

function(iree_add_alias_executable to_target from_target)
  add_executable(${to_target} ALIAS ${from_target})
  iree_record_target_alias(${to_target} ${from_target})
endfunction()

function(iree_install_targets)
  iree_package_source_dir(_IREE_PACKAGE_SOURCE_DIR)
  cmake_parse_arguments(
    _RULE
    ""
    "COMPONENT;EXPORT_SET"
    "HDRS;TARGETS"
    ${ARGN}
  )

  # Determine install component. It can be explicit on the target or implicit
  # from the CMake variable IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT
  # (usually set at the directory level). Note that truthy evaluation is
  # intended: Installation can be suppressed by setting "COMPONENT OFF".
  set(_INSTALL_COMPONENT "${_RULE_COMPONENT}")
  if(NOT _INSTALL_COMPONENT AND IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT)
    set(_INSTALL_COMPONENT "${IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT}")
  endif()
  if(NOT _INSTALL_COMPONENT)
    return()
  endif()

  # Ditto for export set name.
  set(_EXPORT_SET "${_RULE_EXPORT_SET}")
  if(NOT _EXPORT_SET AND IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET)
    set(_EXPORT_SET "${IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET}")
  endif()
  if(NOT _EXPORT_SET)
    message(SEND_ERROR "Installing ${_RULE_TARGETS}: An install COMPONENT was set but EXPORT_SET was not")
  endif()

  # Process targets.
  set_property(TARGET ${_RULE_TARGETS} APPEND PROPERTY EXPORT_PROPERTIES iree_ALIAS_TO)

  # Add it to the global property that will be processed at the end of the build.
  set_property(GLOBAL APPEND PROPERTY "IREE_EXPORT_TARGETS_${_EXPORT_SET}" ${_RULE_TARGETS})

  # The export name is set at a directory level to control export.
  install(
    TARGETS ${_RULE_TARGETS}
    EXPORT IREEExported-${_EXPORT_SET}
    COMPONENT "${_INSTALL_COMPONENT}"
    EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install headers if the rule declares them and the directory level
  # root path is truthy.
  if(_RULE_HDRS AND IREE_HDRS_ROOT_PATH)
    foreach(_hdr_file ${_RULE_HDRS})
      if(IS_ABSOLUTE "${_hdr_file}")
        set(_hdr_abs_path "${_hdr_file}")
        set(_hdr_base_relative "${_hdr_file}")
        cmake_path(IS_PREFIX IREE_BINARY_DIR "${_hdr_file}" NORMALIZE _GENERATED)
        if(_GENERATED)
          cmake_path(RELATIVE_PATH _hdr_base_relative BASE_DIRECTORY "${IREE_BINARY_DIR}")
          set(_hdr_base_relative "${IREE_SOURCE_DIR}/${_hdr_base_relative}")
        endif()
      else()
        set(_hdr_abs_path "${_IREE_PACKAGE_SOURCE_DIR}/${_hdr_file}")
        if(NOT EXISTS "${_hdr_abs_path}")
          # Producers own generated headers at the corresponding binary path.
          set(_hdr_abs_path "${CMAKE_CURRENT_BINARY_DIR}/${_hdr_file}")
        endif()
        set(_hdr_base_relative "${_IREE_PACKAGE_SOURCE_DIR}/${_hdr_file}")
      endif()
      cmake_path(
        RELATIVE_PATH _hdr_base_relative
        BASE_DIRECTORY "${IREE_HDRS_ROOT_PATH}")
      cmake_path(
        GET _hdr_base_relative
        PARENT_PATH _rel_path
      )
      install(
        FILES "${_hdr_abs_path}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_rel_path}"
        COMPONENT "${_INSTALL_COMPONENT}"
        EXCLUDE_FROM_ALL
      )
    endforeach()
  endif()
endfunction()

# Generates build and install tree export CMake files for a given named
# export set.
# This will export any targets added to iree_install_targets above with the
# same export set name (either explicit or at the directory level).
#
# This function must be called at a location after which all targets that are
# part of the export set have been added (typically at the top-level).
# Exports will not be generated if no targets were added for that export set
# name.
#
# If a COMPONENT is specified, the exports will be installed with that
# component. Defaults to IREECMakeExports.
function(iree_generate_export_targets)
  cmake_parse_arguments(
    _RULE
    ""
    "COMPONENT;EXPORT_SET;INSTALL_DESTINATION"
    ""
    ${ARGN}
  )

  if(NOT _RULE_EXPORT_SET)
    message(FATAL_ERROR "EXPORT_SET is required")
    return()
  endif()

  get_property(_export_targets GLOBAL PROPERTY "IREE_EXPORT_TARGETS_${_RULE_EXPORT_SET}")
  if(NOT _export_targets)
    message(STATUS "Skipping generation of export set ${_RULE_EXPORT_SET} (no targets matched)")
  endif()

  set(_component "${_RULE_COMPONENT}")
  if(NOT _component)
    set(_component "IREECMakeExports")
  endif()

  export(TARGETS ${_export_targets}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/IREETargets-${_RULE_EXPORT_SET}.cmake")

  install(
    EXPORT "IREEExported-${_RULE_EXPORT_SET}"
    COMPONENT "${_component}"
    FILE "IREETargets-${_RULE_EXPORT_SET}.cmake"
    DESTINATION "${_RULE_INSTALL_DESTINATION}"
  )

  # Clear the export targets so that innocent aggregating projects don't
  # get in trouble if they use our setup.
  set_property(GLOBAL PROPERTY "IREE_EXPORT_TARGETS_${_RULE_EXPORT_SET}")
endfunction()

# Adds a convenience install target for a component.
function(iree_add_install_target)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;COMPONENT"
    "DEPENDS;ADD_TO"
    ${ARGN}
  )
  set(_depends ${_RULE_DEPENDS})

  if(NOT _RULE_COMPONENT)
    # Just create stub targets.
    add_custom_target(${_RULE_NAME})
    add_custom_target(${_RULE_NAME}-stripped)
  else()
    # Create targets to install a component.
    set(_options -DCMAKE_INSTALL_COMPONENT="${_RULE_COMPONENT}")

    # Non-stripped.
    add_custom_target(
      ${_RULE_NAME}
      COMMAND "${CMAKE_COMMAND}"
              ${_options}
              -P "${IREE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
    set_target_properties(${_RULE_NAME} PROPERTIES FOLDER "Component Install Targets")

    # Stripped.
    add_custom_target(
      ${_RULE_NAME}-stripped
      COMMAND "${CMAKE_COMMAND}"
              ${_options}
              -DCMAKE_INSTALL_DO_STRIP=1
              -P "${IREE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
    set_target_properties(${_RULE_NAME} PROPERTIES FOLDER "Component Install Targets")
  endif()

  if(_depends)
    add_dependencies(${_RULE_NAME} ${_depends})
    add_dependencies(${_RULE_NAME}-stripped ${_depends})
  endif()

  if(_RULE_ADD_TO)
    foreach(_add_to ${_RULE_ADD_TO})
      add_dependencies(${_add_to} ${_RULE_NAME})
      add_dependencies(${_add_to}-stripped ${_RULE_NAME}-stripped)
    endforeach()
  endif()
endfunction()
