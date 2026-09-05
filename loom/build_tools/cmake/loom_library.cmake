# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Relocatable Loom library helpers.

function(_loom_package_label OUTPUT_LABEL NAME)
  file(RELATIVE_PATH _PACKAGE_PATH
    "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  string(REPLACE "\\" "/" _PACKAGE_PATH "${_PACKAGE_PATH}")
  set(${OUTPUT_LABEL} "//${_PACKAGE_PATH}:${NAME}" PARENT_SCOPE)
endfunction()

function(_loom_library_dependency_modules
  OUTPUT_DIRECT_MODULES
  OUTPUT_TRANSITIVE_MODULES
  OUTPUT_TARGETS
  CONSUMER_TARGET
)
  _loom_link_input_paths(_DIRECT_MODULES _TARGETS ${ARGN})
  list(LENGTH ARGN _DEPENDENCY_COUNT)
  list(LENGTH _TARGETS _TARGET_COUNT)
  if(NOT _DEPENDENCY_COUNT EQUAL _TARGET_COUNT)
    message(FATAL_ERROR
      "Loom library consumer ${CONSUMER_TARGET} DEPS must name "
      "loom_library targets")
  endif()
  set(_TRANSITIVE_MODULES
    "$<TARGET_PROPERTY:${CONSUMER_TARGET},LOOM_LIBRARY_TRANSITIVE_MODULES>"
  )
  set(${OUTPUT_DIRECT_MODULES} "${_DIRECT_MODULES}" PARENT_SCOPE)
  set(${OUTPUT_TRANSITIVE_MODULES} "${_TRANSITIVE_MODULES}" PARENT_SCOPE)
  set(${OUTPUT_TARGETS} "${_TARGETS}" PARENT_SCOPE)
endfunction()

function(_loom_resolve_library_dependency_modules TARGET)
  if(NOT TARGET "${TARGET}")
    message(FATAL_ERROR "Loom library target does not exist: ${TARGET}")
  endif()
  get_target_property(
    _STATE "${TARGET}" LOOM_LIBRARY_DEPENDENCY_RESOLUTION_STATE)
  if(_STATE STREQUAL "resolved")
    return()
  elseif(_STATE STREQUAL "resolving")
    message(FATAL_ERROR
      "Loom library dependency cycle reaches ${TARGET}")
  endif()
  get_target_property(_IS_LIBRARY "${TARGET}" LOOM_LIBRARY_TARGET)
  if(NOT _IS_LIBRARY)
    message(FATAL_ERROR
      "Loom library dependency ${TARGET} is not a loom_library target")
  endif()

  set_property(TARGET "${TARGET}" PROPERTY
    LOOM_LIBRARY_DEPENDENCY_RESOLUTION_STATE "resolving")
  get_target_property(
    _DIRECT_DEPENDENCIES "${TARGET}" LOOM_LIBRARY_DIRECT_DEPENDENCIES)
  set(_DEPENDENCY_MODULES)
  foreach(_DEPENDENCY IN LISTS _DIRECT_DEPENDENCIES)
    _loom_resolve_library_dependency_modules("${_DEPENDENCY}")
    get_target_property(_MODULE "${_DEPENDENCY}" LOOM_MODULE_FILE)
    get_target_property(
      _TRANSITIVE_MODULES
      "${_DEPENDENCY}"
      LOOM_LIBRARY_DEPENDENCY_MODULES
    )
    list(APPEND _DEPENDENCY_MODULES "${_MODULE}" ${_TRANSITIVE_MODULES})
  endforeach()
  list(REMOVE_DUPLICATES _DEPENDENCY_MODULES)
  set_property(TARGET "${TARGET}" PROPERTY
    LOOM_LIBRARY_DEPENDENCY_MODULES "${_DEPENDENCY_MODULES}")
  set_property(TARGET "${TARGET}" PROPERTY
    LOOM_LIBRARY_DEPENDENCY_RESOLUTION_STATE "resolved")
endfunction()

function(_loom_finalize_library_dependencies)
  get_property(_CONSUMERS GLOBAL PROPERTY LOOM_LIBRARY_CONSUMERS)
  list(REMOVE_DUPLICATES _CONSUMERS)
  foreach(_CONSUMER IN LISTS _CONSUMERS)
    get_target_property(
      _DIRECT_DEPENDENCIES
      "${_CONSUMER}"
      LOOM_LIBRARY_DIRECT_DEPENDENCIES
    )
    set(_DIRECT_MODULES)
    set(_DEPENDENCY_MODULES)
    foreach(_DEPENDENCY IN LISTS _DIRECT_DEPENDENCIES)
      _loom_resolve_library_dependency_modules("${_DEPENDENCY}")
      get_target_property(_MODULE "${_DEPENDENCY}" LOOM_MODULE_FILE)
      get_target_property(
        _TRANSITIVE_MODULES
        "${_DEPENDENCY}"
        LOOM_LIBRARY_DEPENDENCY_MODULES
      )
      list(APPEND _DIRECT_MODULES "${_MODULE}")
      list(APPEND _DEPENDENCY_MODULES "${_MODULE}" ${_TRANSITIVE_MODULES})
    endforeach()
    list(REMOVE_DUPLICATES _DIRECT_MODULES)
    list(REMOVE_DUPLICATES _DEPENDENCY_MODULES)
    if(_DIRECT_MODULES)
      list(REMOVE_ITEM _DEPENDENCY_MODULES ${_DIRECT_MODULES})
    endif()
    set_property(TARGET "${_CONSUMER}" PROPERTY
      LOOM_LIBRARY_TRANSITIVE_MODULES "${_DEPENDENCY_MODULES}")
  endforeach()
endfunction()

function(_loom_register_library_consumer TARGET)
  if(NOT TARGET "${TARGET}")
    message(FATAL_ERROR "Loom library consumer does not exist: ${TARGET}")
  endif()
  set_property(TARGET "${TARGET}" PROPERTY
    LOOM_LIBRARY_DIRECT_DEPENDENCIES "${ARGN}")
  set_property(GLOBAL APPEND PROPERTY LOOM_LIBRARY_CONSUMERS "${TARGET}")
endfunction()

function(_loom_transitive_library_argument OUTPUT_ARGUMENT MODULES)
  if(MODULES)
    set(_ARGUMENT
      "$<$<BOOL:${MODULES}>:--transitive-library=$<JOIN:${MODULES},$<SEMICOLON>--transitive-library=>>"
    )
  else()
    set(_ARGUMENT)
  endif()
  set(${OUTPUT_ARGUMENT} "${_ARGUMENT}" PARENT_SCOPE)
endfunction()

function(_loom_declare_relocatable_module)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;CONSUMER_TARGET;OUTPUT;DEPENDENCY_REPORT;OUT_DEPENDENCY_TARGETS;OUT_LIBRARY_DEPENDENCY_TARGETS"
    "SRCS;DEPS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Loom relocatable module ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_NAME OR NOT _RULE_CONSUMER_TARGET OR NOT _RULE_OUTPUT OR
     NOT _RULE_DEPENDENCY_REPORT)
    message(FATAL_ERROR
      "Loom relocatable module requires NAME, CONSUMER_TARGET, OUTPUT, and "
      "DEPENDENCY_REPORT"
    )
  endif()
  if(NOT _RULE_SRCS AND NOT _RULE_DEPS)
    message(FATAL_ERROR
      "Loom relocatable module ${_RULE_NAME} requires SRCS or DEPS")
  endif()

  _loom_link_input_paths(_SOURCES _SOURCE_TARGETS ${_RULE_SRCS})
  _loom_library_dependency_modules(
    _DIRECT_MODULES
    _TRANSITIVE_MODULES
    _DEPENDENCY_TARGETS
    "${_RULE_CONSUMER_TARGET}"
    ${_RULE_DEPS}
  )
  _loom_transitive_library_argument(
    _TRANSITIVE_ARGUMENT "${_TRANSITIVE_MODULES}"
  )
  set(_ARGS
    "--mode=merge"
    "--strict-deps"
    "--dependency-component=${_RULE_NAME}"
    "--dependency-report=${_RULE_DEPENDENCY_REPORT}"
    "--to=bc"
    "--output=${_RULE_OUTPUT}"
    ${_SOURCES}
  )
  foreach(_MODULE IN LISTS _DIRECT_MODULES)
    list(APPEND _ARGS "--library=${_MODULE}")
  endforeach()
  if(_TRANSITIVE_ARGUMENT)
    list(APPEND _ARGS "${_TRANSITIVE_ARGUMENT}")
  endif()

  add_custom_command(
    OUTPUT
      "${_RULE_OUTPUT}"
      "${_RULE_DEPENDENCY_REPORT}"
    COMMAND
      "$<TARGET_FILE:loom::tools::loom-link>" ${_ARGS}
    DEPENDS
      loom::tools::loom-link
      ${_SOURCES}
      ${_DIRECT_MODULES}
      ${_TRANSITIVE_MODULES}
    COMMENT
      "Linking relocatable Loom module ${_RULE_NAME}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  set_source_files_properties(
    "${_RULE_OUTPUT}"
    "${_RULE_DEPENDENCY_REPORT}"
    PROPERTIES GENERATED TRUE
  )
  set(_ALL_DEPENDENCY_TARGETS ${_SOURCE_TARGETS} ${_DEPENDENCY_TARGETS})
  list(FILTER _ALL_DEPENDENCY_TARGETS EXCLUDE REGEX "^$")
  set(
    ${_RULE_OUT_DEPENDENCY_TARGETS}
    "${_ALL_DEPENDENCY_TARGETS}"
    PARENT_SCOPE
  )
  set(
    ${_RULE_OUT_LIBRARY_DEPENDENCY_TARGETS}
    "${_DEPENDENCY_TARGETS}"
    PARENT_SCOPE
  )
endfunction()

function(loom_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_library ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_library requires NAME")
  endif()
  if(NOT _RULE_SRCS AND NOT _RULE_DEPS)
    message(FATAL_ERROR "loom_library ${_RULE_NAME} requires SRCS or DEPS")
  endif()

  iree_package_name(_PACKAGE_NAME)
  _loom_package_label(_COMPONENT "${_RULE_NAME}")
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.loombc")
  set(_DEPENDENCY_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.dependencies.json")
  _loom_declare_relocatable_module(
    NAME "${_COMPONENT}"
    CONSUMER_TARGET "${_TARGET}"
    OUTPUT "${_OUTPUT}"
    DEPENDENCY_REPORT "${_DEPENDENCY_REPORT}"
    OUT_DEPENDENCY_TARGETS _DEPENDENCY_TARGETS
    OUT_LIBRARY_DEPENDENCY_TARGETS _LIBRARY_DEPENDENCY_TARGETS
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
  )

  add_custom_target("${_TARGET}" DEPENDS "${_OUTPUT}" "${_DEPENDENCY_REPORT}")
  set_property(TARGET "${_TARGET}" PROPERTY IREE_GENERATED_FILE "${_OUTPUT}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_LIBRARY_TARGET TRUE)
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_FILE "${_OUTPUT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_LIBRARY_DEPENDENCY_REPORT "${_DEPENDENCY_REPORT}")

  _loom_register_library_consumer(
    "${_TARGET}" ${_LIBRARY_DEPENDENCY_TARGETS})
  foreach(_DEPENDENCY_TARGET IN LISTS _DEPENDENCY_TARGETS)
    iree_register_target_dependency(
      TARGET "${_TARGET}"
      DEPENDENCY "${_DEPENDENCY_TARGET}"
    )
  endforeach()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS "${_OUTPUT}" "${_DEPENDENCY_REPORT}"
  )
endfunction()

# Resolves dependency closures after every loom/src package has declared its
# targets. CMake traverses packages by directory rather than dependency order,
# so consumers may be seen before their libraries.
cmake_language(DEFER CALL _loom_finalize_library_dependencies)
