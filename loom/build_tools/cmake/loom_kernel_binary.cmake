# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/loom_module.cmake")

# Mirrors loom_kernel_binary's relocatable source module, resolved link, and
# target-owned kernel emission. Libraries retain their independently declared
# transitive closure until loom_finalize_module_libraries resolves the graph.
function(loom_kernel_binary)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;COMPONENT;TARGET;OUTPUT;INPUT_FORMAT"
    "SRCS;LIBRARIES;DATA;INPUTOPTS;ROOTS;CONFIGS"
    ${ARGN}
  )
  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_kernel_binary has unknown arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _RULE_NAME OR NOT _RULE_TARGET)
    message(FATAL_ERROR "loom_kernel_binary requires NAME and TARGET")
  endif()
  if(NOT _RULE_SRCS AND NOT _RULE_LIBRARIES)
    message(FATAL_ERROR "loom_kernel_binary requires SRCS or LIBRARIES")
  endif()
  if(NOT _RULE_SRCS AND
     (_RULE_DATA OR _RULE_INPUT_FORMAT OR _RULE_INPUTOPTS))
    message(FATAL_ERROR
      "loom_kernel_binary source admission options require SRCS")
  endif()
  if(NOT _RULE_OUTPUT)
    set(_RULE_OUTPUT "${_RULE_NAME}")
  endif()
  if(NOT _RULE_COMPONENT)
    file(RELATIVE_PATH _PACKAGE_PATH "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_RULE_COMPONENT "//${_PACKAGE_PATH}:${_RULE_NAME}")
  endif()

  iree_package_target_name(_PROFILE "${_RULE_TARGET}")
  get_target_property(_AVAILABLE "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
  if(NOT _AVAILABLE)
    message(FATAL_ERROR
      "loom_kernel_binary ${_RULE_NAME} requires available profile ${_RULE_TARGET}")
  endif()
  get_target_property(_COMPILER_TARGET "${_PROFILE}" LOOM_COMPILER_TARGET)
  _loom_host_tool(_LINK_TOOL loom-link)
  _loom_host_tool(_COMPILE_TOOL loom-compile)
  _loom_link_input_paths(_SOURCES _SOURCE_TARGETS ${_RULE_SRCS})
  _loom_link_input_paths(_LIBRARIES _LIBRARY_TARGETS ${_RULE_LIBRARIES})

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_TRANSITIVE_LIBRARIES "$<TARGET_GENEX_EVAL:${_TARGET},$<TARGET_PROPERTY:${_TARGET},LOOM_MODULE_TRANSITIVE_LIBRARIES>>")
  set(_TRANSITIVE_ARGS "$<$<BOOL:${_TRANSITIVE_LIBRARIES}>:--transitive-library=$<JOIN:${_TRANSITIVE_LIBRARIES},$<SEMICOLON>--transitive-library=>>")
  set(_STEM "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}")
  set(_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_OUTPUT}")
  set(_COMPILE_REPORT "${_STEM}.compile.json")
  set(_LINKED_MODULE "${_STEM}.linked.loombc")
  get_filename_component(_OUTPUT_DIRECTORY "${_OUTPUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${_OUTPUT_DIRECTORY}")
  set(_DIRECT_MODULES ${_LIBRARIES})

  if(_SOURCES)
    set(_SOURCE_MODULE "${_STEM}.sources.loombc")
    set(_DEPENDENCY_REPORT "${_STEM}.sources.dependencies.json")
    set(_ARGS
      "--mode=merge"
      "--strict-deps"
      "--dependency-component=${_RULE_COMPONENT}"
      "--dependency-report=${_DEPENDENCY_REPORT}"
      "--to=bc"
      "--output=${_SOURCE_MODULE}"
    )
    if(_RULE_INPUT_FORMAT)
      list(APPEND _ARGS "--input-format=${_RULE_INPUT_FORMAT}")
    endif()
    foreach(_OPTIONS IN LISTS _RULE_INPUTOPTS)
      list(APPEND _ARGS "--input-options=${_OPTIONS}")
    endforeach()
    list(APPEND _ARGS ${_SOURCES})
    foreach(_LIBRARY IN LISTS _LIBRARIES)
      list(APPEND _ARGS "--library=${_LIBRARY}")
    endforeach()
    list(APPEND _ARGS "${_TRANSITIVE_ARGS}")
    add_custom_command(
      OUTPUT "${_SOURCE_MODULE}" "${_DEPENDENCY_REPORT}"
      COMMAND "${_LINK_TOOL}" "${_ARGS}"
      DEPENDS
        "${_LINK_TOOL}"
        ${_SOURCES}
        "${_LIBRARIES}"
        "${_TRANSITIVE_LIBRARIES}"
        ${_RULE_DATA}
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      COMMENT "Assembling Loom sources for ${_RULE_NAME}"
      VERBATIM COMMAND_EXPAND_LISTS
    )
    list(APPEND _DIRECT_MODULES "${_SOURCE_MODULE}")
  endif()

  set(_ARGS
    "--mode=link"
    "--strip-check"
    "--require-resolved-config"
    "--to=bc"
    "--output=${_LINKED_MODULE}"
  )
  if(_RULE_ROOTS)
    foreach(_MODULE IN LISTS _DIRECT_MODULES)
      list(APPEND _ARGS "--library=${_MODULE}")
    endforeach()
    foreach(_ROOT IN LISTS _RULE_ROOTS)
      list(APPEND _ARGS "--root=${_ROOT}")
    endforeach()
  else()
    foreach(_MODULE IN LISTS _DIRECT_MODULES)
      list(APPEND _ARGS "--root-library=${_MODULE}")
    endforeach()
  endif()
  list(APPEND _ARGS "${_TRANSITIVE_ARGS}")
  foreach(_CONFIG IN LISTS _RULE_CONFIGS)
    list(APPEND _ARGS "--config=${_CONFIG}")
  endforeach()
  list(APPEND _ARGS "--target=${_COMPILER_TARGET}")
  add_custom_command(
    OUTPUT "${_LINKED_MODULE}"
    COMMAND "${_LINK_TOOL}" "${_ARGS}"
    DEPENDS "${_LINK_TOOL}" "${_DIRECT_MODULES}" "${_TRANSITIVE_LIBRARIES}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Linking Loom kernel ${_RULE_NAME}"
    VERBATIM COMMAND_EXPAND_LISTS
  )
  add_custom_command(
    OUTPUT "${_OUTPUT}" "${_COMPILE_REPORT}"
    COMMAND "${_COMPILE_TOOL}"
      "${_LINKED_MODULE}"
      "--target=${_COMPILER_TARGET}"
      "--output=${_OUTPUT}"
      "--compile-report=details"
      "--compile-report-output=${_COMPILE_REPORT}"
    DEPENDS "${_COMPILE_TOOL}" "${_LINKED_MODULE}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Compiling Loom kernel ${_RULE_OUTPUT}"
    VERBATIM
  )

  add_custom_target("${_TARGET}" DEPENDS "${_OUTPUT}" "${_COMPILE_REPORT}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_DIRECT_LIBRARIES "${_LIBRARIES}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_LIBRARY_TARGETS "${_LIBRARY_TARGETS}")
  set_property(GLOBAL APPEND PROPERTY LOOM_MODULE_TARGETS "${_TARGET}")
  foreach(_INPUT_TARGET IN LISTS _SOURCE_TARGETS _LIBRARY_TARGETS)
    iree_register_target_dependency(TARGET "${_TARGET}" DEPENDENCY "${_INPUT_TARGET}")
  endforeach()
  foreach(_INPUT IN LISTS _SOURCES _LIBRARIES _RULE_DATA)
    iree_generated_output_add_consumer("${_INPUT}" "${_TARGET}")
  endforeach()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS "${_OUTPUT}" "${_COMPILE_REPORT}")
endfunction()
