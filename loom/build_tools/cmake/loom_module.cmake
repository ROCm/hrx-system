# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom module linking helpers.

function(_loom_link_input_paths OUTPUT_PATHS OUTPUT_TARGETS)
  set(_PATHS)
  set(_TARGETS)
  foreach(_INPUT IN LISTS ARGN)
    if("${_INPUT}" MATCHES "::")
      iree_package_target_name(_TARGET "${_INPUT}")
      list(APPEND _PATHS "$<TARGET_PROPERTY:${_TARGET},LOOM_MODULE_FILE>")
      list(APPEND _TARGETS "${_TARGET}")
    elseif(IS_ABSOLUTE "${_INPUT}" OR "${_INPUT}" MATCHES "^\\$<")
      list(APPEND _PATHS "${_INPUT}")
    else()
      list(APPEND _PATHS "${CMAKE_CURRENT_SOURCE_DIR}/${_INPUT}")
    endif()
  endforeach()
  set(${OUTPUT_PATHS} ${_PATHS} PARENT_SCOPE)
  set(${OUTPUT_TARGETS} ${_TARGETS} PARENT_SCOPE)
endfunction()

function(loom_module)
  cmake_parse_arguments(
    _RULE
    "INCLUDE_INPUT_EXPORTS;STRIP_CHECK;REQUIRE_RESOLVED_CONFIG"
    "NAME;MODE;OUTPUT;OUTPUT_FORMAT"
    "SRCS;LIBRARIES;ROOTS;CONFIGS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_module requires NAME")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR "loom_module requires SRCS")
  endif()
  if(NOT _RULE_MODE)
    set(_RULE_MODE "merge")
  endif()
  if(NOT _RULE_MODE STREQUAL "merge" AND NOT _RULE_MODE STREQUAL "link")
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} has unsupported MODE ${_RULE_MODE}"
    )
  endif()
  if(_RULE_MODE STREQUAL "merge" AND
     (_RULE_ROOTS OR _RULE_INCLUDE_INPUT_EXPORTS))
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} merge mode does not accept roots"
    )
  endif()
  if(_RULE_MODE STREQUAL "link" AND
     NOT _RULE_ROOTS AND NOT _RULE_INCLUDE_INPUT_EXPORTS)
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} link mode requires ROOTS or INCLUDE_INPUT_EXPORTS"
    )
  endif()
  if(NOT _RULE_OUTPUT_FORMAT)
    set(_RULE_OUTPUT_FORMAT "text")
  endif()
  if(NOT _RULE_OUTPUT_FORMAT STREQUAL "text" AND
     NOT _RULE_OUTPUT_FORMAT STREQUAL "bc")
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} has unsupported OUTPUT_FORMAT ${_RULE_OUTPUT_FORMAT}"
    )
  endif()
  if(NOT _RULE_OUTPUT)
    if(_RULE_OUTPUT_FORMAT STREQUAL "bc")
      set(_RULE_OUTPUT "${_RULE_NAME}.loombc")
    else()
      set(_RULE_OUTPUT "${_RULE_NAME}.loom")
    endif()
  endif()

  _loom_link_input_paths(_SOURCES _SOURCE_TARGETS ${_RULE_SRCS})
  _loom_link_input_paths(_LIBRARIES _LIBRARY_TARGETS ${_RULE_LIBRARIES})
  set(_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_OUTPUT}")
  set(_ARGS
    "--mode=${_RULE_MODE}"
    "--to=${_RULE_OUTPUT_FORMAT}"
    ${_SOURCES}
  )
  foreach(_LIBRARY IN LISTS _LIBRARIES)
    list(APPEND _ARGS "--library=${_LIBRARY}")
  endforeach()
  foreach(_ROOT IN LISTS _RULE_ROOTS)
    list(APPEND _ARGS "--root=${_ROOT}")
  endforeach()
  foreach(_CONFIG IN LISTS _RULE_CONFIGS)
    list(APPEND _ARGS "--config=${_CONFIG}")
  endforeach()
  if(_RULE_INCLUDE_INPUT_EXPORTS)
    list(APPEND _ARGS "--include-input-exports=true")
  endif()
  if(_RULE_STRIP_CHECK)
    list(APPEND _ARGS "--strip-check=true")
  endif()
  if(_RULE_REQUIRE_RESOLVED_CONFIG)
    list(APPEND _ARGS "--require-resolved-config=true")
  endif()
  list(APPEND _ARGS "--output=${_OUTPUT}")

  add_custom_command(
    OUTPUT
      "${_OUTPUT}"
    COMMAND
      "$<TARGET_FILE:loom::tools::loom-link>" ${_ARGS}
    DEPENDS
      loom::tools::loom-link
      ${_SOURCES}
      ${_LIBRARIES}
    COMMENT
      "Linking Loom module ${_RULE_OUTPUT}"
    VERBATIM
  )
  set_source_files_properties(
    "${_RULE_OUTPUT}"
    "${_OUTPUT}"
    PROPERTIES GENERATED TRUE
  )

  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET}" DEPENDS "${_OUTPUT}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_FILE "${_OUTPUT}")
  foreach(_INPUT_TARGET IN LISTS _SOURCE_TARGETS _LIBRARY_TARGETS)
    iree_register_target_dependency(
      TARGET "${_TARGET}"
      DEPENDENCY "${_INPUT_TARGET}"
    )
  endforeach()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS "${_OUTPUT}"
  )
endfunction()
