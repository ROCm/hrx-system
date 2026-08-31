# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)

function(_iree_external_hal_driver_property_prefix OUTPUT_VARIABLE DRIVER_NAME)
  string(TOUPPER "${DRIVER_NAME}" _DRIVER_KEY)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _DRIVER_KEY "${_DRIVER_KEY}")
  set(${OUTPUT_VARIABLE} "IREE_EXTERNAL_${_DRIVER_KEY}_HAL_DRIVER"
      PARENT_SCOPE)
endfunction()

# Registers an external HAL driver that may be selected through
# IREE_EXTERNAL_HAL_DRIVERS.
#
# The registration target must expose HEADER through its public include paths
# and implement REGISTER with the following C signature:
#
#   iree_status_t REGISTER(iree_hal_driver_registry_t* registry);
#
# SOURCE_DIR defers adding an external project until the driver is selected. A
# target already defined by the embedding project needs no source directory.
function(iree_register_external_hal_driver)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET;HEADER;REGISTER;SOURCE_DIR;BINARY_DIR"
    ""
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unparsed external HAL driver arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  if(_RULE_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "External HAL driver fields missing values: "
      "${_RULE_KEYWORDS_MISSING_VALUES}")
  endif()
  foreach(_REQUIRED_FIELD NAME TARGET HEADER REGISTER)
    if(NOT _RULE_${_REQUIRED_FIELD})
      message(FATAL_ERROR
        "External HAL driver is missing required field ${_REQUIRED_FIELD}")
    endif()
  endforeach()

  if(NOT _RULE_NAME MATCHES "^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    message(FATAL_ERROR "Invalid external HAL driver name '${_RULE_NAME}'")
  endif()
  if(NOT _RULE_REGISTER MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR
      "Invalid registration function '${_RULE_REGISTER}' for external HAL "
      "driver '${_RULE_NAME}'")
  endif()
  if(IS_ABSOLUTE "${_RULE_HEADER}" OR
     _RULE_HEADER MATCHES "[\\\";]" OR
     _RULE_HEADER MATCHES "(^|/)\\.\\.(/|$)")
    message(FATAL_ERROR
      "External HAL driver '${_RULE_NAME}' HEADER must be a relative C include "
      "path, got '${_RULE_HEADER}'")
  endif()
  if(_RULE_BINARY_DIR AND NOT _RULE_SOURCE_DIR)
    message(FATAL_ERROR
      "External HAL driver '${_RULE_NAME}' has BINARY_DIR without SOURCE_DIR")
  endif()

  _iree_external_hal_driver_property_prefix(_DRIVER_VAR "${_RULE_NAME}")
  get_property(_COLLIDING_NAME GLOBAL PROPERTY ${_DRIVER_VAR}_NAME)
  if(_COLLIDING_NAME AND
     NOT "${_COLLIDING_NAME}" STREQUAL "${_RULE_NAME}")
    message(FATAL_ERROR
      "External HAL driver names '${_COLLIDING_NAME}' and '${_RULE_NAME}' "
      "produce the same configuration key '${_DRIVER_VAR}'")
  endif()

  get_property(_AVAILABLE_DRIVERS GLOBAL
               PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
  if(_RULE_NAME IN_LIST _AVAILABLE_DRIVERS)
    message(FATAL_ERROR
      "External HAL driver '${_RULE_NAME}' has already been registered")
  endif()

  if(_RULE_SOURCE_DIR)
    get_filename_component(
      _RULE_SOURCE_DIR "${_RULE_SOURCE_DIR}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    if(NOT EXISTS "${_RULE_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR
        "External HAL driver '${_RULE_NAME}' has no CMakeLists.txt at "
        "'${_RULE_SOURCE_DIR}'")
    endif()
    if(_RULE_BINARY_DIR)
      get_filename_component(
        _RULE_BINARY_DIR "${_RULE_BINARY_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
      )
    else()
      set(_RULE_BINARY_DIR
          "${CMAKE_BINARY_DIR}/iree_external_hal_drivers/${_RULE_NAME}")
    endif()
  endif()

  set_property(GLOBAL APPEND PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE
               "${_RULE_NAME}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_NAME "${_RULE_NAME}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_TARGET "${_RULE_TARGET}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_HEADER "${_RULE_HEADER}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_REGISTER "${_RULE_REGISTER}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_SOURCE_DIR "${_RULE_SOURCE_DIR}")
  set_property(GLOBAL PROPERTY ${_DRIVER_VAR}_BINARY_DIR "${_RULE_BINARY_DIR}")
endfunction()
