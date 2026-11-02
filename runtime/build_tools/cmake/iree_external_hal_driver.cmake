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

# Emit the selected external registry as one ordinary runtime library.
function(iree_hal_driver_registry)
  iree_package_binary_dir(_PACKAGE_BINARY_DIR)
  cmake_parse_arguments(_RULE "" "NAME" "DRIVERS" ${ARGN})
  set(IREE_EXTERNAL_HAL_DRIVERS_USED)
  get_property(IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE GLOBAL
               PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
  foreach(_DRIVER_NAME ${_RULE_DRIVERS})
    if(NOT _DRIVER_NAME IN_LIST IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
      message(FATAL_ERROR
        "External HAL driver '${_DRIVER_NAME}' was selected but not registered. "
        "Available drivers: ${IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE}")
    endif()
    if(_DRIVER_NAME IN_LIST IREE_EXTERNAL_HAL_DRIVERS_USED)
      message(FATAL_ERROR
        "External HAL driver '${_DRIVER_NAME}' was selected more than once")
    endif()
    _iree_external_hal_driver_property_prefix(_DRIVER_VAR "${_DRIVER_NAME}")

    get_property(_DRIVER_TARGET GLOBAL PROPERTY ${_DRIVER_VAR}_TARGET)
    get_property(_DRIVER_HEADER GLOBAL PROPERTY ${_DRIVER_VAR}_HEADER)
    get_property(_DRIVER_REGISTER GLOBAL PROPERTY ${_DRIVER_VAR}_REGISTER)
    get_property(_DRIVER_SOURCE_DIR GLOBAL PROPERTY ${_DRIVER_VAR}_SOURCE_DIR)
    get_property(_DRIVER_BINARY_DIR GLOBAL PROPERTY ${_DRIVER_VAR}_BINARY_DIR)

    if(_DRIVER_SOURCE_DIR AND NOT TARGET "${_DRIVER_TARGET}")
      message(STATUS "Adding external HAL driver '${_DRIVER_NAME}' from "
                     "${_DRIVER_SOURCE_DIR}")
      add_subdirectory("${_DRIVER_SOURCE_DIR}" "${_DRIVER_BINARY_DIR}")
    endif()
    if(NOT TARGET "${_DRIVER_TARGET}")
      message(FATAL_ERROR
        "External HAL driver '${_DRIVER_NAME}' did not define target "
        "'${_DRIVER_TARGET}'")
    endif()
    list(APPEND IREE_EXTERNAL_HAL_DRIVERS_USED "${_DRIVER_NAME}")
  endforeach()

  # Generate the external registry implementation even when it is empty. This
  # gives init.c one invariant across build systems and configurations.
  set(_INIT_EXTERNAL_C_TPL "${IREE_SOURCE_DIR}/runtime/src/iree/hal/drivers/init_external.c.in")
  set(_INIT_EXTERNAL_C_SRC "${_PACKAGE_BINARY_DIR}/${_RULE_NAME}.c")
  set(_INIT_EXTERNAL_DEPS)
  set(_INIT_EXTERNAL_REGISTER_INCLUDES)
  set(_INIT_EXTERNAL_REGISTER_CALLS)
  if(IREE_EXTERNAL_HAL_DRIVERS_USED)
    message(STATUS
      "Registering external HAL drivers: ${IREE_EXTERNAL_HAL_DRIVERS_USED}")

    foreach(_DRIVER_NAME ${IREE_EXTERNAL_HAL_DRIVERS_USED})
      _iree_external_hal_driver_property_prefix(_DRIVER_VAR "${_DRIVER_NAME}")
      get_property(_DRIVER_TARGET GLOBAL PROPERTY ${_DRIVER_VAR}_TARGET)
      get_property(_DRIVER_HEADER GLOBAL PROPERTY ${_DRIVER_VAR}_HEADER)
      get_property(_DRIVER_REGISTER GLOBAL PROPERTY ${_DRIVER_VAR}_REGISTER)
      list(APPEND _INIT_EXTERNAL_DEPS "${_DRIVER_TARGET}")
      string(APPEND _INIT_EXTERNAL_REGISTER_INCLUDES
             "#include \"${_DRIVER_HEADER}\"\n")
      string(APPEND _INIT_EXTERNAL_REGISTER_CALLS
             "  IREE_RETURN_IF_ERROR(${_DRIVER_REGISTER}(registry));\n")
    endforeach()
  endif()
  configure_file("${_INIT_EXTERNAL_C_TPL}" "${_INIT_EXTERNAL_C_SRC}")

  iree_cc_library(NAME "${_RULE_NAME}" SRCS "${_INIT_EXTERNAL_C_SRC}"
    DEPS iree::hal::drivers::external_registry_api ${_INIT_EXTERNAL_DEPS} PUBLIC)
endfunction()

function(iree_configure_external_hal_drivers)
  if(IREE_EXTERNAL_HAL_DRIVERS)
    iree_hal_driver_registry(NAME configured_external_driver_registry
      DRIVERS ${IREE_EXTERNAL_HAL_DRIVERS})
  else()
    iree_add_alias_library(iree::hal::drivers::configured_external_driver_registry
      iree_hal_drivers_empty_external_driver_registry)
  endif()
endfunction()
