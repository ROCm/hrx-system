# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Resolve the explicitly selected external HAL drivers. Projects register
# available modules with iree_register_external_hal_driver before this
# directory is evaluated. Registration is declarative: a source directory is
# added only when its driver name is selected.
set(IREE_EXTERNAL_HAL_DRIVERS_USED)
get_property(IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE GLOBAL
             PROPERTY IREE_EXTERNAL_HAL_DRIVERS_AVAILABLE)
foreach(_DRIVER_NAME ${IREE_EXTERNAL_HAL_DRIVERS})
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
set(_INIT_EXTERNAL_C_TPL "${CMAKE_CURRENT_SOURCE_DIR}/init_external.c.in")
set(_INIT_EXTERNAL_C_SRC "${CMAKE_CURRENT_BINARY_DIR}/init_external.c")
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
file(READ "${_INIT_EXTERNAL_C_TPL}" _INIT_EXTERNAL_TEMPLATE)
file(CONFIGURE OUTPUT "${_INIT_EXTERNAL_C_SRC}"
     CONTENT "${_INIT_EXTERNAL_TEMPLATE}")

set(_INIT_INTERNAL_DEPS)
if(IREE_HAL_DRIVER_AMDGPU)
  add_subdirectory(amdgpu)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::amdgpu::registration)
endif()
if(IREE_HAL_DRIVER_TASK)
  add_subdirectory(task)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::task::registration)
endif()
if(IREE_HAL_DRIVER_VULKAN)
  add_subdirectory(vulkan)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::vulkan::registration)
endif()
if(IREE_HAL_DRIVER_WEBGPU)
  add_subdirectory(webgpu)
  list(APPEND _INIT_INTERNAL_DEPS iree::hal::drivers::webgpu::registration)
endif()

iree_cc_library(
  NAME
    drivers
  HDRS
    "init.h"
  SRCS
    "init.c"
    ${_INIT_EXTERNAL_C_SRC}
  DEPS
    iree::base
    iree::hal
    ${_INIT_INTERNAL_DEPS}
    ${_INIT_EXTERNAL_DEPS}
  PUBLIC
)
