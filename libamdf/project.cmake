# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED AMDF_BUILD_DEFAULT)
  set(AMDF_BUILD_DEFAULT OFF)
endif()
option(AMDF_BUILD
  "Build the portable AMD GPU and XDNA native device library."
  ${AMDF_BUILD_DEFAULT})

option(AMDF_FAMILY_RDNA "Admit RDNA implementation packages." ON)
option(AMDF_FAMILY_CDNA "Admit CDNA implementation packages." ON)
option(AMDF_FAMILY_XDNA "Admit XDNA implementation packages." ON)
option(AMDF_MEDIA "Admit the optional native media extension." OFF)

set(AMDF_VERSION "0.1.0")

# The installed shared library and its public table negotiate the same ABI.
# Reading the public declaration also makes header changes reconfigure CMake.
set(_AMDF_BASE_HEADER "${CMAKE_CURRENT_LIST_DIR}/include/amdf/base.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_AMDF_BASE_HEADER}")
file(STRINGS "${_AMDF_BASE_HEADER}" _AMDF_ABI_DECLARATION
  REGEX "^#define AMDF_ABI_VERSION_LATEST AMDF_ABI_VERSION_[0-9]+$")
if(NOT _AMDF_ABI_DECLARATION MATCHES "AMDF_ABI_VERSION_([0-9]+)$")
  message(FATAL_ERROR "Cannot determine libamdf ABI from its public header.")
endif()
set(AMDF_ABI_VERSION "${CMAKE_MATCH_1}")
unset(_AMDF_ABI_DECLARATION)
unset(_AMDF_BASE_HEADER)
