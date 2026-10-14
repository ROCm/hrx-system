# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Pinned source consumers enter here only after iree_configure_zlib() creates
# the canonical target. This prevents nested projects from replacing the
# locked dependency with an ambient system installation.
if(NOT TARGET ZLIB::ZLIB)
  set(ZLIB_FOUND FALSE)
  return()
endif()

get_target_property(_zlib_target_include_directories ZLIB::ZLIB
  INTERFACE_INCLUDE_DIRECTORIES)
set(ZLIB_INCLUDE_DIRS "")
foreach(_include_directory IN LISTS _zlib_target_include_directories)
  if(_include_directory MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
    list(APPEND ZLIB_INCLUDE_DIRS "${CMAKE_MATCH_1}")
  elseif(NOT _include_directory MATCHES "^\\$<")
    list(APPEND ZLIB_INCLUDE_DIRS "${_include_directory}")
  endif()
endforeach()
list(GET ZLIB_INCLUDE_DIRS 0 ZLIB_INCLUDE_DIR)
set(ZLIB_LIBRARIES ZLIB::ZLIB)
set(ZLIB_VERSION_STRING "${IREE_DEP_ZLIB_VERSION}")
set(ZLIB_FOUND TRUE)
