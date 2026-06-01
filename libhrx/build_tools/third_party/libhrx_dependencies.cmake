# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

function(_libhrx_assert_fetch_allowed dep_name)
  if(HRX_HERMETIC_BUILD)
    message(FATAL_ERROR
      "${dep_name} was not found via package discovery and HRX_HERMETIC_BUILD=ON "
      "forbids FetchContent. Provide the package via CMAKE_PREFIX_PATH or an "
      "equivalent CMake package path.")
  endif()
endfunction()

function(_libhrx_configure_catch2)
  if(NOT LIBHRX_BUILD_CTS)
    return()
  endif()

  find_package(Catch2 3 CONFIG QUIET)
  if(NOT TARGET Catch2::Catch2)
    _libhrx_assert_fetch_allowed("Catch2")
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.13.0
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(catch2)
  endif()

  if(NOT TARGET Catch2::Catch2)
    message(FATAL_ERROR "Catch2 did not provide a Catch2::Catch2 target")
  endif()
endfunction()

function(libhrx_configure_dependencies)
  _libhrx_configure_catch2()
endfunction()
