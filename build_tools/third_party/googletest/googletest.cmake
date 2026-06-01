# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
include(iree_third_party_helpers)

function(iree_configure_googletest)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  find_package(GTest CONFIG QUIET)
  if(NOT TARGET GTest::gtest)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::gtest)
    iree_fetch_content_assert_allowed("googletest")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
  endif()

  if(TARGET gtest)
    # FetchContent's googletest build already defines the traditional target.
  elseif(TARGET GTest::gtest)
    iree_add_alias_interface(gtest GTest::gtest)
  else()
    message(FATAL_ERROR "googletest did not provide a gtest target")
  endif()

  if(TARGET gtest_main)
    # Provided by FetchContent.
  elseif(TARGET GTest::gtest_main)
    iree_add_alias_interface(gtest_main GTest::gtest_main)
  endif()

  if(TARGET gmock)
    # Provided by FetchContent.
  elseif(TARGET GTest::gmock)
    iree_add_alias_interface(gmock GTest::gmock)
  else()
    # Some packaged GTest installs omit gmock. IREE's test helpers link gmock
    # unconditionally, so provide a compatibility target that forwards to gtest.
    iree_add_alias_interface(gmock gtest)
  endif()

  if(TARGET gmock_main)
    # Provided by FetchContent.
  elseif(TARGET GTest::gmock_main)
    iree_add_alias_interface(gmock_main GTest::gmock_main)
  endif()
endfunction()
