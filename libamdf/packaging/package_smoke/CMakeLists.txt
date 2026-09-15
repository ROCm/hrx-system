# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...4.2)

# This standalone project consumes an installed libamdf package. It is not an
# in-tree package and must never resolve the repository's build-tree targets.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  return()
endif()

project(amdf-package-smoke LANGUAGES C)

find_package(amdf CONFIG REQUIRED)
if(NOT TARGET amdf::headers)
  message(FATAL_ERROR "The installed package does not export amdf::headers.")
endif()

add_executable(amdf_package_shared_smoke main.c)
target_link_libraries(amdf_package_shared_smoke PRIVATE amdf::amdf)

add_executable(amdf_package_static_smoke main.c)
target_link_libraries(amdf_package_static_smoke PRIVATE amdf::amdf_static)

enable_testing()
add_test(NAME amdf_package_shared_smoke COMMAND amdf_package_shared_smoke)
add_test(NAME amdf_package_static_smoke COMMAND amdf_package_static_smoke)
if(WIN32)
  set_tests_properties(amdf_package_shared_smoke PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$<TARGET_FILE_DIR:amdf::amdf>"
  )
endif()
