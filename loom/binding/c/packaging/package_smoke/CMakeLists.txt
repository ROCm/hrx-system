# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# This is an installed-package consumer smoke test, not part of the in-tree
# Loom build. Keep it standalone so it resolves loomc through find_package()
# against the install prefix instead of accidentally linking build-tree targets.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  return()
endif()

project(loomc-package-smoke LANGUAGES C CXX)

if(NOT TARGET loomc::loomc)
  find_package(loomc CONFIG REQUIRED)
endif()

add_executable(loomc_package_smoke main.c)
target_link_libraries(loomc_package_smoke PRIVATE loomc::loomc)
