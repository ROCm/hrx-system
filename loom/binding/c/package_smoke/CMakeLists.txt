# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(loomc-package-smoke LANGUAGES C CXX)

if(NOT TARGET loomc::loomc)
  find_package(loomc CONFIG REQUIRED)
endif()

add_executable(loomc_package_smoke main.c)
target_link_libraries(loomc_package_smoke PRIVATE loomc::loomc)
