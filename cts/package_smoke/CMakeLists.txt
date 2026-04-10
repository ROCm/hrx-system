# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(hrx-package-smoke LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(hrx CONFIG REQUIRED)

add_executable(hrx_package_smoke main.c)
target_link_libraries(hrx_package_smoke PRIVATE hrx::hrx)

add_executable(hrx_package_smoke_cxx main.cc)
target_link_libraries(hrx_package_smoke_cxx PRIVATE hrx::hrx)
