# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(pyre-package-smoke LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(pyre CONFIG REQUIRED)

add_executable(pyre_package_smoke main.c)
target_link_libraries(pyre_package_smoke PRIVATE pyre::pyre)

add_executable(pyre_package_smoke_cxx main.cc)
target_link_libraries(pyre_package_smoke_cxx PRIVATE pyre::pyre)
