# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_cxx)
  if(TARGET iree::third_party::cxx_parser)
    return()
  endif()
  # The importer uses the pinned semantic API and its include ownership/path
  # contracts.
  # An arbitrary installed parser package cannot establish that contract.
  iree_dependency_require_pinned_source_allowed("cxx")
  iree_populate_locked_fetch_content(cxx _cxx_source_dir)
  if(LOOM_CXX_HOST_CODEGEN)
    _iree_configure_cxx_project("${_cxx_source_dir}")
    return()
  endif()
  # The upstream root also configures drivers and unrelated SDK downloads.
  # Build only the independently consumable semantic parser component.
  file(GLOB _cxx_sources CONFIGURE_DEPENDS
    "${_cxx_source_dir}/src/parser/cxx/*.cc")
  add_library(iree_cxx_parser STATIC ${_cxx_sources})
  target_include_directories(iree_cxx_parser SYSTEM PUBLIC
    "${_cxx_source_dir}/src/parser")
  target_compile_features(iree_cxx_parser PRIVATE cxx_std_23)
  target_compile_definitions(iree_cxx_parser PUBLIC CXX_VERSION="1.2.0")
  set_target_properties(iree_cxx_parser PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # The vendored parser owns its implementation warning policy.
  if(MSVC)
    target_compile_options(iree_cxx_parser PRIVATE /EHsc /GR /W0)
  else()
    target_compile_options(iree_cxx_parser PRIVATE -fexceptions -frtti -w)
  endif()
  iree_add_alias_interface(iree::third_party::cxx_parser iree_cxx_parser)
endfunction()

# Single-source host code generation (loom-cxx) needs cxx's MLIR/LLVM back end,
# which only cxx's own CMake project builds. Configure that project for its
# parser, code generator and MLIR libraries, and give the importer cxx's parser
# library as well, so one binary links exactly one copy of the parser.
function(_iree_configure_cxx_project source_dir)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Loom C++ host code generation: cxx with MLIR ${MLIR_DIR}")
  # cxx requires MLIR_VERSION >= 21. Some MLIR packages (conda-forge's mlir)
  # leave PACKAGE_VERSION empty in MLIRConfigVersion.cmake, so MLIR_VERSION is
  # empty although the matching LLVM reports its version. For this configure
  # run, redirect cxx's find_package(MLIR) to the same package with LLVM's
  # version; the redirect directory is cleared at the start of every run.
  set(_mlir_dir "")
  if(NOT MLIR_VERSION AND LLVM_PACKAGE_VERSION)
    set(_mlir_dir "${MLIR_DIR}")
    message(STATUS "MLIR package reports no version; using LLVM's "
                   "${LLVM_PACKAGE_VERSION}")
    file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/mlir-config.cmake"
         "include(\"${_mlir_dir}/MLIRConfig.cmake\")\n")
    file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/mlir-config-version.cmake"
         "set(PACKAGE_VERSION \"${LLVM_PACKAGE_VERSION}\")\n"
         "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n")
  endif()
  set(CXX_ENABLE_MLIR ON)
  set(CXX_BUILD_TESTS OFF)
  set(CXX_INSTALL_TOOLS OFF)
  set(CXX_INSTALL_LSP OFF)
  # The importer and driver run on the process allocator; cxx's mimalloc
  # override is for its own compiler executable.
  set(CXX_ENABLE_MIMALLOC OFF)
  # SYSTEM: cxx's headers are third-party includes for HRX's warning policy.
  # EXCLUDE_FROM_ALL: only libraries that HRX targets link are built.
  add_subdirectory("${source_dir}" "${CMAKE_BINARY_DIR}/third_party/cxx"
                   SYSTEM EXCLUDE_FROM_ALL)
  if(_mlir_dir)
    # cxx's find_package(MLIR) recorded the redirect directory; keep the
    # user's package for the next configure run.
    set(MLIR_DIR "${_mlir_dir}" CACHE PATH
        "The directory containing a CMake configuration file for MLIR." FORCE)
  endif()
  foreach(_target IN ITEMS cxx-parser cxx-codegen cxx-mlir)
    if(NOT TARGET ${_target})
      message(FATAL_ERROR "cxx did not define ${_target}")
    endif()
    set_target_properties(${_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
  iree_add_alias_interface(iree::third_party::cxx_parser cxx-parser)
  iree_add_alias_interface(iree::third_party::cxx_codegen cxx-mlir)
endfunction()
