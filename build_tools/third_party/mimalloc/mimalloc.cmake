# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

set(_IREE_MIMALLOC_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_ALLOCATOR_MIMALLOC_DEPS iree::third_party::mimalloc)

function(iree_configure_mimalloc)
  if(TARGET iree::third_party::mimalloc)
    return()
  endif()

  if(IREE_ENABLE_ASAN OR IREE_ENABLE_MSAN OR IREE_ENABLE_TSAN OR
     IREE_ENABLE_UBSAN)
    message(FATAL_ERROR
      "IREE_ALLOCATOR_SYSTEM=mimalloc is incompatible with sanitizer builds")
  endif()

  # Allocator behavior depends on the pinned build definitions below. An
  # arbitrary installed package cannot establish the same process contract.
  iree_dependency_require_pinned_source_allowed("mimalloc")
  iree_populate_locked_fetch_content(mimalloc _mimalloc_source_dir)

  # A static core is folded into the final executable in ordinary builds. In
  # shared-library builds it would instead be copied into every shared library
  # whose public dependency closure reaches iree::base. Keep one process-wide
  # allocator implementation in a shared provider for that build shape.
  if(BUILD_SHARED_LIBS)
    add_library(iree_mimalloc SHARED
      "${_mimalloc_source_dir}/src/static.c"
    )
    target_compile_definitions(iree_mimalloc PRIVATE
      MI_DEFAULT_ARENA_EAGER_COMMIT=0
      MI_SHARED_LIB=1
      MI_SHARED_LIB_EXPORT=1
    )
  else()
    add_library(iree_mimalloc STATIC
      "${_mimalloc_source_dir}/src/static.c"
    )
    target_compile_definitions(iree_mimalloc PRIVATE
      MI_DEFAULT_ARENA_EAGER_COMMIT=0
      MI_STATIC_LIB=1
    )
  endif()
  target_include_directories(iree_mimalloc SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${_mimalloc_source_dir}/include>"
  )
  target_include_directories(iree_mimalloc SYSTEM PRIVATE
    "${_mimalloc_source_dir}/src"
  )
  set_target_properties(iree_mimalloc PROPERTIES
    C_VISIBILITY_PRESET hidden
    INTERPROCEDURAL_OPTIMIZATION OFF
    POSITION_INDEPENDENT_CODE ON
  )

  if(NOT MSVC)
    target_compile_options(iree_mimalloc PRIVATE -ftls-model=initial-exec)
  endif()

  if(WIN32)
    target_link_libraries(iree_mimalloc PUBLIC
      advapi32
      bcrypt
      psapi
      shell32
      user32
    )
  elseif(TARGET Threads::Threads)
    target_link_libraries(iree_mimalloc PUBLIC Threads::Threads)
  endif()

  iree_add_alias_library(iree::third_party::mimalloc iree_mimalloc)
  iree_install_targets(
    TARGETS
      iree_mimalloc
    COMPONENT
      IREEDevLibraries-Runtime
    EXPORT_SET
      Runtime
  )

  # Global C++ allocation is a final executable ABI concern. Keep the adapter
  # separate from the C allocator core so pure C links do not acquire a C++
  # runtime dependency and every process still contains only one mimalloc
  # implementation.
  add_library(iree_mimalloc_new_delete STATIC
    "${_IREE_MIMALLOC_CMAKE_DIR}/new_delete.cc"
  )
  target_compile_features(iree_mimalloc_new_delete PRIVATE cxx_std_17)
  target_link_libraries(iree_mimalloc_new_delete PRIVATE
    iree::third_party::mimalloc
  )
  set_target_properties(iree_mimalloc_new_delete PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
    POSITION_INDEPENDENT_CODE ON
  )
  if(MSVC)
    target_compile_options(iree_mimalloc_new_delete PRIVATE /EHsc)
  else()
    target_compile_options(iree_mimalloc_new_delete PRIVATE -fexceptions)
  endif()

  iree_add_alias_library(iree::third_party::mimalloc_new_delete
    iree_mimalloc_new_delete)
endfunction()
