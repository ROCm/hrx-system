# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(_iree_add_spirv_tool_alias alias_name target_name)
  if(TARGET ${alias_name})
    return()
  endif()
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "SPIRV-Tools did not provide ${target_name}")
  endif()
  add_executable(${alias_name} ALIAS ${target_name})
  if(COMMAND iree_record_target_alias)
    iree_record_target_alias(${alias_name} ${target_name})
  endif()
endfunction()

function(_iree_add_imported_spirv_tool alias_name target_name executable_path)
  if(TARGET ${alias_name})
    return()
  endif()
  add_executable(${target_name} IMPORTED GLOBAL)
  set_target_properties(${target_name} PROPERTIES
    IMPORTED_LOCATION "${executable_path}")
  add_executable(${alias_name} ALIAS ${target_name})
  if(COMMAND iree_record_target_alias)
    iree_record_target_alias(${alias_name} ${target_name})
  endif()
endfunction()

function(iree_configure_spirv_tools)
  if(TARGET iree::third_party::spirv_as AND
      TARGET iree::third_party::spirv_dis AND
      TARGET iree::third_party::spirv_val)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_program(IREE_SPIRV_AS_EXECUTABLE NAMES spirv-as)
    find_program(IREE_SPIRV_DIS_EXECUTABLE NAMES spirv-dis)
    find_program(IREE_SPIRV_VAL_EXECUTABLE NAMES spirv-val)
    if(IREE_SPIRV_AS_EXECUTABLE AND IREE_SPIRV_DIS_EXECUTABLE AND
        IREE_SPIRV_VAL_EXECUTABLE)
      _iree_add_imported_spirv_tool(
        iree::third_party::spirv_as
        iree_spirv_as
        "${IREE_SPIRV_AS_EXECUTABLE}")
      _iree_add_imported_spirv_tool(
        iree::third_party::spirv_dis
        iree_spirv_dis
        "${IREE_SPIRV_DIS_EXECUTABLE}")
      _iree_add_imported_spirv_tool(
        iree::third_party::spirv_val
        iree_spirv_val
        "${IREE_SPIRV_VAL_EXECUTABLE}")
    endif()
  endif()

  if(NOT TARGET iree::third_party::spirv_as OR
      NOT TARGET iree::third_party::spirv_dis OR
      NOT TARGET iree::third_party::spirv_val)
    iree_dependency_require_pinned_source_allowed("SPIRV-Tools")
    iree_populate_locked_fetch_content(
      spirv_headers _iree_spirv_headers_source_dir)
    set(SPIRV-Headers_SOURCE_DIR "${_iree_spirv_headers_source_dir}"
      CACHE PATH "Pinned SPIRV-Headers source directory." FORCE)

    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    iree_declare_locked_fetch_content(spirv_tools)
    FetchContent_MakeAvailable(spirv_tools)

    _iree_add_spirv_tool_alias(
      iree::third_party::spirv_as spirv-as)
    _iree_add_spirv_tool_alias(
      iree::third_party::spirv_dis spirv-dis)
    _iree_add_spirv_tool_alias(
      iree::third_party::spirv_val spirv-val)
  endif()
endfunction()
