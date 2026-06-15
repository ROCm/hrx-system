# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom AMDGPU target selector helpers.
#
# This file is the Loom-specific facade over the shared AMDGPU selector map.
# Keep target facts in target_config.cmake and build_tools/amdgpu; keep Loom
# target policy here.

include("${CMAKE_CURRENT_LIST_DIR}/../../../build_tools/amdgpu/selectors.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/target_config.cmake")

function(_loom_amdgpu_unknown_selector selector)
  string(JOIN " " _source_selectors ${_LOOM_AMDGPU_TARGET_SOURCE_SELECTORS})
  string(JOIN " " _supported_processors ${_LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS})
  string(JOIN " " _supported_code_objects ${_LOOM_AMDGPU_SUPPORTED_CODE_OBJECT_PROCESSORS})
  message(FATAL_ERROR
    "Unknown or unsupported Loom AMDGPU target selector '${selector}'. "
    "Source selectors: ${_source_selectors}. "
    "Supported exact processors: ${_supported_processors}. "
    "Supported code-object processors: ${_supported_code_objects}.")
endfunction()

function(_loom_amdgpu_append_explicit_exact_selector out_targets selector)
  iree_amdgpu_expand_target_selectors(
    _expanded_exact_targets
    "${IREE_AMDGPU_TARGET_EXPANSION_EXACT}"
    "${selector}"
  )
  foreach(_exact_target ${_expanded_exact_targets})
    if(NOT "${_exact_target}" IN_LIST _LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS)
      _loom_amdgpu_unknown_selector("${selector}")
    endif()
  endforeach()
  set(_targets ${${out_targets}})
  list(APPEND _targets ${_expanded_exact_targets})
  set(${out_targets} "${_targets}" PARENT_SCOPE)
endfunction()

function(_loom_amdgpu_append_supported_exact_selector out_targets selector)
  iree_amdgpu_expand_target_selectors(
    _expanded_exact_targets
    "${IREE_AMDGPU_TARGET_EXPANSION_EXACT}"
    "${selector}"
  )
  set(_targets ${${out_targets}})
  foreach(_exact_target ${_expanded_exact_targets})
    if("${_exact_target}" IN_LIST _LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS)
      list(APPEND _targets "${_exact_target}")
    endif()
  endforeach()
  set(${out_targets} "${_targets}" PARENT_SCOPE)
endfunction()

function(_loom_amdgpu_expand_iree_hal_target_selectors out_targets)
  set(_expanded_targets)
  if(NOT DEFINED IREE_HAL_AMDGPU_TARGETS)
    message(FATAL_ERROR
      "LOOM_TARGET_AMDGPU_TARGETS contains iree_hal but "
      "IREE_HAL_AMDGPU_TARGETS is not defined.")
  endif()
  foreach(_iree_hal_selector ${IREE_HAL_AMDGPU_TARGETS})
    _loom_amdgpu_append_supported_exact_selector(
      _expanded_targets
      "${_iree_hal_selector}"
    )
  endforeach()
  list(REMOVE_DUPLICATES _expanded_targets)
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(loom_amdgpu_expand_target_selectors out_targets)
  set(_expanded_targets)
  foreach(_selector ${ARGN})
    if("${_selector}" STREQUAL "${LOOM_AMDGPU_TARGET_SOURCE_LOOM_DEFAULTS}")
      list(APPEND _expanded_targets ${_LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS})
    elseif("${_selector}" STREQUAL "${LOOM_AMDGPU_TARGET_SOURCE_IREE_HAL}")
      _loom_amdgpu_expand_iree_hal_target_selectors(
        _iree_hal_exact_targets
      )
      list(APPEND _expanded_targets ${_iree_hal_exact_targets})
    else()
      _loom_amdgpu_append_explicit_exact_selector(
        _expanded_targets
        "${_selector}"
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _expanded_targets)
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(_loom_amdgpu_exact_target_selected out_var targets_var processor)
  set(_selected OFF)
  if("${processor}" IN_LIST ${targets_var})
    set(_selected ON)
  endif()
  set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()

function(_loom_amdgpu_descriptor_set_selected out_var targets_var capability)
  set(_selected OFF)
  foreach(_processor ${_LOOM_AMDGPU_DESCRIPTOR_SET_EXACT_PROCESSORS_${capability}})
    if("${_processor}" IN_LIST ${targets_var})
      set(_selected ON)
    endif()
  endforeach()
  set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()

function(loom_amdgpu_configure_target_selectors)
  if(NOT DEFINED LOOM_TARGET_AMDGPU_TARGETS)
    set(LOOM_TARGET_AMDGPU_TARGETS ${_LOOM_AMDGPU_DEFAULT_TARGET_SELECTORS})
  endif()

  loom_amdgpu_expand_target_selectors(
    _loom_amdgpu_exact_targets
    ${LOOM_TARGET_AMDGPU_TARGETS}
  )

  if(DEFINED IREE_HAL_AMDGPU_TARGETS)
    _loom_amdgpu_expand_iree_hal_target_selectors(
      _loom_amdgpu_iree_hal_exact_targets
    )
  else()
    set(_loom_amdgpu_iree_hal_exact_targets)
  endif()

  foreach(_processor ${_LOOM_AMDGPU_SUPPORTED_EXACT_PROCESSORS})
    string(TOUPPER "${_processor}" _processor_upper)
    _loom_amdgpu_exact_target_selected(
      _selected
      _loom_amdgpu_exact_targets
      "${_processor}"
    )
    set("LOOM_TARGET_AMDGPU_PROCESSOR_${_processor_upper}"
        "${_selected}" PARENT_SCOPE)
    _loom_amdgpu_exact_target_selected(
      _iree_hal_selected
      _loom_amdgpu_iree_hal_exact_targets
      "${_processor}"
    )
    set("LOOM_TARGET_AMDGPU_IREE_HAL_PROCESSOR_${_processor_upper}"
        "${_iree_hal_selected}" PARENT_SCOPE)
  endforeach()

  foreach(_capability ${_LOOM_AMDGPU_DESCRIPTOR_SET_CAPABILITIES})
    string(TOUPPER "${_capability}" _capability_upper)
    _loom_amdgpu_descriptor_set_selected(
      _selected
      _loom_amdgpu_exact_targets
      "${_capability}"
    )
    set("LOOM_TARGET_AMDGPU_${_capability_upper}" "${_selected}" PARENT_SCOPE)
    _loom_amdgpu_descriptor_set_selected(
      _iree_hal_selected
      _loom_amdgpu_iree_hal_exact_targets
      "${_capability}"
    )
    set("LOOM_TARGET_AMDGPU_IREE_HAL_${_capability_upper}"
        "${_iree_hal_selected}" PARENT_SCOPE)
  endforeach()
endfunction()
