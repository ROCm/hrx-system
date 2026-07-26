# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include("${CMAKE_CURRENT_LIST_DIR}/target_map.cmake")

set(IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT "code-object")
set(IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY "device-binary")
set(IREE_AMDGPU_TARGET_EXPANSION_EXACT "exact")

function(iree_amdgpu_target_label_fragment out_var target)
  string(REPLACE "-" "_" _fragment "${target}")
  string(REPLACE "." "_" _fragment "${_fragment}")
  set(${out_var} "${_fragment}" PARENT_SCOPE)
endfunction()

function(iree_amdgpu_target_family_var out_var family)
  string(MAKE_C_IDENTIFIER "${family}" _family_identifier)
  set(${out_var} "_IREE_AMDGPU_TARGET_FAMILY_${_family_identifier}" PARENT_SCOPE)
endfunction()

function(iree_amdgpu_target_code_object out_var target)
  set(${out_var} "${_IREE_AMDGPU_TARGET_CODE_OBJECT_${target}}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_available_selectors out_var)
  set(_available_selectors
    ${_IREE_AMDGPU_EXACT_TARGETS}
    ${_IREE_AMDGPU_CODE_OBJECT_TARGETS}
    ${_IREE_AMDGPU_TARGET_FAMILIES}
  )
  list(REMOVE_DUPLICATES _available_selectors)
  list(SORT _available_selectors)
  set(${out_var} "${_available_selectors}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_unknown_selector selector)
  _iree_amdgpu_available_selectors(_available_selectors)
  string(JOIN " " _available_pretty ${_available_selectors})
  message(FATAL_ERROR
    "Unknown AMDGPU target selector '${selector}'. Available: ${_available_pretty}")
endfunction()

function(_iree_amdgpu_append_code_object_selector out_targets selector)
  set(_expanded_targets ${${out_targets}})
  if("${selector}" IN_LIST _IREE_AMDGPU_CODE_OBJECT_TARGETS)
    list(APPEND _expanded_targets "${selector}")
  elseif("${selector}" IN_LIST _IREE_AMDGPU_EXACT_TARGETS)
    iree_amdgpu_target_code_object(_code_object_target "${selector}")
    list(APPEND _expanded_targets "${_code_object_target}")
  elseif("${selector}" IN_LIST _IREE_AMDGPU_TARGET_FAMILIES)
    iree_amdgpu_target_family_var(_family_var "${selector}")
    foreach(_exact_target ${${_family_var}})
      iree_amdgpu_target_code_object(_code_object_target "${_exact_target}")
      list(APPEND _expanded_targets "${_code_object_target}")
    endforeach()
  else()
    _iree_amdgpu_unknown_selector("${selector}")
  endif()
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_append_exact_selector out_targets selector)
  set(_expanded_targets ${${out_targets}})
  if("${selector}" IN_LIST _IREE_AMDGPU_EXACT_TARGETS)
    list(APPEND _expanded_targets "${selector}")
  elseif("${selector}" IN_LIST _IREE_AMDGPU_CODE_OBJECT_TARGETS)
    foreach(_exact_target ${_IREE_AMDGPU_EXACT_TARGETS})
      iree_amdgpu_target_code_object(_code_object_target "${_exact_target}")
      if("${_code_object_target}" STREQUAL "${selector}")
        list(APPEND _expanded_targets "${_exact_target}")
      endif()
    endforeach()
  elseif("${selector}" IN_LIST _IREE_AMDGPU_TARGET_FAMILIES)
    iree_amdgpu_target_family_var(_family_var "${selector}")
    list(APPEND _expanded_targets ${${_family_var}})
  else()
    _iree_amdgpu_unknown_selector("${selector}")
  endif()
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(_iree_amdgpu_append_device_binary_selector out_targets selector)
  set(_expanded_targets ${${out_targets}})
  set(_exact_targets)
  _iree_amdgpu_append_exact_selector(_exact_targets "${selector}")
  foreach(_exact_target ${_exact_targets})
    list(APPEND _expanded_targets
      ${_IREE_AMDGPU_TARGET_DEVICE_BINARY_VARIANTS_${_exact_target}}
    )
    iree_amdgpu_target_code_object(_code_object_target "${_exact_target}")
    list(APPEND _expanded_targets "${_code_object_target}")
  endforeach()
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()

function(iree_amdgpu_expand_target_selectors out_targets mode)
  set(_expanded_targets)
  foreach(_selection ${ARGN})
    if("${mode}" STREQUAL "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}")
      _iree_amdgpu_append_code_object_selector(_expanded_targets "${_selection}")
    elseif("${mode}" STREQUAL
           "${IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY}")
      _iree_amdgpu_append_device_binary_selector(
        _expanded_targets "${_selection}")
    elseif("${mode}" STREQUAL "${IREE_AMDGPU_TARGET_EXPANSION_EXACT}")
      _iree_amdgpu_append_exact_selector(_expanded_targets "${_selection}")
    else()
      message(FATAL_ERROR
        "Unknown AMDGPU target expansion mode '${mode}'. Available: "
        "${IREE_AMDGPU_TARGET_EXPANSION_CODE_OBJECT}, "
        "${IREE_AMDGPU_TARGET_EXPANSION_DEVICE_BINARY}, "
        "${IREE_AMDGPU_TARGET_EXPANSION_EXACT}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _expanded_targets)
  set(${out_targets} "${_expanded_targets}" PARENT_SCOPE)
endfunction()
