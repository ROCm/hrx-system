# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom target-profile and product-format descriptor helpers.

function(_loom_declare_descriptor_target OUTPUT_TARGET NAME KIND)
  if(NOT NAME)
    message(FATAL_ERROR "Loom descriptor target requires NAME")
  endif()

  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_TARGET "${_PACKAGE_NAME}_${NAME}")
  set(_ALIAS "${_PACKAGE_NS}::${NAME}")
  if(TARGET "${_TARGET}" OR TARGET "${_ALIAS}")
    message(FATAL_ERROR "Loom descriptor target already exists: ${_ALIAS}")
  endif()

  add_library("${_TARGET}" INTERFACE)
  iree_add_alias_library("${_ALIAS}" "${_TARGET}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_DESCRIPTOR_KIND "${KIND}")
  set(${OUTPUT_TARGET} "${_TARGET}" PARENT_SCOPE)
endfunction()

function(_loom_validate_product_extension CONTEXT EXTENSION)
  if(NOT EXTENSION MATCHES "^\\.[^/\\\\]+$")
    message(FATAL_ERROR
      "${CONTEXT} extension must begin with '.' and contain no path: "
      "${EXTENSION}"
    )
  endif()
endfunction()

function(_loom_resolve_descriptor_target OUTPUT_TARGET INPUT_TARGET KIND CONTEXT)
  if(NOT INPUT_TARGET)
    message(FATAL_ERROR "${CONTEXT} requires a descriptor target")
  endif()
  iree_package_target_name(_TARGET "${INPUT_TARGET}")
  if(NOT TARGET "${_TARGET}")
    message(FATAL_ERROR "${CONTEXT} references missing target ${INPUT_TARGET}")
  endif()
  get_target_property(_KIND "${_TARGET}" LOOM_DESCRIPTOR_KIND)
  if(NOT _KIND STREQUAL "${KIND}")
    message(FATAL_ERROR
      "${CONTEXT} target ${INPUT_TARGET} is not a Loom ${KIND} descriptor"
    )
  endif()
  set(${OUTPUT_TARGET} "${_TARGET}" PARENT_SCOPE)
endfunction()

function(loom_file_product_format)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;PRODUCT;FORMAT;OUTPUT_EXTENSION"
    ""
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_file_product_format ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_PRODUCT)
    message(FATAL_ERROR "loom_file_product_format ${_RULE_NAME} requires PRODUCT")
  endif()
  if(NOT _RULE_FORMAT)
    message(FATAL_ERROR "loom_file_product_format ${_RULE_NAME} requires FORMAT")
  endif()
  _loom_validate_product_extension(
    "loom_file_product_format ${_RULE_NAME} output"
    "${_RULE_OUTPUT_EXTENSION}"
  )

  _loom_declare_descriptor_target(_TARGET "${_RULE_NAME}" "product_format")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_PRODUCT "${_RULE_PRODUCT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_NAME "${_RULE_FORMAT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_OUTPUT_KIND "file")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_OUTPUT_EXTENSION "${_RULE_OUTPUT_EXTENSION}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_ARTIFACT_DIRECTORY_EXTENSION "")
endfunction()

function(loom_artifact_set_product_format)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;PRODUCT;FORMAT;MANIFEST_EXTENSION;ARTIFACT_DIRECTORY_EXTENSION"
    ""
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_artifact_set_product_format ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_PRODUCT)
    message(FATAL_ERROR
      "loom_artifact_set_product_format ${_RULE_NAME} requires PRODUCT")
  endif()
  if(NOT _RULE_FORMAT)
    message(FATAL_ERROR
      "loom_artifact_set_product_format ${_RULE_NAME} requires FORMAT")
  endif()
  _loom_validate_product_extension(
    "loom_artifact_set_product_format ${_RULE_NAME} manifest"
    "${_RULE_MANIFEST_EXTENSION}"
  )
  _loom_validate_product_extension(
    "loom_artifact_set_product_format ${_RULE_NAME} artifact directory"
    "${_RULE_ARTIFACT_DIRECTORY_EXTENSION}"
  )

  _loom_declare_descriptor_target(_TARGET "${_RULE_NAME}" "product_format")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_PRODUCT "${_RULE_PRODUCT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_NAME "${_RULE_FORMAT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_OUTPUT_KIND "artifact_set")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_OUTPUT_EXTENSION "${_RULE_MANIFEST_EXTENSION}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_PRODUCT_FORMAT_ARTIFACT_DIRECTORY_EXTENSION
    "${_RULE_ARTIFACT_DIRECTORY_EXTENSION}")
endfunction()

function(loom_target_profile)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;FAMILY;SELECTOR"
    "CANONICAL_FORMATS;FORMATS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_target_profile ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_FAMILY OR _RULE_FAMILY MATCHES ":")
    message(FATAL_ERROR
      "loom_target_profile ${_RULE_NAME} requires a nonempty FAMILY without ':'"
    )
  endif()
  if(NOT _RULE_SELECTOR)
    message(FATAL_ERROR "loom_target_profile ${_RULE_NAME} requires SELECTOR")
  endif()

  set(_CANONICAL_FORMATS)
  set(_CANONICAL_PRODUCTS)
  foreach(_FORMAT IN LISTS _RULE_CANONICAL_FORMATS)
    _loom_resolve_descriptor_target(
      _FORMAT_TARGET "${_FORMAT}" "product_format"
      "loom_target_profile ${_RULE_NAME} canonical format"
    )
    get_target_property(
      _PRODUCT "${_FORMAT_TARGET}" LOOM_PRODUCT_FORMAT_PRODUCT)
    if(_PRODUCT IN_LIST _CANONICAL_PRODUCTS)
      message(FATAL_ERROR
        "loom_target_profile ${_RULE_NAME} declares multiple canonical "
        "formats for product ${_PRODUCT}"
      )
    endif()
    list(APPEND _CANONICAL_PRODUCTS "${_PRODUCT}")
    list(APPEND _CANONICAL_FORMATS "${_FORMAT_TARGET}")
  endforeach()

  set(_FORMATS)
  set(_FORMAT_IDENTITIES)
  foreach(_FORMAT IN LISTS _RULE_CANONICAL_FORMATS _RULE_FORMATS)
    _loom_resolve_descriptor_target(
      _FORMAT_TARGET "${_FORMAT}" "product_format"
      "loom_target_profile ${_RULE_NAME} format"
    )
    get_target_property(
      _PRODUCT "${_FORMAT_TARGET}" LOOM_PRODUCT_FORMAT_PRODUCT)
    get_target_property(
      _FORMAT_NAME "${_FORMAT_TARGET}" LOOM_PRODUCT_FORMAT_NAME)
    set(_FORMAT_IDENTITY "${_PRODUCT}|${_FORMAT_NAME}")
    if(_FORMAT_IDENTITY IN_LIST _FORMAT_IDENTITIES)
      message(FATAL_ERROR
        "loom_target_profile ${_RULE_NAME} declares product format "
        "${_PRODUCT}/${_FORMAT_NAME} more than once"
      )
    endif()
    list(APPEND _FORMAT_IDENTITIES "${_FORMAT_IDENTITY}")
    list(APPEND _FORMATS "${_FORMAT_TARGET}")
  endforeach()

  _loom_declare_descriptor_target(_TARGET "${_RULE_NAME}" "target_profile")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_TARGET_PROFILE_FAMILY "${_RULE_FAMILY}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_TARGET_PROFILE_SELECTOR "${_RULE_SELECTOR}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_TARGET_PROFILE_CANONICAL_FORMATS "${_CANONICAL_FORMATS}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_TARGET_PROFILE_FORMATS "${_FORMATS}")
endfunction()

function(_loom_target_profile_identity OUTPUT_IDENTITY TARGET)
  _loom_resolve_descriptor_target(
    _TARGET "${TARGET}" "target_profile" "Loom binary target profile"
  )
  get_target_property(_FAMILY "${_TARGET}" LOOM_TARGET_PROFILE_FAMILY)
  get_target_property(_SELECTOR "${_TARGET}" LOOM_TARGET_PROFILE_SELECTOR)
  set(${OUTPUT_IDENTITY} "${_FAMILY}:${_SELECTOR}" PARENT_SCOPE)
endfunction()

function(_loom_resolve_target_product_format OUTPUT_FORMAT)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;PRODUCT;OUTPUT_KIND;FORMAT"
    ""
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Loom product format resolution has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_PRODUCT OR NOT _RULE_OUTPUT_KIND)
    message(FATAL_ERROR
      "Loom product format resolution requires PRODUCT and OUTPUT_KIND")
  endif()
  _loom_resolve_descriptor_target(
    _TARGET "${_RULE_TARGET}" "target_profile" "Loom product format resolution"
  )
  get_target_property(
    _SUPPORTED_FORMATS "${_TARGET}" LOOM_TARGET_PROFILE_FORMATS)

  if(_RULE_FORMAT)
    _loom_resolve_descriptor_target(
      _SELECTED_FORMAT "${_RULE_FORMAT}" "product_format"
      "Loom product format resolution"
    )
  else()
    get_target_property(
      _CANONICAL_FORMATS "${_TARGET}" LOOM_TARGET_PROFILE_CANONICAL_FORMATS)
    unset(_SELECTED_FORMAT)
    foreach(_FORMAT_TARGET IN LISTS _CANONICAL_FORMATS)
      get_target_property(
        _PRODUCT "${_FORMAT_TARGET}" LOOM_PRODUCT_FORMAT_PRODUCT)
      if(_PRODUCT STREQUAL _RULE_PRODUCT)
        if(_SELECTED_FORMAT)
          message(FATAL_ERROR
            "Target profile ${_RULE_TARGET} has multiple canonical formats "
            "for product ${_RULE_PRODUCT}"
          )
        endif()
        set(_SELECTED_FORMAT "${_FORMAT_TARGET}")
      endif()
    endforeach()
    if(NOT _SELECTED_FORMAT)
      message(FATAL_ERROR
        "Target profile ${_RULE_TARGET} has no canonical format for product "
        "${_RULE_PRODUCT}"
      )
    endif()
  endif()

  if(NOT _SELECTED_FORMAT IN_LIST _SUPPORTED_FORMATS)
    message(FATAL_ERROR
      "Target profile ${_RULE_TARGET} does not support product format "
      "${_RULE_FORMAT}"
    )
  endif()
  get_target_property(
    _PRODUCT "${_SELECTED_FORMAT}" LOOM_PRODUCT_FORMAT_PRODUCT)
  get_target_property(
    _OUTPUT_KIND "${_SELECTED_FORMAT}" LOOM_PRODUCT_FORMAT_OUTPUT_KIND)
  if(NOT _PRODUCT STREQUAL _RULE_PRODUCT)
    message(FATAL_ERROR
      "Product format ${_SELECTED_FORMAT} describes product ${_PRODUCT}; "
      "expected ${_RULE_PRODUCT}"
    )
  endif()
  if(NOT _OUTPUT_KIND STREQUAL _RULE_OUTPUT_KIND)
    message(FATAL_ERROR
      "Product format ${_SELECTED_FORMAT} has output kind ${_OUTPUT_KIND}; "
      "expected ${_RULE_OUTPUT_KIND}"
    )
  endif()
  set(${OUTPUT_FORMAT} "${_SELECTED_FORMAT}" PARENT_SCOPE)
endfunction()
