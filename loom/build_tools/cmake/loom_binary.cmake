# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Closed Loom deployment product helpers.

function(_loom_declare_binary_linked_module)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;COMPONENT;OWNER_TARGET;TARGET;OUTPUT;OUT_DEPENDENCY_TARGETS;OUT_DEPENDENCY_REPORT;OUT_LIBRARY_DEPENDENCY_TARGETS"
    "SRCS;DEPS;ROOTS;CONFIGS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Loom binary ${_RULE_NAME} has unknown link arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_NAME OR NOT _RULE_COMPONENT OR NOT _RULE_OWNER_TARGET OR
     NOT _RULE_TARGET OR NOT _RULE_OUTPUT)
    message(FATAL_ERROR
      "Loom binary link requires NAME, COMPONENT, OWNER_TARGET, TARGET, and "
      "OUTPUT")
  endif()
  if(NOT _RULE_SRCS AND NOT _RULE_DEPS)
    message(FATAL_ERROR "Loom binary ${_RULE_NAME} requires SRCS or DEPS")
  endif()

  _loom_library_dependency_modules(
    _DIRECT_MODULES
    _TRANSITIVE_MODULES
    _DEPENDENCY_TARGETS
    "${_RULE_OWNER_TARGET}"
    ${_RULE_DEPS}
  )
  set(_LIBRARY_DEPENDENCY_TARGETS ${_DEPENDENCY_TARGETS})
  set(_DEPENDENCY_REPORT)
  if(_RULE_SRCS)
    set(_SOURCE_MODULE
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.sources.loombc")
    set(_DEPENDENCY_REPORT
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.sources.dependencies.json")
    _loom_declare_relocatable_module(
      NAME "${_RULE_COMPONENT}"
      CONSUMER_TARGET "${_RULE_OWNER_TARGET}"
      OUTPUT "${_SOURCE_MODULE}"
      DEPENDENCY_REPORT "${_DEPENDENCY_REPORT}"
      OUT_DEPENDENCY_TARGETS _SOURCE_DEPENDENCY_TARGETS
      OUT_LIBRARY_DEPENDENCY_TARGETS _UNUSED_LIBRARY_DEPENDENCY_TARGETS
      SRCS ${_RULE_SRCS}
      DEPS ${_RULE_DEPS}
    )
    list(APPEND _DIRECT_MODULES "${_SOURCE_MODULE}")
    list(APPEND _DEPENDENCY_TARGETS ${_SOURCE_DEPENDENCY_TARGETS})
  endif()

  _loom_target_profile_identity(_TARGET_IDENTITY "${_RULE_TARGET}")
  set(_ARGS
    "--mode=link"
    "--strip-check"
    "--require-resolved-config"
    "--to=bc"
    "--output=${_RULE_OUTPUT}"
    "--target=${_TARGET_IDENTITY}"
  )
  if(_RULE_ROOTS)
    foreach(_MODULE IN LISTS _DIRECT_MODULES)
      list(APPEND _ARGS "--library=${_MODULE}")
    endforeach()
    foreach(_ROOT IN LISTS _RULE_ROOTS)
      list(APPEND _ARGS "--root=${_ROOT}")
    endforeach()
  else()
    foreach(_MODULE IN LISTS _DIRECT_MODULES)
      list(APPEND _ARGS "--root-library=${_MODULE}")
    endforeach()
  endif()
  _loom_transitive_library_argument(
    _TRANSITIVE_ARGUMENT "${_TRANSITIVE_MODULES}"
  )
  if(_TRANSITIVE_ARGUMENT)
    list(APPEND _ARGS "${_TRANSITIVE_ARGUMENT}")
  endif()
  set(_CONFIGS ${_RULE_CONFIGS})
  list(SORT _CONFIGS)
  foreach(_CONFIG IN LISTS _CONFIGS)
    list(APPEND _ARGS "--config=${_CONFIG}")
  endforeach()

  add_custom_command(
    OUTPUT
      "${_RULE_OUTPUT}"
    COMMAND
      "$<TARGET_FILE:loom::tools::loom-link>" ${_ARGS}
    DEPENDS
      loom::tools::loom-link
      ${_SOURCE_MODULE}
      ${_DIRECT_MODULES}
      ${_TRANSITIVE_MODULES}
    COMMENT
      "Linking Loom binary ${_RULE_NAME}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  set_source_files_properties("${_RULE_OUTPUT}" PROPERTIES GENERATED TRUE)
  list(REMOVE_DUPLICATES _DEPENDENCY_TARGETS)
  set(${_RULE_OUT_DEPENDENCY_TARGETS}
    "${_DEPENDENCY_TARGETS}" PARENT_SCOPE)
  set(${_RULE_OUT_DEPENDENCY_REPORT}
    "${_DEPENDENCY_REPORT}" PARENT_SCOPE)
  set(${_RULE_OUT_LIBRARY_DEPENDENCY_TARGETS}
    "${_LIBRARY_DEPENDENCY_TARGETS}" PARENT_SCOPE)
endfunction()

function(_loom_declare_kernel_product)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET;FORMAT;LINKED_MODULE;OUTPUT_STEM;OUT_ARTIFACT;OUT_REPORT"
    ""
    ${ARGN}
  )
  get_target_property(
    _FORMAT_NAME "${_RULE_FORMAT}" LOOM_PRODUCT_FORMAT_NAME)
  get_target_property(
    _OUTPUT_EXTENSION "${_RULE_FORMAT}" LOOM_PRODUCT_FORMAT_OUTPUT_EXTENSION)
  _loom_target_profile_identity(_TARGET_IDENTITY "${_RULE_TARGET}")
  set(_ARTIFACT "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_OUTPUT_STEM}${_OUTPUT_EXTENSION}")
  set(_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_OUTPUT_STEM}.compile.json")
  add_custom_command(
    OUTPUT
      "${_ARTIFACT}"
      "${_REPORT}"
    COMMAND
      "$<TARGET_FILE:loom::tools::loom-compile>"
      "${_RULE_LINKED_MODULE}"
      "--product=kernel"
      "--format=${_FORMAT_NAME}"
      "--target=${_TARGET_IDENTITY}"
      "--output=${_ARTIFACT}"
      "--compile-report=details"
      "--compile-report-output=${_REPORT}"
    DEPENDS
      loom::tools::loom-compile
      "${_RULE_LINKED_MODULE}"
    COMMENT
      "Compiling Loom kernel product ${_RULE_NAME}"
    VERBATIM
  )
  set_source_files_properties(
    "${_ARTIFACT}"
    "${_REPORT}"
    PROPERTIES GENERATED TRUE
  )
  set(${_RULE_OUT_ARTIFACT} "${_ARTIFACT}" PARENT_SCOPE)
  set(${_RULE_OUT_REPORT} "${_REPORT}" PARENT_SCOPE)
endfunction()

function(_loom_declare_command_product)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;FORMAT;LINKED_MODULE;OUT_MANIFEST;OUT_ARTIFACTS;OUT_REPORT"
    ""
    ${ARGN}
  )
  get_target_property(
    _FORMAT_NAME "${_RULE_FORMAT}" LOOM_PRODUCT_FORMAT_NAME)
  get_target_property(
    _OUTPUT_EXTENSION "${_RULE_FORMAT}" LOOM_PRODUCT_FORMAT_OUTPUT_EXTENSION)
  get_target_property(
    _ARTIFACT_DIRECTORY_EXTENSION "${_RULE_FORMAT}"
    LOOM_PRODUCT_FORMAT_ARTIFACT_DIRECTORY_EXTENSION)
  set(_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}${_OUTPUT_EXTENSION}")
  set(_ARTIFACTS
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}${_ARTIFACT_DIRECTORY_EXTENSION}")
  set(_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.commands.compile.json")
  add_custom_command(
    OUTPUT
      "${_MANIFEST}"
      "${_REPORT}"
    BYPRODUCTS
      "${_ARTIFACTS}"
    COMMAND
      "${CMAKE_COMMAND}" -E rm -rf "${_ARTIFACTS}"
    COMMAND
      "$<TARGET_FILE:loom::tools::loom-compile>"
      "${_RULE_LINKED_MODULE}"
      "--product=command"
      "--format=${_FORMAT_NAME}"
      "--output=${_MANIFEST}"
      "--emit-command-artifacts=${_ARTIFACTS}"
      "--compile-report=details"
      "--compile-report-output=${_REPORT}"
    DEPENDS
      loom::tools::loom-compile
      "${_RULE_LINKED_MODULE}"
    COMMENT
      "Compiling Loom command product ${_RULE_NAME}"
    VERBATIM
  )
  set_source_files_properties(
    "${_MANIFEST}"
    "${_REPORT}"
    PROPERTIES GENERATED TRUE
  )
  set(${_RULE_OUT_MANIFEST} "${_MANIFEST}" PARENT_SCOPE)
  set(${_RULE_OUT_ARTIFACTS} "${_ARTIFACTS}" PARENT_SCOPE)
  set(${_RULE_OUT_REPORT} "${_REPORT}" PARENT_SCOPE)
endfunction()

function(_loom_declare_binary_target)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;KIND;PRIMARY_ARTIFACT;LINKED_MODULE;DEPENDENCY_REPORT"
    "ARTIFACTS;REPORTS;DEPENDENCY_TARGETS;LIBRARY_DEPENDENCY_TARGETS"
    ${ARGN}
  )
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_TARGET}" DEPENDS ${_RULE_ARTIFACTS} ${_RULE_REPORTS})
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_BINARY_KIND "${_RULE_KIND}")
  set_property(TARGET "${_TARGET}" PROPERTY
    IREE_GENERATED_FILE "${_RULE_PRIMARY_ARTIFACT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_BINARY_FILE "${_RULE_PRIMARY_ARTIFACT}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_BINARY_ARTIFACTS "${_RULE_ARTIFACTS}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_BINARY_LINKED_MODULE "${_RULE_LINKED_MODULE}")
  set_property(TARGET "${_TARGET}" PROPERTY
    LOOM_BINARY_REPORTS "${_RULE_REPORTS}")
  _loom_register_library_consumer(
    "${_TARGET}" ${_RULE_LIBRARY_DEPENDENCY_TARGETS})
  if(_RULE_DEPENDENCY_REPORT)
    set_property(TARGET "${_TARGET}" PROPERTY
      LOOM_BINARY_DEPENDENCY_REPORT "${_RULE_DEPENDENCY_REPORT}")
  endif()
  foreach(_DEPENDENCY_TARGET IN LISTS _RULE_DEPENDENCY_TARGETS)
    iree_register_target_dependency(
      TARGET "${_TARGET}"
      DEPENDENCY "${_DEPENDENCY_TARGET}"
    )
  endforeach()
  set(_GENERATED_OUTPUTS
    ${_RULE_ARTIFACTS}
    ${_RULE_REPORTS}
    "${_RULE_LINKED_MODULE}"
  )
  if(_RULE_DEPENDENCY_REPORT)
    list(APPEND _GENERATED_OUTPUTS "${_RULE_DEPENDENCY_REPORT}")
  endif()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS ${_GENERATED_OUTPUTS})
  set(_LOOM_DECLARED_BINARY_TARGET "${_TARGET}" PARENT_SCOPE)
endfunction()

function(loom_kernel_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET;FORMAT"
    "SRCS;DEPS;ROOTS;CONFIGS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_kernel_binary ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_NAME OR NOT _RULE_TARGET)
    message(FATAL_ERROR "loom_kernel_binary requires NAME and TARGET")
  endif()
  _loom_resolve_target_product_format(
    _FORMAT
    TARGET "${_RULE_TARGET}"
    PRODUCT "kernel"
    OUTPUT_KIND "file"
    FORMAT "${_RULE_FORMAT}"
  )
  set(_LINKED_MODULE
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.linked.loombc")
  iree_package_name(_PACKAGE_NAME)
  _loom_package_label(_COMPONENT "${_RULE_NAME}")
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  _loom_declare_binary_linked_module(
    NAME "${_RULE_NAME}"
    COMPONENT "${_COMPONENT}"
    OWNER_TARGET "${_TARGET}"
    TARGET "${_RULE_TARGET}"
    OUTPUT "${_LINKED_MODULE}"
    OUT_DEPENDENCY_TARGETS _DEPENDENCY_TARGETS
    OUT_DEPENDENCY_REPORT _DEPENDENCY_REPORT
    OUT_LIBRARY_DEPENDENCY_TARGETS _LIBRARY_DEPENDENCY_TARGETS
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    ROOTS ${_RULE_ROOTS}
    CONFIGS ${_RULE_CONFIGS}
  )
  _loom_declare_kernel_product(
    NAME "${_RULE_NAME}"
    TARGET "${_RULE_TARGET}"
    FORMAT "${_FORMAT}"
    LINKED_MODULE "${_LINKED_MODULE}"
    OUTPUT_STEM "${_RULE_NAME}"
    OUT_ARTIFACT _ARTIFACT
    OUT_REPORT _REPORT
  )
  _loom_declare_binary_target(
    NAME "${_RULE_NAME}"
    KIND "kernel"
    PRIMARY_ARTIFACT "${_ARTIFACT}"
    LINKED_MODULE "${_LINKED_MODULE}"
    DEPENDENCY_REPORT "${_DEPENDENCY_REPORT}"
    ARTIFACTS "${_ARTIFACT}"
    REPORTS "${_REPORT}"
    DEPENDENCY_TARGETS ${_DEPENDENCY_TARGETS}
    LIBRARY_DEPENDENCY_TARGETS ${_LIBRARY_DEPENDENCY_TARGETS}
  )
  set_property(TARGET "${_LOOM_DECLARED_BINARY_TARGET}" PROPERTY
    LOOM_BINARY_KERNEL_FORMAT "${_FORMAT}")
endfunction()

function(loom_command_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET;FORMAT;KERNEL_FORMAT"
    "SRCS;DEPS;ROOTS;CONFIGS"
    ${ARGN}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "loom_command_binary ${_RULE_NAME} has unknown arguments: "
      "${_RULE_UNPARSED_ARGUMENTS}"
    )
  endif()
  if(NOT _RULE_NAME OR NOT _RULE_TARGET)
    message(FATAL_ERROR "loom_command_binary requires NAME and TARGET")
  endif()
  _loom_resolve_target_product_format(
    _COMMAND_FORMAT
    TARGET "${_RULE_TARGET}"
    PRODUCT "command"
    OUTPUT_KIND "artifact_set"
    FORMAT "${_RULE_FORMAT}"
  )
  _loom_resolve_target_product_format(
    _KERNEL_FORMAT
    TARGET "${_RULE_TARGET}"
    PRODUCT "kernel"
    OUTPUT_KIND "file"
    FORMAT "${_RULE_KERNEL_FORMAT}"
  )
  set(_LINKED_MODULE
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.linked.loombc")
  iree_package_name(_PACKAGE_NAME)
  _loom_package_label(_COMPONENT "${_RULE_NAME}")
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  _loom_declare_binary_linked_module(
    NAME "${_RULE_NAME}"
    COMPONENT "${_COMPONENT}"
    OWNER_TARGET "${_TARGET}"
    TARGET "${_RULE_TARGET}"
    OUTPUT "${_LINKED_MODULE}"
    OUT_DEPENDENCY_TARGETS _DEPENDENCY_TARGETS
    OUT_DEPENDENCY_REPORT _DEPENDENCY_REPORT
    OUT_LIBRARY_DEPENDENCY_TARGETS _LIBRARY_DEPENDENCY_TARGETS
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    ROOTS ${_RULE_ROOTS}
    CONFIGS ${_RULE_CONFIGS}
  )
  _loom_declare_command_product(
    NAME "${_RULE_NAME}"
    FORMAT "${_COMMAND_FORMAT}"
    LINKED_MODULE "${_LINKED_MODULE}"
    OUT_MANIFEST _MANIFEST
    OUT_ARTIFACTS _COMMAND_ARTIFACTS
    OUT_REPORT _COMMAND_REPORT
  )
  _loom_declare_kernel_product(
    NAME "${_RULE_NAME} kernels"
    TARGET "${_RULE_TARGET}"
    FORMAT "${_KERNEL_FORMAT}"
    LINKED_MODULE "${_LINKED_MODULE}"
    OUTPUT_STEM "${_RULE_NAME}.kernels"
    OUT_ARTIFACT _KERNEL_ARTIFACT
    OUT_REPORT _KERNEL_REPORT
  )
  _loom_declare_binary_target(
    NAME "${_RULE_NAME}"
    KIND "command"
    PRIMARY_ARTIFACT "${_MANIFEST}"
    LINKED_MODULE "${_LINKED_MODULE}"
    DEPENDENCY_REPORT "${_DEPENDENCY_REPORT}"
    ARTIFACTS "${_MANIFEST}" "${_COMMAND_ARTIFACTS}" "${_KERNEL_ARTIFACT}"
    REPORTS "${_COMMAND_REPORT}" "${_KERNEL_REPORT}"
    DEPENDENCY_TARGETS ${_DEPENDENCY_TARGETS}
    LIBRARY_DEPENDENCY_TARGETS ${_LIBRARY_DEPENDENCY_TARGETS}
  )
  set_property(TARGET "${_LOOM_DECLARED_BINARY_TARGET}" PROPERTY
    LOOM_BINARY_COMMAND_FORMAT "${_COMMAND_FORMAT}")
  set_property(TARGET "${_LOOM_DECLARED_BINARY_TARGET}" PROPERTY
    LOOM_BINARY_COMMAND_ARTIFACT_DIRECTORY "${_COMMAND_ARTIFACTS}")
  set_property(TARGET "${_LOOM_DECLARED_BINARY_TARGET}" PROPERTY
    LOOM_BINARY_KERNEL_FORMAT "${_KERNEL_FORMAT}")
  set_property(TARGET "${_LOOM_DECLARED_BINARY_TARGET}" PROPERTY
    LOOM_BINARY_KERNEL_FILE "${_KERNEL_ARTIFACT}")
endfunction()
