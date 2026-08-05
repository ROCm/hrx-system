# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_library(
  NAME
    dependency_expression
  SRCS
    "__init__.py"
    "dependency_expression.py"
)

iree_py_library(
  NAME
    registry_source
  SRCS
    "__init__.py"
    "registry_source.py"
)

iree_py_library(
  NAME
    spirv_core_grammar
  SRCS
    "spirv_core_grammar.py"
  DEPS
    ::registry_source
)

iree_py_test(
  NAME
    dependency_expression_test
  SRCS
    "dependency_expression_test.py"
  DEPS
    ::dependency_expression
  PACKAGE_DIRS
    "${PROJECT_SOURCE_DIR}"
)

iree_py_test(
  NAME
    spirv_core_grammar_test
  SRCS
    "spirv_core_grammar_test.py"
  DEPS
    ::registry_source
    ::spirv_core_grammar
  PACKAGE_DIRS
    "${PROJECT_SOURCE_DIR}"
)

iree_dependency_pinned_source_allowed(_pinned_source_allowed)
if(IREE_BUILD_TESTS AND _pinned_source_allowed AND
    (IREE_HAL_DRIVER_VULKAN OR
      (LOOM_BUILD AND LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV)))
  iree_populate_locked_fetch_content(
    spirv_headers _spirv_headers_source_dir)
  set(_spirv_core_grammar
    "${_spirv_headers_source_dir}/include/spirv/unified1/spirv.core.grammar.json")
  if(NOT EXISTS "${_spirv_core_grammar}")
    message(FATAL_ERROR
      "Pinned SPIRV-Headers source has no core grammar: "
      "${_spirv_headers_source_dir}")
  endif()
  iree_py_test(
    NAME
      spirv_core_grammar_registry_test
    SRCS
      "spirv_core_grammar_registry_test.py"
    ARGS
      "${_spirv_core_grammar}"
    DEPS
      ::spirv_core_grammar
    PACKAGE_DIRS
      "${PROJECT_SOURCE_DIR}"
  )

  iree_populate_locked_fetch_content(
    vulkan_headers _vulkan_headers_source_dir)
  set(_vulkan_xml_registry
    "${_vulkan_headers_source_dir}/registry/vk.xml")
  if(NOT EXISTS "${_vulkan_xml_registry}")
    message(FATAL_ERROR
      "Pinned Vulkan-Headers source has no registry/vk.xml: "
      "${_vulkan_headers_source_dir}")
  endif()
  iree_py_test(
    NAME
      dependency_expression_registry_test
    SRCS
      "dependency_expression_registry_test.py"
    ARGS
      "${_vulkan_xml_registry}"
    DEPS
      ::dependency_expression
    PACKAGE_DIRS
      "${PROJECT_SOURCE_DIR}"
  )
endif()
