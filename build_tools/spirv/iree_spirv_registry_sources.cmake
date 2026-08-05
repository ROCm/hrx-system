# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# Returns the exact locked Khronos source files used by SPIR-V generators.
function(iree_get_spirv_registry_sources out_spirv_grammar out_vulkan_registry)
  if(NOT out_spirv_grammar OR NOT out_vulkan_registry)
    message(FATAL_ERROR
      "iree_get_spirv_registry_sources requires two output variables")
  endif()

  iree_dependency_require_pinned_source_allowed("SPIR-V registry generation")
  iree_populate_locked_fetch_content(
    spirv_headers _spirv_headers_source_dir)
  set(_spirv_grammar
    "${_spirv_headers_source_dir}/include/spirv/unified1/spirv.core.grammar.json")
  if(NOT EXISTS "${_spirv_grammar}")
    message(FATAL_ERROR
      "Pinned SPIRV-Headers source has no core grammar: "
      "${_spirv_headers_source_dir}")
  endif()

  iree_populate_locked_fetch_content(
    vulkan_headers _vulkan_headers_source_dir)
  set(_vulkan_registry
    "${_vulkan_headers_source_dir}/registry/vk.xml")
  if(NOT EXISTS "${_vulkan_registry}")
    message(FATAL_ERROR
      "Pinned Vulkan-Headers source has no registry/vk.xml: "
      "${_vulkan_headers_source_dir}")
  endif()

  set(${out_spirv_grammar} "${_spirv_grammar}" PARENT_SCOPE)
  set(${out_vulkan_registry} "${_vulkan_registry}" PARENT_SCOPE)
endfunction()
