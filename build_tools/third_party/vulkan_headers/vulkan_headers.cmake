# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(_iree_configure_vulkan_headers_from_include_dir include_dir)
  if(NOT TARGET iree_vulkan_headers)
    add_library(iree_vulkan_headers INTERFACE)
    target_include_directories(iree_vulkan_headers SYSTEM INTERFACE
      "${include_dir}")
  endif()
  if(NOT TARGET Vulkan::Headers)
    add_library(Vulkan::Headers ALIAS iree_vulkan_headers)
  endif()
endfunction()

function(iree_configure_vulkan_headers)
  if(TARGET iree::third_party::vulkan_headers)
    return()
  endif()

  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(VulkanHeaders CONFIG QUIET)
    if(NOT TARGET Vulkan::Headers)
      find_package(Vulkan QUIET)
    endif()
    if(NOT TARGET Vulkan::Headers AND Vulkan_INCLUDE_DIRS)
      _iree_configure_vulkan_headers_from_include_dir("${Vulkan_INCLUDE_DIRS}")
    endif()
  endif()

  if(NOT TARGET Vulkan::Headers)
    iree_dependency_require_pinned_source_allowed("Vulkan-Headers")
    iree_declare_locked_fetch_content(vulkan_headers)
    FetchContent_MakeAvailable(vulkan_headers)
  endif()

  if(NOT TARGET Vulkan::Headers)
    message(FATAL_ERROR "Vulkan headers did not provide Vulkan::Headers")
  endif()
  iree_add_alias_interface(
    iree::third_party::vulkan_headers Vulkan::Headers)
endfunction()
