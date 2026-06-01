# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

set(_IREE_LIBBACKTRACE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(iree_runtime_configure_libbacktrace)
  if(NOT IREE_ENABLE_LIBBACKTRACE)
    set(IREE_LIBBACKTRACE_TARGET "" CACHE INTERNAL
      "libbacktrace target (empty when disabled)" FORCE)
    return()
  endif()
  iree_fetch_content_assert_allowed("libbacktrace")
  add_subdirectory(
    "${_IREE_LIBBACKTRACE_CMAKE_DIR}"
    "${CMAKE_BINARY_DIR}/runtime/build_tools/third_party/libbacktrace"
    EXCLUDE_FROM_ALL
  )
endfunction()
