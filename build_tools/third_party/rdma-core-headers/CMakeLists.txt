# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

if(TARGET rdma_core::headers)
  return()
endif()

iree_dependency_require_pinned_source_allowed("rdma_core_headers")
iree_populate_locked_fetch_content(rdma_core_headers RDMA_CORE_HEADERS_ROOT)

add_library(rdma_core_headers INTERFACE)
target_include_directories(rdma_core_headers SYSTEM
  INTERFACE
    "$<BUILD_INTERFACE:${RDMA_CORE_HEADERS_ROOT}/include>"
)
add_library(rdma_core::headers ALIAS rdma_core_headers)
if(COMMAND iree_record_target_alias)
  iree_record_target_alias(rdma_core::headers rdma_core_headers)
endif()
