# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

# IREE textually compiles the pinned client with its own tracing configuration
# and worker lifecycle patches. A separately compiled Tracy package cannot
# provide that implementation contract.
iree_dependency_require_pinned_source_allowed("tracy_client")
iree_populate_locked_fetch_content(tracy_client _tracy_client_source_dir)
add_library(tracy_client_runtime INTERFACE)
target_include_directories(tracy_client_runtime SYSTEM INTERFACE
  "${_tracy_client_source_dir}/public")
add_library(tracy_client::runtime ALIAS tracy_client_runtime)
