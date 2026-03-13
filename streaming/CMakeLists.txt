# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    streaming
  HDRS
    "internal.h"
  SRCS
    "context.c"
    "device.c"
    "event.c"
    "graph.c"
    "graph.h"
    "graph_analysis.c"
    "graph_exec.c"
    "init.c"
    "mem_pool.c"
    "memory.c"
    "module.c"
    "peer.c"
    "registry.c"
    "stream.c"
  COPTS
    "-fPIC"
  DEPS
    ::internal
    iree::base
    iree::base::internal
    iree::base::internal::arena
    iree::base::threading
    streaming::util
    iree::hal
    iree::hal::drivers
    iree::hal::utils::resource_set
  PUBLIC
)

iree_cc_library(
  NAME
    internal
  HDRS
    "internal.h"
  DEPS
    iree::base
    iree::base::internal::arena
    iree::base::threading
    iree::hal
  PUBLIC
)
