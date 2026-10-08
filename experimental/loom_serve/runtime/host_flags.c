// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/host_flags.h"

#include "iree/base/tooling/flags.h"

IREE_FLAG(int32_t, workers, 8,
          "Shared CPU workers for compilation, scheduling and media jobs.");

iree_status_t loom_serve_host_create_from_flags(loom_serve_host_t** out_host,
                                                iree_allocator_t allocator) {
  return loom_serve_host_create((iree_host_size_t)FLAG_workers, out_host,
                                allocator);
}
