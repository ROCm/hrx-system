// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_FLAGS_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_FLAGS_H_

#include "loom/serve/runtime/host/host.h"

#ifdef __cplusplus
extern "C" {
#endif

// CLI leaf creating shared application workers and I/O infrastructure.
iree_status_t loom_serve_host_create_from_flags(loom_serve_host_t** out_host,
                                                iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_FLAGS_H_
