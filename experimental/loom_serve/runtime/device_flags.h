// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_FLAGS_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_FLAGS_H_

#include "experimental/loom_serve/runtime/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// CLI leaf selecting the shared device, backing strategy and physical budget.
// Model constructors borrow this owner; the entry point destroys it last.
iree_status_t loom_serve_device_create_from_flags(
    loom_serve_device_t** out_device, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_FLAGS_H_
