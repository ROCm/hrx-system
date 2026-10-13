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

// CLI leaf constructing a single-device application group from the required
// device URI, backing strategy and physical budget flags. Models borrow its
// first record; the entry point destroys the group after every model retires.
iree_status_t loom_serve_device_group_create_from_flags(
    loom_serve_host_t* host, loom_serve_device_group_t** out_group,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_FLAGS_H_
